from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    from deepagents import create_deep_agent
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_chroma import Chroma
    OPIK_DEPS_AVAILABLE = True
except ImportError:
    create_deep_agent = None
    AIMessage = None
    BaseMessage = None
    HumanMessage = None
    ChatOpenAI = None
    OpenAIEmbeddings = None
    Chroma = None
    OPIK_DEPS_AVAILABLE = False

try:
    import opik
    from opik import configure
    from opik.evaluation.metrics import AnswerRelevance, GEval, Hallucination
    OPIK_AVAILABLE = True
except ImportError:
    opik = None
    OPIK_AVAILABLE = False

logger = logging.getLogger(__name__)
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VORTEX_PATH = PROJECT_ROOT / "ichep2026.vortex"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "ichep_2026"
DEFAULT_VECTORSTORE_DIR = PROJECT_ROOT / "evaluation" / "vectorstore" / "ichep_2026"
DEFAULT_BENCHMARK_CSV = PROJECT_ROOT / "evaluation" / "benchmarks" / "ichep_2026_questions.csv"


class BenchmarkResult(BaseModel):
    prompt: str
    expected_answer: str
    agent_response: str
    scores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        values = [score.get("value", 0.0) for score in self.scores.values()]
        return sum(values) / len(values) if values else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "agent_response": self.agent_response,
            "scores": self.scores,
            "average_score": self.average_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class SearchRunResult:
    answer: str
    tool_calls: int
    context_length_chars: int
    latency_seconds: float
    raw_output: Dict[str, Any]


class BenchmarkEvaluator:
    def __init__(
        self,
        judge_model: str = "gpt-5-mini",
        use_opik: bool = True,
        project_name: str = "deepagents-benchmark",
    ):
        self.judge_model = judge_model
        self.use_opik = use_opik and OPIK_AVAILABLE
        self.project_name = project_name
        if self.use_opik:
            configure()
        self.metrics = self._build_metrics()

    def _build_metrics(self) -> List[Any]:
        if not self.use_opik:
            return []
        return [
            AnswerRelevance(model=self.judge_model, project_name=self.project_name),
            Hallucination(model=self.judge_model, project_name=self.project_name),
            GEval(
                name="g_eval_grounded_correctness",
                model=self.judge_model,
                project_name=self.project_name,
                task_introduction=(
                    "You are grading answers from retrieval agents over ICHEP 2026 presentation documents."
                ),
                evaluation_criteria=(
                    "Score the answer from 0 to 1 based on factual correctness, completeness, and grounding in the expected answer. "
                    "Prefer answers that directly answer the question, preserve important qualifiers, and avoid unsupported claims."
                ),
            ),
        ]

    def evaluate_response(self, prompt: str, expected_answer: str, agent_response: str) -> Dict[str, Dict[str, Any]]:
        if not self.metrics:
            return {}
        dataset_item = {
            "input": prompt,
            "expected_output": expected_answer,
            "output": agent_response,
        }
        scores: Dict[str, Dict[str, Any]] = {}
        for metric in self.metrics:
            score = metric.score(dataset_item)
            scores[score.name] = {
                "value": float(score.value),
                "reason": getattr(score, "reason", "") or "",
            }
        return scores

    def print_summary(self, results: List[BenchmarkResult]) -> None:
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return
        metric_scores: Dict[str, List[float]] = {}
        for result in results:
            for metric_name, score_data in result.scores.items():
                metric_scores.setdefault(metric_name, []).append(score_data["value"])
        avg_scores = {
            name: sum(values) / len(values)
            for name, values in metric_scores.items()
            if values
        }
        overall_avg = sum(r.average_score for r in results) / len(results)
        avg_latency = sum(r.metadata.get("latency_seconds", 0.0) for r in results) / len(results)
        avg_context = sum(r.metadata.get("context_length_chars", 0) for r in results) / len(results)

        table = Table(title="Benchmark Evaluation Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right", style="green")
        table.add_row("Overall Average", f"{overall_avg:.3f}")
        for metric_name, avg_score in sorted(avg_scores.items()):
            table.add_row(metric_name, f"{avg_score:.3f}")
        table.add_row("Avg Latency (s)", f"{avg_latency:.3f}")
        table.add_row("Avg Context (chars)", f"{avg_context:.0f}")
        table.add_row("Total Items", str(len(results)))
        console.print(table)


def count_tool_calls(messages: List[BaseMessage]) -> int:
    total = 0
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            total += len(tool_calls)
    return total


def build_model(model_name: str):
    if ChatOpenAI is None:
        raise RuntimeError("LangChain/OpenAI dependencies are not installed")
    return ChatOpenAI(model=model_name, temperature=0)


@tool
def sql_vortex_search(question: str, limit: int = 8, vortex_path: str = str(DEFAULT_VORTEX_PATH)) -> str:
    """Search the ICHEP 2026 Vortex slide database with SQL and return matching rows."""
    question_lit = question.lower().replace("'", "''")
    sql = f"""
    SELECT stem, slide_number, contribution_title, author, extracted_text, visual_description
    FROM data
    WHERE lower(extracted_text) LIKE '%{question_lit}%'
       OR lower(visual_description) LIKE '%{question_lit}%'
       OR lower(contribution_title) LIKE '%{question_lit}%'
       OR lower(author) LIKE '%{question_lit}%'
    LIMIT {int(limit)}
    """.strip()
    cmd = [
        "pixi",
        "run",
        "vx",
        "query",
        "--output",
        "json",
        "--sql",
        sql,
        vortex_path,
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


class RagSearchToolFactory:
    def __init__(self, vectorstore_dir: Path, embeddings_model: str):
        self.vectorstore_dir = vectorstore_dir
        self.embeddings_model = embeddings_model
        self._vectorstore: Optional[Chroma] = None

    def _get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            embeddings = OpenAIEmbeddings(model=self.embeddings_model)
            self._vectorstore = Chroma(
                persist_directory=str(self.vectorstore_dir),
                embedding_function=embeddings,
            )
        return self._vectorstore

    def build_tool(self):
        vectorstore_dir = self.vectorstore_dir

        @tool
        def rag_vector_search(question: str, k: int = 6) -> str:
            """Search the ICHEP 2026 vector store and return the most relevant chunks."""
            vectorstore = self._get_vectorstore()
            docs = vectorstore.similarity_search(question, k=int(k))
            payload = []
            for doc in docs:
                payload.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                )
            return json.dumps({"vectorstore_dir": str(vectorstore_dir), "matches": payload}, ensure_ascii=False)

        return rag_vector_search


def build_search_agent(search_mode: str, llm_model: str, vectorstore_dir: Path, embeddings_model: str):
    model = build_model(llm_model)
    if create_deep_agent is None:
        raise RuntimeError("deepagents is not installed")

    if search_mode == "sql":
        tools = [sql_vortex_search]
        system_prompt = (
            "You answer questions about ICHEP 2026 presentation documents. "
            "Use the sql_vortex_search tool to gather evidence before answering. "
            "Summarize the answer concisely and cite slide identifiers when possible."
        )
    elif search_mode == "rag":
        tools = [RagSearchToolFactory(vectorstore_dir, embeddings_model).build_tool()]
        system_prompt = (
            "You answer questions about ICHEP 2026 presentation documents. "
            "Use the rag_vector_search tool to gather relevant chunks before answering. "
            "Summarize the answer concisely and cite source metadata when possible."
        )
    else:
        raise ValueError(f"Unsupported search mode: {search_mode}")

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        name=f"ichep_2026_{search_mode}_agent",
    )


async def run_agent_once(agent: Any, prompt: str) -> SearchRunResult:
    start = time.perf_counter()
    response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    latency_seconds = time.perf_counter() - start
    messages: List[BaseMessage] = response["messages"]
    tool_calls = count_tool_calls(messages)
    answer = ""
    context_length_chars = 0
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            context_length_chars += len(content)
        elif isinstance(content, list):
            context_length_chars += len(json.dumps(content, ensure_ascii=False))
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            answer = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
            break
    return SearchRunResult(
        answer=answer,
        tool_calls=tool_calls,
        context_length_chars=context_length_chars,
        latency_seconds=latency_seconds,
        raw_output=response,
    )


async def run_benchmark(
    csv_file: Path,
    judge_model: str = "gpt-5-mini",
    output_file: Optional[Path] = None,
    max_items: Optional[int] = None,
    use_opik: bool = True,
    experiment_name: Optional[str] = None,
    project_name: str = "deepagents-benchmark",
    llm_model: str = "gpt-5-mini",
    embeddings_model: str = "text-embedding-3-small",
    vectorstore_dir: Path = DEFAULT_VECTORSTORE_DIR,
) -> List[BenchmarkResult]:
    if not OPIK_DEPS_AVAILABLE:
        raise RuntimeError("Required evaluation dependencies are not installed")

    df = pd.read_csv(csv_file)
    if max_items is not None:
        df = df.head(max_items)
    dataset = df.to_dict(orient="records")

    evaluator = BenchmarkEvaluator(
        judge_model=judge_model,
        use_opik=use_opik,
        project_name=project_name,
    )

    sql_agent = build_search_agent("sql", llm_model, vectorstore_dir, embeddings_model)
    rag_agent = build_search_agent("rag", llm_model, vectorstore_dir, embeddings_model)

    results: List[BenchmarkResult] = []
    experiment_name = experiment_name or f"ichep_2026_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task_id = progress.add_task("Running benchmark", total=len(dataset) * 2)
        for row in dataset:
            prompt = row["prompt"]
            expected_answer = row["expected_answer"]
            for search_mode, agent in (("sql", sql_agent), ("rag", rag_agent)):
                run_result = await run_agent_once(agent, prompt)
                scores = evaluator.evaluate_response(prompt, expected_answer, run_result.answer)
                result = BenchmarkResult(
                    prompt=prompt,
                    expected_answer=expected_answer,
                    agent_response=run_result.answer,
                    scores=scores,
                    metadata={
                        "search_mode": search_mode,
                        "experiment_name": experiment_name,
                        "latency_seconds": run_result.latency_seconds,
                        "tool_calls": run_result.tool_calls,
                        "context_length_chars": run_result.context_length_chars,
                    },
                )
                results.append(result)
                progress.advance(task_id)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=False))

    evaluator.print_summary(results)
    return results


def generate_questions_from_documents(
    docs_dir: Path,
    output_csv: Path,
    model_name: str,
    max_documents: int = 24,
    questions_per_document: int = 2,
) -> Path:
    pdf_paths = sorted(docs_dir.glob("*.pdf"))[:max_documents]
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")

    snippets = []
    for pdf_path in pdf_paths:
        text = extract_pdf_text(pdf_path, max_chars=6000)
        snippets.append(
            {
                "source": pdf_path.name,
                "text": text,
            }
        )

    model = build_model(model_name)
    prompt = (
        "Generate benchmark QA pairs for evaluating retrieval agents over ICHEP 2026 slides. "
        f"Create exactly {questions_per_document} questions per document snippet. "
        "Questions should be answerable from the snippet, specific, and varied across facts, methods, goals, or results. "
        "Return JSON list items with keys prompt, expected_answer, source_document.\n\n"
        f"Snippets:\n{json.dumps(snippets, ensure_ascii=False)}"
    )
    response = model.invoke(prompt)
    content = response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)
    qa_pairs = json.loads(extract_json_block(content))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "expected_answer", "source_document"])
        writer.writeheader()
        for item in qa_pairs:
            writer.writerow(item)
    return output_csv


def extract_json_block(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in model output")
    return text[start:end + 1]


def extract_pdf_text(pdf_path: Path, max_chars: int = 6000) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    chunks: List[str] = []
    for page in reader.pages[:6]:
        chunks.append(page.extract_text() or "")
        if sum(len(chunk) for chunk in chunks) >= max_chars:
            break
    return "\n".join(chunks)[:max_chars]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge benchmark for SQL vs RAG deepagents")
    parser.add_argument("--csv-file", type=Path, default=DEFAULT_BENCHMARK_CSV)
    parser.add_argument("--output-file", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "ichep_2026_eval.json")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--embeddings-model", default="text-embedding-3-small")
    parser.add_argument("--vectorstore-dir", type=Path, default=DEFAULT_VECTORSTORE_DIR)
    parser.add_argument("--generate-questions", action="store_true")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--no-opik", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.generate_questions:
        generate_questions_from_documents(
            docs_dir=args.docs_dir,
            output_csv=args.csv_file,
            model_name=args.llm_model,
        )

    asyncio.run(
        run_benchmark(
            csv_file=args.csv_file,
            judge_model=args.judge_model,
            output_file=args.output_file,
            max_items=args.max_items,
            use_opik=not args.no_opik,
            llm_model=args.llm_model,
            embeddings_model=args.embeddings_model,
            vectorstore_dir=args.vectorstore_dir,
        )
    )
