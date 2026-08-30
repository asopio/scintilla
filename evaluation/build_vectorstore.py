from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "ichep_2026"
DEFAULT_MD_DIR = DEFAULT_DOCS_DIR / "md"
DEFAULT_VECTORSTORE_DIR = PROJECT_ROOT / "evaluation" / "vectorstore" / "ichep_2026"


def read_md_documents(md_dir: Path) -> List[Document]:
    documents: List[Document] = []
    for md_path in sorted(md_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": md_path.name,
                    "path": str(md_path),
                    "input_type": "markdown",
                },
            )
        )
    return documents


def read_pdf_documents(docs_dir: Path) -> List[Document]:
    documents: List[Document] = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number,
                            "path": str(pdf_path),
                            "input_type": "raw_pdf",
                        },
                    )
                )
        documents.extend(pages)
    return documents


def build_vectorstore(
    docs_dir: Path,
    output_dir: Path,
    embeddings_model: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    raw_pdf_inputs: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw_pdf_inputs:
        documents = read_pdf_documents(docs_dir)
    else:
        md_dir = docs_dir / "md" if docs_dir == DEFAULT_DOCS_DIR or (docs_dir / "md").exists() else docs_dir
        documents = read_md_documents(md_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(output_dir),
    )
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a persisted Chroma vector store for ICHEP 2026 documents")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VECTORSTORE_DIR)
    parser.add_argument("--embeddings-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--raw-pdf-inputs", action="store_true")
    args = parser.parse_args()

    path = build_vectorstore(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
        embeddings_model=args.embeddings_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        raw_pdf_inputs=args.raw_pdf_inputs,
    )
    print(path)
