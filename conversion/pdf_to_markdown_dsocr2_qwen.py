#!/usr/bin/env python3
"""DSOCR2 OCR + Qwen3.8 collation and review: PDF -> canonical Markdown.

Combines the two-model workflow from
``plan_dsocr_qwen38_combined_workflow.md`` into one entry point:

  Stage 1   page render + DSOCR2 per-page grounding OCR + figure crops
            (reused from ``pdf_to_markdown_dsocr2.py``; DSOCR2 pod,
            default ``http://127.0.0.1:18000/v1``)
  Stage 1.5 ``dsocr2_postprocess.postprocess()`` -> ``md/<stem>_dsocr2.md``
  Stage 2   Qwen3.8 collation (``:8008``).  The document is split into
            page-bounded jobs of at most ``--max-pages`` (default 40);
            each job is further split by a MANDATORY client-side token
            budget guard (the server admits oversized calls and fails
            late — feasibility study, finding 3).  Two contracts:

            * ``diff`` (default): the model emits exact find/replace
              edits as JSON lines; the client applies them by exact
              substring match within the page section.  Output budget
              16384 tokens -> ~3-5 min per 120k-token doc at the
              measured 24.5 tok/s decode.
            * ``full``: the model re-emits the whole chunk.  Output
              budget ceil(1.1 x input) + 4096.  Fallback when diff
              application rate < 95% (or a diff guard fails).

            Each chunk is written to ``md_chunks/NN_<stem>.md``.
  Stage 3   Qwen3.8 agentic review per chunk (DeepAgents, reusing
            ``create_review_agent`` / ``validate_markdown`` from
            ``pdf_to_markdown_agentic.py``), driven by deterministic
            extended validation (markdown lib, pandoc, math balance,
            KaTeX compile, table columns, heading ladder, figure refs
            vs manifest, OCR-artifact scan).  The agent's
            validate_markdown tool also compiles every math span with
            the ``markdown_katex`` package's KaTeX CLI (per formula,
            not a full ``markdown`` conversion) so it sees parse
            errors directly.  Each chunk's full agent conversation
  is traced to ``review_traces/NN_<stem>.jsonl`` via the LangChain
  callback API (LLM prompts/responses, tool calls).
  Stage 4   deterministic merge (no LLM call) -> ``md/<stem>_qwen_final.md``
            + global checks + ``collate_report.json``.

Defaults come from the feasibility study
(``feasibility_qwen38_context_window.md``): native 262144-token window,
``enable_thinking`` off for collation (thinking burns the output
budget), ``temperature=0`` everywhere in Stage 2.

Usage:
  cd conversion
  pixi run python pdf_to_markdown_dsocr2_qwen.py input.pdf -o out/
  pixi run python pdf_to_markdown_dsocr2_qwen.py input.pdf --collate-mode full
  pixi run python pdf_to_markdown_dsocr2_qwen.py already_ocred.pdf --skip-ocr

Resume: stage 1 caches in ``pages/`` + ``raw_pages/`` (existing
semantics); stage 2 skips chunks whose ``md_chunks/NN_<stem>.md``
already exists; stage 3 skips chunks that already pass the extended
validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import httpx
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

import dsocr2_postprocess
from pdf_to_markdown_agentic import (
    WORKSPACE_MOUNT,
    _validate_markdown_text,
    create_review_agent,
    get_pdf_page_count,
)
from pdf_to_markdown_dsocr2 import (
    build_raw_dump,
    crop_figures,
    ensure_page_images,
    ocr_pages,
    page_width,
)

# ---------------------------------------------------------------------------
# Constants (feasibility study numbers — see plan §4)
# ---------------------------------------------------------------------------

DIFF_OUTPUT_BUDGET = 16384          # tokens, diff-mode output cap
FULL_OUTPUT_FACTOR = 1.1             # full-mode output ~= 1.1 x input
FULL_OUTPUT_MARGIN = 4096            # + headroom
RETRY_OUTPUT_BUDGET = 4096           # per-page diff retry
MIN_DIFF_APPLY_RATE = 0.95           # else fall back to full for the chunk
DIFF_WORD_DELTA = (-0.02, 0.05)      # post-hoc guard
FULL_WORD_DELTA = (-0.05, 0.15)      # post-hoc guard

DOC_OPEN = "<<<document>>>"
DOC_CLOSE = "<<<end document>>>"

DIFF_SYSTEM = (
    "You are a scientific document editor. Review the OCR'd document "
    "section below (slides {a}-{b} of a larger document) and fix only: "
    "OCR artifacts (garbled, split, or duplicated text), cross-page "
    "inconsistencies (heading numbering, style), pipe tables broken "
    "across page breaks, and unbalanced $ / $$ math delimiters. Do NOT "
    "restructure, reword, or translate. Do NOT change any '## Slide N' "
    "heading, do NOT change any ![Figure k](...) image reference, and do "
    "NOT change or drop the '<!-- Figure k, description: ... -->' comment "
    "line that follows an image reference (it is figure metadata).\n\n"
    "Output ONLY a list of edits, one JSON object per line:\n"
    '{{"page": N, "find": "<exact span copied verbatim from the document>", '
    '"replace": "<corrected span>"}}\n'
    "Each \"find\" must be copied character-for-character from the "
    "document and must occur within the section of page N. Keep edits "
    "minimal and precise. No code fences, no preamble, no commentary. "
    "If nothing needs fixing, output exactly:\nNO_EDITS"
)

FULL_SYSTEM = (
    "You are a scientific document editor. Convert the OCR'd document "
    "section below into one clean, canonical Markdown document.\n\n"
    "Hard rules:\n"
    "1. Preserve every word of the body text; fix only OCR artifacts "
    "(garbled tokens, words split across line breaks, duplicated "
    "lines).\n"
    "2. Keep every '## Slide N' heading and their order exactly as "
    "given; never drop content; do not add content (no title, no "
    "summaries, no notes).\n"
    "3. Keep every ![Figure k](...) image reference byte-identical, and "
    "keep the '<!-- Figure k, description: ... -->' comment line that "
    "follows it (metadata; do not reword or drop it).\n"
    "4. Keep $...$ and $$...$$ math; only fix unbalanced delimiters.\n"
    "5. Pipe tables: keep columns and rows; fix misaligned cells and "
    "broken '| --- |' separator lines; if a table is split across a "
    "page break, merge it into one table.\n"
    "6. Keep the page separator exactly as in the input: blank line, "
    "'---', blank line, between slides.\n"
    "7. Output the full document only - no code fences, no preamble, "
    "no commentary."
)

RAW_FULL_SYSTEM = FULL_SYSTEM + (
    "\n\nThe input is a RAW OCR dump. Additional conversion rules:\n"
    "A. Replace each '<!-- ==================== PAGE N ==================== -->' "
    "marker with the heading '## Slide N' (drop the marker comment).\n"
    "B. Drop every '<|ref|>...<|/ref|><|det|>...<|/det|>' marker line; the "
    "coordinates carry no text.\n"
    "C. Where the dump has an image block, insert the figure reference "
    "given in the list after the document (the list is in document "
    "order; one reference per image block).\n"
    "D. Convert HTML <table> blocks into pipe tables and convert "
    "LaTeX-style math delimiters (\\( \\), \\[ \\]) to $ / $$."
)

RETRY_SYSTEM = (
    "You are a scientific document editor. You are given the exact text "
    "of one page (slide) of an OCR'd document, plus a list of intended "
    "find/replace edits whose 'find' spans did not match the page text. "
    "Output corrected edits, one JSON object per line: "
    '{"page": N, "find": "<exact span copied verbatim from the page '
    'text>", "replace": "<corrected span>"}. Each "find" must now match '
    "the page text character-for-character. If an intended edit is "
    "already correct in the page text, omit it. No code fences, no "
    "commentary. If no edits apply, output exactly:\nNO_EDITS"
)

REVIEW_SYSTEM_PROMPT = (
    "You are a scientific Markdown auditor and fixer for OCR'd "
    "documents. You receive one page-range chunk of a document "
    "converted from PDF by an OCR model. It must validate cleanly and "
    "stay faithful to the OCR content.\n\n"
    "Hard rules:\n"
    "- Preserve every word of the body text. Never add, remove, or "
    "reword content — except the two OCR failure artifacts listed "
    "below, which are model errors, not content.\n"
    "- Keep every \"## Slide N\" heading exactly as is; never add or "
    "drop slides.\n"
    "- Keep every ![Figure k](...) image reference byte-identical, and "
    "keep the '<!-- Figure k, description: ... -->' comment line that "
    "follows it (metadata, not content; never reword or drop it).\n"
    "- Keep $...$ and $$...$$ math; only fix unbalanced delimiters.\n"
    "- Pipe tables: keep rows and columns; only fix syntax (separator "
    "lines, escaped pipes, column counts).\n"
    "- Make targeted edits with edit_file on specific spans; never "
    "rewrite the whole file.\n\n"
    "Permitted OCR failure repairs (model errors, not content — use "
    "only for the two cases below, never to drop distinct content):\n"
    "1. Repeated-line loops: the same non-empty line emitted 3 or more "
    "times in a row (an OCR decoding loop) — collapse the run to a "
    "single copy.\n"
    "2. A pipe table corrupted beyond repair (mismatched column "
    "counts, loop debris, empty/missing cells) — replace the entire "
    "table with the single line '<!-- unparseable table -->', the "
    "established placeholder for this document family.\n\n"
    "You have tools: read_file, edit_file, validate_markdown (a Python "
    "markdown parser check that also compiles every $...$ / $$...$$ "
    "formula with KaTeX and reports any parse error). The figure "
    "manifest JSON path is given in the task. All file paths are "
    "relative to your /workspace directory.\n\n"
    "Workflow:\n"
    "1. read_file the target file.\n"
    "2. Fix each item in the deterministic problem list given in the "
    "task.\n"
    "3. Call validate_markdown; fix anything it reports.\n"
    "4. Repeat until validate_markdown returns 'VALIDATION PASSED' and "
    "every deterministic problem is resolved."
)

SLIDE_RE = re.compile(r"^## Slide (\d+)\s*$", re.M)
RAW_PAGE_RE = re.compile(
    r"^<!--\s*=+\s*PAGE (\d+)\s*=+\s*-->\s*$", re.M)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
TRAILING_SEP_RE = re.compile(r"\n*-{3,}\s*$")

RETRY_TMP = Path("/tmp/opencode/qwen_collate_retries")


# ---------------------------------------------------------------------------
# Token counting (budget guard — mandatory, plan §5.1)
# ---------------------------------------------------------------------------

class TokenCounter:
    """Count Qwen3.8 tokens: local tokenizer first (verified identical
    to the server), server ``/tokenize`` second, char/2.5 heuristic
    last (conservative for LaTeX-heavy OCR text).
    """

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._tk = None
        self._local_tried = False

    def _load_local(self) -> None:
        if self._local_tried:
            return
        self._local_tried = True
        try:
            from transformers import AutoTokenizer
            self._tk = AutoTokenizer.from_pretrained(self.model)
            print("  tokenizer: local transformers")
        except Exception as exc:
            print(f"  warning: local tokenizer unavailable ({exc}); "
                  f"will use server /tokenize", file=sys.stderr)

    def _server_count(self, text: str) -> int | None:
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(
                    f"{self.base_url}/tokenize",
                    json={"model": self.model, "prompt": text})
                if r.status_code == 200:
                    return int(r.json()["count"])
        except Exception:
            pass
        return None

    def __call__(self, text: str) -> int:
        self._load_local()
        if self._tk is not None:
            return len(self._tk.encode(text, add_special_tokens=False))
        n = self._server_count(text)
        if n is not None:
            return n
        return max(1, math.ceil(len(text) / 2.5))


# ---------------------------------------------------------------------------
# Slide / page-section handling
# ---------------------------------------------------------------------------

def split_slides(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Split a post-processed doc into (preamble, [(page, section)]).

    A section runs from its ``## Slide N`` heading to the start of the
    next slide heading, with the trailing ``---`` separator stripped.
    """
    matches = list(SLIDE_RE.finditer(text))
    if not matches:
        raise ValueError("no '## Slide N' sections found")
    preamble = text[:matches[0].start()].strip("\n")
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = TRAILING_SEP_RE.sub("", text[m.start():end]).rstrip("\n")
        sections.append((int(m.group(1)), body))
    return preamble, sections


def join_slides(sections: list[tuple[int, str]]) -> str:
    return "\n\n---\n\n".join(body for _, body in sections)


def split_raw_pages(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Split a raw grounding dump into (preamble, [(page, section)]).

    A section runs from its ``<!-- ==== PAGE N ==== -->`` marker (kept —
    the model converts it to ``## Slide N``) to the next marker.
    """
    matches = list(RAW_PAGE_RE.finditer(text))
    if not matches:
        raise ValueError("no PAGE markers found in raw dump")
    preamble = text[:matches[0].start()].strip("\n")
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((int(m.group(1)), text[m.start():end].rstrip("\n")))
    return preamble, sections


def join_raw(sections: list[tuple[int, str]]) -> str:
    return "\n\n".join(body for _, body in sections)


def section_spans(chunk_text: str) -> dict[int, tuple[int, int]]:
    """Map slide number -> (start, end) char span within chunk_text."""
    matches = list(SLIDE_RE.finditer(chunk_text))
    spans: dict[int, tuple[int, int]] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk_text)
        spans[int(m.group(1))] = (m.start(), end)
    return spans


def heading_signature(text: str) -> list[tuple[int, str]]:
    return [(len(m.group(1)), m.group(2).strip())
            for m in HEADING_RE.finditer(text)]


def slide_numbers(text: str) -> list[int]:
    return [int(m.group(1)) for m in SLIDE_RE.finditer(text)]


def image_refs(text: str) -> Counter[str]:
    return Counter(r.strip() for r in IMAGE_REF_RE.findall(text))


def word_count(text: str) -> int:
    return len(text.split())


def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2:
            t = "\n".join(lines[1:-1]).strip()
    return t


# ---------------------------------------------------------------------------
# Chunk planning (plan §4.1 + the 40-page job cap)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    index: int
    sections: list = field(repr=False)      # [(page, text)]
    mode: str = "diff"                      # 'diff' or 'full'
    raw: bool = False                       # raw-dump input sections

    @property
    def pages(self) -> tuple[int, int]:
        return self.sections[0][0], self.sections[-1][0]

    def text(self) -> str:
        return join_raw(self.sections) if self.raw else join_slides(self.sections)


def max_input_tokens(mode: str, context_limit: int) -> int:
    """Largest framed input (system+user) that satisfies the budget."""
    if mode == "diff":
        return context_limit - DIFF_OUTPUT_BUDGET
    # input + ceil(1.1*input) + 4096 <= limit  ->  input <= (limit-4096)/2.1
    return math.floor((context_limit - FULL_OUTPUT_MARGIN) / 2.1)


def build_messages(chunk: Chunk, raw_appendix: str = "") -> tuple[str, str]:
    text = chunk.text()
    a, b = chunk.pages
    if chunk.raw:
        system = RAW_FULL_SYSTEM
    elif chunk.mode == "diff":
        system = DIFF_SYSTEM.format(a=a, b=b)
    else:
        system = FULL_SYSTEM
    user = f"{DOC_OPEN}\n{text}\n{DOC_CLOSE}"
    if raw_appendix:
        user += raw_appendix
    return system, user


def raw_figure_appendix(manifest: list[dict], pages: list[int]) -> str:
    """Per-page figure refs for raw-mode full collation (rule C)."""
    page_set = set(pages)
    lines = []
    k: dict[int, int] = {}
    for e in manifest:
        if e["page"] not in page_set:
            continue
        k[e["page"]] = k.get(e["page"], 0) + 1
        lines.append(f"  page {e['page']} (image {k[e['page']]}): "
                     f"![Figure {k[e['page']]}](../{e['file']})")
    if not lines:
        return ""
    return ("\n\nFigure references to insert for the image blocks "
            "(document order):\n" + "\n".join(lines))


def figure_basenames_for_pages(manifest: list[dict],
                                pages: set[int]) -> set[str]:
    return {os.path.basename(e["file"]) for e in manifest
            if e["page"] in pages}


def plan_chunks(
    sections: list[tuple[int, str]],
    max_pages: int,
    counter: TokenCounter,
    mode: str,
    context_limit: int,
    raw: bool = False,
    raw_appendix: str = "",
) -> list[Chunk]:
    """Bisect a page group into chunks whose framed message fits the
    token budget for ``mode``.  Indices are assigned by the caller.
    """
    limit = max_input_tokens(mode, context_limit)

    def fits(group: list[tuple[int, str]]) -> tuple[bool, int]:
        c = Chunk(index=0, sections=group, mode=mode, raw=raw)
        system, user = build_messages(c, raw_appendix)
        toks = counter(system + "\n" + user)
        return toks <= limit, toks

    def split(group: list[tuple[int, str]]) -> list[Chunk]:
        ok, toks = fits(group)
        if ok:
            return [Chunk(index=0, sections=group, mode=mode, raw=raw)]
        if len(group) == 1:
            raise ValueError(
                f"slide {group[0][0]} alone needs ~{toks} tokens "
                f"(framed), exceeding the {mode}-mode budget of "
                f"{limit} for --context-limit {context_limit}; "
                "raise --context-limit (330000 is verified for diff) "
                "or use --collate-input postprocessed")
        mid = len(group) // 2
        return split(group[:mid]) + split(group[mid:])

    chunks: list[Chunk] = []
    for i in range(0, len(sections), max_pages):
        chunks.extend(split(sections[i:i + max_pages]))
    return chunks


def group_pages(sections: list[tuple[int, str]], max_pages: int,
                overlap: int) -> list[list[tuple[int, str]]]:
    if overlap > 0:
        step = max(1, max_pages - overlap)
        return [sections[i:i + max_pages]
                for i in range(0, len(sections), step)]
    return [sections[i:i + max_pages]
            for i in range(0, len(sections), max_pages)]


# ---------------------------------------------------------------------------
# Stage 2 — collation
# ---------------------------------------------------------------------------

def qwen_complete_stream(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    part_path: Path,
) -> tuple[str, str]:
    """One streaming chat completion; writes to part_path.
    Returns (text, finish_reason).
    """
    finish_reason = ""
    with part_path.open("w", encoding="utf-8") as f:
        with client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ) as stream:
            for piece in stream:
                if not piece.choices:
                    continue
                delta = piece.choices[0].delta
                if delta is not None and delta.content:
                    f.write(delta.content)
                if piece.choices[0].finish_reason:
                    finish_reason = piece.choices[0].finish_reason
    return part_path.read_text(encoding="utf-8"), finish_reason


def qwen_call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    final_path: Path,
    retries: int,
    label: str,
) -> tuple[str, str]:
    """qwen_complete_stream with whole-call retries."""
    part_path = final_path.with_name(final_path.name + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            text, finish = qwen_complete_stream(
                client, model, system, user, max_tokens, part_path)
            os.replace(part_path, final_path)
            return text, finish
        except Exception as exc:
            last_exc = exc
            part_path.unlink(missing_ok=True)
            print(f"  {label}: attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"{label}: {retries} attempts failed: {last_exc}")


def parse_edits(text: str) -> tuple[list[dict], int]:
    """Parse diff-mode output into edit dicts; (edits, bad_lines)."""
    t = text.strip()
    if not t:
        return [], 0
    if t == "NO_EDITS":
        return [], 0
    t = strip_code_fence(t)
    items: list = []
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            items = [obj]
    except json.JSONDecodeError:
        for line in t.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    edits, bad = [], 0
    for it in items:
        if (isinstance(it, dict) and "find" in it
                and isinstance(it["find"], str) and it["find"]):
            page = it.get("page")
            if isinstance(page, str) and page.isdigit():
                page = int(page)
            replace = it.get("replace", "")
            edits.append({
                "page": page if isinstance(page, int) else None,
                "find": it["find"],
                "replace": replace if isinstance(replace, str)
                else str(replace),
            })
        else:
            bad += 1
    return edits, bad


def apply_edits(
    chunk_text: str,
    edits: list[dict],
) -> tuple[str, int, list[dict], int]:
    """Apply edits by exact substring match.  Primary scope: the page's
    own section; unpaginated edits fall back to the whole chunk.
    Returns (new_text, applied, failed_edits, out_of_page).
    """
    spans = section_spans(chunk_text)
    out = chunk_text
    applied = out_of_page = 0
    failed: list[dict] = []
    by_page: dict[int, list[dict]] = {}
    unpaginated: list[dict] = []
    for e in edits:
        (by_page.setdefault(e["page"], []) if e["page"] is not None
         else unpaginated).append(e)

    for page in sorted(by_page):
        if page not in spans:
            for e in by_page[page]:
                e["_why"] = "page not in this chunk"
                failed.append(e)
            continue
        s, epos = spans[page]
        section = out[s:epos]
        for e in by_page[page]:
            if e["find"] in section:
                section = section.replace(e["find"], e["replace"], 1)
                applied += 1
            else:
                e["_why"] = "find span not in page section"
                failed.append(e)
        out = out[:s] + section + out[epos:]

    for e in unpaginated:
        if e["find"] in out:
            out = out.replace(e["find"], e["replace"], 1)
            out_of_page += 1
        else:
            e["_why"] = "no page given and find span not in chunk"
            failed.append(e)
    return out, applied, failed, out_of_page


def retry_failed_pages(
    chunk: Chunk,
    current_text: str,
    failed_edits: list[dict],
    client: OpenAI,
    args: argparse.Namespace,
    label: str,
) -> tuple[str, int, int, list[int]]:
    """Re-send each failed edit's page individually; apply what matches.
    Returns (text, applied, still_failed, pages_retried).
    """
    # A unique per-call directory: concurrent collation workers (the
    # batch script runs several documents at once) must not share
    # retry temp files.
    RETRY_TMP.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"retry_{label}_",
                                    dir=RETRY_TMP))
    spans = section_spans(current_text)
    applied = 0
    still_failed = 0
    pages_retried: list[int] = []
    by_page: dict[int, list[dict]] = {}
    for e in failed_edits:
        if e["page"] is not None:
            by_page.setdefault(e["page"], []).append(e)
        else:
            still_failed += 1

    for page in sorted(by_page):
        if page not in spans:
            still_failed += len(by_page[page])
            continue
        s, epos = spans[page]
        section = current_text[s:epos]
        body = "\n".join(
            json.dumps({"page": page, "find": e["find"],
                        "replace": e["replace"]})
            for e in by_page[page])
        user = (f"<<<page {page}>>>\n{section}\n<<<end page>>>\n\n"
                f"Intended edits that did not match:\n{body}")
        tmp = tmp_dir / f"page{page}.md"
        try:
            text, _ = qwen_call(client, args.qwen_model, RETRY_SYSTEM, user,
                                 RETRY_OUTPUT_BUDGET, tmp, args.qwen_retries,
                                 f"{label} page {page} retry")
        except Exception as exc:
            print(f"  retry for page {page} failed: {exc}", file=sys.stderr)
            still_failed += len(by_page[page])
            continue
        finally:
            tmp.unlink(missing_ok=True)
        pages_retried.append(page)
        new_edits, _ = parse_edits(text)
        new_section = section
        page_ok = 0
        for e in new_edits:
            if e["page"] not in (None, page):
                continue
            if e["find"] and e["find"] in new_section:
                new_section = new_section.replace(e["find"], e["replace"], 1)
                page_ok += 1
        applied += page_ok
        still_failed += len(by_page[page]) - page_ok
        if new_section != section:
            current_text = (current_text[:s] + new_section
                            + current_text[epos:])
            spans = section_spans(current_text)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return current_text, applied, still_failed, pages_retried


def diff_guards(chunk: Chunk, result_text: str) -> list[str]:
    """Post-hoc guards for a diff-collated chunk (plan §5.1)."""
    problems = []
    src = chunk.text()
    w_in, w_out = word_count(src), word_count(result_text)
    delta = (w_out - w_in) / w_in if w_in else 0.0
    lo, hi = DIFF_WORD_DELTA
    if not (lo <= delta <= hi):
        problems.append(f"word delta {delta:+.1%} outside [{lo:+.0%},{hi:+.0%}]")
    if heading_signature(src) != heading_signature(result_text):
        problems.append("heading signature changed")
    if image_refs(src) != image_refs(result_text):
        problems.append("image reference set changed")
    if "<|" in result_text:
        problems.append("stray layout token in output")
    return problems


def full_guards(chunk: Chunk, result_text: str,
                expected_pages: list[int],
                expected_figure_basenames: set[str] | None) -> list[str]:
    """Post-hoc guards for a full-collated chunk (plan §5.1)."""
    problems = []
    if not chunk.raw:
        w_in, w_out = word_count(chunk.text()), word_count(result_text)
        delta = (w_out - w_in) / w_in if w_in else 0.0
        lo, hi = FULL_WORD_DELTA
        if not (lo <= delta <= hi):
            problems.append(
                f"word delta {delta:+.1%} outside [{lo:+.0%},{hi:+.0%}]")
    got = slide_numbers(result_text)
    if sorted(set(got)) != sorted(set(expected_pages)):
        missing = sorted(set(expected_pages) - set(got))
        extra = sorted(set(got) - set(expected_pages))
        if missing:
            problems.append(f"missing slides: {missing}")
        if extra:
            problems.append(f"unexpected slides: {extra}")
    if len(got) != len(set(got)):
        problems.append("duplicate ## Slide headings")
    if expected_figure_basenames is not None:
        refs = Counter(os.path.basename(p.split(" ")[0])
                       for p in image_refs(result_text))
        for name in sorted(expected_figure_basenames):
            if refs.get(name, 0) == 0:
                problems.append(f"figure not referenced: {name}")
            elif refs[name] > 1:
                problems.append(f"figure referenced {refs[name]}x: {name}")
    if re.search(r"<\|", result_text):
        problems.append("stray layout token in output")
    return problems


def fallback_sections_text(chunk: Chunk,
                           post_fallback_sections: list[tuple[int, str]]) -> str:
    """Deterministic fallback: the post-processed sections for this
    chunk's page range; the raw chunk's own text if unavailable."""
    a, b = chunk.pages
    sel = [s for s in post_fallback_sections if a <= s[0] <= b]
    if len(sel) == b - a + 1:
        return join_slides(sel)
    return chunk.text()


def collate_chunk(
    chunk: Chunk,
    client: OpenAI,
    args: argparse.Namespace,
    counter: TokenCounter,
    out_path: Path,
    manifest: list[dict],
    post_fallback_sections: list[tuple[int, str]],
) -> tuple[str, dict]:
    """Collate one chunk.  Returns (final_chunk_text, report_entry).
    ``final_chunk_text`` always carries ``## Slide N`` headers (model
    output when guards pass, post-processed fallback otherwise).
    """
    a, b = chunk.pages
    page_list = [p for p, _ in chunk.sections]
    raw_appendix = (raw_figure_appendix(manifest, page_list)
                    if chunk.raw else "")
    system, user = build_messages(chunk, raw_appendix)
    input_tokens = counter(system + "\n" + user)
    mode = "full" if chunk.raw else chunk.mode
    max_tokens = (DIFF_OUTPUT_BUDGET if mode == "diff"
                  else math.ceil(FULL_OUTPUT_FACTOR * input_tokens)
                  + FULL_OUTPUT_MARGIN)
    if input_tokens + max_tokens > args.context_limit:
        raise RuntimeError(
            f"chunk {chunk.index}: budget guard tripped "
            f"({input_tokens} input + {max_tokens} output > "
            f"{args.context_limit})")

    entry: dict = {
        "index": chunk.index, "pages": f"{a}-{b}",
        "input_tokens": input_tokens, "max_tokens": max_tokens,
        "mode": mode, "raw_input": chunk.raw,
    }
    model_cache = out_path.with_name(out_path.name + ".model")

    def run(m: str) -> tuple[str, str]:
        m_max = (DIFF_OUTPUT_BUDGET if m == "diff"
                 else math.ceil(FULL_OUTPUT_FACTOR * input_tokens)
                 + FULL_OUTPUT_MARGIN)
        return qwen_call(client, args.qwen_model, system, user, m_max,
                         model_cache, args.qwen_retries,
                         f"chunk {chunk.index} ({m})")

    t0 = time.time()
    final_text: str | None = None

    if mode == "diff":
        text, finish = run("diff")
        entry["finish_reason"] = finish
        entry["output_tokens"] = counter(text)
        edits, bad_lines = parse_edits(text)
        result = chunk.text()
        applied, out_of_page = 0, 0
        n_failed = 0
        if edits:
            result, applied, failed, out_of_page = apply_edits(result, edits)
            n_failed = len(failed)
            if failed:
                result, n_applied, n_failed, pages_retried = \
                    retry_failed_pages(chunk, result, failed, client, args,
                                       f"chunk{chunk.index}")
                applied += n_applied
                entry["retry"] = {"pages": pages_retried,
                                  "applied": n_applied}
        total = applied + n_failed
        rate = applied / total if total else 1.0
        entry["edits"] = {
            "requested": len(edits), "applied": applied, "failed": n_failed,
            "out_of_page": out_of_page, "unparsed_lines": bad_lines,
            "rate": round(rate, 4)}
        guards = diff_guards(chunk, result)
        entry["guards"] = guards
        if not guards and rate >= MIN_DIFF_APPLY_RATE:
            entry["status"] = "ok"
            final_text = result
        else:
            entry["status"] = ("diff_guards_failed" if guards
                               else "diff_apply_rate_low")

    # Full mode (direct or as the diff fallback).
    if final_text is None:
        if mode == "diff":
            print(f"  chunk {chunk.index}: {entry['status']} -> "
                  f"full-mode fallback")
        if input_tokens <= max_input_tokens("full", args.context_limit):
            entry["full_fallback"] = mode == "diff"
            text, finish = run("full")
            entry["finish_reason"] = finish
            entry["output_tokens"] = counter(text)
            text = strip_code_fence(text)
            guards = full_guards(
                chunk, text, page_list,
                figure_basenames_for_pages(manifest, set(page_list)))
            entry["guards"] = guards
            if not guards:
                entry["status"] = ("ok_via_full_fallback"
                                   if mode == "diff" else "ok")
                final_text = text
            else:
                entry["status"] = ("full_fallback_guards_failed"
                                   if mode == "diff" else "full_guards_failed")
                final_text = fallback_sections_text(
                    chunk, post_fallback_sections)
        else:
            entry["status"] += "_no_full_budget"
            final_text = fallback_sections_text(
                chunk, post_fallback_sections)

    out_path.write_text(final_text.rstrip("\n") + "\n", encoding="utf-8")
    model_cache.unlink(missing_ok=True)
    entry["output_path"] = str(out_path)
    entry["elapsed_s"] = round(time.time() - t0, 1)
    return final_text, entry


# ---------------------------------------------------------------------------
# Stage 3 — extended deterministic validation + agentic review
# ---------------------------------------------------------------------------

def check_math(text: str) -> list[str]:
    problems = []
    in_fence = False
    in_display = False
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if in_display:
            if s == "$$" or s.endswith("$$"):
                in_display = False
            continue
        if s == "$$":
            in_display = True
            continue
        if s.startswith("$$"):
            if s.endswith("$$") and len(s) > 4:
                continue  # complete single-line display math
            in_display = True
            continue
        s2 = re.sub(r"\$\$[^$]*\$\$", "", line)
        if s2.count("$") % 2:
            problems.append(f"line {i}: odd number of $ delimiters")
    if in_display:
        problems.append("unclosed $$ display math block")
    return problems


# ---------------------------------------------------------------------------
# KaTeX compilation check (markdown_katex CLI)
#
# The web renderer compiles our $...$ / $$...$$ math with KaTeX; verify it
# deterministically here.  markdown_katex shells out to its bundled KaTeX
# CLI once per formula (24 h content-addressed cache in /tmp/mdkatex) and
# raises wrapper.KatexError on a parse failure — the same backend the
# ``markdown_katex`` python-markdown extension uses internally, called
# directly per formula so that ALL bad spans are reported (a single
# ``markdown.convert`` run aborts at the first error, which would force an
# O(broken x total) re-convert harvest loop on math-heavy chunks).
#
# Performance note: markdown_katex's get_bin_cmd() re-scans PATH (~1 s) on
# every call unless its LOCAL_CMD_CACHE is primed.  _ensure_katex_bin()
# primes it once, after which a formula costs ~110 ms cold (node start-up)
# and ~5 ms warm (cache hit).
# ---------------------------------------------------------------------------

_KATEX_CODE_SPAN_RE = re.compile(r"`[^`\n]*`")
_KATEX_INLINE_RUN_RE = re.compile(r"\$\$[^$]+\$\$")
_KATEX_PAIR_RE = re.compile(r"\$([^$\n]+)\$")
_KATEX_BIN_LOCK = threading.Lock()
_KATEX_BIN_PRIMED = False


def _ensure_katex_bin() -> bool:
    """Resolve the KaTeX binary once and prime markdown_katex's
    ``LOCAL_CMD_CACHE`` so ``get_bin_cmd()`` is O(1) instead of re-scanning
    PATH (~1 s) on every formula.  Returns False when no usable binary is
    available (missing extension or unsupported platform)."""
    global _KATEX_BIN_PRIMED
    with _KATEX_BIN_LOCK:
        if _KATEX_BIN_PRIMED:
            return True
        try:
            from markdown_katex import wrapper
            bin_path = str(wrapper._get_pkg_bin_path())
            if not Path(bin_path).exists():
                return False
            wrapper.LOCAL_CMD_CACHE.parent.mkdir(parents=True, exist_ok=True)
            wrapper.LOCAL_CMD_CACHE.write_text(bin_path, encoding="utf-8")
        except (ImportError, NotImplementedError, OSError):
            return False
        _KATEX_BIN_PRIMED = True
        return True


def _katex_transform(text: str) -> list[tuple[str, bool]]:
    """Extract every math span in a chunk as ``(tex, display)`` pairs.

    * ``$$...$$`` (single- or multi-line, incl. mid-line runs) ->
      ``(tex, True)``;
    * ``$...$`` inline -> ``(tex, False)``;
    * fenced code blocks and inline code spans are ignored.

    Used only for the compile check; the original text is never modified.
    """
    spans: list[tuple[str, bool]] = []
    in_code = False
    in_display = False
    display: list[str] = []

    def flush_display() -> None:
        body = "\n".join(display)
        if body.strip():
            spans.append((body, True))

    def inline_pass(line: str) -> None:
        masked = _KATEX_CODE_SPAN_RE.sub("[code]", line)
        last = 0
        for dm in _KATEX_INLINE_RUN_RE.finditer(masked):
            for m in _KATEX_PAIR_RE.finditer(masked[last:dm.start()]):
                spans.append((m.group(1), False))
            tex = dm.group(0)[2:-2].strip()
            if tex:
                spans.append((tex, True))
            last = dm.end()
        for m in _KATEX_PAIR_RE.finditer(masked[last:]):
            spans.append((m.group(1), False))

    for line in text.splitlines():
        s = line.strip()
        if in_code:
            if s.startswith("```"):
                in_code = False
            continue
        if in_display:
            if s.endswith("$$"):
                if len(s) > 2:
                    display.append(s[:-2])
                flush_display()
                in_display, display = False, []
            else:
                display.append(line)
            continue
        if s.startswith("```"):
            in_code = True
            continue
        if s == "$$" or (s.startswith("$$") and not s.endswith("$$")):
            rest = s[2:].strip()
            display = [rest] if rest else []
            in_display = True
            continue
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            spans.append((s[2:-2].strip(), True))
            continue
        inline_pass(line)
    if in_display:  # unclosed block: compile what we have (check_math flags
        flush_display()  # the missing closer)
    return spans


# markdown_katex's tex2html() runs _cleanup_cache_dir() in a `finally`
# block on *every* formula render: a full iterdir()+stat() scan of the
# shared /tmp/mdkatex cache.  With thousands of entries and ~17
# concurrent review threads each validating formula-heavy chunks, that
# scan -- not the KaTeX compile -- dominated validate_markdown: one
# 3,400-formula chunk measured 372 s in isolation and 13,000+ s in a
# live run.  The cache has a 24 h TTL and /tmp is large, so the real
# cleanup is safe to run at most once per hour.
_katex_cleanup_state = {
    "installed": False,
    "real": None,
    "last": 0.0,
    "interval_s": 3600.0,
    "lock": threading.Lock(),
}


def _throttled_katex_cleanup() -> None:
    state = _katex_cleanup_state
    with state["lock"]:
        now = time.time()
        if now - state["last"] < state["interval_s"]:
            return
        state["last"] = now
    state["real"]()


def _install_throttled_katex_cleanup(wrapper) -> None:
    """Replace markdown_katex's per-call cache cleanup with a version
    that runs at most once per hour (see ``_katex_cleanup_state``)."""
    state = _katex_cleanup_state
    if state["installed"]:
        return
    with state["lock"]:
        if state["installed"]:
            return
        state["real"] = wrapper._cleanup_cache_dir
        wrapper._cleanup_cache_dir = _throttled_katex_cleanup
        state["installed"] = True


def check_katex(text: str) -> list[str]:
    """Compile every math span with KaTeX and return the failures."""
    if not _ensure_katex_bin():
        print("warning: markdown_katex / KaTeX binary unavailable; "
              "skipping the KaTeX compile check", file=sys.stderr)
        return []
    from markdown_katex import wrapper
    _install_throttled_katex_cleanup(wrapper)
    problems: list[str] = []
    for tex, display in _katex_transform(text):
        opts = {"display-mode": True} if display else {}
        try:
            wrapper.tex2html(tex, opts)
        except wrapper.KatexError as exc:
            msg = str(exc)
            m = re.match(r"Error processing '(.*?)': (.*)", msg,
                         re.DOTALL)
            output = m.group(2) if m else msg
            i = output.find("KaTeX parse error:")
            if i >= 0:
                lines = [l for l in output[i:i + 300].splitlines()
                         if not l.strip().startswith("at ")]
                detail = "\n".join(lines).strip() or output.strip()[:300]
            else:
                detail = output.strip()[:300]
            problems.append(f"KaTeX compile error "
                            f"(formula {tex.strip()[:60]!r}): {detail}")
        except NotImplementedError:
            print("warning: katex binary unavailable (markdown_katex); "
                  "skipping the KaTeX compile check", file=sys.stderr)
            return []
        except Exception as exc:  # subprocess / I/O failure: likely systemic
            problems.append(f"KaTeX check error "
                            f"(formula {tex.strip()[:60]!r}): {exc!r}")
            return problems
    return problems


def check_tables(text: str) -> list[str]:
    problems = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            block = lines[i:j]

            def ncols(r: str) -> int:
                r = r.strip()
                if r.startswith("|"):
                    r = r[1:]
                if r.endswith("|"):
                    r = r[:-1]
                return len(re.split(r"(?<!\\)\|", r))

            if len(block) < 2:
                problems.append(f"line {i + 1}: single-row pipe table")
            else:
                first = ncols(block[0])
                if not re.match(r"^\s*\|?[\s:|*-]+\|?\s*$", block[1]):
                    problems.append(
                        f"line {i + 2}: table separator row malformed")
                for k, r in enumerate(block):
                    n = ncols(r)
                    if n != first:
                        problems.append(
                            f"line {i + 1 + k}: table column count {n} "
                            f"!= header {first}")
                        break
            i = j
        else:
            i += 1
    return problems


def check_headings(text: str, expected_pages: list[int] | None = None
                   ) -> list[str]:
    problems = []
    levels = []
    for m in HEADING_RE.finditer(text):
        levels.append(len(m.group(1)))
    for a, b in zip(levels, levels[1:]):
        if b > a + 1:
            problems.append(f"heading level skip {a} -> {b}")
    pages = slide_numbers(text)
    if expected_pages is not None:
        if sorted(set(pages)) != sorted(set(expected_pages)):
            missing = sorted(set(expected_pages) - set(pages))
            extra = sorted(set(pages) - set(expected_pages))
            if missing:
                problems.append(f"missing slides: {missing}")
            if extra:
                problems.append(f"unexpected slides: {extra}")
    if len(pages) != len(set(pages)):
        problems.append("duplicate ## Slide headings")
    if pages and pages != sorted(pages):
        problems.append("## Slide headings out of order")
    return problems


def check_figures(text: str, base_dir: Path, manifest: list[dict],
                  pages: set[int] | None) -> list[str]:
    problems = []
    refs = image_refs(text)
    for path in sorted(set(refs)):
        clean = path.split(" ")[0]  # strip optional "title"
        if not os.path.isfile(
                os.path.normpath(os.path.join(base_dir, clean))):
            problems.append(f"image ref not found on disk: {clean}")
    expected = {os.path.basename(e["file"])
                for e in manifest if pages is None or e["page"] in pages}
    basenames = Counter(os.path.basename(p.split(" ")[0]) for p in refs)
    for name in sorted(expected):
        n = basenames.get(name, 0)
        if n == 0:
            problems.append(f"manifest figure not referenced: {name}")
        elif n > 1:
            problems.append(f"figure referenced {n}x: {name}")
    return problems


def check_artifacts(text: str) -> list[str]:
    problems = []
    if re.search(r"<\|", text):
        problems.append("stray layout token <|...|>")
    if "NO_EDITS" in text:
        problems.append("literal NO_EDITS leaked into document")
    if DOC_OPEN in text or DOC_CLOSE in text:
        problems.append("prompt framing leaked into document")
    prev, run = None, 0
    for line in text.splitlines():
        s = line.strip()
        if s and s == prev:
            run += 1
            if run == 2:
                problems.append(f"repeated-line loop: {s[:60]!r}")
        else:
            prev, run = s, 0
    return problems


def check_pandoc(path: str) -> list[str]:
    try:
        r = subprocess.run(
            ["pandoc", "-f", "markdown", "-t", "html", path],
            capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return [f"pandoc failed (exit {r.returncode}): "
                    f"{r.stderr.strip()[:300]}"]
    except FileNotFoundError:
        return ["pandoc not found on PATH"]
    except subprocess.TimeoutExpired:
        return ["pandoc timed out"]
    return []


def extended_validate(path: Path, manifest: list[dict],
                      pages: list[int]) -> list[str]:
    """All deterministic checks (plan §5.2) for one chunk file."""
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    problems += _validate_markdown_text(text)
    problems += check_pandoc(str(path))
    problems += check_math(text)
    problems += check_katex(text)
    problems += check_tables(text)
    problems += check_headings(text, pages)
    problems += check_figures(text, path.parent, manifest, set(pages))
    problems += check_artifacts(text)
    return problems


class ReviewTraceHandler(BaseCallbackHandler):
    """Write a LangSmith-style trace of the agentic review to JSONL.

    The deepagents reviewer is a LangGraph, so every node, LLM call and
    tool call emits callbacks.  This handler records each as one JSON
    line (event type, ``run_id``/``parent_run_id`` for the run tree, the
    run name, and a JSON-safe input/output/error payload).  The result
    is a replayable log of the whole agent conversation: the prompt sent
    to the model, the model's response (including any ``tool_calls``),
    and each tool's input and result.

    LangChain's callback API is the same hook LangSmith itself uses for
    tracing; here we simply write it to a local file instead of the
    LangSmith cloud, so no account or network is required.

    One instance writes one file; it is safe to construct one per
    ``review_chunk`` call.  ``run_inline`` keeps events in call order,
    and a lock guards the append so concurrent callbacks cannot
    interleave.
    """

    run_inline = True

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        # RLock: close() emits trace_end while holding the lock.
        self._lock = threading.RLock()
        self._run_names: dict[str, str] = {}
        self._start = time.time()
        self._emit("trace_start", name=str(self.path))

    # -- serialization helpers -------------------------------------------
    @staticmethod
    def _jsonable(obj, depth=0):
        """Convert ``obj`` into a JSON-serialisable structure."""
        if depth > 10:
            return repr(obj)
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(k): ReviewTraceHandler._jsonable(v, depth + 1)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [ReviewTraceHandler._jsonable(v, depth + 1) for v in obj]
        # langchain messages (AIMessage, HumanMessage, ToolMessage, ...)
        if hasattr(obj, "type") and hasattr(obj, "content"):
            d: dict = {"type": obj.type,
                       "content": ReviewTraceHandler._jsonable(obj.content,
                                                               depth + 1)}
            tc = getattr(obj, "tool_calls", None)
            if tc:
                d["tool_calls"] = ReviewTraceHandler._jsonable(tc, depth + 1)
            nm = getattr(obj, "name", None)
            if nm:
                d["name"] = nm
            return d
        return repr(obj)

    def _name(self, serialized, metadata, kwargs) -> str:
        name = ((serialized or {}).get("name")
                or (metadata or {}).get("langgraph_node")
                or kwargs.get("name"))
        return name

    def _emit(self, event, run_id=None, parent_run_id=None, name=None,
              tags=None, metadata=None, payload=None):
        rec = {
            "event": event,
            "t": round(time.time() - self._start, 6),
            "run_id": str(run_id) if run_id is not None else None,
            "parent_run_id": (str(parent_run_id)
                              if parent_run_id is not None else None),
            "name": name,
        }
        if tags:
            rec["tags"] = list(tags)
        if metadata:
            rec["metadata"] = self._jsonable(metadata)
        if payload is not None:
            rec["data"] = self._jsonable(payload)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._emit("trace_end")
                self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- graph nodes ------------------------------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id,
                       parent_run_id=None, tags=None, metadata=None,
                       **kwargs):
        name = self._name(serialized, metadata, kwargs)
        if run_id is not None and name:
            self._run_names[str(run_id)] = name
        self._emit("chain_start", run_id, parent_run_id, name, tags, metadata,
                   {"inputs": inputs})

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        name = self._run_names.get(str(run_id)) if run_id else None
        self._emit("chain_end", run_id, parent_run_id, name, None, None,
                   {"outputs": outputs})

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        name = self._run_names.get(str(run_id)) if run_id else None
        self._emit("chain_error", run_id, parent_run_id, name, None, None,
                   {"error": repr(error)})

    # -- LLM calls (the agent's conversation) ----------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id,
                            parent_run_id=None, tags=None, metadata=None,
                            **kwargs):
        name = self._name(serialized, metadata, kwargs)
        if run_id is not None and name:
            self._run_names[str(run_id)] = name
        self._emit("llm_start", run_id, parent_run_id, name, tags, metadata,
                   {"messages": messages})

    def on_llm_start(self, serialized, prompts, *, run_id,
                     parent_run_id=None, tags=None, metadata=None,
                     **kwargs):
        name = self._name(serialized, metadata, kwargs)
        if run_id is not None and name:
            self._run_names[str(run_id)] = name
        self._emit("llm_start", run_id, parent_run_id, name, tags, metadata,
                   {"prompts": prompts})

    def on_llm_end(self, response, *, run_id, parent_run_id=None, tags=None,
                   **kwargs):
        name = self._run_names.get(str(run_id)) if run_id else None
        gens = []
        for batch in response.generations:
            for g in batch:
                item: dict = {"text": getattr(g, "text", "")}
                msg = getattr(g, "message", None)
                if msg is not None:
                    item["message"] = self._jsonable(msg)
                gens.append(item)
        payload: dict = {"generations": gens}
        llm_output = getattr(response, "llm_output", None)
        if llm_output:
            payload["llm_output"] = llm_output
        self._emit("llm_end", run_id, parent_run_id, name, tags, None,
                   payload)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, tags=None,
                     **kwargs):
        name = self._run_names.get(str(run_id)) if run_id else None
        self._emit("llm_error", run_id, parent_run_id, name, tags, None,
                   {"error": repr(error)})

    # -- tool calls (read_file / edit_file / validate_markdown) ----------
    def on_tool_start(self, serialized, input_str, *, run_id,
                      parent_run_id=None, tags=None, metadata=None,
                      inputs=None, **kwargs):
        name = self._name(serialized, metadata, kwargs) or "tool"
        if run_id is not None:
            self._run_names[str(run_id)] = name
        data = inputs if inputs is not None else input_str
        self._emit("tool_start", run_id, parent_run_id, name, tags, metadata,
                   {"input": data})

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        name = self._run_names.get(str(run_id)) if run_id else None
        self._emit("tool_end", run_id, parent_run_id, name, None, None,
                   {"output": output})

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        name = self._run_names.get(str(run_id)) if run_id else None
        self._emit("tool_error", run_id, parent_run_id, name, None, None,
                   {"error": repr(error)})


def review_chunk(
    chunk_path: Path,
    pages: list[int],
    manifest_path: Path,
    manifest: list[dict],
    llm: ChatOpenAI,
    max_iterations: int,
    log: list[dict],
    trace_path: Path | None = None,
) -> str:
    """validate -> (agent fix -> validate)*, up to max_iterations agent
    runs.  Returns 'clean', 'dirty', or 'agent_error' and appends
    iteration records to ``log``.

    When ``trace_path`` is given, the full agent conversation (every LLM
    call and tool call, via the LangChain callback API) is additionally
    written to that file as a JSONL trace.
    """
    abs_path = Path(chunk_path).resolve()
    a, b = pages[0], pages[-1]
    # Mount the real directory that holds both the chunk and the figure
    # manifest (the document outdir) at /workspace inside the agent's
    # virtual filesystem.  Without this, the built-in file tools run
    # on deepagents' default empty in-memory StateBackend and cannot
    # see (or change) the target file on disk, while validate_markdown
    # -- a plain Python tool -- reads the real path: the agent would
    # get errors it is unable to fix.
    manifest_abs = Path(manifest_path).resolve()
    workspace = Path(os.path.commonpath([str(abs_path), str(manifest_abs)]))
    if not workspace.is_dir():
        workspace = abs_path.parent
    rel_chunk = abs_path.relative_to(workspace).as_posix()
    rel_manifest = manifest_abs.relative_to(workspace).as_posix()
    agent = None
    tracer = ReviewTraceHandler(trace_path) if trace_path is not None else None
    invoke_config = ({"callbacks": [tracer]} if tracer is not None else None)

    def current_problems() -> list[str]:
        return extended_validate(chunk_path, manifest, pages)

    try:
        problems = current_problems()
        if not problems:
            log.append({"chunk": chunk_path.name, "iteration": 0,
                        "status": "clean_on_entry", "problems": 0})
            return "clean"

        for it in range(1, max_iterations + 1):
            record = {"chunk": chunk_path.name, "iteration": it,
                      "problems_before": len(problems),
                      "problems": problems[:50]}
            if agent is None:
                agent = create_review_agent(llm, REVIEW_SYSTEM_PROMPT,
                                            extra_checks=check_katex,
                                            workspace=workspace)
            task = (
                f"Review and fix markdown syntax in the file "
                f"'{WORKSPACE_MOUNT}/{rel_chunk}'.\n"
                f"This file is a page-range chunk (slides {a}..{b}) of an "
                "OCR'd scientific document.\n"
                f"Figure manifest for the whole document: "
                f"'{WORKSPACE_MOUNT}/{rel_manifest}' "
                '(JSON list of {"page": N, "file": "figures/..."}; you may '
                "read_file it).\n\n"
                "Deterministic validation currently reports:\n"
                + "\n".join(f"{i}. {p}" for i, p in enumerate(problems[:50], 1))
                + "\n\nFix each problem with a targeted edit_file, then call "
                "validate_markdown (it re-checks markdown syntax AND compiles "
                "every $...$ / $$...$$ formula with KaTeX) until it returns "
                "VALIDATION PASSED and all problems above are resolved. For a "
                "KaTeX compile error, fix the TeX so it parses (matched "
                "braces, closed environments, valid commands) while keeping "
                "the formula's meaning. Preserve all words, all '## Slide N' "
                "headings, all image references, and all "
                "'<!-- Figure k, description: ... -->' comment lines."
            )
            try:
                result = agent.invoke(
                    {"messages": [HumanMessage(content=task)]},
                    config=invoke_config)
                output = (result.content if hasattr(result, "content")
                          else str(result))
                record["agent_output_tail"] = (output or "")[-400:]
                record["status"] = "agent_ran"
            except Exception as exc:
                record["status"] = "agent_error"
                record["error"] = str(exc)[:400]
                log.append(record)
                return "agent_error"
            log.append(record)
            problems = current_problems()
            if not problems:
                log.append({"chunk": chunk_path.name,
                            "iteration": it + 1, "status": "clean",
                            "problems": 0})
                return "clean"
        log.append({"chunk": chunk_path.name,
                    "iteration": max_iterations + 1,
                    "status": "dirty_after_max_iterations",
                    "problems": len(problems)})
        return "dirty"
    finally:
        if tracer is not None:
            tracer.close()


# ---------------------------------------------------------------------------
# Stage 4 — merge + global checks
# ---------------------------------------------------------------------------

def merge_chunks(chunk_texts: list[tuple[int, str]], pdf_stem: str) -> str:
    """Deterministic concatenation.  Slides are kept in document order;
    a slide is dropped if its number does not exceed the last one kept
    (dedupes full-mode chunk overlap)."""
    merged: list[str] = []
    last_page = 0
    for _idx, text in chunk_texts:
        try:
            _, sections = split_slides(text)
        except ValueError:
            sections = []
        for page, body in sections:
            if page > last_page:
                merged.append(body)
                last_page = page
    body = "\n\n---\n\n".join(merged)
    return f"# {pdf_stem}\n\n{body}\n\n---\n"


def global_checks(final_path: Path, manifest: list[dict],
                  expected_pages: list[int],
                  baseline_words: int) -> dict:
    text = final_path.read_text(encoding="utf-8")
    problems: list[str] = []
    problems += _validate_markdown_text(text)
    problems += check_pandoc(str(final_path))
    problems += check_math(text)
    problems += check_katex(text)
    problems += check_tables(text)
    problems += check_headings(text, expected_pages)
    problems += check_figures(text, final_path.parent, manifest, None)
    problems += check_artifacts(text)
    w = word_count(text)
    delta = (w - baseline_words) / baseline_words if baseline_words else 0.0
    return {
        "slides": slide_numbers(text),
        "word_count": w,
        "word_delta_vs_postprocessed": round(delta, 4),
        "problems": problems,
        "status": "ok" if not problems else "dirty",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  pixi run python pdf_to_markdown_dsocr2_qwen.py input.pdf -o out/\n"
            "  pixi run python pdf_to_markdown_dsocr2_qwen.py input.pdf "
            "--collate-mode full --context-limit 262144\n"
            "  pixi run python pdf_to_markdown_dsocr2_qwen.py done.pdf "
            "--skip-ocr --skip-review\n"))
    p.add_argument("pdf", help="Path to the input PDF file")
    p.add_argument("-o", "--output", default=None,
                   help="Output directory (default: the PDF's directory)")
    # stage 1 (DSOCR2)
    p.add_argument("--dpi", type=int, default=150,
                   help="Page render DPI (default: 150)")
    p.add_argument("--start-page", type=int, default=0,
                   help="First page, 0-indexed, inclusive (default: 0)")
    p.add_argument("--end-page", type=int, default=None,
                   help="Last page, 0-indexed, exclusive (default: end)")
    p.add_argument("--npages-parallel", type=int, default=8,
                   help="Parallelism for rendering + OCR (default: 8)")
    p.add_argument("--dsocr2-base-url", default="http://127.0.0.1:18000/v1",
                   help="OpenAI-compatible base URL of the DSOCR2 vLLM pod "
                        "(default: %(default)s)")
    p.add_argument("--dsocr2-model", default="deepseek-ocr2",
                   help="Served DSOCR2 model name (default: %(default)s)")
    p.add_argument("--dsocr2-api-key", default="EMPTY")
    p.add_argument("--dsocr2-temperature", type=float, default=0.0)
    p.add_argument("--dsocr2-max-tokens", type=int, default=8180)
    p.add_argument("--dsocr2-timeout", type=float, default=1800.0)
    p.add_argument("--force", action="store_true",
                   help="Re-render, re-OCR, and re-collate cached items")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Skip pages whose OCR request fails")
    p.add_argument("--skip-figures", action="store_true",
                   help="Do not crop figures (empty manifest)")
    p.add_argument("--skip-ocr", action="store_true",
                   help="Reuse the existing <stem>_dsocr2.md raw dump + "
                        "figures (only re-run postprocess and later stages)")
    # stage 2 (Qwen collation)
    p.add_argument("--qwen-base-url", default="http://127.0.0.1:8008/v1",
                   help="OpenAI-compatible base URL of the Qwen3.8 vLLM "
                        "pod (default: %(default)s)")
    p.add_argument("--qwen-model", default="Qwen/Qwen3.8-27B-FP8",
                   help="Served Qwen model name (default: %(default)s)")
    p.add_argument("--qwen-api-key", default="EMPTY")
    p.add_argument("--qwen-timeout", type=float, default=7200.0,
                   help="Per-collation-call timeout in seconds (default: "
                        "%(default)s; full-mode 100k-token decodes take ~70 min)")
    p.add_argument("--qwen-retries", type=int, default=3,
                   help="Whole-chunk retries on HTTP failure (default: 3)")
    p.add_argument("--collate-mode", choices=["diff", "full"],
                   default="diff",
                   help="Collation contract (default: diff; see plan §4.0)")
    p.add_argument("--collate-input",
                   choices=["postprocessed", "raw"],
                   default="postprocessed",
                   help="Collation input (default: postprocessed; raw "
                        "implies full mode)")
    p.add_argument("--context-limit", type=int, default=262144,
                   help="Token budget for prompt+max_tokens (default: "
                        "262144 native; 330000 is verified YARN headroom, "
                        "diff mode only)")
    p.add_argument("--max-pages", type=int, default=40,
                   help="Max pages per Qwen collation job (default: 40)")
    p.add_argument("--chunk-overlap", type=int, default=0,
                   help="Overlap slides between chunks, full mode only "
                        "(default: 0)")
    p.add_argument("--skip-collate", action="store_true",
                   help="Skip Qwen collation (pure dsocr2 add-on mode)")
    p.add_argument("--plan-only", action="store_true",
                   help="Print the chunk plan and exit (no LLM calls)")
    # stage 3 (review)
    p.add_argument("--skip-review", action="store_true",
                   help="Skip the per-chunk agentic review loop")
    p.add_argument("--max-review-iterations", type=int, default=3,
                   help="Agentic review cap per chunk (default: 3)")
    p.add_argument("--no-cache-dir", action="store_true",
                   help="Delete rendered page PNGs after completion")
    return p.parse_args(argv)


def write_report(path: Path, payload, as_lines: bool = False) -> None:
    if as_lines:
        path.write_text("\n".join(payload) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.collate_input == "raw" and args.collate_mode == "diff":
        print("error: --collate-input raw only supports --collate-mode full "
              "(the diff contract needs post-processed slide sections)",
              file=sys.stderr)
        return 1
    if args.chunk_overlap > 0 and args.collate_mode != "full" \
            and args.collate_input != "raw":
        print("error: --chunk-overlap is only supported in full mode",
              file=sys.stderr)
        return 1

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    outdir = (Path(args.output).expanduser().resolve()
              if args.output else pdf_path.parent)
    stem = pdf_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    total_pages = get_pdf_page_count(str(pdf_path))
    first_page = args.start_page + 1
    last_page = (args.end_page if args.end_page is not None
                 else total_pages)
    if first_page < 1 or last_page > total_pages or last_page < first_page:
        print(f"error: page range {args.start_page}-{args.end_page} out of "
              f"bounds for {total_pages} pages", file=sys.stderr)
        return 1
    width = page_width(total_pages)
    expected_pages = list(range(first_page, last_page + 1))

    t_start = time.time()
    print(f"DSOCR2 + Qwen3.8 workflow: {pdf_path.name} "
          f"({total_pages} pages, using {first_page}-{last_page})")
    report: dict = {
        "pdf": str(pdf_path), "pages": f"{first_page}-{last_page}",
        "collate_mode": args.collate_mode,
        "collate_input": args.collate_input,
        "context_limit": args.context_limit,
        "max_pages": args.max_pages,
        "dsocr2": {"base_url": args.dsocr2_base_url,
                   "model": args.dsocr2_model},
        "qwen": {"base_url": args.qwen_base_url,
                 "model": args.qwen_model},
    }

    # ------------------------------------------------------------------
    # Stage 1 — OCR (existing dsocr2 pipeline functions)
    # ------------------------------------------------------------------
    dump_path = outdir / f"{stem}_dsocr2.md"
    manifest_path = outdir / "figures_manifest.json"
    figures_dir = outdir / "figures"
    raw_pages_dir = outdir / "raw_pages"

    if args.skip_ocr:
        if not dump_path.is_file():
            print(f"error: --skip-ocr but no raw dump at {dump_path}",
                  file=sys.stderr)
            return 1
        manifest = (json.load(open(manifest_path, encoding="utf-8"))
                    if manifest_path.is_file() else [])
        report["ocr"] = "skipped (reused raw dump)"
        print("  stage 1: skipped (--skip-ocr)")
    else:
        client_dsocr = OpenAI(
            base_url=args.dsocr2_base_url, api_key=args.dsocr2_api_key,
            timeout=args.dsocr2_timeout)
        try:
            models = [m.id for m in client_dsocr.models.list().data]
        except Exception as exc:
            print(f"error: cannot reach DSOCR2 server at "
                  f"{args.dsocr2_base_url}: {exc}", file=sys.stderr)
            print("hint: is the dsocr2 pod up and the port forward "
                  "(18000) alive?", file=sys.stderr)
            return 1
        if args.dsocr2_model not in models:
            print(f"warning: model {args.dsocr2_model!r} not in served "
                  f"models {models}", file=sys.stderr)
        print(f"  stage 1: DSOCR2 OCR via {args.dsocr2_base_url} "
              f"({args.dsocr2_model})")

        page_images = ensure_page_images(
            str(pdf_path), outdir / "pages", args.dpi, first_page,
            last_page, total_pages, args.npages_parallel, args.force)
        pages_text = ocr_pages(
            page_images, raw_pages_dir, width, client_dsocr,
            args.dsocr2_model, args.dsocr2_temperature,
            args.dsocr2_max_tokens, args.npages_parallel, args.force,
            args.continue_on_error)
        if not pages_text:
            print("error: no pages OCRed", file=sys.stderr)
            return 1
        dump_path.write_text(build_raw_dump(
            pdf_path.name, pages_text, total_pages, args.dpi,
            args.dsocr2_model, args.dsocr2_max_tokens,
            args.dsocr2_temperature), encoding="utf-8")
        print(f"  wrote raw dump {dump_path}")

        if args.skip_figures:
            manifest = []
            figures_dir.mkdir(parents=True, exist_ok=True)
        else:
            manifest = crop_figures(pages_text, page_images, width,
                                    figures_dir)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        print(f"  {len(manifest)} figure crop(s) -> {figures_dir}")
        report["ocr"] = {"pages_ocred": len(pages_text),
                         "figures": len(manifest)}

    # ------------------------------------------------------------------
    # Stage 1.5 — postprocess (existing)
    # ------------------------------------------------------------------
    dsocr2_md = outdir / "md" / f"{stem}_dsocr2.md"
    print(f"  stage 1.5: postprocess -> {dsocr2_md}")
    pp_result = dsocr2_postprocess.postprocess(
        str(dump_path), str(figures_dir), str(manifest_path),
        str(dsocr2_md))
    report["postprocess"] = {k: pp_result[k] for k in
                             ("slides", "images", "tables", "equations")}
    if pp_result["problems"]:
        report["postprocess"]["problems"] = pp_result["problems"]
        print("  postprocess problems:", file=sys.stderr)
        for prob in pp_result["problems"]:
            print(f"    - {prob}", file=sys.stderr)

    post_text = dsocr2_md.read_text(encoding="utf-8")
    _preamble, post_sections = split_slides(post_text)
    baseline_words = word_count(post_text)

    # ------------------------------------------------------------------
    # Stage 2 — collation
    # ------------------------------------------------------------------
    chunks_dir = outdir / "md_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    qwen_final_md = outdir / "md" / f"{stem}_qwen_final.md"
    collate_mode = "full" if (args.collate_input == "raw"
                              or args.collate_mode == "full") else "diff"

    if args.skip_collate:
        print("  stage 2: skipped (--skip-collate)")
        report["collate"] = "skipped"
        chunk_texts: list[tuple[int, str]] = [
            (1, join_slides(post_sections))]
        chunk_width = 2
    else:
        counter = TokenCounter(args.qwen_model, args.qwen_base_url)
        counter("calibration")  # load the tokenizer up front

        if args.collate_input == "raw":
            raw_text = dump_path.read_text(encoding="utf-8")
            _rp, raw_sections = split_raw_pages(raw_text)
            sections = [s for s in raw_sections if s[0] in expected_pages]
            raw = True
        else:
            sections = [s for s in post_sections if s[0] in expected_pages]
            raw = False

        groups = group_pages(sections, args.max_pages,
                             args.chunk_overlap if collate_mode == "full"
                             else 0)
        all_chunks: list[Chunk] = []
        for group in groups:
            appendix = (raw_figure_appendix(manifest, [p for p, _ in group])
                        if raw else "")
            all_chunks.extend(
                plan_chunks(sections=group, max_pages=args.max_pages,
                            counter=counter, mode=collate_mode,
                            context_limit=args.context_limit, raw=raw,
                            raw_appendix=appendix))
        for n, c in enumerate(all_chunks, 1):
            c.index = n
        chunk_width = max(2, len(str(len(all_chunks))))

        print(f"  stage 2: Qwen3.8 collation ({collate_mode} mode, "
              f"{args.collate_input} input, <= {args.max_pages} pages/job, "
              f"plan: {len(all_chunks)} chunk(s), context limit "
              f"{args.context_limit})")
        for c in all_chunks:
            a, b = c.pages
            print(f"    chunk {c.index:0{chunk_width}d}: slides {a}-{b} "
                  f"({b - a + 1} pages, mode={c.mode})")

        if args.plan_only:
            report["collate"] = ([
                {"index": c.index, "pages": f"{c.pages[0]}-{c.pages[1]}",
                 "status": "planned"} for c in all_chunks]
                if not args.skip_collate else "skipped")
            report["plan_only"] = True
            write_report(outdir / "collate_report.json", report)
            print("plan-only: no LLM calls made.")
            return 0

        client_qwen = OpenAI(
            base_url=args.qwen_base_url, api_key=args.qwen_api_key,
            timeout=httpx.Timeout(args.qwen_timeout, connect=30.0))
        try:
            qmodels = [m.id for m in client_qwen.models.list().data]
        except Exception as exc:
            print(f"error: cannot reach Qwen server at "
                  f"{args.qwen_base_url}: {exc}", file=sys.stderr)
            return 1
        if args.qwen_model not in qmodels:
            print(f"warning: model {args.qwen_model!r} not in served "
                  f"models {qmodels}", file=sys.stderr)

        chunk_texts = []
        collate_entries = []
        for chunk in all_chunks:
            label = f"{chunk.index:0{chunk_width}d}"
            chunk_path = chunks_dir / f"{label}_{stem}.md"
            post_fb = [s for s in post_sections
                       if chunk.pages[0] <= s[0] <= chunk.pages[1]]
            if chunk_path.exists() and not args.force:
                text = chunk_path.read_text(encoding="utf-8")
                entry = {"index": chunk.index,
                         "pages": f"{chunk.pages[0]}-{chunk.pages[1]}",
                         "status": "cached"}
                print(f"  chunk {label}: cached")
            else:
                try:
                    text, entry = collate_chunk(
                        chunk, client_qwen, args, counter, chunk_path,
                        manifest, post_fb)
                except RuntimeError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
                print(f"  chunk {label}: {entry['status']} "
                      f"({entry['input_tokens']} input tokens, "
                      f"{entry.get('output_tokens', '?')} output, "
                      f"{entry['elapsed_s']}s)")
            chunk_texts.append((chunk.index, text))
            collate_entries.append(entry)
        report["collate"] = collate_entries
        write_report(outdir / "collate_report.json", report)

    # ------------------------------------------------------------------
    # Stage 3 — per-chunk agentic review
    # ------------------------------------------------------------------
    review_log: list[dict] = []
    if args.skip_review:
        print("  stage 3: skipped (--skip-review)")
        report["review"] = "skipped"
    else:
        print(f"  stage 3: per-chunk review (<= "
              f"{args.max_review_iterations} agent iterations/chunk)")
        # Bounded per-call timeout: a request silently lost in the
        # ssh port-forward chain used to hang the whole sweep in the
        # review agent (observed 2026-10-30: ~45 min in read()).
        # 900 s is ample for a short edit/validate agent turn.
        llm_review = ChatOpenAI(
            model=args.qwen_model, api_key=args.qwen_api_key,
            base_url=args.qwen_base_url, temperature=0,
            max_retries=1, timeout=900)
        for idx, text in chunk_texts:
            label = f"{idx:0{chunk_width}d}"
            chunk_path = chunks_dir / f"{label}_{stem}.md"
            if not chunk_path.exists():
                chunk_path.write_text(text.rstrip("\n") + "\n",
                                      encoding="utf-8")
            try:
                _pre, secs = split_slides(chunk_path.read_text(
                    encoding="utf-8"))
                pages = [p for p, _ in secs]
            except ValueError:
                pages = expected_pages
            if not pages:
                pages = expected_pages
            status = review_chunk(chunk_path, pages, manifest_path,
                                  manifest, llm_review,
                                  args.max_review_iterations, review_log,
                                  trace_path=outdir / "review_traces" /
                                  f"{label}_{stem}.jsonl")
            print(f"  chunk {label}: review {status}")
        report["review"] = review_log
        write_report(outdir / "review_log.jsonl",
                     [json.dumps(r) for r in review_log], as_lines=True)

    # ------------------------------------------------------------------
    # Stage 4 — merge + global checks
    # ------------------------------------------------------------------
    print("  stage 4: deterministic merge")
    merged = merge_chunks(chunk_texts, stem)
    qwen_final_md.parent.mkdir(parents=True, exist_ok=True)
    qwen_final_md.write_text(merged, encoding="utf-8")
    print(f"  wrote {qwen_final_md}")

    gcheck = global_checks(qwen_final_md, manifest, expected_pages,
                           baseline_words)
    report["merge"] = {
        "slides": len(gcheck["slides"]),
        "word_count": gcheck["word_count"],
        "word_delta_vs_postprocessed":
            gcheck["word_delta_vs_postprocessed"],
        "problems": gcheck["problems"],
        "status": gcheck["status"],
        "final_md": str(qwen_final_md),
    }
    write_report(outdir / "collate_report.json", report)

    if args.no_cache_dir:
        shutil.rmtree(outdir / "pages", ignore_errors=True)

    dt = time.time() - t_start
    print(f"\nDone in {dt / 60:.1f} min. Final markdown: {qwen_final_md}")
    if gcheck["status"] != "ok":
        print(f"WARNING: global checks report {len(gcheck['problems'])} "
              f"problem(s); see {outdir / 'collate_report.json'}")
        return 2
    print("Global validation: clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
