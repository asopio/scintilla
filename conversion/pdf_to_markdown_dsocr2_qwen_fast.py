#!/usr/bin/env python3
"""Batch DSOCR2 OCR + Qwen3.8 collation: PDFs -> canonical Markdown,
parallel page processing + dedicated collation workers.

A batch-oriented re-architecture of ``pdf_to_markdown_dsocr2_qwen.py``
following the workflow of ``pdf_md_convert_vlm_fast.py``:

  * one thread pool converts pages (from one or many documents) in
    parallel.  Each page job renders the page to ``pages/page-NN.png``
    (cached), OCRed it via the DSOCR2 vLLM pod (raw output cached in
    ``raw_pages/page-NN.txt``), and crops the figure det boxes into
    ``figures/`` plus a per-page manifest entry in
    ``raw_pages/figures_page-NN.json``.
  * as soon as a page's OCR completes, a figure-description job (same
    pool by default; ``--description-workers N`` with N > 0 gives them
    a dedicated N-thread pool so they never queue behind OCR) sends
    the whole rendered page image plus the page's OCR text to Qwen3.8,
    which returns one concise description per figure block in a fixed
    JSON schema (``desc_figure_1``, ``desc_figure_2``, ...); the values
    are cached in the per-page manifest (a ``"description"`` field on
    each entry).  ``--skip-figure-descriptions`` opts out.  A page
    whose description fails (one corrective retry on unparseable JSON)
    warns and continues with undescribed figures.
  * a small dedicated set of collation workers runs in parallel with
    the pool (``--collation-workers``, default ``workers//16 + 1``;
    the OCR page pool gets ``workers - collation-workers`` threads).
    A document is handed to a worker as soon as all of its pages are
    OCR'd and all of its figure pages described (the worker re-checks
    the raw pages on disk, so it can never build a partial deck).  It
    first injects the cached descriptions as
    ``<!-- Figure k, description: ... -->`` comments after each
    ``![Figure k](...)`` image tag in the post-processed markdown, then
    performs the single-document
    script's per-document pipeline, reusing that script's stage
    machinery (``pdf_to_markdown_dsocr2_qwen``):

      1.5  raw dump + figure-manifest assembly +
           ``dsocr2_postprocess.postprocess()`` -> ``md/<stem>_dsocr2.md``
      2    Qwen3.8 collation (diff/full, chunked by the mandatory
           client-side token budget guard) -> ``md_chunks/NN_<stem>.md``
      3    Qwen3.8 agentic per-chunk review (deepagents)
      4    deterministic merge -> ``md/<stem>_qwen_final.md`` + global
           checks + ``collate_report.json``

  Because the pool and the collation workers hit different servers
  (DSOCR2 ``:18000`` vs Qwen ``:8008``), OCR of later documents
  overlaps with collation of earlier ones.  Each document's pipeline
  stays sequential (one worker owns it at a time); with more than one
  collation worker, different documents' Qwen jobs may also run
  concurrently.

Output layout (per document; with ``--output-dir`` each document goes
to ``<output-dir>/<rel-parent>/<stem>/``, mirroring the input layout;
without it, to the PDF's own directory as in the single-document
script):
  <outdir>/
    pages/page-NN.png               rendered page images (cache)
    raw_pages/page-NN.txt           raw per-page OCR output (cache)
    raw_pages/figures_page-NN.json  per-page figure manifest entries
                                    ("description" per entry once the
                                    Qwen pass has run)
    <stem>_dsocr2.md                raw grounding dump
    figures/page-NN-fKK.png         cropped figures
    figures_manifest.json           assembled crop manifest
    md/<stem>_dsocr2.md             post-processed markdown
    md_chunks/NN_<stem>.md          collated chunks (resume cache)
    md/<stem>_qwen_final.md         final merged markdown
    collate_report.json             per-stage report
    review_log.jsonl                per-chunk review log
    review_traces/NN_<stem>.jsonl   per-chunk agent trace (LangChain
                                    callback events: LLM calls + tools)

Usage:
  cd conversion
  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py \
      --input-dir ./pdfs --output-dir ./out --workers 8
  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py a.pdf b.pdf
  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py a.pdf -o out/
  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py a.pdf --plan-only
  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py a.pdf --skip-ocr

Resume: page jobs skip pages whose ``raw_pages/page-NN.txt`` exists
(unless ``--force``); figure-description jobs skip pages whose
per-page manifest entries all already carry a ``"description"``
(unless ``--force``); a collation worker skips chunks whose
``md_chunks/NN_<stem>.md`` exists; a document whose
``md/<stem>_qwen_final.md`` exists is skipped entirely.

``--force`` semantics: re-render, re-OCR, re-crop, and re-collate
everything.  ``--force --skip-ocr`` keeps the cached raw pages but
re-collates/re-reviews/re-merges.  A document whose page pool aborted
(a page failed without ``--continue-on-error``) is handed to the
worker, which reports it ``incomplete``; a re-run retries only the
missing pages, then finishes the document.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
from langchain_openai import ChatOpenAI
from openai import OpenAI

import dsocr2_postprocess
from pdf_to_markdown_agentic import get_pdf_page_count, render_pdf_page
from pdf_to_markdown_dsocr2 import (
    build_raw_dump,
    crop_figures,
    ocr_page_image,
    page_width,
)
from pdf_to_markdown_dsocr2_qwen import (
    Chunk,
    TokenCounter,
    collate_chunk,
    global_checks,
    group_pages,
    join_slides,
    merge_chunks,
    plan_chunks,
    raw_figure_appendix,
    review_chunk,
    split_raw_pages,
    split_slides,
    word_count,
    write_report,
)

# ---------------------------------------------------------------------------
# Job / document records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PageJob:
    """Everything a pool thread needs to convert one page."""

    pdf_path: str
    pdf_name: str
    page_number: int
    outdir: str
    width: int
    dpi: int
    force: bool
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    retries: int
    skip_figures: bool
    dry_run: bool


@dataclass(frozen=True)
class DescJob:
    """Everything a pool thread needs to describe one page's figures
    (whole-page image + page text -> one Qwen call, fixed JSON schema).
    Runs in the same pool as the page jobs, right after the page's OCR
    completes."""

    pdf_path: str
    pdf_name: str
    page_number: int
    outdir: str
    width: int
    dpi: int
    force: bool
    base_url: str
    api_key: str
    model: str
    retries: int


@dataclass(frozen=True)
class DocumentRecord:
    """Bookkeeping for one PDF (main thread only)."""

    pdf_path: str
    pdf_name: str
    stem: str
    num_pages: int          # total pages in the PDF
    first_page: int         # 1-indexed, inclusive (requested range)
    last_page: int          # 1-indexed, inclusive (requested range)
    outdir: str

    @property
    def expected_pages(self) -> list[int]:
        return list(range(self.first_page, self.last_page + 1))

    @property
    def width(self) -> int:
        return page_width(self.num_pages)


class DocState:
    """Main-thread bookkeeping for one document.

    ``finished`` counts the document's requested pages that are done --
    already on disk at startup, completed in this run, replaced by a
    blank placeholder (``--continue-on-error``), or cancelled because
    the document aborted.  ``desc_requested`` / ``desc_pending`` count
    the figure-description jobs seeded for the document's pages
    (``desc_pending`` reaches zero as they complete).  Once
    ``finished`` reaches the expected count AND ``desc_pending`` is
    zero, the document is handed to the collation workers (which
    re-verify on disk, so a partial deck can never be built).
    """

    __slots__ = ("aborted", "desc_done", "desc_failed", "desc_pending",
                 "desc_requested", "doc", "finished", "handed_off")

    def __init__(self, doc: DocumentRecord) -> None:
        self.doc = doc
        self.finished = 0
        self.handed_off = False
        self.aborted = False
        self.desc_requested = 0
        self.desc_pending = 0
        self.desc_done = 0
        self.desc_failed = 0


# ---------------------------------------------------------------------------
# Shared LLM clients (one per endpoint + timeout; the OpenAI client is
# thread-safe, so the pool shares them)
# ---------------------------------------------------------------------------

_CLIENT_CACHE: dict[tuple, OpenAI] = {}
_CLIENT_LOCK = threading.Lock()


def _get_llm_client(base_url: str, api_key: str, timeout: float,
                    retries: int) -> OpenAI:
    key = (base_url, api_key, timeout, retries)
    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(key)
        if client is None:
            client = OpenAI(base_url=base_url, api_key=api_key,
                            timeout=timeout, max_retries=retries)
            _CLIENT_CACHE[key] = client
        return client


def _get_dsocr2_client(base_url: str, api_key: str, timeout: float,
                       retries: int) -> OpenAI:
    return _get_llm_client(base_url, api_key, timeout, retries)


def _get_qwen_client(base_url: str, api_key: str, timeout: float,
                     retries: int) -> OpenAI:
    """Qwen client for the figure-description calls.  Shares the client
    cache; the key includes the timeout, so the description client
    (DESC_TIMEOUT) never collides with the collation client (7200 s)."""
    return _get_llm_client(base_url, api_key, timeout, retries)


# ---------------------------------------------------------------------------
# Page worker (runs in the thread pool)
# ---------------------------------------------------------------------------

def _page_paths(job: PageJob) -> tuple[Path, Path, Path]:
    outdir = Path(job.outdir)
    n, w = job.page_number, job.width
    return (outdir / "pages" / f"page-{n:0{w}d}.png",
            outdir / "raw_pages" / f"page-{n:0{w}d}.txt",
            outdir / "raw_pages" / f"figures_page-{n:0{w}d}.json")


def _figures_cached(figs_json: Path, outdir: Path) -> list[dict] | None:
    """Entries of an existing per-page figure manifest whose PNGs are all
    on disk; None when the file is missing or incomplete."""
    if not figs_json.is_file():
        return None
    try:
        entries = json.loads(figs_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not all((outdir / e["file"]).is_file() for e in entries):
        return None
    return entries


def run_page_job(job: PageJob) -> dict:
    """Pool entry point: render (if needed) + OCR (if needed) + figure
    crop (if needed) for one page.  Returns small stats.  Raises on a
    failed OCR request (the main thread decides placeholder vs abort).
    """
    img_path, raw_path, figs_json = _page_paths(job)
    outdir = Path(job.outdir)
    n = job.page_number
    t0 = time.time()

    def ensure_image() -> None:
        if img_path.exists() and not job.force:
            return
        if job.force and img_path.exists():
            img_path.unlink()
        img_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_pdf_page(job.pdf_path, str(img_path.parent / "page"),
                                   job.dpi, n)
        if rendered != str(img_path):
            os.replace(rendered, str(img_path))

    if job.dry_run:
        ensure_image()
        return {"status": "dry", "page": n, "pdf": job.pdf_name,
                "image_bytes": img_path.stat().st_size,
                "elapsed": time.time() - t0}

    ensure_image()

    ocred = job.force or not raw_path.exists()
    if ocred:
        client = _get_dsocr2_client(job.base_url, job.api_key,
                                    job.timeout, job.retries)
        text = ocr_page_image(client, job.model, str(img_path),
                              job.temperature, job.max_tokens)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(text, encoding="utf-8")
    else:
        text = raw_path.read_text(encoding="utf-8")

    if job.skip_figures:
        if job.force or not figs_json.exists():
            figs_json.parent.mkdir(parents=True, exist_ok=True)
            figs_json.write_text("[]\n", encoding="utf-8")
        n_figs = 0
    else:
        entries = None if job.force else _figures_cached(figs_json, outdir)
        if entries is None:
            # Reuse the existing per-document cropper on this single page;
            # it warns (not raises) on image blocks without det boxes.
            entries = crop_figures({n: text}, {n: str(img_path)},
                                    job.width, outdir / "figures")
            figs_json.parent.mkdir(parents=True, exist_ok=True)
            figs_json.write_text(json.dumps(entries, indent=1) + "\n",
                                 encoding="utf-8")
        n_figs = len(entries)

    return {"status": "ok", "page": n, "pdf": job.pdf_name,
            "chars": len(text), "figures": n_figs,
            "ocred": ocred, "elapsed": time.time() - t0}


def _write_placeholder(job: PageJob) -> None:
    """Replace a failed page with an empty raw file + empty figure
    manifest so the document completes with a blank slide
    (--continue-on-error)."""
    _img, raw_path, figs_json = _page_paths(job)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("", encoding="utf-8")
    if not job.skip_figures:
        figs_json.write_text("[]\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure-description worker (runs in the same pool as the page jobs,
# launched as soon as a page's OCR completes)
# ---------------------------------------------------------------------------

DESC_TIMEOUT = 600.0        # per description request (seconds)
DESC_MAX_TOKENS = 8192      # headroom for several figures per page
DESC_MARKER_RE = re.compile(r"^\s*<\|ref\|>.*<\|/(?:det|ref)\|>\s*$")


def _page_content(raw: str) -> str:
    """The page's OCR text with the layout marker lines stripped."""
    content = "\n".join(line for line in raw.splitlines()
                        if not DESC_MARKER_RE.match(line)).strip()
    return content or "(no text content extracted for this page)"


def _figure_desc_system(n: int) -> str:
    """System prompt for one page: the model must describe exactly the
    n figure blocks the OCR job extracted, in a fixed JSON schema."""
    keys = ", ".join(f'"desc_figure_{k}": ""' for k in range(1, n + 1))
    return (
        "The following image is a page from a document that contains exactly "
        f"{n} figure(s). Describe each figure: chart type, axes, legend, "
        "curves/bars, and the most salient values, maxima, minima or trends. "
        "Do not invent content that is not visible in the plots. Number the "
        "figures in reading order (top to bottom, then left to right). For "
        "context, the text content of the document page is also provided.\n\n"
        "Respond with ONLY a valid JSON object with exactly "
        f"{n} keys, in this form:\n{{{keys}}}\n"
        "Fill each value with one or two concise sentences describing that "
        "figure. No markdown fences, no commentary outside the JSON."
    )


def _parse_desc_json(text: str, n: int) -> dict[str, str] | None:
    """Parse the model's JSON response.  Returns None when unparseable;
    otherwise a (possibly partial) mapping ``desc_figure_k -> value`` --
    missing keys simply mean those figures stay undescribed."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    return {k: obj[k] for k in (f"desc_figure_{k}" for k in range(1, n + 1))
            if isinstance(obj.get(k), str) and obj.get(k).strip()}


def _page_needs_desc(outdir: Path, n: int, width: int,
                     force: bool) -> bool:
    """True when an already-OCR'd page still needs a description job:
    its per-page manifest has figure entries and (unless ``force``) at
    least one lacks a cached ``"description"``.  A missing file (e.g. a
    page OCR'd before per-page manifests existed) or an empty list means
    there is nothing to describe."""
    figs = outdir / "raw_pages" / f"figures_page-{n:0{width}d}.json"
    if not figs.is_file():
        return False
    try:
        entries = json.loads(figs.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not entries:
        return False
    if force:
        return True
    return any(not (e.get("description") or "").strip() for e in entries)


def run_figure_desc_job(job: DescJob) -> dict:
    """Pool entry point: one Qwen vision call describing every figure on
    one page.  The whole rendered page image plus the page's OCR text go
    in; a fixed-schema JSON object comes out.  ``desc_figure_k`` maps to
    the k-th entry of the per-page figure manifest (crop order = OCR
    reading order).  The values are written back into that manifest, so
    the collation stage picks them up with no extra plumbing.  Raises
    on a failed request or on JSON that stays unparseable after one
    corrective retry (the main thread warns and continues)."""
    outdir = Path(job.outdir)
    n = job.page_number
    w = job.width
    t0 = time.time()

    figs_json = outdir / "raw_pages" / f"figures_page-{n:0{w}d}.json"
    if not figs_json.is_file():
        raise RuntimeError(f"no per-page figure manifest {figs_json}")
    entries = json.loads(figs_json.read_text(encoding="utf-8"))
    if not entries:
        return {"status": "none", "page": n, "pdf": job.pdf_name,
                "figures": 0, "described": 0, "elapsed": time.time() - t0}

    raw_path = outdir / "raw_pages" / f"page-{n:0{w}d}.txt"
    raw = raw_path.read_text(encoding="utf-8") if raw_path.is_file() else ""

    img_path = outdir / "pages" / f"page-{n:0{w}d}.png"
    if not img_path.exists():
        # Re-render for a resume after --no-cache-dir removed pages/.
        img_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_pdf_page(job.pdf_path, str(img_path.parent / "page"),
                                   job.dpi, n)
        if rendered != str(img_path):
            os.replace(rendered, str(img_path))
    data_url = ("data:image/png;base64,"
                + base64.b64encode(img_path.read_bytes()).decode())

    client = _get_qwen_client(job.base_url, job.api_key, DESC_TIMEOUT,
                              job.retries)
    messages = [
        {"role": "system", "content": _figure_desc_system(len(entries))},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": data_url}},
            {"type": "text",
             "text": "<begin page content>\n" + _page_content(raw)
                     + "\n<end page content>"},
        ]},
    ]
    resp = client.chat.completions.create(
        model=job.model, messages=messages,
        temperature=0, max_tokens=DESC_MAX_TOKENS)
    content = resp.choices[0].message.content or ""
    parsed = _parse_desc_json(content, len(entries))
    if parsed is None:
        # One corrective retry: show the model its unparseable answer.
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user",
                         "content": "Your previous response was not valid "
                                     "JSON. Respond again with ONLY the "
                                     "JSON object."})
        resp = client.chat.completions.create(
            model=job.model, messages=messages,
            temperature=0, max_tokens=DESC_MAX_TOKENS)
        parsed = _parse_desc_json(resp.choices[0].message.content or "",
                                  len(entries))
        if parsed is None:
            raise RuntimeError(
                "figure description response was not valid JSON "
                "(after one corrective retry)")

    for k, entry in enumerate(entries, 1):
        v = parsed.get(f"desc_figure_{k}")
        if v is not None:
            entry["description"] = re.sub(r"\s+", " ", v).strip()
    figs_json.write_text(json.dumps(entries, indent=1) + "\n",
                         encoding="utf-8")
    return {"status": "ok", "page": n, "pdf": job.pdf_name,
            "figures": len(entries), "described": len(parsed),
            "elapsed": time.time() - t0}


# ---------------------------------------------------------------------------
# Figure-manifest assembly (collation thread)
# ---------------------------------------------------------------------------

def assemble_manifest(outdir: Path, doc: DocumentRecord,
                      note) -> list[dict]:
    """Assemble the document's figure manifest.

    Prefers the per-page entries written by the page pool
    (``raw_pages/figures_page-NN.json``); when those are not complete
    (e.g. pages OCR'd by the old single-document script), falls back to a
    pre-existing ``figures_manifest.json``, filtered to the requested
    page range.
    """
    raw_dir = outdir / "raw_pages"
    manifest_path = outdir / "figures_manifest.json"
    expected = doc.expected_pages
    per_page = [raw_dir / f"figures_page-{n:0{doc.width}d}.json"
                for n in expected]
    if all(p.is_file() for p in per_page):
        manifest: list[dict] = []
        for p in per_page:
            manifest.extend(json.loads(p.read_text(encoding="utf-8")))
        return manifest
    if manifest_path.is_file():
        legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
        have = set(expected)
        manifest = [e for e in legacy if e["page"] in have]
        note(f"using pre-existing {manifest_path.name} ({len(manifest)} "
             f"figure(s)); per-page manifest files are incomplete")
        return manifest
    manifest = []
    for p in per_page:
        if p.is_file():
            manifest.extend(json.loads(p.read_text(encoding="utf-8")))
    if manifest:
        note(f"assembling a partial figure manifest ({len(manifest)} "
             f"figure(s)) from the per-page files present")
    return manifest


# ---------------------------------------------------------------------------
# Figure-description injection (collation thread)
# ---------------------------------------------------------------------------

_IMG_TAG_RE = re.compile(r"^!\[Figure (\d+)\]\(([^)]+)\)\s*$")


def inject_figure_descriptions(text: str,
                               manifest: list[dict]) -> tuple[str, int]:
    """Insert a ``<!-- Figure k, description: ... -->`` comment line
    after each ``![Figure k](rel)`` image tag whose manifest entry
    carries a cached description.  The relative path is computed exactly
    the way ``dsocr2_postprocess`` builds it (``os.path.relpath(file,
    "md")`` -> ``../figures/...``), so the comment always lands on the
    matching image line.  Returns the new text and the number of
    comments injected."""
    desc_by_rel: dict[str, str] = {}
    for e in manifest:
        desc = (e.get("description") or "").strip()
        if not desc:
            continue
        desc_by_rel[os.path.relpath(e["file"], "md")] = desc

    lines = text.splitlines()
    out: list[str] = []
    injected = 0
    for line in lines:
        out.append(line)
        m = _IMG_TAG_RE.match(line)
        if m:
            desc = desc_by_rel.get(m.group(2))
            if desc:
                out.append("<!-- Figure " + m.group(1)
                           + ", description: "
                           + desc.replace("-->", "-- >") + " -->")
                injected += 1
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + trailing, injected


# ---------------------------------------------------------------------------
# Collation-thread resources
# ---------------------------------------------------------------------------

class _CollateContext:
    """Lazily created collation resources, shared by all collation
    threads.

    Each resource is created at most once (double-checked under a
    lock); the OpenAI clients and the local transformers tokenizer
    are thread-safe, so the workers share them.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._lock = threading.Lock()
        self._counter: TokenCounter | None = None
        self._qwen: OpenAI | None = None
        self._review_llm: ChatOpenAI | None = None

    def counter(self) -> TokenCounter:
        if self._counter is None:
            with self._lock:
                if self._counter is None:
                    self._counter = TokenCounter(self.args.qwen_model,
                                                 self.args.qwen_base_url)
                    self._counter(
                        "calibration")  # load the tokenizer up front
        return self._counter

    def qwen_client(self) -> OpenAI:
        if self._qwen is None:
            with self._lock:
                if self._qwen is None:
                    self._qwen = OpenAI(
                        base_url=self.args.qwen_base_url,
                        api_key=self.args.qwen_api_key,
                        timeout=httpx.Timeout(self.args.qwen_timeout,
                                              connect=30.0))
        return self._qwen

    def review_llm(self) -> ChatOpenAI:
        # Bounded per-call timeout: a request silently lost in the ssh
        # port-forward chain used to hang the review agent (observed
        # 2026-10-30: ~45 min in read()). 900 s is ample for a short
        # edit/validate agent turn.
        if self._review_llm is None:
            with self._lock:
                if self._review_llm is None:
                    self._review_llm = ChatOpenAI(
                        model=self.args.qwen_model,
                        api_key=self.args.qwen_api_key,
                        base_url=self.args.qwen_base_url,
                        temperature=0, max_retries=1, timeout=900)
        return self._review_llm


# ---------------------------------------------------------------------------
# Per-document pipeline (runs on a collation worker thread)
# ---------------------------------------------------------------------------

def process_document(doc: DocumentRecord, ctx: _CollateContext,
                     state: DocState) -> dict:
    """The single-document script's stages 1.5-4 for one document.

    Returns a small result record whose ``status`` is one of
    ``incomplete`` (raw pages missing on disk), ``error`` (pipeline
    failed), ``planned`` (--plan-only), or ``ok`` / ``dirty`` (the
    global-checks outcome of the merged document).
    """
    args = ctx.args
    outdir = Path(doc.outdir)
    width = doc.width
    expected = doc.expected_pages
    t0 = time.time()

    def say(msg: str) -> None:
        print(f"[{doc.pdf_name}] {msg}")

    # -- verify the raw pages on disk (never build a partial deck) --------
    raw_dir = outdir / "raw_pages"
    pages_text: dict[int, str] = {}
    missing: list[int] = []
    for n in expected:
        f = raw_dir / f"page-{n:0{width}d}.txt"
        if f.is_file():
            pages_text[n] = f.read_text(encoding="utf-8")
        else:
            missing.append(n)
    if missing:
        say(f"INCOMPLETE: {len(missing)} raw page(s) missing "
            f"({missing[:10]}{' ...' if len(missing) > 10 else ''}); "
            f"re-run to retry")
        return {"status": "incomplete", "missing": missing}

    # -- stage 1.5: raw dump + manifest + postprocess ----------------------
    dump_path = outdir / f"{doc.stem}_dsocr2.md"
    figures_dir = outdir / "figures"
    manifest_path = outdir / "figures_manifest.json"

    dump_path.write_text(build_raw_dump(
        doc.pdf_name, pages_text, doc.num_pages, args.dpi,
        args.dsocr2_model, args.dsocr2_max_tokens,
        args.dsocr2_temperature), encoding="utf-8")
    say(f"raw dump -> {dump_path}")

    manifest = assemble_manifest(outdir, doc,
                                 note=lambda m: say("note: " + m))
    manifest.sort(key=lambda e: (e["page"], e["file"]))
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n",
                             encoding="utf-8")
    figures_dir.mkdir(parents=True, exist_ok=True)

    dsocr2_md = outdir / "md" / f"{doc.stem}_dsocr2.md"
    pp = dsocr2_postprocess.postprocess(
        str(dump_path), str(figures_dir), str(manifest_path), str(dsocr2_md))
    say(f"postprocess: {pp['slides']} slides, {pp['images']} figures, "
        f"{pp['tables']} tables, {pp['equations']} equations -> "
        f"{dsocr2_md}")
    for prob in pp["problems"]:
        say(f"postprocess problem: {prob}")

    post_text = dsocr2_md.read_text(encoding="utf-8")
    desc_comments = 0
    if not args.skip_figure_descriptions:
        post_text, desc_comments = inject_figure_descriptions(post_text,
                                                              manifest)
        if desc_comments:
            dsocr2_md.write_text(post_text, encoding="utf-8")
            say(f"figure descriptions: {desc_comments} comment(s) "
                f"injected into {dsocr2_md.name}")
    try:
        _pre, post_sections = split_slides(post_text)
    except ValueError:
        say("postprocess produced no '## Slide N' sections")
        return {"status": "error",
                "error": "no slide sections in postprocessed markdown"}
    baseline_words = word_count(post_text)

    report: dict = {
        "document": doc.pdf_name,
        "pdf": doc.pdf_path,
        "pages": f"{doc.first_page}-{doc.last_page}",
        "collate_mode": args.collate_mode,
        "collate_input": args.collate_input,
        "context_limit": args.context_limit,
        "max_pages": args.max_pages,
        "dsocr2": {"base_url": args.dsocr2_base_url,
                   "model": args.dsocr2_model},
        "qwen": {"base_url": args.qwen_base_url,
                 "model": args.qwen_model},
        "ocr": {"pages_ocred": len(pages_text),
                "figures": len(manifest)},
        "figure_descriptions": {"requested": state.desc_requested,
                                "done": state.desc_done,
                                "failed": state.desc_failed,
                                "comments": desc_comments},
        "postprocess": {k: pp[k] for k in
                        ("slides", "images", "tables", "equations")},
    }
    if pp["problems"]:
        report["postprocess"]["problems"] = pp["problems"]

    # -- stage 2: collation -------------------------------------------------
    chunks_dir = outdir / "md_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    qwen_final_md = outdir / "md" / f"{doc.stem}_qwen_final.md"
    collate_mode = ("full" if (args.collate_input == "raw"
                               or args.collate_mode == "full")
                    else "diff")

    if args.skip_collate:
        say("collation skipped (--skip-collate)")
        report["collate"] = "skipped"
        chunk_texts: list[tuple[int, str]] = [
            (1, join_slides(post_sections))]
        chunk_width = 2
    else:
        counter = ctx.counter()
        page_set = set(expected)
        if args.collate_input == "raw":
            _rp, raw_sections = split_raw_pages(
                dump_path.read_text(encoding="utf-8"))
            sections = [s for s in raw_sections if s[0] in page_set]
            raw = True
        else:
            sections = [s for s in post_sections if s[0] in page_set]
            raw = False

        groups = group_pages(sections, args.max_pages,
                             args.chunk_overlap if collate_mode == "full"
                             else 0)
        all_chunks: list[Chunk] = []
        for group in groups:
            appendix = (raw_figure_appendix(manifest,
                                             [p for p, _ in group])
                        if raw else "")
            all_chunks.extend(plan_chunks(
                sections=group, max_pages=args.max_pages,
                counter=counter, mode=collate_mode,
                context_limit=args.context_limit, raw=raw,
                raw_appendix=appendix))
        for i, c in enumerate(all_chunks, 1):
            c.index = i
        chunk_width = max(2, len(str(len(all_chunks))))

        say(f"collation ({collate_mode} mode, {args.collate_input} input, "
            f"<= {args.max_pages} pages/job, {len(all_chunks)} chunk(s), "
            f"context limit {args.context_limit})")
        for c in all_chunks:
            a, b = c.pages
            say(f"  chunk {c.index:0{chunk_width}d}: slides {a}-{b} "
                f"({b - a + 1} pages)")

        if args.plan_only:
            report["collate"] = [
                {"index": c.index,
                 "pages": f"{c.pages[0]}-{c.pages[1]}",
                 "status": "planned"} for c in all_chunks]
            report["plan_only"] = True
            write_report(outdir / "collate_report.json", report)
            say(f"plan-only: {len(all_chunks)} chunk(s) planned, "
                f"no LLM calls")
            return {"status": "planned", "chunks": len(all_chunks)}

        client = ctx.qwen_client()
        chunk_texts = []
        collate_entries = []
        for chunk in all_chunks:
            label = f"{chunk.index:0{chunk_width}d}"
            chunk_path = chunks_dir / f"{label}_{doc.stem}.md"
            post_fb = [s for s in post_sections
                       if chunk.pages[0] <= s[0] <= chunk.pages[1]]
            if chunk_path.exists() and not args.force:
                text = chunk_path.read_text(encoding="utf-8")
                entry = {"index": chunk.index,
                         "pages": f"{chunk.pages[0]}-{chunk.pages[1]}",
                         "status": "cached"}
                say(f"chunk {label}: cached")
            else:
                try:
                    text, entry = collate_chunk(chunk, client, args, counter,
                                                chunk_path, manifest,
                                                post_fb)
                except RuntimeError as exc:
                    report["collate"] = collate_entries
                    write_report(outdir / "collate_report.json", report)
                    say(f"ERROR: {exc}")
                    return {"status": "error", "error": str(exc)}
                say(f"chunk {label}: {entry['status']} "
                    f"({entry['input_tokens']} input tokens, "
                    f"{entry.get('output_tokens', '?')} output, "
                    f"{entry['elapsed_s']}s)")
            chunk_texts.append((chunk.index, text))
            collate_entries.append(entry)
        report["collate"] = collate_entries
        write_report(outdir / "collate_report.json", report)

    # -- stage 3: per-chunk agentic review -----------------------------------
    review_log: list[dict] = []
    if args.skip_review:
        say("review skipped (--skip-review)")
        report["review"] = "skipped"
    else:
        say(f"review (<= {args.max_review_iterations} agent "
            f"iterations/chunk)")
        llm = ctx.review_llm()
        for idx, text in chunk_texts:
            label = f"{idx:0{chunk_width}d}"
            chunk_path = chunks_dir / f"{label}_{doc.stem}.md"
            if not chunk_path.exists():
                chunk_path.write_text(text.rstrip("\n") + "\n",
                                      encoding="utf-8")
            try:
                _pre, secs = split_slides(chunk_path.read_text(
                    encoding="utf-8"))
                pages = [p for p, _ in secs]
            except ValueError:
                pages = []
            if not pages:
                pages = expected
            status = review_chunk(chunk_path, pages, manifest_path,
                                  manifest, llm,
                                  args.max_review_iterations, review_log,
                                  trace_path=outdir / "review_traces" /
                                  f"{label}_{doc.stem}.jsonl")
            say(f"chunk {label}: review {status}")
        report["review"] = review_log
        write_report(outdir / "review_log.jsonl",
                     [json.dumps(r) for r in review_log], as_lines=True)

    # -- stage 4: merge + global checks ---------------------------------------
    say("merge")
    merged = merge_chunks(chunk_texts, doc.stem)
    qwen_final_md.parent.mkdir(parents=True, exist_ok=True)
    qwen_final_md.write_text(merged, encoding="utf-8")
    say(f"wrote {qwen_final_md}")

    gcheck = global_checks(qwen_final_md, manifest, expected,
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

    dt = time.time() - t0
    if gcheck["status"] != "ok":
        say(f"done in {dt / 60:.1f} min; global checks: "
            f"{len(gcheck['problems'])} problem(s)")
    else:
        say(f"done in {dt / 60:.1f} min; global checks clean")
    return {"status": gcheck["status"], "final_md": str(qwen_final_md),
            "problems": len(gcheck["problems"]),
            "elapsed_s": round(dt, 1)}


def _collation_worker(requests: queue.Queue, results: queue.Queue,
                      ctx: _CollateContext) -> None:
    """One collation worker thread (N of them consume the same request
    queue), mirroring ``pdf_md_convert_vlm_fast._collation_worker``.

    Pulls whole documents (all of their pages done), runs the
    per-document post-OCR pipeline, and reports the outcome.  Each
    document is processed by exactly one worker, so its stages stay
    sequential; different documents' Qwen jobs may overlap.  A worker
    simply stays idle while the page pool is busy.
    """
    while True:
        state: DocState | None = requests.get()
        if state is None:  # sentinel: shut down
            break
        doc = state.doc
        try:
            result = process_document(doc, ctx, state)
        except Exception as exc:
            print(f"COLLATE FAILED {doc.pdf_name}:\n"
                  f"{traceback.format_exc()}", file=sys.stderr)
            results.put(("error", doc.pdf_name, str(exc)))
            continue
        status = result.get("status", "error")
        if status == "incomplete":
            info = (f"{len(result.get('missing', []))} page(s) missing; "
                    f"re-run to retry")
        elif status == "planned":
            info = f"{result.get('chunks', 0)} chunk(s) planned"
        else:
            info = result.get("final_md") or result.get("error") or ""
        results.put((status, doc.pdf_name, info))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py \\\n"
            "      --input-dir ./pdfs --output-dir ./out --workers 8\n"
            "  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py a.pdf b.pdf\n"
            "  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py a.pdf -o out/\n"
            "  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py a.pdf "
            "--plan-only\n"
            "  pixi run python pdf_to_markdown_dsocr2_qwen_fast.py a.pdf "
            "--skip-ocr\n"))
    p.add_argument("pdfs", nargs="*",
                   help="explicit PDF files (otherwise --input-dir is "
                        "scanned recursively)")
    p.add_argument("--input-dir", default="./pdfs",
                   help="directory scanned for PDFs when no explicit files "
                        "are given (default: %(default)s)")
    p.add_argument("--pattern", default="*.pdf",
                   help="glob pattern for PDF discovery "
                        "(default: %(default)s)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="output root; each document is converted into "
                        "<output-dir>/<rel-parent>/<stem>/, mirroring the "
                        "input layout (default: each PDF's own directory)")
    # worker budget: OCR page pool + collation workers
    p.add_argument("--workers", type=int, default=8,
                   help="total worker threads: the OCR page pool gets "
                        "workers - collation-workers (default: %(default)s)")
    p.add_argument("--collation-workers", type=int, default=None,
                   help="dedicated collation + review threads (default: "
                        "workers//16 + 1; the rest of --workers do OCR)")
    p.add_argument("--description-workers", type=int, default=0,
                   help="dedicated thread pool for figure-description "
                        "jobs, so they never queue behind OCR (default: "
                        "0 = share the OCR page pool)")
    p.add_argument("--dpi", type=int, default=150,
                   help="page render DPI (default: %(default)s)")
    p.add_argument("--start-page", type=int, default=0,
                   help="first page, 0-indexed, inclusive (default: 0)")
    p.add_argument("--end-page", type=int, default=None,
                   help="last page, 0-indexed, exclusive (default: end)")
    p.add_argument("--max-doc-pages", type=int, default=0,
                   help="skip documents with more than this many pages "
                        "(default: 0 = no limit; --max-pages is the "
                        "per-Qwen-job cap, a different thing)")
    p.add_argument("--dsocr2-base-url", default="http://127.0.0.1:18000/v1",
                   help="OpenAI-compatible base URL of the DSOCR2 vLLM pod "
                        "(default: %(default)s)")
    p.add_argument("--dsocr2-model", default="deepseek-ocr2",
                   help="served DSOCR2 model name (default: %(default)s)")
    p.add_argument("--dsocr2-api-key", default="EMPTY")
    p.add_argument("--dsocr2-temperature", type=float, default=0.0)
    p.add_argument("--dsocr2-max-tokens", type=int, default=8180)
    p.add_argument("--dsocr2-timeout", type=float, default=1800.0,
                   help="per OCR request timeout in seconds "
                        "(default: %(default)s)")
    p.add_argument("--dsocr2-retries", type=int, default=2,
                   help="OpenAI client retries per OCR request "
                        "(default: %(default)s)")
    p.add_argument("--force", action="store_true",
                   help="re-render, re-OCR, re-crop, and re-collate cached "
                        "items")
    p.add_argument("--continue-on-error", action="store_true",
                   help="replace a page whose OCR fails with a blank "
                        "placeholder so its document still completes")
    p.add_argument("--fail-fast", action="store_true",
                   help="stop at the first page error")
    p.add_argument("--skip-figures", action="store_true",
                   help="do not crop figures (empty manifest)")
    p.add_argument("--skip-figure-descriptions", action="store_true",
                   help="do not ask Qwen for figure descriptions (by "
                        "default every figure page is described by one "
                        "whole-page Qwen call as soon as its OCR "
                        "completes, and the descriptions are injected "
                        "as <!-- Figure k, description: ... --> comments)")
    p.add_argument("--skip-ocr", action="store_true",
                   help="no page jobs; hand every document straight to the "
                        "collation workers (its raw pages must be on disk; "
                        "with --force: keep raw pages, redo collation and "
                        "later stages)")
    p.add_argument("--dry-run", action="store_true",
                   help="render the pages in the pool and report sizes; no "
                        "OCR, no collation")
    # stage 2 (Qwen collation, on the collation worker threads)
    p.add_argument("--qwen-base-url", default="http://127.0.0.1:8008/v1",
                   help="OpenAI-compatible base URL of the Qwen3.8 vLLM "
                        "pod (default: %(default)s)")
    p.add_argument("--qwen-model", default="Qwen/Qwen3.8-27B-FP8",
                   help="served Qwen model name (default: %(default)s)")
    p.add_argument("--qwen-api-key", default="EMPTY")
    p.add_argument("--qwen-timeout", type=float, default=7200.0,
                   help="per collation call timeout in seconds (default: "
                        "%(default)s; a full-mode 100k-token decode takes "
                        "~70 min)")
    p.add_argument("--qwen-retries", type=int, default=3,
                   help="whole-chunk retries on HTTP failure "
                        "(default: %(default)s)")
    p.add_argument("--collate-mode", choices=["diff", "full"],
                   default="diff",
                   help="collation contract (default: diff)")
    p.add_argument("--collate-input",
                   choices=["postprocessed", "raw"],
                   default="postprocessed",
                   help="collation input (default: postprocessed; raw "
                        "implies full mode)")
    p.add_argument("--context-limit", type=int, default=262144,
                   help="token budget for prompt+max_tokens (default: "
                        "%(default)s native; 330000 is verified YARN "
                        "headroom, diff mode only)")
    p.add_argument("--max-pages", type=int, default=40,
                   help="max pages per Qwen collation job "
                        "(default: %(default)s)")
    p.add_argument("--chunk-overlap", type=int, default=0,
                   help="overlap slides between chunks, full mode only "
                        "(default: %(default)s)")
    p.add_argument("--skip-collate", action="store_true",
                   help="skip Qwen collation (pure dsocr2 add-on mode)")
    p.add_argument("--plan-only", action="store_true",
                   help="build the chunk plans and exit (no Qwen calls; "
                        "the page pool still OCRs unless its cache is "
                        "warm)")
    # stage 3 (review)
    p.add_argument("--skip-review", action="store_true",
                   help="skip the per-chunk agentic review loop")
    p.add_argument("--max-review-iterations", type=int, default=3,
                   help="agentic review cap per chunk (default: %(default)s)")
    p.add_argument("--no-cache-dir", action="store_true",
                   help="delete rendered page PNGs after each document "
                        "completes")
    return p.parse_args(argv)


def outdir_for(pdf_path: Path, input_dir: Path | None,
               output_dir: Path | None) -> Path:
    """Per-document output directory.

    With ``--output-dir``: ``<output-dir>/<rel-parent>/<stem>`` (mirrors
    the input layout; explicit files have no rel-parent).  Without it:
    the PDF's own directory, as in the single-document script.
    """
    if output_dir is None:
        return pdf_path.parent
    if input_dir is not None:
        return output_dir / pdf_path.parent.relative_to(input_dir) \
            / pdf_path.stem
    return output_dir / pdf_path.stem


def _make_job(doc: DocumentRecord, n: int, args: argparse.Namespace,
              dry_run: bool = False) -> PageJob:
    return PageJob(
        pdf_path=doc.pdf_path,
        pdf_name=doc.pdf_name,
        page_number=n,
        outdir=doc.outdir,
        width=doc.width,
        dpi=args.dpi,
        force=args.force,
        base_url=args.dsocr2_base_url,
        api_key=args.dsocr2_api_key,
        model=args.dsocr2_model,
        temperature=args.dsocr2_temperature,
        max_tokens=args.dsocr2_max_tokens,
        timeout=args.dsocr2_timeout,
        retries=args.dsocr2_retries,
        skip_figures=args.skip_figures,
        dry_run=dry_run,
    )


def _make_desc_job(doc: DocumentRecord, n: int,
                   args: argparse.Namespace) -> DescJob:
    return DescJob(
        pdf_path=doc.pdf_path,
        pdf_name=doc.pdf_name,
        page_number=n,
        outdir=doc.outdir,
        width=doc.width,
        dpi=args.dpi,
        force=args.force,
        base_url=args.qwen_base_url,
        api_key=args.qwen_api_key,
        model=args.qwen_model,
        retries=args.qwen_retries,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

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

    # ---- worker budget: OCR page pool + collation workers ---------------
    total_workers = max(1, args.workers)
    if args.collation_workers is None:
        collation_workers = total_workers // 16 + 1
    else:
        collation_workers = args.collation_workers
    page_workers = total_workers - collation_workers
    if collation_workers < 1 or page_workers < 1:
        print(f"error: --workers {total_workers} must leave at least one "
              f"OCR worker after reserving --collation-workers "
              f"{collation_workers}; raise --workers or lower "
              f"--collation-workers", file=sys.stderr)
        return 1
    description_workers = args.description_workers
    if description_workers < 0:
        print("error: --description-workers must be >= 0 (0 = share the "
              "OCR page pool with the figure-description jobs)",
              file=sys.stderr)
        return 1

    # ---- discover documents ----------------------------------------------
    if args.pdfs:
        input_dir: Path | None = None
        pdf_paths: list[Path] = []
        for spec in args.pdfs:
            pp = Path(spec).expanduser().resolve()
            if not pp.is_file():
                print(f"error: PDF not found: {pp}", file=sys.stderr)
                return 1
            pdf_paths.append(pp)
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            print(f"error: input directory does not exist: {input_dir}",
                  file=sys.stderr)
            return 2
        pdf_paths = sorted(p for p in input_dir.rglob(args.pattern)
                           if p.is_file())
        if not pdf_paths:
            print(f"No PDFs found in {input_dir} matching {args.pattern}")
            return 0

    output_dir = (Path(args.output_dir).expanduser().resolve()
                  if args.output_dir else None)

    documents: list[DocumentRecord] = []
    for pdf_path in pdf_paths:
        try:
            num_pages = get_pdf_page_count(str(pdf_path))
        except Exception as exc:
            print(f"warning: could not read page count for "
                  f"{pdf_path.name}: {exc}", file=sys.stderr)
            continue
        if args.max_doc_pages and num_pages > args.max_doc_pages:
            print(f"skip {pdf_path.name}: {num_pages} pages > "
                  f"--max-doc-pages {args.max_doc_pages}", file=sys.stderr)
            continue
        first = args.start_page + 1
        last = args.end_page if args.end_page is not None else num_pages
        last = min(last, num_pages)
        if first > last:
            print(f"skip {pdf_path.name}: page range "
                  f"{args.start_page}-{args.end_page} covers none of its "
                  f"{num_pages} pages", file=sys.stderr)
            continue
        documents.append(DocumentRecord(
            pdf_path=str(pdf_path),
            pdf_name=pdf_path.name,
            stem=pdf_path.stem,
            num_pages=num_pages,
            first_page=first,
            last_page=last,
            outdir=str(outdir_for(pdf_path, input_dir, output_dir)),
        ))

    if not documents:
        print("no usable PDFs")
        return 0

    # Two documents must never share an output directory: their pages/,
    # raw_pages/ and figures/ caches are keyed by page number only.
    seen: dict[str, str] = {}
    for doc in documents:
        key = str(Path(doc.outdir).resolve())
        if key in seen:
            print(f"error: {doc.pdf_name} and {seen[key]} would share "
                  f"output directory {key}; pass --output-dir so each "
                  f"document gets its own <stem>/ directory",
                  file=sys.stderr)
            return 1
        seen[key] = doc.pdf_name

    # ---- build page jobs + per-document state ------------------------------
    jobs: list[PageJob] = []
    # Description jobs for pages already on disk (cached raw / --skip-ocr);
    # pages OCR'd in this run seed their own description job as soon as
    # the page job completes.  Counters below must be set BEFORE the
    # initial hand-off loop, or cached documents would be handed over
    # before their description jobs are seeded.
    desc_jobs: list[DescJob] = []
    desc_enabled = (not args.dry_run and not args.plan_only
                    and not args.skip_figure_descriptions)
    doc_states: list[DocState] = []
    skipped_docs: list[str] = []
    pages_total = 0

    for doc in documents:
        final_md = Path(doc.outdir) / "md" / f"{doc.stem}_qwen_final.md"
        if final_md.exists() and not args.force:
            skipped_docs.append(doc.pdf_name)
            print(f"skip {doc.pdf_name}: {final_md} already exists "
                  f"(use --force to redo)")
            continue
        state = DocState(doc)
        doc_states.append(state)
        for n in range(doc.first_page, doc.last_page + 1):
            pages_total += 1
            if args.dry_run:
                jobs.append(_make_job(doc, n, args, dry_run=True))
                continue
            done = args.skip_ocr
            if not done:
                raw = (Path(doc.outdir) / "raw_pages"
                       / f"page-{n:0{doc.width}d}.txt")
                done = raw.exists() and not args.force
            if done:
                state.finished += 1
                if desc_enabled and _page_needs_desc(
                        Path(doc.outdir), n, doc.width, args.force):
                    state.desc_requested += 1
                    state.desc_pending += 1
                    desc_jobs.append(_make_desc_job(doc, n, args))
                continue
            jobs.append(_make_job(doc, n, args))

    # ---- reachability checks (fail early, like the single-doc script) ------
    need_ocr = not args.dry_run and not args.skip_ocr and bool(jobs)
    need_qwen = ((not args.dry_run and bool(doc_states) and not args.plan_only
                  and not (args.skip_collate and args.skip_review))
                 or (desc_enabled and bool(desc_jobs)))

    if need_ocr:
        client = _get_dsocr2_client(args.dsocr2_base_url,
                                    args.dsocr2_api_key,
                                    args.dsocr2_timeout,
                                    args.dsocr2_retries)
        try:
            models = [m.id for m in client.models.list().data]
        except Exception as exc:
            print(f"error: cannot reach DSOCR2 server at "
                  f"{args.dsocr2_base_url}: {exc}", file=sys.stderr)
            print("hint: is the dsocr2 pod up and the port forward "
                  "(18000) alive?", file=sys.stderr)
            return 1
        if args.dsocr2_model not in models:
            print(f"warning: model {args.dsocr2_model!r} not in served "
                  f"models {models}", file=sys.stderr)

    if need_qwen:
        qclient = OpenAI(base_url=args.qwen_base_url,
                         api_key=args.qwen_api_key,
                         timeout=httpx.Timeout(60.0, connect=10.0))
        try:
            qmodels = [m.id for m in qclient.models.list().data]
        except Exception as exc:
            print(f"error: cannot reach Qwen server at "
                  f"{args.qwen_base_url}: {exc}", file=sys.stderr)
            return 1
        if args.qwen_model not in qmodels:
            print(f"warning: model {args.qwen_model!r} not in served "
                  f"models {qmodels}", file=sys.stderr)

    # ---- banner -------------------------------------------------------------
    t_start = time.time()
    # The OCR page pool hosts the description jobs too unless
    # --description-workers gives them a separate pool of their own.
    if description_workers:
        pool_workers = max(1, min(page_workers, len(jobs) or 1))
    else:
        pool_workers = max(1, min(page_workers, len(jobs) + len(desc_jobs)
                                  or 1))
    n_collators = (0 if (args.dry_run or not doc_states)
                   else min(collation_workers, len(doc_states)))
    print(f"DSOCR2 + Qwen3.8 batch workflow: {len(documents)} document(s), "
          f"{pages_total} page(s) requested, {len(skipped_docs)} "
          f"already done")
    if args.dry_run:
        print(f"Dry run: rendering {len(jobs)} page image(s) with up to "
              f"{pool_workers} parallel pdftoppm workers "
              f"(no OCR, no collation)")
    else:
        desc_note = (f" + {description_workers} dedicated description "
                     f"thread(s)" if description_workers else "")
        print(f"Page pool: {len(jobs)} OCR job(s) + {len(desc_jobs)} "
              f"figure-description job(s) with up to {pool_workers} "
              f"threads{desc_note} -> {args.dsocr2_base_url} "
              f"({args.dsocr2_model}) / {args.qwen_base_url} "
              f"({args.qwen_model}); {n_collators} collation thread(s) "
              f"for {len(doc_states)} document(s)")

    # ---- collation workers (run in parallel with the page pool) ----------
    requests: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    ctx = _CollateContext(args)
    collators: list[threading.Thread] = []
    handed = 0

    def hand_off(state: DocState) -> None:
        """Hand a document to the collation workers once all of its
        pages are done AND every figure page is described (a worker
        re-verifies on disk before building)."""
        nonlocal handed
        if not collators or state.handed_off:
            return
        expected_n = state.doc.last_page - state.doc.first_page + 1
        if state.finished >= expected_n and state.desc_pending == 0:
            state.handed_off = True
            handed += 1
            desc_note = (f"; {state.desc_done}/{state.desc_requested} "
                         f"figure page(s) described") \
                if state.desc_requested else ""
            print(f"handing off {state.doc.pdf_name} to the collation "
                  f"worker(s) ({state.finished}/{expected_n} pages "
                  f"ready{desc_note})")
            requests.put(state)

    if n_collators:
        for i in range(n_collators):
            collators.append(threading.Thread(
                target=_collation_worker, args=(requests, results, ctx),
                name=f"collation-worker-{i + 1}", daemon=True))
        for t in collators:
            t.start()
        # Documents whose pages are all already on disk (or --skip-ocr):
        # hand them over right away.
        for state in doc_states:
            hand_off(state)

    # ---- page pool ------------------------------------------------------------
    state_by_pdf = {s.doc.pdf_path: s for s in doc_states}
    pages_ok = 0
    pages_failed = 0
    placeholders = 0
    dry_count = 0
    desc_ok = 0
    desc_failed = 0
    stopping = False

    if jobs or desc_jobs:
        with contextlib.ExitStack() as stack:
            page_pool = stack.enter_context(
                ThreadPoolExecutor(max_workers=pool_workers,
                                   thread_name_prefix="page"))
            # Figure-description jobs either share the OCR pool (default)
            # or run in a dedicated --description-workers pool so they
            # never queue behind OCR work.  Both pools feed the same
            # completion queue below.
            desc_pool = (stack.enter_context(
                ThreadPoolExecutor(max_workers=description_workers,
                                   thread_name_prefix="desc"))
                if description_workers else None)
            # A page job that succeeds can spawn a figure-description
            # job, so a static as_completed over the initial batch could
            # never see the later futures.  Instead every submitted
            # future pushes (kind, state, job, payload) onto a
            # completion queue, and the loop below drains it.
            futures: dict = {}
            completions: queue.Queue = queue.Queue()
            outstanding = 0

            def track(kind: str, state: DocState | None, job, fn) -> None:
                nonlocal outstanding
                pool_ = (desc_pool
                         if (kind == "desc" and description_workers)
                         else page_pool)
                fut = pool_.submit(fn, job)
                futures[fut] = (kind, state, job)
                outstanding += 1

                def done(f) -> None:
                    nonlocal outstanding
                    outstanding -= 1
                    try:
                        payload = f.result()
                    except CancelledError:
                        payload = "cancelled"
                    except Exception as error:
                        payload = error
                    completions.put((kind, state, job, payload))

                fut.add_done_callback(done)

            for djob in desc_jobs:
                track("desc", state_by_pdf.get(djob.pdf_path), djob,
                      run_figure_desc_job)
            for job in jobs:
                track("page", state_by_pdf.get(job.pdf_path), job,
                      run_page_job)

            pages_done = 0

            def progress() -> None:
                """Consolidated progress line: pages, figure-description
                pages, and hand-offs.  Printed on every description
                completion and every 10 page completions."""
                elapsed = time.time() - t_start
                d_requested = sum(s.desc_requested for s in doc_states)
                d_settled = sum(s.desc_done + s.desc_failed
                                for s in doc_states)
                d_failed = sum(s.desc_failed for s in doc_states)
                if len(jobs):
                    page_part = (f"{pages_done}/{len(jobs)} pages "
                                 f"({pages_ok} ok, {pages_failed} failed, "
                                 f"{placeholders} placeholders)"
                                 + (f", avg {elapsed / pages_done:.2f}s/page"
                                    if pages_done else "")
                                 + "; ")
                else:
                    page_part = ""
                print(f"Progress: {page_part}"
                      f"{d_settled}/{d_requested} pages described"
                      + (f" ({d_failed} failed)" if d_failed else "")
                      + f"; {handed}/{len(doc_states)} document(s) "
                        f"handed off for review")

            while outstanding > 0:
                kind, state, job, payload = completions.get()

                if kind == "desc":
                    if payload == "cancelled":
                        if state is not None:
                            state.desc_pending -= 1
                            hand_off(state)
                        continue
                    if isinstance(payload, Exception):
                        desc_failed += 1
                        if state is not None:
                            state.desc_pending -= 1
                            state.desc_failed += 1
                            hand_off(state)
                        print(f"[{job.pdf_name}] figure descriptions for "
                              f"page {job.page_number} failed: {payload} "
                              f"(document continues without them)",
                              file=sys.stderr)
                    else:
                        desc_ok += 1
                        if state is not None:
                            state.desc_pending -= 1
                            state.desc_done += 1
                            hand_off(state)
                        print(f"[{job.pdf_name}] described "
                              f"{payload['described']}/{payload['figures']} "
                              f"figure(s) on page {job.page_number} "
                              f"({payload['elapsed']:.1f}s)")
                    progress()
                    continue

                pages_done += 1
                if payload == "cancelled":
                    # Page never ran (its document aborted); count it so
                    # the hand-off can trigger.
                    if state is not None:
                        state.finished += 1
                        hand_off(state)
                    continue

                if isinstance(payload, Exception):
                    error = payload
                    pages_failed += 1
                    if state is not None:
                        state.finished += 1
                        if args.continue_on_error:
                            _write_placeholder(job)
                            placeholders += 1
                            print(f"[{pages_done}/{len(jobs)}] placeholder "
                                  f"{job.pdf_name} page {job.page_number} "
                                  f"(OCR failed: {error})")
                        else:
                            state.aborted = True
                            # Cancel the document's remaining page AND
                            # description futures.
                            for fut, (_fkind, _fst, fj) in futures.items():
                                if fj.pdf_path == job.pdf_path:
                                    fut.cancel()
                            print(f"[{pages_done}/{len(jobs)}] FAILED "
                                  f"{job.pdf_name} page {job.page_number}: "
                                  f"{error} (document aborted; re-run to "
                                  f"retry)", file=sys.stderr)
                            if args.fail_fast:
                                stopping = True
                                for fut in futures:
                                    fut.cancel()
                                break
                        hand_off(state)
                    else:
                        print(f"[{pages_done}/{len(jobs)}] FAILED "
                              f"{job.pdf_name} page {job.page_number}: "
                              f"{error}", file=sys.stderr)
                        if args.fail_fast:
                            stopping = True
                            for fut in futures:
                                fut.cancel()
                            break
                    continue

                result = payload
                if result["status"] == "dry":
                    dry_count += 1
                    print(f"[{pages_done}/{len(jobs)}] dry-run "
                          f"{job.pdf_name} page {job.page_number}: "
                          f"{result['image_bytes'] // 1024} KB "
                          f"({result['elapsed']:.1f}s)")
                else:
                    pages_ok += 1
                    verb = "ocr" if result["ocred"] else "cache"
                    print(f"[{pages_done}/{len(jobs)}] {verb} "
                          f"{job.pdf_name} page {job.page_number}: "
                          f"{result['chars']} chars, "
                          f"{result['figures']} figure(s), "
                          f"{result['elapsed']:.1f}s")
                    if (desc_enabled and state is not None
                            and result["figures"] > 0):
                        djob = _make_desc_job(state.doc, result["page"],
                                              args)
                        state.desc_requested += 1
                        state.desc_pending += 1
                        track("desc", state, djob, run_figure_desc_job)
                if state is not None:
                    state.finished += 1
                    hand_off(state)

                if pages_done % 10 == 0 or pages_done == len(jobs):
                    progress()

    # ---- drain the collation workers ---------------------------------------
    doc_results: list[tuple[str, str, str]] = []
    if collators:
        for _ in collators:
            requests.put(None)
        for t in collators:
            t.join()
        while True:
            try:
                doc_results.append(results.get_nowait())
            except queue.Empty:
                break

    # ---- summary ----------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} min: pages ok={pages_ok} "
          f"failed={pages_failed} placeholder={placeholders} "
          f"dry={dry_count} figure-descs ok={desc_ok} "
          f"failed={desc_failed}")
    by_name = {name: (status, info) for status, name, info in doc_results}
    for doc in documents:
        if doc.pdf_name in by_name:
            status, info = by_name[doc.pdf_name]
            print(f"  {doc.pdf_name}: {status} {info}")
        elif doc.pdf_name in skipped_docs:
            print(f"  {doc.pdf_name}: skipped (already done)")
        elif args.dry_run:
            print(f"  {doc.pdf_name}: dry-run (rendered, no collation)")
        else:
            print(f"  {doc.pdf_name}: not collated "
                  f"(fail-fast or aborted before hand-off)",
                  file=sys.stderr)

    if stopping or pages_failed or any(s == "error"
                                       for s, _n, _d in doc_results):
        return 1
    if any(s == "incomplete" for s, _n, _d in doc_results):
        print("Re-run to retry the incomplete document(s) "
              "(missing pages only).", file=sys.stderr)
        return 1
    if any(s == "dirty" for s, _n, _d in doc_results):
        print("WARNING: some document(s) finished with global-check "
              "problems; see their collate_report.json", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
