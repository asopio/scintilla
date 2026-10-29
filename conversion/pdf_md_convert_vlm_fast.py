#!/usr/bin/env python3

# New python script for converting pdfs to markdown
# Key design decisions:
# - minimal dependencies
#   - openai for LLM/VLM calls
#   - PIL (optional) for in-memory image downscaling
#   - multiprocessing for concurrent jobs
#   - external poppler-utils scripts handle all PDF parsing
# - pdfs are processed entirely in memory: pdftotext and pdftoppm stream their
#   output to stdout, so no intermediate text/image files are written to disk
#   - pdftotext: 'pdftotext -layout -f <pagenumber> -l <pagenumber> <filename>.pdf -'
#   - pdftoppm:  'pdftoppm -png -r <dpi> -f <pagenumber> -l <pagenumber> <filename>.pdf'
#     (or -jpeg -jpegopt quality=N with --jpeg, which compresses the image and
#     reduces token usage)
#     (poppler writes the image to stdout when no output prefix is given; on
#     older poppler releases that require a prefix, a short-lived temp file is
#     used as a fallback and removed immediately)
# - image + extracted text are passed together to the VLM with a prompt that
#   produces structured markdown including descriptions of figures/plots.
#   Images are passed as base64 strings created from the stdout of pdftoppm.
#   The VLM must not use '#' or '##' headings (reserved for the document
#   title and slide tags added by the collation step below).
# - one page = one job, fanned out over a multiprocessing pool
# - page markdown files are named '<stem>.md.pageNNNN'; as soon as all pages
#   of a given document are converted (not after every document is done), the
#   single collation worker (a dedicated thread running in parallel with the
#   conversion pool) merges its pages into a single presentation markdown
#   '<stem>.md':
#     '# <Title> — <Author>' header (from pdfinfo metadata, falling back to
#     the file name), a '## Slide XX' tag per slide (number only, no title),
#     and a '---' delimiter at the end of each slide.
#   The worker knows each document's expected page count from the startup
#   pdfinfo scan (passed with the collation request) and re-checks on disk
#   that all of it is present before merging, so it can never build a
#   partial deck.  The per-page files are deleted once the presentation
#   document is written.  More post-processing can be added to the worker's
#   queue later.
# - progress is printed every 10 pages, with process-tree profiling stats:
#   peak memory utilisation (GB, % of the machine's RAM; the parent process,
#   the pool workers and active CLI subprocesses, shared forked memory
#   counted once via Pss) and cumulative average CPU utilisation (%) of the
#   same tree (may exceed 100% while several workers are busy)

"""Batch-convert PDFs to markdown using pdftotext + pdftoppm + a VLM.

For each page, raw text is extracted with the ``pdftotext`` CLI and the page is
rendered to an image by ``pdftoppm``.  Both stay in memory (stdout capture) and
are sent to a VLM via an OpenAI-compatible API, which produces clean markdown
with LaTeX equations and descriptions of figures/plots.  Each page is written
to ``<stem>.md.pageNNNN``; as soon as all pages of a document are converted, a
single dedicated collation worker merges them into a single presentation
markdown ``<stem>.md`` (``# Title — Author`` header, ``## Slide XX`` tag per
slide, ``---`` at the end of each slide) and removes the per-page files.  The
worker verifies on disk that every expected page file is present before
merging (the page count comes from the startup pdfinfo scan), so one document
never waits for the pages of other documents.

Usage:
    python pdf_md_convert_vlm_fast.py \\
        --input-dir ./pdfs \\
        --output-dir ./md \\
        --base-url http://localhost:4141/v1 \\
        --model "nvidia/Qwen3.6-35B-A3B-NVFP4" \\
        --workers 4
"""

from __future__ import annotations

import argparse
import base64
import io
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    import resource  # Unix-only; used for process-tree profiling stats

except ImportError:  # pragma: no cover - non-Unix platforms
    resource = None  # type: ignore[assignment]

try:
    from openai import OpenAI

    _OPENAI_IMPORT_ERROR: Exception | None = None
except ImportError as error:  # pragma: no cover - depends on environment
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = error

try:
    from PIL import Image  # optional, used for in-memory downscaling
except ImportError:  # pragma: no cover - depends on environment
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="./pdfs",
        help="Directory containing PDF files to convert",
    )
    parser.add_argument(
        "--output-dir",
        default="./md",
        help="Directory where Markdown files will be written (mirrors input layout)",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for PDF discovery (used recursively)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8008/v1",
        help="OpenAI-compatible base URL for VLM calls",
    )
    parser.add_argument(
        "--api-key",
        default="local",
        help="API key for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.8-27B-FP8",
        help="VLM model name used for page-level conversion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first conversion error",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=150,
        help="Skip PDFs with more pages than this",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of pages to convert in parallel (worker processes)",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=200,
        help="DPI for pdftoppm page rendering (higher = better quality, more tokens)",
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=2200,
        help=(
            "Downscale the rendered page in memory (PIL) so its longest edge "
            "does not exceed this many pixels. 0 disables downscaling."
        ),
    )
    parser.add_argument(
        "--jpeg",
        action="store_true",
        help=(
            "Render pages as JPEG instead of PNG (pdftoppm -jpeg). "
            "JPEG compression typically produces much smaller images, "
            "reducing token usage at some loss of sharpness."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for pdftoppm -jpegopt (1-100, only used with --jpeg)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for the VLM response per page",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts per VLM call (exponential backoff between attempts)",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds for VLM calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only run pdftotext/pdftoppm and report extracted sizes. "
            "No VLM calls, no files written."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a document-to-markdown converter.  You will receive:\n"
    "  1. The raw text extracted from a single PDF page (may contain OCR-like "
    "artifacts, repeated headers/footers, and mis-ordered text).\n"
    "  2. A screenshot image of the same page.\n"
    "\n"
    "Produce a clean, well-structured markdown document that faithfully "
    "represents the page content.  Follow these rules exactly:\n"
    "\n"
    "1. **Valid markdown only** — the output must render correctly as markdown. "
    "Use proper syntax for all elements.\n"
    "\n"
    "2. **Headings** — Respect the following document hierarchy: "
    "Slide titles, subheadings, etc. should use ``###`` or lower-order heading tags. "
    "The top and second-level headings ``#`` and ``##`` are reserved for "
    "the document title and slide titles (to be added externally), "
    "they should NOT be used anywhere for any slide content."
    "\n"
    "3. **Equations and Greek letters** — render ALL mathematical expressions, "
    "equations, and formulas using LaTeX syntax inside ``$...$`` for inline "
    "math and ``$$...$$`` for display equations.  Also render any Greek "
    "letters (e.g. alpha, beta, gamma, Delta, Omega) using LaTeX commands "
    "(e.g. ``\\alpha``, ``\\beta``, ``\\Gamma``, ``\\Omega``) inside ``$`` or "
    "``$$`` delimiters.  Never leave raw Greek text unrendered when it "
    "represents a mathematical or physics variable. Non-cursive math text, i.e. units, "
    "should be written inside ``$\\mathrm{...}$`` tags, eg. ``$\\mathrm{fb^{-1}}$`` "
    "instead of fb$^{-1}$.\n"
    "\n"
    "4. **Tables** — use standard markdown table syntax with pipe ``|`` "
    "separators and a header separator row (``---|---|---``).  Preserve all "
    "column data.\n"
    "\n"
    "5. **Block quotes** — use ``> `` for quoted text, pull quotes, or "
    "emphasis blocks.\n"
    "\n"
    "6. **Footnotes** — use standard markdown footnote syntax: "
    "``[^1]`` in-line and ``[^1]: footnote text`` at the bottom of the page.\n"
    "\n"
    "7. **Lists** — use ``-`` or ``*`` for unordered lists and ``1.`` for "
    "ordered lists.  Nest properly for sub-items.\n"
    "\n"
    "8. **Figures, plots, and images** — the extracted text does not contain "
    "the content of figures.  For every figure, plot, chart, photo, table "
    "screenshot, or diagram visible in the page image, insert a block quote at "
    "the position where it appears on the page, prefixed by a label such as "
    "``> **Figure 3:**``, and describe it: chart type, axes, legend, "
    "curves/bars, and the most salient values or trends.  Do not invent "
    "content that is not visible.\n"
    "\n"
    "9. **Clean up** — remove duplicate headers/footers, page numbers, and "
    "repeated labels.  Keep only the meaningful content of the page.\n"
    "\n"
    "10. **Do not** add any introductory text, meta-commentary, or "
    "explanation outside the markdown itself.  Output only the markdown.\n"
)

USER_PROMPT_TEMPLATE = (
    "Here is the raw text from a single PDF page, followed by the page image. "
    "Convert this page into clean, valid markdown.\n"
    "Extracted text:\n\n"
    "```text\n"
    "{raw_text}"
    "\n```\n\n"
    "Convert this page to markdown:"
)


# ---------------------------------------------------------------------------
# External tools (poppler-utils)
# ---------------------------------------------------------------------------


def require_tool(name: str, env_var: str) -> str:
    """Locate an external tool, honouring an env-var path override."""
    path = os.environ.get(env_var) or shutil.which(name)
    if path is None:
        raise SystemExit(
            f"{name} not found on PATH (override with the {env_var} env var). "
            "Install poppler-utils:\n"
            "  conda install -c conda-forge poppler-utils\n"
            "  apt-get install -y poppler-utils\n"
            "  yum install -y poppler-utils"
        )
    return path


def pdf_metadata(pdfinfo_path: str, pdf_path: Path) -> tuple[int, str, str]:
    """Page count, title, and author of a PDF, read from ``pdfinfo``.

    No python PDF dependencies needed.  Title/author may be empty strings if
    the PDF has no metadata.
    """
    result = subprocess.run(
        [pdfinfo_path, str(pdf_path)],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"pdfinfo failed: {stderr or 'unknown error'}")
    num_pages, title, author = 0, "", ""
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "pages":
            num_pages = int(value)
        elif key == "title":
            title = value
        elif key == "author":
            author = value
    if num_pages == 0:
        raise RuntimeError("pdfinfo output did not contain a 'Pages' field")
    return num_pages, title, author


def extract_page_text(pdftotext_path: str, pdf_path: Path, page_number: int) -> str:
    """Extract the text of one page (1-indexed) into memory via stdout."""
    result = subprocess.run(
        [
            pdftotext_path,
            "-layout",
            "-f", str(page_number),
            "-l", str(page_number),
            str(pdf_path),
            "-",
        ],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"pdftotext failed on page {page_number}: {stderr or 'unknown error'}"
        )
    return result.stdout.decode("utf-8", "replace")


_IMAGE_MAGIC = {
    "png": (b"\x89PNG\r\n\x1a\n", None),   # (start marker, end marker)
    "jpeg": (b"\xff\xd8\xff", b"\xff\xd9"),
}


def _single_image(data: bytes, image_format: str) -> bytes | None:
    """Return the data if it is exactly one complete image of the given format."""
    start_marker, end_marker = _IMAGE_MAGIC[image_format]
    if not data.startswith(start_marker):
        return None
    if image_format == "png" and data.count(start_marker) != 1:
        return None
    if end_marker is not None and not data.rstrip(b"\x00").endswith(end_marker):
        return None
    return data


def render_page_image(
    pdftoppm_path: str,
    pdf_path: Path,
    page_number: int,
    dpi: int,
    image_format: str,
    jpeg_quality: int,
    timeout_seconds: int=600,
) -> bytes:
    """Render one page (1-indexed) to image bytes entirely in memory.

    ``image_format`` is "png" or "jpeg".  Modern poppler writes the image to
    stdout when no output prefix is given.  Older releases require a prefix
    argument; in that case we fall back to a short-lived temp file that is
    read and removed immediately.
    """
    cmd = [
        pdftoppm_path,
        f"-{image_format}",
    ]
    if image_format == "jpeg":
        cmd += ["-jpegopt", f"quality={max(1, min(100, jpeg_quality))}"]
    cmd += [
        "-r", str(dpi),
        "-f", str(page_number),
        "-l", str(page_number),
        str(pdf_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"pdftoppm timed out on page {page_number}") from error
    if result.returncode == 0:
        image = _single_image(result.stdout, image_format)
        if image is not None:
            return image

    # Fallback: prefix mode in a temp directory (cleaned up automatically).
    extension = "jpg" if image_format == "jpeg" else "png"
    with tempfile.TemporaryDirectory(prefix="pdftoppm_") as tmp_dir:
        prefix = Path(tmp_dir) / "page"
        result = subprocess.run(cmd + [str(prefix)], capture_output=True, timeout=timeout_seconds)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"pdftoppm failed on page {page_number}: {stderr or 'unknown error'}"
            )
        candidates = sorted(Path(tmp_dir).glob(f"page-*.{extension}"))
        if not candidates:
            raise RuntimeError(
                f"pdftoppm produced no {image_format.upper()} for page {page_number} "
                f"of {pdf_path.name}"
            )
        return candidates[0].read_bytes()


def downscale_image(image: bytes, max_dimension: int, image_format: str) -> bytes:
    """Optionally shrink the image in memory so its longest edge <= max_dimension."""
    if max_dimension <= 0 or Image is None:
        return image
    pil_image = Image.open(io.BytesIO(image))
    width, height = pil_image.size
    if max(width, height) <= max_dimension:
        return image
    scale = max_dimension / max(width, height)
    pil_image = pil_image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.LANCZOS,
    )
    # JPEG has no alpha channel.
    if image_format == "jpeg" and pil_image.mode not in ("L", "RGB"):
        pil_image = pil_image.convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format=image_format.upper(), optimize=True)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageJob:
    """Everything a worker process needs to convert one page. Picklable."""

    pdf_path: str
    page_number: int
    output_path: str
    pdftotext: str
    pdftoppm: str
    base_url: str
    api_key: str
    model: str
    image_resolution: int
    max_image_dimension: int
    image_format: str
    jpeg_quality: int
    max_tokens: int
    api_timeout: float
    retries: int
    dry_run: bool


@dataclass(frozen=True)
class DocumentRecord:
    """Bookkeeping for one PDF (kept in the parent process only)."""

    pdf_path: str
    pdf_name: str
    stem: str
    num_pages: int
    title: str
    author: str
    presentation_path: str


class DocState:
    """Parent-process-only bookkeeping for one document.

    ``finished`` counts the document's pages that are done — either because
    their page file was already on disk at startup, or because the page job
    completed in this run.  As soon as ``finished`` reaches the document's
    expected page count (known from the startup pdfinfo scan), the document
    is handed to the collation worker.
    """

    __slots__ = ("doc", "finished", "handed_off")

    def __init__(self, doc: DocumentRecord) -> None:
        self.doc = doc
        self.finished = 0
        self.handed_off = False


# One OpenAI client per worker process (created lazily inside the pool).
_CLIENT_CACHE: dict[tuple[str, str, float], object] = {}


def _get_client(base_url: str, api_key: str, api_timeout: float):
    key = (base_url, api_key, api_timeout)
    client = _CLIENT_CACHE.get(key)
    if client is None:
        client = OpenAI(
            base_url=base_url, api_key=api_key, max_retries=0, timeout=api_timeout
        )
        _CLIENT_CACHE[key] = client
    return client


def _strip_code_fences(markdown: str) -> str:
    """Remove a wrapping ``` fence if the model added one."""
    text = markdown.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            body = text[first_newline + 1 :]
            if body.rstrip().endswith("```"):
                return body.rstrip()[:-3].rstrip()
    return text


def _chat_with_retries(client, job: PageJob, messages: list[dict]) -> str:
    last_error: Exception | None = None
    for attempt in range(1, job.retries + 1):
        try:
            response = client.chat.completions.create(
                model=job.model,
                messages=messages,
                max_tokens=job.max_tokens,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise RuntimeError("VLM returned empty content")
            return content.strip()
        except Exception as error:  # noqa: BLE001 - any API failure is retried
            last_error = error
            if attempt < job.retries:
                time.sleep(2.0 * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"VLM call failed after {job.retries} attempt(s): {last_error}"
    )


def run_page_job(job: PageJob) -> dict:
    """Worker entry point: text + image (in memory) -> VLM -> markdown file.

    Runs inside the multiprocessing pool; must stay picklable (module level).
    """
    text = extract_page_text(job.pdftotext, Path(job.pdf_path), job.page_number)
    image = render_page_image(
        job.pdftoppm,
        Path(job.pdf_path),
        job.page_number,
        job.image_resolution,
        job.image_format,
        job.jpeg_quality,
    )
    image = downscale_image(image, job.max_image_dimension, job.image_format)
    b64_image = base64.b64encode(image).decode("ascii")

    if job.dry_run:
        return {
            "status": "dry",
            "text_chars": len(text),
            "image_bytes": len(image),
            "base64_chars": len(b64_image),
            "image_format": job.image_format,
            "worker_pid": os.getpid(),
            "worker_cli_cpu_seconds": _reaped_children_cpu_seconds(),
        }

    client = _get_client(job.base_url, job.api_key, job.api_timeout)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT_TEMPLATE.format(raw_text=text),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{job.image_format};base64,{b64_image}",
                    },
                },
            ],
        },
    ]
    markdown = _strip_code_fences(_chat_with_retries(client, job, messages))

    output_path = Path(job.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "text_chars": len(text),
        "image_bytes": len(image),
        "base64_chars": len(b64_image),
        "image_format": job.image_format,
        "markdown_chars": len(markdown),
        "worker_pid": os.getpid(),
        "worker_cli_cpu_seconds": _reaped_children_cpu_seconds(),
    }


def _document_header(title: str, author: str) -> str:
    if author:
        return f"# {title} — {author}"
    return f"# {title}"


def collate_document(doc: DocumentRecord) -> dict:
    """Merge one document's pages into its presentation markdown.

    Runs in the collation worker thread.  ``doc.num_pages`` is the expected
    page count from the startup pdfinfo scan; the page files are re-checked
    on disk before merging, so a partial document is reported as ``skipped``
    (nothing is deleted) instead of being turned into a partial deck.

    Layout of the resulting presentation document:

        # <Title> — <Author>

        ## Slide 1

        <page 1 markdown>

        ---

        ## Slide 2

        <page 2 markdown>

        ---

    The per-page files are deleted only after the presentation document has
    been written successfully, so a failed collation loses nothing and the
    document can be re-collated after re-running the missing pages.
    """
    page_dir = Path(doc.presentation_path).parent
    page_files = [
        page_dir / f"{doc.stem}.md.page{page:04d}"
        for page in range(1, doc.num_pages + 1)
    ]
    missing = [p.name for p in page_files if not p.is_file()]
    if missing:
        return {
            "status": "skipped",
            "missing": missing,
            "output": doc.presentation_path,
        }

    lines = [_document_header(doc.title, doc.author), ""]
    for page, page_file in enumerate(page_files, start=1):
        content = page_file.read_text(encoding="utf-8").strip()
        lines.extend([f"## Slide {page}", "", content, "", "---", ""])
    text = "\n".join(lines).rstrip("\n") + "\n"

    presentation_path = Path(doc.presentation_path)
    presentation_path.parent.mkdir(parents=True, exist_ok=True)
    presentation_path.write_text(text, encoding="utf-8")

    for page_file in page_files:
        page_file.unlink()

    return {
        "status": "collated",
        "slides": doc.num_pages,
        "output": str(presentation_path),
        "bytes": len(text.encode("utf-8")),
    }


def _collation_worker(requests: queue.Queue, results: queue.Queue) -> None:
    """Single dedicated post-processing worker (one thread).

    Pulls documents to collate, verifies on disk that all expected page
    files are present, merges them, and reports the outcome.  A single
    worker means two requests can never race on the same document, and the
    worker simply stays idle while the conversion pool is busy — collation
    is cheap local file I/O.  Further post-processing steps can be added to
    this queue later.
    """
    while True:
        doc = requests.get()
        if doc is None:  # sentinel: shut down
            break
        try:
            result = collate_document(doc)
        except Exception as error:
            print(f"COLLATE FAILED {doc.pdf_name}: {error}", file=sys.stderr)
            results.put(("error", doc.pdf_name, str(error)))
            continue
        if result["status"] == "collated":
            print(
                f"collated {doc.pdf_name}: {result['slides']} slides -> "
                f"{result['output']} ({result['bytes'] // 1024} KB)"
            )
        else:
            print(
                f"collation skipped for {doc.pdf_name}: missing page file(s) "
                f"{', '.join(result['missing'])}",
                file=sys.stderr,
            )
        results.put((result["status"], doc.pdf_name, None))


# ---------------------------------------------------------------------------
# Job discovery / output layout
# ---------------------------------------------------------------------------


def discover_pdfs(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_dir.rglob(pattern) if path.is_file())


def output_path_for(pdf_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Path of the collated presentation markdown, mirroring the input layout."""
    relative_parent = pdf_path.relative_to(input_dir).parent
    return (output_dir / relative_parent / f"{pdf_path.stem}.md")


def page_path_for(pdf_path: Path, input_dir: Path, output_dir: Path, page_number: int) -> Path:
    """Path of one page's markdown file: '<stem>.md.pageNNNN'."""
    relative_parent = pdf_path.relative_to(input_dir).parent
    return (
        output_dir / relative_parent / f"{pdf_path.stem}.md.page{page_number:04d}"
    )


def doc_page_files_present(doc: DocumentRecord) -> bool:
    """True if at least one of the document's page files exists on disk."""
    page_dir = Path(doc.presentation_path).parent
    return any(
        (page_dir / f"{doc.stem}.md.page{page:04d}").is_file()
        for page in range(1, doc.num_pages + 1)
    )


# ---------------------------------------------------------------------------
# Profiling stats (process tree: parent + pool workers + CLI subprocesses)
# ---------------------------------------------------------------------------


def _mem_total_kb() -> int:
    """Total physical memory of the machine in kB (0 if not measurable)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _tree_usage() -> tuple[int, float] | None:
    """Memory and CPU usage of the whole process tree, via /proc.

    Returns ``(rss_kb, cpu_seconds)`` where ``rss_kb`` is the sum of the
    proportional memory (Pss, falling back to VmRSS) of this process, the
    pool workers, and any active CLI subprocesses — Pss counts memory shared
    by forked workers only once — and ``cpu_seconds`` is their cumulative
    user+system CPU time plus the rusage of already-reaped children (e.g.
    replaced pool workers).  Returns ``None`` when /proc is unavailable.
    """
    if not Path("/proc/self/stat").is_file():
        return None
    tick = os.sysconf("SC_CLK_TCK")
    rss_kb = 0
    cpu_seconds = 0.0
    seen: set[int] = set()
    stack = [os.getpid()]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split():
                stack.append(int(child))
        except (OSError, ValueError):
            pass
        try:
            parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            cpu_seconds += (int(parts[11]) + int(parts[12])) / tick
        except (OSError, ValueError, IndexError):
            pass
        rss = None
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    rss = int(line.split()[1])
                    break
        except (OSError, ValueError):
            pass
        if rss is None:
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
                        break
            except (OSError, ValueError):
                pass
        rss_kb += rss or 0
    return rss_kb, cpu_seconds


def _reaped_children_cpu_seconds() -> float:
    """Cumulative user+system CPU seconds of this process's terminated (and
    waited-for) children.  Called inside a pool worker this is the CPU time
    of the short-lived pdftotext/pdftoppm subprocesses it has reaped — which
    no longer show up in any live /proc entry between checkpoints."""
    if resource is None:
        return 0.0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def _profiling_suffix(
    peak_mem_kb: int,
    mem_total_kb: int,
    elapsed: float,
    total_cpu_seconds: float,
) -> str:
    """Profiling suffix for the progress/summary lines.

    'peak_mem' is the process-tree memory high-water mark so far (Pss based,
    so memory shared by forked workers is counted once); 'avg_cpu' is the
    cumulative average CPU utilisation of the whole process tree (parent,
    pool workers and their pdftotext/pdftoppm subprocesses) over the run so
    far, so it may exceed 100% while several workers are busy.  Returns an
    empty string when the stats are not measurable.
    """
    if peak_mem_kb <= 0 or elapsed <= 0 or not mem_total_kb:
        return ""
    peak_gb = peak_mem_kb / 1024 / 1024
    mem_pct = 100.0 * peak_mem_kb / mem_total_kb
    cpu_pct = 100.0 * total_cpu_seconds / elapsed
    return (
        f" | peak_mem={peak_gb:.2f} GB ({mem_pct:.1f}% of RAM)"
        f" | avg_cpu={cpu_pct:.1f}%"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if not args.dry_run and OpenAI is None:
        raise SystemExit(
            "Missing dependency 'openai'. Install it with:\n"
            "  pixi run uv pip install openai\n"
            f"(import error: {_OPENAI_IMPORT_ERROR})"
        )

    pdftotext = require_tool("pdftotext", "PDFTOTEXT")
    pdftoppm = require_tool("pdftoppm", "PDFTOPPM")
    pdfinfo = require_tool("pdfinfo", "PDFINFO")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    pdf_paths = discover_pdfs(input_dir, args.pattern)
    if not pdf_paths:
        print(f"No PDFs found in {input_dir} matching {args.pattern}")
        return 0

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build the per-page job list, the per-document records, and the
    # per-document in-flight state used to trigger collation early.
    jobs: list[PageJob] = []
    documents: list[DocumentRecord] = []
    doc_states: list[DocState] = []
    skipped_pages = 0
    total_pages = 0

    for pdf_path in pdf_paths:
        try:
            num_pages, title, author = pdf_metadata(pdfinfo, pdf_path)
        except Exception as error:
            print(f"warning: could not read info for {pdf_path}: {error}",
                  file=sys.stderr)
            continue
        if num_pages > args.max_pages:
            skipped_pages += num_pages
            print(
                f"skip {pdf_path.name} (has {num_pages} pages, max is {args.max_pages})",
                file=sys.stderr,
            )
            continue
        total_pages += num_pages
        presentation_path = output_path_for(pdf_path, input_dir, output_dir)
        doc = DocumentRecord(
            pdf_path=str(pdf_path),
            pdf_name=pdf_path.name,
            stem=pdf_path.stem,
            num_pages=num_pages,
            title=title or pdf_path.stem,
            author=author,
            presentation_path=str(presentation_path),
        )
        documents.append(doc)

        # A document is finished once its presentation file exists (the
        # collation step deletes the per-page files), so skip it entirely —
        # unless stale page files are left behind (a previous run crashed
        # between collation and cleanup), in which case handing it to the
        # collation worker refreshes the deck and removes them.
        doc_done = (
            not args.dry_run
            and presentation_path.exists()
            and not args.overwrite
        )
        if doc_done and not doc_page_files_present(doc):
            skipped_pages += num_pages
            continue

        state = None if args.dry_run else DocState(doc)
        if state is not None:
            doc_states.append(state)

        for page_number in range(1, num_pages + 1):
            if doc_done:
                skipped_pages += 1
                if state is not None:
                    state.finished += 1
                continue
            out_path = page_path_for(pdf_path, input_dir, output_dir, page_number)
            if not args.dry_run and out_path.exists() and not args.overwrite:
                skipped_pages += 1
                if state is not None:
                    state.finished += 1
                continue
            jobs.append(
                PageJob(
                    pdf_path=str(pdf_path),
                    page_number=page_number,
                    output_path=str(out_path),
                    pdftotext=pdftotext,
                    pdftoppm=pdftoppm,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    image_resolution=args.image_resolution,
                    max_image_dimension=args.max_image_dimension,
                    image_format="jpeg" if args.jpeg else "png",
                    jpeg_quality=args.jpeg_quality,
                    max_tokens=args.max_tokens,
                    api_timeout=args.api_timeout,
                    retries=args.retries,
                    dry_run=args.dry_run,
                )
            )

    # Nothing to convert and nothing left to collate (e.g. a document whose
    # pages exist but was never collated)?
    if not jobs and (args.dry_run or not any(doc_page_files_present(d) for d in documents)):
        print(
            f"Done. pages={total_pages} skipped={skipped_pages} "
            f"converted=0 collated=0 failed=0"
        )
        return 0

    workers = max(1, min(args.workers, len(jobs) or 1))
    mode = "Dry run" if args.dry_run else "Converting"
    print(
        f"{mode}: {len(jobs)} page(s) across {len(pdf_paths)} file(s) "
        f"(skipped {skipped_pages} existing page(s))"
    )
    if doc_states:
        print(
            f"Collating each of the {len(doc_states)} document(s) as soon as "
            f"all of its pages are converted"
        )
    print(
        f"Using {workers} worker process(es) "
        f"(multiprocessing start method: {multiprocessing.get_start_method()})"
    )

    start_time = time.perf_counter()
    completed = 0
    converted = 0
    failed = 0
    peak_mem_kb = 0
    mem_total_kb = _mem_total_kb()
    # pool worker pid -> cumulative CPU seconds of the CLI subprocesses it
    # has reaped (reported by the jobs, since they vanish from /proc between
    # checkpoints); the max per worker is what gets added to the live walk
    worker_cli_cpu: dict[int, float] = {}
    state_by_pdf = {state.doc.pdf_path: state for state in doc_states}

    # The single collation worker runs in parallel with the conversion pool:
    # it is handed a document the moment the last of that document's pages
    # is done, so a document's collation can never wait behind the pool's
    # backlog of still-pending page jobs.
    requests: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    collator: threading.Thread | None = None
    if doc_states:
        collator = threading.Thread(
            target=_collation_worker,
            args=(requests, results),
            name="collation-worker",
            daemon=True,
        )
        collator.start()

        def hand_off(state: DocState) -> None:
            """Hand a document to the collation worker once all of its pages
            are done (the worker re-verifies on disk before merging)."""
            if not state.handed_off and state.finished == state.doc.num_pages:
                state.handed_off = True
                requests.put(state.doc)

        # Documents whose pages are all already on disk (e.g. a previous run
        # stopped before collating): hand them over right away.
        for state in doc_states:
            hand_off(state)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_page_job, job): job for job in jobs}
        stopping = False
        for future in as_completed(futures):
            if stopping:
                future.cancel()
                continue
            job: PageJob = futures[future]
            completed += 1
            pdf_name = Path(job.pdf_path).name
            state = state_by_pdf.get(job.pdf_path)

            try:
                result = future.result()
            except Exception as error:
                failed += 1
                if state is not None:
                    state.finished += 1
                    hand_off(state)
                print(
                    f"[{completed}/{len(jobs)}] FAILED {pdf_name} "
                    f"page {job.page_number}: {error}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    stopping = True
                    for pending in futures:
                        pending.cancel()
                    break
                continue

            fmt = result["image_format"].upper()
            if result["status"] == "dry":
                print(
                    f"[{completed}/{len(jobs)}] dry-run {pdf_name} "
                    f"page {job.page_number}: "
                    f"text={result['text_chars']} chars, "
                    f"{fmt}={result['image_bytes'] // 1024} KB, "
                    f"base64={result['base64_chars'] // 1024} KB"
                )
            else:
                converted += 1
                print(
                    f"[{completed}/{len(jobs)}] done {pdf_name} "
                    f"page {job.page_number}: "
                    f"text={result['text_chars']} chars, "
                    f"{fmt}={result['image_bytes'] // 1024} KB, "
                    f"markdown={result['markdown_chars']} chars"
                )
            if state is not None:
                state.finished += 1
                hand_off(state)

            if result["worker_pid"]:
                worker_cli_cpu[result["worker_pid"]] = max(
                    worker_cli_cpu.get(result["worker_pid"], 0.0),
                    result["worker_cli_cpu_seconds"],
                )

            if completed % 10 == 0:
                elapsed = time.perf_counter() - start_time
                usage = _tree_usage()
                if usage is not None:
                    peak_mem_kb = max(peak_mem_kb, usage[0])
                    # live tree (parent + workers + CLI subprocesses running
                    # right now) plus the CLI CPU the workers have already
                    # reaped between checkpoints
                    total_cpu = usage[1] + sum(worker_cli_cpu.values())
                else:
                    total_cpu = 0.0
                print(
                    f"Progress: {completed}/{len(jobs)} pages processed, "
                    f"avg {elapsed / completed:.2f}s/page"
                    + _profiling_suffix(peak_mem_kb, mem_total_kb, elapsed, total_cpu)
                )

    # Let the collation worker finish what it has been handed, then collect
    # its outcomes for the summary.
    collated = 0
    collate_errors = 0
    collation_skipped = 0
    if collator is not None:
        requests.put(None)
        collator.join()
        while True:
            try:
                status, _name, _detail = results.get_nowait()
            except queue.Empty:
                break
            if status == "collated":
                collated += 1
            elif status == "error":
                collate_errors += 1
            else:  # "skipped"
                collation_skipped += 1
        failed += collate_errors

    elapsed = time.perf_counter() - start_time
    usage = _tree_usage()
    if usage is not None:
        peak_mem_kb = max(peak_mem_kb, usage[0])
        # the pool workers are reaped by now, so RUSAGE_CHILDREN gives their
        # complete CPU time (including all of their CLI subprocesses)
        total_cpu = usage[1] + _reaped_children_cpu_seconds()
    else:
        total_cpu = 0.0
    print(
        f"Done. converted={converted} collated={collated} skipped={skipped_pages} "
        f"failed={failed} total_time={elapsed:.1f}s"
        + _profiling_suffix(peak_mem_kb, mem_total_kb, elapsed, total_cpu)
    )
    if collation_skipped:
        print(
            f"Collation skipped for {collation_skipped} document(s) due to "
            f"missing page file(s); re-run to retry them",
            file=sys.stderr,
        )
    if completed:
        print(f"Average: {elapsed / completed:.2f}s/page")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
