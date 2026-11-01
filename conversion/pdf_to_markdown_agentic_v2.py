#!/usr/bin/env python3
"""Convert a directory of PDFs to Markdown (batch, in-memory, agentic).

V2 pipeline: combines the fast batch architecture of
``pdf_md_convert_vlm_fast.py`` with the figure-extraction + agentic review
workflow of ``pdf_to_markdown_agentic.py``.

Design
------
- Pages are processed entirely in memory: ``pdftotext -layout`` and
  ``pdftoppm`` stream their output to stdout, so no intermediate
  text/image files are written.  A rendered page image exists only inside
  a worker's memory (optional in-memory downscale and JPEG).
- One page = one job, fanned out over a ``ProcessPoolExecutor`` across
  documents *and* pages.  Each worker:
    1. extracts raw page text and renders the page image (in memory),
    2. sends both to the VLM (retries with exponential backoff,
       temperature 0) to get page markdown with ``<!-- Figure: ... -->``
       markers,
    3. runs the in-memory figure pipeline: marker detection -> bbox
       detection (VLM call, or qpdf heuristic boxes precomputed per
       document in the parent) -> crop -> crop review with one retry,
       saving ``figures/*.png`` and replacing the markers,
    4. writes only ``<stem>.md.pageNNNN`` (plus figure crops / debug logs).
- A single dedicated collation worker (a thread running in parallel with
  the pool) is handed a document as soon as all of its pages are done and
  never builds a partial deck: it verifies on disk that every expected
  page file is present, merges them into ``<stem>.md``
    - default layout: ``# <Title> — <Author>`` header, ``## Slide NN`` tag
      per page, trailing ``---`` (consumed by
      ``database/build_database.py::parse_slides_md``);
    - ``--legacy-page-format``: ``# <stem>.pdf`` header + bare ``---``
      page blocks (the layout of ``pdf_to_markdown_agentic.py``),
  then runs the agentic DeepAgents review loop (``validate_markdown``
  tool, read_file / edit_file) on the merged file, and only after that
  deletes the per-page files.
- Resume: existing page files are skipped; a document whose presentation
  file exists and has no stray page files is skipped entirely; stray page
  files are re-collated.  Re-running after a crash converges.
- Progress is printed every 10 pages with process-tree profiling stats
  (peak Pss-based memory, cumulative average CPU of parent + workers +
  CLI subprocesses).

Usage:
    python pdf_to_markdown_agentic_v2.py \\
        --input-dir ./pdfs --output-dir ./md \\
        --base-url http://127.0.0.1:8008/v1 \\
        --model "Qwen/Qwen3.8-27B-FP8" --workers 4
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import multiprocessing
import os
import queue
import re
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
    from PIL import Image

except ImportError:  # pragma: no cover - depends on environment
    Image = None  # type: ignore[assignment]

from list_pdf_figures import analyze_pdf

# Ensure pixi environment's bin is on PATH so subprocess calls find
# pixi-installed tools (pdftoppm, pdftotext, pdfinfo, qpdf).
_pixi_bin = str(Path(__file__).resolve().parent / ".pixi" / "envs" / "default" / "bin")
if os.path.isdir(_pixi_bin) and _pixi_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _pixi_bin + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Batch / I/O (from pdf_md_convert_vlm_fast.py)
    parser.add_argument("--input-dir", default="./pdfs",
                        help="Directory containing PDF files to convert (searched recursively)")
    parser.add_argument("--output-dir", default="./md",
                        help="Directory where Markdown files will be written (mirrors input layout)")
    parser.add_argument("--pattern", default="*.pdf",
                        help="Glob pattern for PDF discovery (used recursively)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing Markdown / page files")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop immediately on the first conversion error")
    parser.add_argument("--max-pages", type=int, default=150,
                        help="Skip PDFs with more pages than this")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of pages to convert in parallel (worker processes)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run pdftotext/pdftoppm and report extracted sizes. No VLM calls, no files written.")
    # VLM / API
    parser.add_argument("--llm-base-url", "--base-url",
                        default="http://127.0.0.1:8008/v1",
                        help="OpenAI-compatible base URL for VLM calls")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key for the OpenAI-compatible endpoint")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8",
                        help="VLM model name used for page conversion and figure calls")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Maximum tokens for the VLM response per call (default: 8192)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of attempts per VLM call (exponential backoff between attempts)")
    parser.add_argument("--api-timeout", type=float, default=300.0,
                        help="Per-request timeout in seconds for VLM calls")
    # Rendering
    parser.add_argument("--dpi", type=int, default=200,
                        help="DPI for pdftoppm page rendering (higher = better quality, more tokens)")
    parser.add_argument("--max-image-dimension", type=int, default=2200,
                        help="Downscale the rendered page in memory (PIL) so its longest edge does not exceed this many pixels. 0 disables downscaling.")
    parser.add_argument("--jpeg", action="store_true",
                        help="Render pages as JPEG instead of PNG (pdftoppm -jpeg); smaller images, fewer tokens")
    parser.add_argument("--jpeg-quality", type=int, default=90,
                        help="JPEG quality for pdftoppm -jpegopt (1-100, only used with --jpeg)")
    # Figure extraction (from pdf_to_markdown_agentic.py)
    parser.add_argument("--skip-figure-extraction", action="store_true",
                        help="Skip figure detection and extraction")
    parser.add_argument("--figure-extraction-mode", choices=["vlm", "heuristic"], default="vlm",
                        help="Figure detection backend: VLM bbox prediction, or qpdf/XObject boxes paired in reading order (default: vlm)")
    parser.add_argument("--no-figure-debug", action="store_true",
                        help="Do not write per-page figure_debug text logs")
    # Agentic review (from pdf_to_markdown_agentic.py)
    parser.add_argument("--skip-review", action="store_true",
                        help="Skip the agentic markdown review step")
    parser.add_argument("--max-review-iterations", type=int, default=5,
                        help="Max agentic review->fix cycles per document (default: 5)")
    # Output format
    parser.add_argument("--legacy-page-format", action="store_true",
                        help="Emit the legacy '# <stem>.pdf' + bare '---' page-block layout instead of '# Title - Author' + '## Slide NN'. "
                             "Matches pdf_to_markdown_agentic.py output for markdown_to_book/preprocess.py. "
                             "Note (same as v1): a bare '---' generated inside a page's markdown splits into an extra page break there; the default format is unaffected by that.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# System prompt (merged: fast-script rules + agentic figure-marker convention)
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
    "they should NOT be used anywhere for any slide content.\n"
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
    "the content of figures.  For every figure, plot, chart, photo, diagram, "
    "logo, or other non-text visual element visible in the page image, insert "
    "exactly one HTML comment of the form "
    "``<!-- Figure: <brief description> -->`` at the position where the "
    "element appears on the page.  The description is one or two short "
    "sentences (chart type, axes, legend, most salient values or trends).  "
    "Do NOT transcribe the contents of figures (axis labels, legend entries, "
    "in-figure annotations) as markdown text.  Do NOT use ``> **Figure N:**`` "
    "block quotes or markdown image syntax for figures — only the "
    "``<!-- Figure: ... -->`` comment.  Do not invent content that is not "
    "visible.\n"
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
    """Page count, title, and author of a PDF, read from ``pdfinfo``."""
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
    timeout_seconds: int = 600,
) -> bytes:
    """Render one page (1-indexed) to image bytes entirely in memory.

    ``image_format`` is "png" or "jpeg".  Modern poppler writes the image to
    stdout when no output prefix is given.  Older releases require a prefix
    argument; in that case a short-lived temp file is used as a fallback and
    removed immediately.
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
# Job / document bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageJob:
    """Everything a worker process needs to convert one page.  Picklable."""

    pdf_path: str
    stem: str
    page_number: int
    output_path: str            # '<stem>.md.pageNNNN'
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
    figure_enabled: bool
    figure_mode: str            # "vlm" | "heuristic"
    heuristic_boxes: tuple      # per-page qpdf placements (heuristic mode)
    figure_debug: bool


@dataclass(frozen=True)
class DocumentRecord:
    """Bookkeeping for one PDF (kept in the parent process / collation queue)."""

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
    completed in this run.  As soon as ``finished`` reaches the expected page
    count the document is handed to the collation worker.
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
            body = text[first_newline + 1:]
            if body.rstrip().endswith("```"):
                return body.rstrip()[:-3].rstrip()
    return text


def _chat_with_retries(client, job: PageJob, messages: list[dict]) -> str:
    """Chat completion with per-job retries and exponential backoff."""
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
            # Also treat as empty content that becomes empty after
            # code-fence stripping (observed model quirk under server
            # load); retrying instead of caching a blank page keeps
            # collation from producing silently empty slides.
            if not content or not _strip_code_fences(content).strip():
                raise RuntimeError("VLM returned empty content")
            return content.strip()
        except Exception as error:  # noqa: BLE001 - any API failure is retried
            last_error = error
            if attempt < job.retries:
                time.sleep(2.0 * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"VLM call failed after {job.retries} attempt(s): {last_error}"
    )


# ---------------------------------------------------------------------------
# Figure pipeline (ported from pdf_to_markdown_agentic.py, in-memory bytes)
# ---------------------------------------------------------------------------


def _figure_debug_dir(job: PageJob) -> Path | None:
    if not job.figure_debug:
        return None
    debug_dir = Path(job.output_path).parent / "figure_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def write_figure_debug(job: PageJob, name: str, content: str) -> None:
    """Write a per-page figure-debug text artifact (namespaced by stem)."""
    debug_dir = _figure_debug_dir(job)
    if debug_dir is None:
        return
    path = debug_dir / f"{job.stem}_p{job.page_number:04d}_{name}.txt"
    path.write_text(content, encoding="utf-8")


def extract_figure_descriptions_from_markdown(md_text: str) -> list[dict]:
    """Parse figure markers from markdown in document order."""
    figures = []
    marker_index = 0
    for line_no, line in enumerate(md_text.splitlines()):
        for match in re.finditer(r"<!--\s*Figure:\s*(.*?)\s*-->", line):
            figures.append({
                "marker_index": marker_index,
                "description": match.group(1).strip(),
                "line_no": line_no,
            })
            marker_index += 1
    return figures


def summarize_figure_description(description: str) -> str:
    """Reduce a verbose figure description to a short matching prompt."""
    summary = description.strip()
    summary = re.split(r"(?<=[.!?])\s+", summary)[0].strip()
    summary = re.split(r"\bThe data-to-background ratio\b", summary, maxsplit=1)[0].strip()
    summary = re.split(r"\bIn \([a-z]\)\b", summary, maxsplit=1)[0].strip()
    return summary or description.strip()


def scale_bbox_to_pixels(bbox, image_size) -> tuple[int, int, int, int]:
    """Convert bbox values that may be normalized or 0-1000 scaled into pixels."""
    x1, y1, x2, y2 = bbox
    width, height = image_size
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.5:
        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )
    if max_coord <= 1000:
        return (
            int((x1 / 1000.0) * width),
            int((y1 / 1000.0) * height),
            int((x2 / 1000.0) * width),
            int((y2 / 1000.0) * height),
        )
    return (int(x1), int(y1), int(x2), int(y2))


def normalize_bbox_to_image_space(bbox, image_size):
    """Return top-left and bottom-left interpretations of a reported bbox."""
    x1, y1, x2, y2 = scale_bbox_to_pixels(bbox, image_size)
    width, height = image_size
    top_left = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    bottom_left = (max(0, x1), max(0, height - y2), min(width, x2), min(height, height - y1))
    return top_left, bottom_left


def clamp_bbox(bbox, image_size):
    """Clamp bbox to image bounds."""
    x1, y1, x2, y2 = bbox
    width, height = image_size
    return (
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )


def choose_best_bbox(img, bbox) -> tuple[tuple[int, int, int, int], dict]:
    """Choose between top-left and bottom-left bbox interpretations."""
    import PIL.ImageStat as ImageStat

    top_left, bottom_left = normalize_bbox_to_image_space(bbox, img.size)

    def _score(candidate):
        cx1, cy1, cx2, cy2 = candidate
        if cx2 <= cx1 or cy2 <= cy1:
            return -1.0
        crop = img.crop(candidate).convert("L")
        stat = ImageStat.Stat(crop)
        mean = stat.mean[0] if stat.mean else 255.0
        return 255.0 - mean

    top_score = _score(top_left)
    bottom_score = _score(bottom_left)
    chosen = bottom_left if bottom_score > top_score else top_left
    return chosen, {
        "top_left": top_left,
        "bottom_left": bottom_left,
        "top_score": top_score,
        "bottom_score": bottom_score,
        "chosen": chosen,
    }


def parse_json_object(text: str) -> dict | None:
    """Parse the first JSON object found in model output."""
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def normalize_review_bbox(raw_bbox, image_size):
    """Normalize a review-suggested bbox into pixel coordinates."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw_bbox)
    except (TypeError, ValueError):
        return None
    scaled = scale_bbox_to_pixels((x1, y1, x2, y2), image_size)
    return clamp_bbox(scaled, image_size)


def review_figure_crop(job: PageJob, client, page_b64: str, crop_b64: str,
                       description: str) -> tuple[dict | None, str]:
    """Review a crop against the full page and return structured feedback."""
    match_description = summarize_figure_description(description)
    prompt = (
        "You are reviewing a candidate figure crop extracted from a PDF page. "
        "You will be given two images: first the full page, then the extracted crop. "
        "Decide whether the crop correctly captures the figure described in the markdown. "
        "Focus on the visual figure region itself: plots, charts, diagrams, logos, photos, or multi-panel figure areas. "
        "If the crop is wrong, blank, mostly whitespace, too small, only partial, or includes surrounding body text or a figure caption that sits below the figure, suggest a better bounding box on the FULL PAGE image. "
        "Figure captions underneath a figure should usually be excluded from the crop unless the caption text is visually embedded inside the figure artwork itself. "
        "A crop that includes a separate caption block below the figure should generally be treated as a mismatch and tightened to exclude that caption region. "
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  "match": true or false,\n'
        '  "issue": "ok" | "blank_crop" | "mostly_whitespace" | "partial_crop" | "wrong_region" | "low_confidence",\n'
        '  "summary": "brief description of the crop",\n'
        '  "suggested_bbox": [x1, y1, x2, y2] or null\n'
        "}\n"
        "Use FULL PAGE coordinates for suggested_bbox. Use pixel coordinates if possible; normalized [0,1] or [0,1000] coordinates are also acceptable. "
        "When refining the crop, prefer a tight bounding box around the actual visual figure content and exclude nearby paragraph text and external captions. "
        "If the crop already looks correct, set match=true and suggested_bbox=null.\n\n"
        f"Markdown figure description: {match_description}"
    )
    try:
        response_text = _chat_with_retries(client, job, [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{job.image_format};base64,{page_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/{job.image_format};base64,{crop_b64}"}},
            ],
        }])
    except Exception as exc:
        return None, f"ERROR: {exc}"
    return parse_json_object(response_text), response_text


def detect_figure_candidates_vlm(job: PageJob, client, page_b64: str,
                                 figure_markers: list[dict]) -> list[dict]:
    """Locate described figures on the page using a VLM (in-memory page bytes)."""
    if not figure_markers:
        return []

    figure_list = "\n".join(
        f"FIGURE {fig['marker_index']}: {fig['description']}"
        for fig in figure_markers
    )
    prompt = (
        "This page has already been converted to markdown, and the markdown contains "
        "the following figure descriptions. For each figure below, locate the full matching "
        "visual region on the page image and output one or more lines in this format:\n"
        "FIGURE <index> | COORD: <x1>,<y1>,<x2>,<y2> | DESC: <description>\n"
        "If a described figure has multiple panels, output one line per panel using the same figure index. "
        "Logos should also be localized if they were marked as figures in the markdown. "
        "Use pixel coordinates relative to the full page image. If a figure cannot be "
        "located confidently, omit it. Output ONLY lines in the specified format.\n\n"
        f"{figure_list}"
    )
    response_text = _chat_with_retries(client, job, [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/{job.image_format};base64,{page_b64}"}},
        ],
    }])

    write_figure_debug(
        job, "coordinate_extraction_vlm",
        "PROMPT:\n" + prompt + "\n\nRESPONSE:\n" + response_text,
    )

    by_index = {fig["marker_index"]: fig for fig in figure_markers}
    grouped_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    grouped_desc: dict[int, str] = {}
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("FIGURE ") or "COORD:" not in line:
            continue
        try:
            parts = [part.strip() for part in line.split("|")]
            marker_index = int(parts[0].replace("FIGURE", "").strip())
            coords_part = next(part for part in parts if part.startswith("COORD:"))
            desc_part = next((part for part in parts if part.startswith("DESC:")), None)
            x1, y1, x2, y2 = map(float, coords_part.replace("COORD:", "").strip().split(","))
            if x2 <= x1 or y2 <= y1:
                continue
            original = by_index.get(marker_index)
            if original is None:
                continue
            grouped_boxes.setdefault(marker_index, []).append((x1, y1, x2, y2))
            grouped_desc[marker_index] = (
                desc_part.replace("DESC:", "").strip() if desc_part else original["description"]
            )
        except (StopIteration, ValueError, IndexError):
            continue

    candidates = []
    for marker_index, boxes in grouped_boxes.items():
        original = by_index.get(marker_index)
        if original is None:
            continue
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        candidates.append({
            "marker_index": marker_index,
            "description": original["description"],
            "bbox": (x1, y1, x2, y2),
            "reported_description": grouped_desc.get(marker_index, original["description"]),
            "source_boxes": boxes,
            "detection_mode": "vlm",
        })

    candidates.sort(key=lambda item: item["marker_index"])
    write_figure_debug(
        job, "coordinate_candidates_vlm",
        "\n".join(str(candidate) for candidate in candidates) or "NO_CANDIDATES",
    )
    return candidates


def normalized_pdf_box_to_pixel_bbox(box: dict, image_size) -> tuple[int, int, int, int] | None:
    """Convert qpdf normalized boxes (bottom-left origin) to pixel bbox (top-left)."""
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    width, height = image_size
    px1 = int(round(x1 * width))
    py1 = int(round((1.0 - y2) * height))
    px2 = int(round(x2 * width))
    py2 = int(round((1.0 - y1) * height))
    bbox = clamp_bbox((px1, py1, px2, py2), image_size)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def build_heuristic_page_map(pdf_path: str) -> dict[int, list[dict]]:
    """Analyze the PDF once with qpdf; normalized figure placements by page (1-indexed).

    Runs in the parent process (one qpdf pass per document); per-page slices
    are shipped to workers inside ``PageJob.heuristic_boxes``.
    """
    analysis = analyze_pdf(pdf_path)
    page_map: dict[int, list[dict]] = {}
    for page_record in analysis.get("pages", []):
        page_num = int(page_record["page"])
        placements: list[dict] = []
        for fig in page_record.get("figures", []):
            for placement in fig.get("placements", []):
                box = placement.get("normalized_box")
                if not box:
                    continue
                placements.append({
                    "normalized_box": box,
                    "object": fig.get("object"),
                    "name": fig.get("name"),
                    "subtype": fig.get("subtype"),
                    "figure_reasons": fig.get("figure_reasons", []),
                })
        placements.sort(
            key=lambda item: (
                round(1.0 - float(item["normalized_box"]["y2"]), 6),
                round(float(item["normalized_box"]["x1"]), 6),
                round(1.0 - float(item["normalized_box"]["y1"]), 6),
                round(float(item["normalized_box"]["x2"]), 6),
            )
        )
        page_map[page_num] = placements
    return page_map


def detect_figure_candidates_heuristic(job: PageJob, image_size,
                                       figure_markers: list[dict]) -> list[dict]:
    """Pair qpdf-derived figure boxes (from the job) with markers in reading order."""
    if not figure_markers or not job.heuristic_boxes:
        return []

    pixel_boxes: list[dict] = []
    for raw_box in job.heuristic_boxes:
        bbox = normalized_pdf_box_to_pixel_bbox(raw_box.get("normalized_box", {}), image_size)
        if bbox is None:
            continue
        pixel_boxes.append({
            "bbox": bbox,
            "normalized_box": raw_box.get("normalized_box"),
            "object": raw_box.get("object"),
            "name": raw_box.get("name"),
            "subtype": raw_box.get("subtype"),
            "figure_reasons": raw_box.get("figure_reasons", []),
        })

    pair_count = min(len(figure_markers), len(pixel_boxes))
    candidates: list[dict] = []
    pairing_logs: list[str] = [
        f"mode=heuristic page={job.page_number} markers={len(figure_markers)} "
        f"boxes={len(pixel_boxes)} paired={pair_count}"
    ]

    for idx in range(pair_count):
        marker = figure_markers[idx]
        box_info = pixel_boxes[idx]
        pairing_logs.append(
            f"marker_index={marker['marker_index']} desc={marker['description']!r} -> "
            f"bbox={box_info['bbox']} normalized={box_info['normalized_box']} "
            f"object={box_info['object']} name={box_info['name']} "
            f"reasons={box_info['figure_reasons']}"
        )
        candidates.append({
            "marker_index": marker["marker_index"],
            "description": marker["description"],
            "bbox": box_info["bbox"],
            "reported_description": marker["description"],
            "source_boxes": [box_info["bbox"]],
            "detection_mode": "heuristic",
            "normalized_box": box_info["normalized_box"],
            "object": box_info["object"],
            "name": box_info["name"],
            "figure_reasons": box_info["figure_reasons"],
        })

    for marker in figure_markers[pair_count:]:
        pairing_logs.append(
            f"marker_index={marker['marker_index']} desc={marker['description']!r} -> NO_HEURISTIC_BOX"
        )
    for box_info in pixel_boxes[pair_count:]:
        pairing_logs.append(
            f"UNUSED_HEURISTIC_BOX bbox={box_info['bbox']} normalized={box_info['normalized_box']} "
            f"object={box_info['object']} name={box_info['name']} reasons={box_info['figure_reasons']}"
        )

    write_figure_debug(job, "coordinate_candidates_heuristic", "\n".join(pairing_logs))
    return candidates


def validate_and_extract_figures(job: PageJob, client, img, page_b64: str,
                                 candidates: list[dict]) -> list[dict]:
    """Validate candidate figure crops, retry once on bad reviews, save figures."""
    if not candidates:
        return []

    figures_dir = Path(job.output_path).parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    validation_logs: list[str] = []
    for candidate in sorted(candidates, key=lambda item: item["marker_index"]):
        chosen_bbox, bbox_debug = choose_best_bbox(img, candidate["bbox"])
        attempts: list[dict] = []
        current_bbox = chosen_bbox
        selected_bbox = chosen_bbox
        selected_crop = None
        selected_review = None
        selected_review_raw = ""

        for attempt_num in range(1, 3):
            x1, y1, x2, y2 = current_bbox
            crop = img.crop((x1, y1, x2, y2))

            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            review_data, review_raw = review_figure_crop(
                job, client, page_b64, crop_b64, candidate["description"],
            )

            attempts.append({
                "attempt": attempt_num,
                "bbox": current_bbox,
                "crop_size": crop.size,
                "review": review_data,
                "review_raw": review_raw,
            })

            if selected_crop is None:
                selected_bbox = current_bbox
                selected_crop = crop.copy()
                selected_review = review_data
                selected_review_raw = review_raw

            if review_data and review_data.get("match") is True:
                selected_bbox = current_bbox
                selected_crop = crop.copy()
                selected_review = review_data
                selected_review_raw = review_raw
                break

            suggested_bbox = normalize_review_bbox(
                review_data.get("suggested_bbox") if review_data else None,
                img.size,
            )
            if attempt_num == 1 and suggested_bbox is not None and suggested_bbox != current_bbox:
                current_bbox = suggested_bbox
                continue
            break

        if selected_crop is None:
            continue

        fig_name = f"fig_{job.stem}_p{job.page_number:04d}_{candidate['marker_index']}.png"
        selected_crop.save(str(figures_dir / fig_name))
        figure_record = {
            "page_num": job.page_number,
            "marker_index": candidate["marker_index"],
            "description": candidate["description"],
            "reported_description": candidate.get("reported_description", candidate["description"]),
            "saved_path": f"figures/{fig_name}",
            "raw_bbox": candidate["bbox"],
            "bbox": selected_bbox,
            "bbox_debug": bbox_debug,
            "source_boxes": candidate.get("source_boxes", [candidate["bbox"]]),
            "crop_size": selected_crop.size,
            "validation_response": selected_review_raw,
            "validation_result": selected_review,
            "validation_attempts": attempts,
        }
        figures.append(figure_record)
        validation_logs.append(str(figure_record))

    write_figure_debug(
        job, "validation",
        "\n\n".join(validation_logs) if validation_logs else "NO_VALIDATION_LOGS",
    )
    return figures


def replace_figure_markers(job: PageJob, md_text: str, figures: list[dict]) -> str:
    """Replace figure markers with markdown image tags by marker order."""
    figures_by_index = {fig["marker_index"]: fig for fig in figures}
    marker_counter = 0
    replacement_logs: list[str] = []

    def _replace(match: re.Match) -> str:
        nonlocal marker_counter
        description = match.group(1).strip()
        current_index = marker_counter
        fig = figures_by_index.get(current_index)
        marker_counter += 1
        if fig is None:
            replacement_logs.append(f"marker_index={current_index} desc={description!r} -> NO_MATCH")
            return match.group(0)
        replacement_logs.append(
            f"marker_index={current_index} desc={description!r} -> {fig['saved_path']} bbox={fig['bbox']}"
        )
        return f"![{description}]({fig['saved_path']})"

    replaced = re.sub(r"<!--\s*Figure:\s*(.*?)\s*-->", _replace, md_text)
    write_figure_debug(
        job, "replacement",
        "\n".join(replacement_logs) if replacement_logs else "NO_REPLACEMENTS",
    )
    return replaced


def process_page_figures(job: PageJob, client, image_bytes: bytes, page_b64: str,
                         markdown: str) -> tuple[str, int]:
    """Run the in-memory figure stage for one page.

    Returns ``(markdown_with_image_tags, number_of_saved_figures)``.
    """
    if Image is None:
        return markdown, 0

    markers = extract_figure_descriptions_from_markdown(markdown)
    if not markers:
        return markdown, 0

    write_figure_debug(
        job, "markers",
        "mode=" + job.figure_mode + "\n" + "\n".join(str(m) for m in markers),
    )

    img = Image.open(io.BytesIO(image_bytes))

    if job.figure_mode == "vlm":
        candidates = detect_figure_candidates_vlm(job, client, page_b64, markers)
    elif job.figure_mode == "heuristic":
        candidates = detect_figure_candidates_heuristic(job, img.size, markers)
    else:
        raise ValueError(f"Unsupported figure extraction mode: {job.figure_mode}")

    figures = validate_and_extract_figures(job, client, img, page_b64, candidates)
    if not figures:
        return markdown, 0

    return replace_figure_markers(job, markdown, figures), len(figures)


# ---------------------------------------------------------------------------
# Worker: one page, in memory
# ---------------------------------------------------------------------------


def run_page_job(job: PageJob) -> dict:
    """Worker entry point: text + image (in memory) -> VLM -> markdown page file.

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
                {"type": "text", "text": USER_PROMPT_TEMPLATE.format(raw_text=text)},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{job.image_format};base64,{b64_image}"},
                },
            ],
        },
    ]
    markdown = _strip_code_fences(_chat_with_retries(client, job, messages))

    figures_count = 0
    if job.figure_enabled:
        markdown, figures_count = process_page_figures(
            job, client, image, b64_image, markdown
        )

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
        "figures": figures_count,
        "worker_pid": os.getpid(),
        "worker_cli_cpu_seconds": _reaped_children_cpu_seconds(),
    }


# ---------------------------------------------------------------------------
# Agentic markdown review (ported from pdf_to_markdown_agentic.py)
# ---------------------------------------------------------------------------


def _validate_markdown_text(text: str) -> list[str]:
    """Validate markdown syntax by attempting to convert it to HTML.

    Returns a list of error strings; empty list means the markdown is valid.
    """
    import markdown as markdown_lib

    errors: list[str] = []

    md = markdown_lib.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "footnotes",
            "attr_list",
            "md_in_html",
        ]
    )

    try:
        html = md.convert(text)
    except Exception as exc:
        errors.append(f"markdown.convert raised: {exc}")
        return errors

    broken_comments = re.findall(r"<!--(.*?)-->", html, re.DOTALL)
    for idx, block in enumerate(broken_comments):
        block_stripped = block.strip()
        if block_stripped.startswith("Figure:") or block_stripped.startswith("Table:"):
            continue
        if not re.search(r"<[^/>]", block_stripped):
            continue
        errors.append(
            f"Unclosed HTML tag detected (comment block {idx + 1}): "
            f"{block_stripped[:200]}"
        )

    return errors


def _validate_markdown_tool(file_path: str) -> str:
    """Validate markdown syntax in a file (exposed to the review agent)."""
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"ERROR reading file: {exc}"

    errors = _validate_markdown_text(text)

    if errors:
        return "VALIDATION ERRORS:\n" + "\n".join(errors)
    return "VALIDATION PASSED: file is valid markdown."


REVIEW_SYSTEM_PROMPT = (
    "You are a Markdown syntax auditor and fixer. Your job is to read a "
    "Markdown file, validate its syntax, fix any issues you find, and "
    "re-validate until the file passes.\n\n"
    "You have three tools available:\n"
    "- read_file: read the current contents of a file\n"
    "- edit_file: make targeted fixes to a file\n"
    "- validate_markdown: run the Python markdown parser on a file and get "
    "structured error output (or a PASS message)\n"
    "All file paths are relative to your /workspace directory.\n\n"
    "Your workflow:\n"
    "1. Call validate_markdown on the target file to get a list of errors\n"
    "2. If validation passed, report success and stop\n"
    "3. If validation failed, read the file with read_file to see the current content\n"
    "4. Use edit_file to fix each error you identified\n"
    "5. Call validate_markdown again to check your fixes\n"
    "6. Repeat steps 3-5 until validation passes or you determine no more fixes are possible\n\n"
    "Common markdown rules:\n"
    "- Headings must use consistent # syntax (e.g. # Title, ## Section)\n"
    "- List items must have proper indentation and spacing\n"
    "- Table rows must align columns\n"
    "- Fenced code blocks (```) must be closed\n"
    "- Blockquotes (>) must be on their own lines\n"
    "- Mathematical expressions using $...$ or $$...$$ must be balanced\n"
    "- Bold (**text**) and italic (*text*) markers must be balanced\n"
    "- Links must be in valid [text](url) format\n"
    "- Footnotes must use proper [^1] and [^1]: syntax\n\n"
    "Be thorough — this markdown is the authoritative text output "
    "of a PDF-to-Markdown conversion pipeline. Every conversion artifact "
    "must be cleaned."
)


def review_markdown_file(file_path: str, model: str, api_key: str, base_url: str,
                         max_iterations: int) -> None:
    """Agentic review loop: validate markdown, fix errors until clean.

    Runs in the collation worker thread (parent process), on the merged
    presentation file.  Raises on unrecoverable agent errors.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        StateBackend,
    )

    abs_path = Path(file_path).resolve()
    file_name = abs_path.name
    # Mount the file's directory at /workspace so the built-in file
    # tools (read_file/edit_file) and validate_markdown share one
    # namespace with the real disk.  deepagents' default backend is an
    # empty in-memory StateBackend, where the file tools can never
    # see the target file (and edits would be lost on disk).
    workspace = abs_path.parent
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/workspace/": FilesystemBackend(
            root_dir=str(workspace), virtual_mode=True)},
    )

    def _resolve(vpath: str) -> Path:
        rel = vpath
        if rel.startswith("/workspace/"):
            rel = rel[len("/workspace/"):]
        return workspace / rel.lstrip("/")

    @tool
    def validate_markdown(file_path: str) -> str:
        """Validate markdown syntax in a file using the Python markdown library.

        Returns the full validation result (errors or success message).
        """
        return _validate_markdown_tool(_resolve(file_path))

    print(f"\n=== Agentic Markdown Review ({file_name}) ===")

    review_model = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        use_responses_api=False,
    )
    agent = create_deep_agent(
        model=review_model,
        tools=[validate_markdown],
        system_prompt=REVIEW_SYSTEM_PROMPT,
        backend=backend,
    )

    task_desc = (
        f"Review and fix markdown syntax in the file "
        f"'/workspace/{file_name}'.\n\n"
        "Use your validate_markdown tool to check the file, then read_file to see "
        "the content, and edit_file to fix any errors. Re-validate after each fix "
        "cycle. Continue until validate_markdown returns 'VALIDATION PASSED' or you "
        "determine no more fixes are possible."
    )

    for iteration in range(1, max_iterations + 1):
        print(f"  Review iteration {iteration}/{max_iterations}...")
        try:
            result = agent.invoke({"messages": [HumanMessage(content=task_desc)]})
        except Exception as exc:
            print(f"  Agent error on iteration {iteration}: {exc}")
            return

        output = result.content if hasattr(result, "content") else str(result)
        if output is None:
            print(f"  No output from agent on iteration {iteration}.")
            return

        if "VALIDATION PASSED" in output:
            print(f"  Markdown validation passed on iteration {iteration}.")
            return

    print("  Review completed (max iterations reached).")


# ---------------------------------------------------------------------------
# Collation (merge pages -> presentation markdown -> review -> cleanup)
# ---------------------------------------------------------------------------


def _document_header(title: str, author: str) -> str:
    if author:
        return f"# {title} — {author}"
    return f"# {title}"


def _page_files_for(doc: DocumentRecord) -> list[Path]:
    page_dir = Path(doc.presentation_path).parent
    return [
        page_dir / f"{doc.stem}.md.page{page:04d}"
        for page in range(1, doc.num_pages + 1)
    ]


def collate_document(doc: DocumentRecord, legacy_page_format: bool = False) -> dict:
    """Merge one document's pages into its presentation markdown.

    Runs in the collation worker thread.  ``doc.num_pages`` is the expected
    page count from the startup pdfinfo scan; the page files are re-checked
    on disk before merging, so a partial document is reported as ``skipped``
    (nothing is deleted) instead of being turned into a partial deck.

    Default layout:

        # <Title> — <Author>

        ## Slide 1

        <page 1 markdown>

        ---

        ## Slide 2
        ...

    Legacy layout (``legacy_page_format=True``, matching
    ``pdf_to_markdown_agentic.py``):

        # <stem>.pdf

        <page 1 markdown>

        ---

        <page 2 markdown>

        ---

    Page files are NOT deleted here — the caller deletes them after the
    agentic review step completes.
    """
    page_files = _page_files_for(doc)
    missing = [p.name for p in page_files if not p.is_file()]
    if missing:
        return {"status": "skipped", "missing": missing, "output": doc.presentation_path}

    lines: list[str]
    if legacy_page_format:
        lines = [f"# {doc.stem}.pdf", ""]
        for page_file in page_files:
            content = page_file.read_text(encoding="utf-8").strip()
            lines.extend([content, "", "---", ""])
    else:
        lines = [_document_header(doc.title, doc.author), ""]
        for page, page_file in enumerate(page_files, start=1):
            content = page_file.read_text(encoding="utf-8").strip()
            lines.extend([f"## Slide {page}", "", content, "", "---", ""])
    text = "\n".join(lines).rstrip("\n") + "\n"

    presentation_path = Path(doc.presentation_path)
    presentation_path.parent.mkdir(parents=True, exist_ok=True)
    presentation_path.write_text(text, encoding="utf-8")

    return {
        "status": "collated",
        "slides": doc.num_pages,
        "output": str(presentation_path),
        "bytes": len(text.encode("utf-8")),
    }


def _delete_page_files(doc: DocumentRecord) -> None:
    for page_file in _page_files_for(doc):
        try:
            page_file.unlink()
        except FileNotFoundError:
            pass


def _collation_worker(requests: queue.Queue, results: queue.Queue,
                      review_cfg: tuple[str, str, str, int, bool, bool]) -> None:
    """Single dedicated post-processing worker (one thread).

    Pulls documents to collate, verifies on disk that all expected page
    files are present, merges them, runs the agentic review on the merged
    file, and only then deletes the page files.  A single worker means two
    requests can never race on the same document; the worker stays idle
    while the conversion pool is busy.
    """
    model, api_key, base_url, max_review_iterations, skip_review, legacy = review_cfg
    while True:
        doc = requests.get()
        if doc is None:  # sentinel: shut down
            break
        try:
            result = collate_document(doc, legacy_page_format=legacy)
            if result["status"] != "collated":
                print(
                    f"collation skipped for {doc.pdf_name}: missing page file(s) "
                    f"{', '.join(result['missing'])}",
                    file=sys.stderr,
                )
                results.put(("skipped", doc.pdf_name, None))
                continue

            if not skip_review:
                review_markdown_file(
                    doc.presentation_path, model, api_key, base_url,
                    max_review_iterations,
                )

            _delete_page_files(doc)
            print(
                f"collated {doc.pdf_name}: {result['slides']} slides -> "
                f"{result['output']} ({result['bytes'] // 1024} KB)"
            )
            results.put(("collated", doc.pdf_name, None))
        except Exception as error:
            print(f"COLLATE/REVIEW FAILED {doc.pdf_name}: {error}", file=sys.stderr)
            results.put(("error", doc.pdf_name, str(error)))


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
    return any(p.is_file() for p in _page_files_for(doc))


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
    pool workers, and any active CLI subprocesses — Pss counts memory
    shared by forked workers only once — and ``cpu_seconds`` is their
    cumulative user+system CPU time plus the rusage of already-reaped
    children.  Returns ``None`` when /proc is unavailable.
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
    of the short-lived pdftotext/pdftoppm subprocesses it has reaped."""
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
    cumulative average CPU utilisation of the whole process tree over the
    run so far (may exceed 100% while several workers are busy).  Returns
    an empty string when the stats are not measurable.
    """
    # The CPU average is only meaningful once the run has actually elapsed
    # (near-zero denominators otherwise produce absurd percentages).
    if peak_mem_kb <= 0 or elapsed < 1.0 or not mem_total_kb:
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

    if not args.dry_run and not args.skip_review:
        missing = [
            module
            for module in ("langchain_openai", "deepagents")
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            raise SystemExit(
                f"Agentic review is enabled but missing dependencies: {missing}. "
                "Install them (pixi install) or pass --skip-review."
            )

    pdftotext = require_tool("pdftotext", "PDFTOTEXT")
    pdftoppm = require_tool("pdftoppm", "PDFTOPPM")
    pdfinfo = require_tool("pdfinfo", "PDFINFO")
    if not args.dry_run and args.figure_extraction_mode == "heuristic":
        require_tool("qpdf", "QPDF")

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

    figure_enabled = not args.skip_figure_extraction
    figure_mode = args.figure_extraction_mode
    image_format = "jpeg" if args.jpeg else "png"

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
            print(f"warning: could not read info for {pdf_path}: {error}", file=sys.stderr)
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

        # Heuristic figure mode: one qpdf pass per document (parent process);
        # per-page box slices are shipped to the workers inside PageJob.
        page_map: dict[int, list[dict]] | None = None
        needs_conversion = not doc_done and any(
            not (page_path_for(pdf_path, input_dir, output_dir, page).exists()
                 or (args.dry_run))
            for page in range(1, num_pages + 1)
        )
        if (
            not args.dry_run
            and needs_conversion
            and figure_enabled
            and figure_mode == "heuristic"
        ):
            try:
                page_map = build_heuristic_page_map(str(pdf_path))
                total_boxes = sum(len(v) for v in page_map.values())
                print(
                    f"  heuristic: {total_boxes} figure-like placement(s) across "
                    f"{len(page_map)} page(s) in {pdf_path.name}"
                )
            except Exception as error:
                print(
                    f"warning: qpdf heuristic analysis failed for {pdf_path.name}: "
                    f"{error} (figures will keep their markers)",
                    file=sys.stderr,
                )
                page_map = {}

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
                    stem=pdf_path.stem,
                    page_number=page_number,
                    output_path=str(out_path),
                    pdftotext=pdftotext,
                    pdftoppm=pdftoppm,
                    base_url=args.llm_base_url,
                    api_key=args.api_key,
                    model=args.model,
                    image_resolution=args.dpi,
                    max_image_dimension=args.max_image_dimension,
                    image_format=image_format,
                    jpeg_quality=args.jpeg_quality,
                    max_tokens=args.max_tokens,
                    api_timeout=args.api_timeout,
                    retries=args.retries,
                    dry_run=args.dry_run,
                    figure_enabled=figure_enabled and not args.dry_run,
                    figure_mode=figure_mode,
                    heuristic_boxes=tuple(page_map.get(page_number, ())) if page_map is not None else (),
                    figure_debug=not args.no_figure_debug,
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
        f"{mode}: {len(jobs)} page(s) across {len(documents)} file(s) "
        f"(skipped {skipped_pages} existing page(s))"
    )
    if doc_states:
        print(
            f"Collating each of the {len(doc_states)} document(s) as soon as "
            f"all of its pages are converted"
        )
        if not args.skip_review:
            print("Running agentic markdown review after each collation")
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
    # is done, so a document's collation/review can never wait behind the
    # pool's backlog of still-pending page jobs.
    requests: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    collator: threading.Thread | None = None
    if doc_states:
        review_cfg = (
            args.model,
            args.api_key,
            args.llm_base_url,
            args.max_review_iterations,
            args.skip_review,
            args.legacy_page_format,
        )
        collator = threading.Thread(
            target=_collation_worker,
            args=(requests, results, review_cfg),
            name="collation-worker",
            daemon=True,
        )
        collator.start()

        def hand_off(state: DocState) -> None:
            """Hand a document to the collation worker once all pages are done
            (the worker re-verifies on disk before merging)."""
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
                    f"markdown={result['markdown_chars']} chars, "
                    f"figures={result['figures']}"
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
