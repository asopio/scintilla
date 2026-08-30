#!/usr/bin/env python3

"""Batch-convert PDFs to markdown using pdftotext + VLM vision-language model.

For each page, extracts raw text with the pdftotext CLI and renders the page
as an image.  Both are passed to a VLM (via an OpenAI-compatible API) with a
system prompt that instructs it to produce valid, well-structured markdown with
LaTeX equations and Greek letters rendered via ``$...$`` / ``$$...$$`` delimiters.

Usage:
    python extract_pdf_markdown_vlm.py \\
        --input-dir ./pdfs \\
        --output-dir ./md \\
        --base-url http://localhost:4141/v1 \\
        --model "nvidia/Qwen3.6-35B-A3B-NVFP4" \\
        --workers 4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="./pdfs",
        help="Directory containing PDF files to convert",
    )
    parser.add_argument(
        "--output-dir",
        default="./md",
        help="Directory where Markdown files will be written",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for PDF discovery (used recursively)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL for VLM calls",
    )
    parser.add_argument(
        "--api-key",
        default="local",
        help="API key for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="VLM model name to use for page-level conversion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first conversion error",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=150,
        help="Maximum number of PDF pages to process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of files to process in parallel",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=200,
        help="DPI for page image rendering (higher = better quality, more tokens)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a document-to-markdown converter.  You will receive:\n"
    "  1. The raw text extracted from a PDF page (may contain OCR artifacts, "
    "repeated headers/footers, and mis-ordered text).\n"
    "  2. A screenshot image of the same page.\n"
    "\n"
    "Produce a clean, well-structured markdown document that faithfully "
    "represents the page content.  Follow these rules exactly:\n"
    "\n"
    "1. **Valid markdown only** — the output must render correctly as markdown. "
    "Use proper syntax for all elements.\n"
    "\n"
    "2. **Headings** — use ``#``, ``##``, ``###``, etc. to reflect the "
    "document hierarchy on the page.  The main title should be ``#``, section "
    "headings ``##``, subsections ``###``, etc.\n"
    "\n"
    "3. **Equations and Greek letters** — render ALL mathematical expressions, "
    "equations, and formulas using LaTeX syntax inside ``$...$`` for inline "
    "math and ``$$...$$`` for display equations.  Also render any Greek "
    "letters (e.g. alpha, beta, gamma, Delta, Omega) using LaTeX commands "
    "(e.g. ``\\alpha``, ``\\beta``, ``\\Gamma``, ``\\Omega``) inside ``$`` or "
    "$$`` delimiters.  Never leave raw Greek text unrendered when it represents "
    "a mathematical or physics variable.\n"
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
    "8. **Clean up** — remove duplicate headers/footers, page numbers, "
    "repeated labels.  Keep only the meaningful content of the page.\n"
    "\n"
    "9. **Do not** add any introductory text, meta-commentary, or explanation "
    "outside the markdown itself.  Output only the markdown.\n"
)


# ---------------------------------------------------------------------------
# pdftotext wrapper (single page via -f/-l flags)
# ---------------------------------------------------------------------------


def extract_page_text_with_pdftotext(pdf_path: Path, page_number: int) -> str:
    """Extract raw text from exactly one PDF page using pdftotext.

    pdftotext ``-f`` and ``-l`` are 1-indexed, so we pass ``page_number``
    directly.  This avoids reading the entire PDF and is much faster when
    processing pages in parallel.
    """
    pdftotext_path = shutil.which("pdftotext")
    if pdftotext_path is None:
        raise SystemExit(
            "pdftotext not found. Install poppler-utils:\n"
            "  conda install -c conda-forge poppler-utils\n"
            "  apt-get install -y poppler-utils\n"
            "  yum install -y poppler-utils"
        )

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
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr.strip()}")
    return result.stdout


# ---------------------------------------------------------------------------
# Single-page VLM conversion
# ---------------------------------------------------------------------------


def convert_page(
    pdf_path: Path,
    page_number: int,
    output_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    image_resolution: int,
):
    """Convert one page of a PDF to markdown via pdftotext + VLM.

    Each worker gets its own temporary directory so concurrent workers
    never collide on temp files.  VLM calls are issued in parallel across
    workers (the bottleneck step).  We use pdftotext -f/-l for per-page
    text extraction and pdfplumber's direct per-page rendering — no
    intermediate splitting.
    """
    # Unique temp dir per-page so concurrent workers never collide
    page_tmp = output_dir / ".tmp_pages" / f"{pdf_path.name}_page_{page_number:04d}"
    page_tmp.mkdir(parents=True, exist_ok=True)

    image_path = page_tmp / "page.png"
    md_path = output_dir / f"{pdf_path.stem}_{page_number:04d}.md"

    # Render one page to an image (pdfplumber supports per-page rendering)
    render_page_to_image(pdf_path, page_number - 1, image_path, image_resolution)

    # Extract text from THIS page only (pdftotext -f/-l are 1-indexed)
    raw_text = extract_page_text_with_pdftotext(pdf_path, page_number)

    # Build the VLM prompt with text + image
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Here is the raw text from a single PDF page, followed by "
                        "the page image.  Convert this into clean, valid markdown.  "
                        "Extracted text:\n\n"
                        "```text\n"
                        + raw_text
                        + "\n```\n\n"
                        "Convert this page to markdown:"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_image_to_base64(image_path)}",
                    },
                },
            ],
        },
    ]

    # Call the VLM — this is where concurrency happens
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit(
            "Missing dependency. Install with: pixi run uv pip install openai"
        ) from error

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=8192,
        temperature=0.0,
    )
    markdown_text = response.choices[0].message.content.strip()

    md_path.write_text(markdown_text + "\n", encoding="utf-8")
    return md_path


def _image_to_base64(image_path: Path) -> str:
    import base64
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Page-level parallel processing (multiple pages per file)
# ---------------------------------------------------------------------------


def convert_all_pages(args) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    pdf_paths = discover_pdfs(input_dir, args.pattern)
    if not pdf_paths:
        print(f"No PDFs found in {input_dir} matching {args.pattern}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    skipped = 0
    failed = 0
    jobs = []

    for pdf_path in pdf_paths:
        md_path = output_path_for(pdf_path, input_dir, output_dir)

        if md_path.exists() and not args.overwrite:
            # Skip entire file if output already exists
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path), strict=False)
                num_pages = len(reader.pages)
            except Exception:
                num_pages = 0
            skipped += num_pages
            continue

        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path), strict=False)
            num_pages = len(reader.pages)
            if num_pages > args.max_pages:
                skipped += num_pages
                print(
                    f"skip {pdf_path} (has {num_pages} pages, max is {args.max_pages})",
                    file=sys.stderr,
                )
                continue
        except Exception as error:
            print(
                f"warning: could not check page count for {pdf_path}: {error}",
                file=sys.stderr,
            )
            continue

        total_pages += num_pages
        for page_idx in range(num_pages):
            jobs.append((pdf_path, page_idx + 1))  # page_number is 1-based

    if not jobs:
        print(f"Done. converted={total_pages} skipped={skipped} failed={failed}")
        return 0

    print(f"Converting {len(jobs)} page(s) across {len(pdf_paths)} file(s)")

    state = threading.local()
    start_time = time.perf_counter()
    completed = 0

    workers = max(1, args.workers)
    print(f"Using {workers} worker thread(s)")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                convert_page,
                pdf_path,
                page_number,
                output_dir,
                args.base_url,
                args.api_key,
                args.model,
                args.image_resolution,
            ): (pdf_path, page_number)
            for pdf_path, page_number in jobs
        }

        for future in as_completed(futures):
            completed += 1
            pdf_path, page_number = futures[future]

            error = future.exception()
            if error is None:
                print(
                    f"[{completed}/{len(jobs)}] done {pdf_path.name} page {page_number}"
                )
            else:
                failed += 1
                print(
                    f"[{completed}/{len(jobs)}] FAILED {pdf_path.name} page {page_number}: {error}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    for pending in futures:
                        pending.cancel()
                    break

            if completed % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"Progress: {completed}/{len(jobs)} processed, "
                    f"avg {elapsed / completed:.2f}s/page"
                )

    elapsed = time.perf_counter() - start_time
    print(f"Done. completed={completed} skipped={skipped} failed={failed} total_time={elapsed:.1f}s")
    print(f"Average: {elapsed / max(completed, 1):.2f}s/page")
    return 1 if failed > 0 else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_pdfs(input_dir: Path, pattern: str):
    return sorted(path for path in input_dir.rglob(pattern) if path.is_file())


def output_path_for(pdf_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = pdf_path.relative_to(input_dir)
    return (output_dir / relative).with_suffix(".md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    raise SystemExit(convert_all_pages(args))


if __name__ == "__main__":
    main()
