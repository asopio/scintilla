#!/usr/bin/env python3

"""Batch-convert PDF files to OCR-enabled Markdown using MarkItDown."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# def _suppress_ort_affinity_warnings():
#    # ONNX Runtime's pip package is not built with OpenMP, so OMP_NUM_THREADS has no
#    # effect on its internal Eigen thread pool.  On HPC clusters (e.g. Perlmutter) the
#    # SLURM cpuset exposes a non-contiguous set of CPU cores; ORT tries to pin threads
#    # to cores outside that set and logs a noisy ERROR for each failure.  Raising the
#    # default logger severity to FATAL suppresses those messages.  The threads still
#    # run correctly — they just won't have CPU affinity set.
#    try:
#        import onnxruntime as ort
#
#        ort.set_default_logger_severity(4)  # 4 = FATAL; suppresses ERROR-level affinity warnings
#    except ImportError:
#        pass

# _suppress_ort_affinity_warnings()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="./egam_contribs_test",
        help="Directory containing PDF files to convert",
    )
    parser.add_argument(
        "--output-dir",
        default="./egam_contribs_test_md",
        help="Directory where Markdown files will be written",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="Glob pattern for PDF discovery (used recursively)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:4141/v1",
        help="OpenAI-compatible base URL for OCR/model calls",
    )
    parser.add_argument(
        "--api-key",
        default="local",
        help="API key for the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        # default="gpt-4.1",
        # default="goldeneye-free-auto" # I think this is the github internal name for 'raptor mini'
        help="Model name used by MarkItDown OCR/conversion",
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
        "--workflow",
        choices=["ocr", "visual-description", "no-ocr"],
        default="ocr",
        help=(
            "Conversion workflow: 'ocr' uses OCR-enhanced PDF conversion; "
            "'visual-description' uses per-slide text extraction plus image descriptions; "
            "'no-ocr' uses plain text extraction without LLM calls"
        ),
    )
    return parser.parse_args()


def build_converter(base_url: str, api_key: str, model: str):
    try:
        from markitdown import MarkItDown
        from markitdown_ocr import LLMVisionOCRService, PdfConverterWithOCR
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit(
            "Missing dependencies. Install with: pixi run uv pip install markitdown markitdown-ocr openai"
        ) from error

    llm_client = OpenAI(base_url=base_url, api_key=api_key)
    converter = MarkItDown(
        enable_plugins=False,
        llm_client=llm_client,
        llm_model=model,
    )

    # Explicitly register OCR PDF converter to avoid relying on plugin discovery.
    ocr_service = LLMVisionOCRService(client=llm_client, model=model, default_prompt=None)
    converter.register_converter(PdfConverterWithOCR(ocr_service=ocr_service), priority=-1.0)
    return converter


def build_text_only_converter():
    try:
        from markitdown import MarkItDown
    except ImportError as error:
        raise SystemExit("Missing dependency. Install with: pixi run uv pip install markitdown") from error
    return MarkItDown(enable_plugins=False)


def build_image_description_converter(base_url: str, api_key: str, model: str):
    try:
        from markitdown import MarkItDown
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Missing dependencies. Install with: pixi run uv pip install markitdown openai") from error

    return MarkItDown(
        enable_plugins=False,
        llm_client=OpenAI(base_url=base_url, api_key=api_key),
        llm_model=model,
        llm_prompt=(
            "Provide a concise visual description of this presentation slide. "
            "Focus on layout, charts, plots, tables, and key visual elements. "
            "Do not perform OCR or transcribe slide text verbatim,"
            "unless it is essential for understanding the visual content."
            "Ignore decorative elements: backgrounds, headers footers, logos,"
            "slide numbers, except on the title slide where they may be relevant."
            "Descriptions of charts and plots should be informative rather than superficial."
            "Example: 'a histogram peaking at 150 GeV with a long tail to 500 GeV'"
            "is better than 'a stepped histogram with blue fill and a dark blue outline'."
            "Exceptions are permitted where style elements are essential "
            "to distinguishing different data series or specific features."
            "Use labels and legends to inform your descriptions of plots, "
            "but do not feel compelled to transcribe every label unless "
            "they aid in understanding the data presented in the plots. "
            "Example: 'The tW contribution to the background dominates "
            "over the Z+jets at low jet multiplicities' is better than "
            "'The green area of the histogram stack is larger than "
            "the orange area in the left half of the plot.'"
            "Do not put any markdown headers in the description."
        ),
    )


def split_pdf_into_page_pdfs(pdf_path: Path, output_dir: Path):
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as error:
        raise SystemExit("Missing dependency. Install with: pixi run uv pip install pypdf") from error

    # Some CERN slide PDFs have mildly malformed xref pointers; suppress noisy parser warnings.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(pdf_path), strict=False)
    page_paths = []
    for index, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        page_path = output_dir / f"page_{index:04d}.pdf"
        with page_path.open("wb") as handle:
            writer.write(handle)
        page_paths.append(page_path)
    return page_paths


def render_page_pdf_to_image(page_pdf_path: Path, image_path: Path):
    try:
        import pdfplumber
    except ImportError as error:
        raise SystemExit("Missing dependency. Install with: pixi run uv pip install pdfplumber") from error

    with pdfplumber.open(str(page_pdf_path)) as pdf:
        if not pdf.pages:
            raise ValueError(f"No pages found in {page_pdf_path}")
        page_image = pdf.pages[0].to_image(resolution=200)
        page_image.original.save(str(image_path), format="PNG")


def extract_presentation_markdown_no_ocr(
    pdf_path: Path,
    output_markdown_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    llm_workers: int,
):
    """Alternative workflow: per-page text extraction + visual description, then merge."""
    text_converter = build_text_only_converter()
    llm_workers = max(1, llm_workers)

    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="slides_extract_") as temp_dir:
        temp_root = Path(temp_dir)
        pages_dir = temp_root / "pages"
        images_dir = temp_root / "images"
        pages_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        page_pdfs = split_pdf_into_page_pdfs(pdf_path, pages_dir)
        slides = []

        # Phase 1 (serial): split, render, and text extraction.
        for slide_number, page_pdf in enumerate(page_pdfs, start=1):
            page_image = images_dir / (f"slide_{slide_number:04d}.png")
            render_page_pdf_to_image(page_pdf, page_image)

            text_result = text_converter.convert(str(page_pdf))
            slide_text = (text_result.text_content or "").strip()

            slides.append(
                {
                    "slide_number": slide_number,
                    "slide_text": slide_text,
                    "page_image": page_image,
                }
            )

        # Phase 2 (parallel): visual descriptions from slide images.
        def describe_slide(slide):
            converter = build_image_description_converter(base_url, api_key, model)
            image_result = converter.convert(str(slide["page_image"]))
            return slide["slide_number"], (image_result.text_content or "").strip()

        visual_descriptions = {}
        with ThreadPoolExecutor(max_workers=llm_workers) as llm_pool:
            future_to_slide = {llm_pool.submit(describe_slide, slide): slide for slide in slides}
            for future in as_completed(future_to_slide):
                slide_number, description = future.result()

                # remove any markdown headers from the description to avoid interfering with final assembly
                description = description.replace("# Description:", "").strip()

                visual_descriptions[slide_number] = description

        sections = [f"# {pdf_path.stem}", ""]

        # Phase 3 (serial): final markdown assembly in slide order.
        for slide in slides:
            slide_number = slide["slide_number"]
            slide_text = slide["slide_text"]
            visual_description = visual_descriptions.get(slide_number, "")

            sections.append(f"## Slide {slide_number}")
            sections.append("")
            sections.append("### Extracted Text")
            sections.append(slide_text if slide_text else "(no text extracted)")
            sections.append("")
            sections.append("### Visual Description")
            sections.append(visual_description if visual_description else "(no visual description extracted)")
            sections.append("")

        output_markdown_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return output_markdown_path


def discover_pdfs(input_dir: Path, pattern: str):
    return sorted(path for path in input_dir.rglob(pattern) if path.is_file())


def output_path_for(pdf_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = pdf_path.relative_to(input_dir)
    return (output_dir / relative).with_suffix(".md")


def get_thread_converter(args, state):
    converter = getattr(state, "converter", None)
    if converter is None:
        if args.workflow == "ocr":
            converter = build_converter(args.base_url, args.api_key, args.model)
        elif args.workflow == "no-ocr":
            converter = build_text_only_converter()
        else:
            raise ValueError(f"Unsupported workflow for thread converter: {args.workflow}")
        state.converter = converter
    return converter


def convert_one_pdf(pdf_path: Path, input_dir: Path, output_dir: Path, args, state) -> Path:
    md_path = output_path_for(pdf_path, input_dir, output_dir)
    if args.workflow == "visual-description":
        extract_presentation_markdown_no_ocr(
            pdf_path=pdf_path,
            output_markdown_path=md_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            llm_workers=args.workers,
        )
        return md_path

    converter = get_thread_converter(args, state)
    result = converter.convert(str(pdf_path))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(result.text_content or "", encoding="utf-8")
    return md_path


def convert_all(args) -> int:
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

    # Pre-scan the output tree once to avoid N individual stat() RPCs on the network filesystem.
    existing_md: set = set() if args.overwrite else {p for p in output_dir.rglob("*.md") if p.is_file()}

    converted = 0
    skipped = 0
    failed = 0
    jobs = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        md_path = output_path_for(pdf_path, input_dir, output_dir)

        if md_path in existing_md:
            skipped += 1
            print(f"[{index}/{len(pdf_paths)}] skip {md_path} (exists)")
            continue

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path), strict=False)
            num_pages = len(reader.pages)
            if num_pages > args.max_pages:
                skipped += 1
                print(f"[{index}/{len(pdf_paths)}] skip {pdf_path} (has {num_pages} pages, max is {args.max_pages})")
                continue
        except Exception as error:
            print(
                f"[{index}/{len(pdf_paths)}] warning: could not check page count for {pdf_path}: {error}",
                file=sys.stderr,
            )

        jobs.append(pdf_path)

    if not jobs:
        print(f"Done. converted={converted} skipped={skipped} failed={failed}")
        return 0

    if args.workflow == "visual-description":
        print(
            f"Converting {len(jobs)} file(s) with serial file processing; LLM image-description workers={max(1, args.workers)}"
        )
        state = threading.local()
        start_time = time.perf_counter()
        completed = 0
        for completed, pdf_path in enumerate(jobs, start=1):
            try:
                md_path = convert_one_pdf(pdf_path, input_dir, output_dir, args, state)
                converted += 1
                print(f"[{completed}/{len(jobs)}] done {pdf_path} -> {md_path}")
            except Exception as error:
                failed += 1
                print(f"[{completed}/{len(jobs)}] failed {pdf_path}: {error}", file=sys.stderr)
                if args.fail_fast:
                    break
            if completed % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"Progress: {completed}/{len(jobs)} processed, average {elapsed / completed:.2f}s/file")

        if completed > 0:
            elapsed = time.perf_counter() - start_time
            print(f"Average conversion time: {elapsed / completed:.2f}s/file over {completed} file(s)")
    else:
        workers = max(1, args.workers)
        print(f"Converting {len(jobs)} file(s) with {workers} worker(s)".format(len(jobs), workers))
        state = threading.local()
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(convert_one_pdf, pdf_path, input_dir, output_dir, args, state): pdf_path
                for pdf_path in jobs
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                pdf_path = futures[future]
                error = future.exception()
                if error is None:
                    converted += 1
                    md_path = future.result()
                    print(f"[{completed}/{len(jobs)}] done {pdf_path}: {error}")
                else:
                    failed += 1
                    print(f"[{completed}/{len(jobs)}] failed {pdf_path}: {error}", file=sys.stderr)
                    if args.fail_fast:
                        for pending in futures:
                            pending.cancel()
                        break

                if completed % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    print(f"Progress: {completed}/{len(jobs)} processed, average {elapsed / completed:.2f}s/file")

        if completed > 0:
            elapsed = time.perf_counter() - start_time
            print(f"Average conversion time: {elapsed / completed:.2f}s/file over {completed} file(s)")

    print(f"Done. converted={converted} skipped={skipped} failed={failed}")
    return 1 if failed > 0 else 0


def main():
    args = parse_args()
    raise SystemExit(convert_all(args))


if __name__ == "__main__":
    main()
