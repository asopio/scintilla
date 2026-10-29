#!/usr/bin/env python3
"""Convert a PDF to scintilla markdown via DeepSeek-OCR2 on a vLLM server.

Pipeline:
  1. pdfinfo page count
  2. render each requested page to pages/page-NN.png with pdftoppm
  3. one chat completion per page image (grounding prompt, temperature 0,
     max_tokens 8180, skip_special_tokens false)
  4. write the raw grounding dump <stem>_dsocr2.md (PAGE markers + page
     notes), the same format the sample Evian1992LHC dump uses
  5. crop figure det boxes (normalized 0-999 of the page image) from the
     rendered pages -> figures/page-NN-fKK.png + figures_manifest.json
  6. dsocr2_postprocess.postprocess() -> md/<stem>_dsocr2.md

Pages are sent to the model in their native orientation; rotated pages
are not handled (no sideways-content retry).

Server contract (see vllm_deepseek_ocr2.yaml at the repo root):
  model name `deepseek-ocr2`, prompt
  `<|grounding|>Convert the document to markdown.`, one page image per
  request; the server keeps the layout tokens in the output because
  skip_special_tokens is false.  The tokenizer defines no chat template,
  so each request also passes a per-request `chat_template` (BOS +
  echo-of-content) with `add_generation_prompt: false`, and the image
  part precedes the text part.

Output layout (default: the PDF's own directory, as in
example_docs/evian/):
  <outdir>/
    pages/page-NN.png            rendered page images (cache)
    raw_pages/page-NN.txt        raw per-page model output (cache, resume)
    <stem>_dsocr2.md             raw grounding dump
    figures/page-NN-fKK.png      cropped figures
    figures_manifest.json        crop boxes (same schema as evian's)
    md/<stem>_dsocr2.md          final standard markdown

Usage:
  pixi run python pdf_to_markdown_dsocr2.py input.pdf
  pixi run python pdf_to_markdown_dsocr2.py input.pdf -o out/ --dpi 150
  pixi run python pdf_to_markdown_dsocr2.py input.pdf --start-page 0 --end-page 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from pdf_to_markdown_agentic import encode_image, get_pdf_page_count, render_pdf_page
import dsocr2_postprocess

GROUNDING_PROMPT = "<|grounding|>Convert the document to markdown."
# The dsocr2 tokenizer defines no chat template, so vLLM's chat endpoint
# rejects requests unless one is supplied.  The model expects the raw
# reference prompt with no chat framing: BOS + "<image>\n<|grounding|>...".
# vLLM renders the image part as the placeholder "<image>" in the
# (string) content, so the template just echoes the message content.
BOS = "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>"
CHAT_TEMPLATE = (BOS +
                 "{% for message in messages %}"
                 "{{ message['content'] }}{% endfor %}")
IMAGE_MARKER_RE = re.compile(r"<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>")


def page_width(total_pages: int) -> int:
    """Zero-padding width for page numbers (>= 2, matches the sample)."""
    return max(2, len(str(total_pages)))


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def ensure_page_images(
    pdf_path: str,
    pages_dir: Path,
    dpi: int,
    first_page: int,
    last_page: int,
    total_pages: int,
    npages_parallel: int,
    force: bool,
) -> dict[int, str]:
    """Render pages first..last (1-indexed) to pages/page-NN.png.

    Existing page files are reused unless ``force`` is set.  Returns
    ``{1-indexed page: png path}``.
    """
    width = page_width(total_pages)
    pages_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(pages_dir / "page")

    paths: dict[int, str] = {}
    todo: list[int] = []
    for n in range(first_page, last_page + 1):
        p = pages_dir / f"page-{n:0{width}d}.png"
        if force and p.exists():
            p.unlink()
        if p.exists():
            paths[n] = str(p)
        else:
            todo.append(n)

    if todo:
        print(f"  Rendering {len(todo)} page image(s) at {dpi} dpi "
              f"(up to {npages_parallel} parallel pdftoppm workers)")
        with ThreadPoolExecutor(max_workers=npages_parallel) as executor:
            future_to_page = {
                executor.submit(render_pdf_page, pdf_path, prefix, dpi, n): n
                for n in todo
            }
            for future in as_completed(future_to_page):
                n = future_to_page[future]
                rendered = future.result()
                canonical = pages_dir / f"page-{n:0{width}d}.png"
                if rendered != str(canonical):
                    os.replace(rendered, str(canonical))
                paths[n] = str(canonical)
                print(f"  Rendered page {n}")
    print(f"  {len(paths)} page image(s) ready in {pages_dir}")
    return paths


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_page_image(
    client: OpenAI,
    model: str,
    image_path: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Send one page image to the model and return the raw markdown text.

    The image part MUST come before the text part: the reference prompt is
    ``<image>\n<|grounding|>Convert the document to markdown.``.
    """
    b64 = encode_image(image_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": GROUNDING_PROMPT},
            ],
        }],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            "skip_special_tokens": False,
            "chat_template": CHAT_TEMPLATE,
            "add_generation_prompt": False,
        },
    )
    return (resp.choices[0].message.content or "").strip("\n")


def ocr_pages(
    page_images: dict[int, str],
    raw_dir: Path,
    width: int,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    npages_parallel: int,
    force: bool,
    continue_on_error: bool,
) -> dict[int, str]:
    """OCR the requested pages, resuming from raw_pages/page-NN.txt."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, str] = {}
    todo: list[tuple[int, str]] = []
    for n, img in sorted(page_images.items()):
        raw_file = raw_dir / f"page-{n:0{width}d}.txt"
        if raw_file.exists() and not force:
            results[n] = raw_file.read_text(encoding="utf-8")
        else:
            todo.append((n, img))

    if todo:
        print(f"  OCRed {len(todo)} page(s) via {model} "
              f"(up to {npages_parallel} parallel requests); "
              f"{len(results)} cached")
        with ThreadPoolExecutor(max_workers=npages_parallel) as executor:
            future_to_page = {
                executor.submit(ocr_page_image, client, model, img,
                                 temperature, max_tokens): n
                for n, img in todo
            }
            for future in as_completed(future_to_page):
                n = future_to_page[future]
                try:
                    text = future.result()
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    print(f"  page {n}: OCR failed ({exc}); skipping",
                          file=sys.stderr)
                    continue
                (raw_dir / f"page-{n:0{width}d}.txt").write_text(
                    text, encoding="utf-8")
                results[n] = text
                print(f"  page {n}: {len(text)} chars")
    else:
        print(f"  All {len(results)} raw page(s) cached in {raw_dir}")
    return results


# ---------------------------------------------------------------------------
# Raw grounding dump
# ---------------------------------------------------------------------------

def build_raw_dump(
    pdf_name: str,
    pages_text: dict[int, str],
    total_pages: int,
    dpi: int,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    first, last = min(pages_text), max(pages_text)
    count = len(pages_text)
    if count == total_pages:
        range_desc = f"{count} pages"
    else:
        range_desc = f"{count} of {total_pages} pages (pages {first}-{last})"

    lines = [f"# DeepSeek-OCR2 output — {pdf_name}", ""]
    lines.append(f"{range_desc}, one page image ({dpi} dpi) per request, "
                 f"grounding mode (`{GROUNDING_PROMPT}`), "
                 f"temperature {temperature:g}.")
    lines.append(f"Model: {model} via vLLM OpenAI server "
                 f"(max_model_len 8192, max_tokens {max_tokens}).")
    lines.append("")
    blank = [n for n, t in sorted(pages_text.items()) if not t.strip()]
    if blank:
        lines.append("Page notes:")
        lines.append("- Pages " + ", ".join(map(str, blank)) +
                     " are blank scans (model emitted EOS).")
        lines.append("")
    for n, text in sorted(pages_text.items()):
        lines.append(f"<!-- ==================== PAGE {n} ==================== -->")
        t = text.strip("\n")
        if t.strip():
            lines.append(t)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Figure crops
# ---------------------------------------------------------------------------

def crop_figures(
    pages_text: dict[int, str],
    page_images: dict[int, str],
    width: int,
    figures_dir: Path,
) -> list[dict]:
    """Crop each image-block det box (0-999 normalized) from the page image.

    Returns the figures manifest (same schema as
    example_docs/evian/figures_manifest.json); rotated_input is always
    false because pages are sent in their native orientation.
    """
    from PIL import Image

    figures_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for n, text in sorted(pages_text.items()):
        if not text or n not in page_images:
            continue
        img = Image.open(page_images[n])
        W, H = img.size
        k = 0
        for m in IMAGE_MARKER_RE.finditer(text):
            boxes = dsocr2_postprocess.parse_boxes(m.group(1))
            if not boxes:
                print(f"warning: page {n}: image block without det box; "
                      f"not cropped", file=sys.stderr)
                continue
            x1, y1, x2, y2 = boxes[0]
            X1 = min(W, max(0, round(x1 * W / 1000)))
            Y1 = min(H, max(0, round(y1 * H / 1000)))
            X2 = min(W, max(0, round(x2 * W / 1000)))
            Y2 = min(H, max(0, round(y2 * H / 1000)))
            if X2 <= X1 or Y2 <= Y1:
                print(f"warning: page {n}: degenerate image box {boxes[0]}; "
                      f"skipped", file=sys.stderr)
                continue
            k += 1
            fname = f"page-{n:0{width}d}-f{k:02d}.png"
            img.crop((X1, Y1, X2, Y2)).save(figures_dir / fname)
            manifest.append({
                "page": n,
                "file": f"figures/{fname}",
                "box_model": [x1, y1, x2, y2],
                "rotated_input": False,
                "crop_px": [X1, Y1, X2, Y2],
                "size": [X2 - X1, Y2 - Y1],
            })
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  pixi run python pdf_to_markdown_dsocr2.py input.pdf\n"
               "  pixi run python pdf_to_markdown_dsocr2.py input.pdf "
               "--start-page 0 --end-page 10 -o out/\n")
    p.add_argument("pdf", help="Path to the input PDF file")
    p.add_argument("-o", "--output", default=None,
                   help="Output directory (default: the PDF's directory)")
    p.add_argument("--dpi", type=int, default=150,
                   help="Page render DPI (default: 150, as in the sample)")
    p.add_argument("--start-page", type=int, default=0,
                   help="First page, 0-indexed, inclusive (default: 0)")
    p.add_argument("--end-page", type=int, default=None,
                   help="Last page, 0-indexed, exclusive (default: end of PDF)")
    p.add_argument("--npages-parallel", type=int, default=8,
                   help="Parallelism for page rendering and OCR (default: 8)")
    p.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1",
                   help="OpenAI-compatible base URL of the vLLM dsocr2 pod "
                        "(default: %(default)s)")
    p.add_argument("--model", default="deepseek-ocr2",
                   help="Served model name (default: %(default)s)")
    p.add_argument("--api-key", default="EMPTY",
                   help="API key (default: %(default)s)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: %(default)s)")
    p.add_argument("--max-tokens", type=int, default=8180,
                   help="Max tokens per page (default: %(default)s)")
    p.add_argument("--request-timeout", type=float, default=1800.0,
                   help="Per-request timeout in seconds (default: %(default)s)")
    p.add_argument("--force", action="store_true",
                   help="Re-render and re-OCR pages even if cached")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Skip pages whose OCR request fails")
    p.add_argument("--skip-figures", action="store_true",
                   help="Do not crop figures (writes an empty manifest)")
    p.add_argument("--no-postprocess", action="store_true",
                   help="Stop after writing the raw dump and figure crops")
    p.add_argument("--no-cache-dir", action="store_true",
                   help="Delete rendered page PNGs after completion")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    outdir = (Path(args.output).expanduser().resolve()
              if args.output else pdf_path.parent)

    total_pages = get_pdf_page_count(str(pdf_path))
    first_page = args.start_page + 1                      # 1-indexed
    last_page = args.end_page if args.end_page is not None else total_pages
    if first_page < 1 or last_page > total_pages or last_page < first_page:
        print(f"error: page range {args.start_page}-{args.end_page} is out of "
              f"bounds for {total_pages} pages", file=sys.stderr)
        return 1
    width = page_width(total_pages)

    print(f"Converting {pdf_path.name} ({total_pages} pages, "
          f"pages {first_page}-{last_page}) with {args.model} "
          f"at {args.llm_base_url}")

    client = OpenAI(base_url=args.llm_base_url, api_key=args.api_key,
                    timeout=args.request_timeout)
    try:
        client.models.list()
    except Exception as exc:
        print(f"error: cannot reach the vLLM server at {args.llm_base_url}: {exc}",
              file=sys.stderr)
        print("hint: start the dsocr2 pod (deploy_dsocr2_retry.sh) and make "
              "port 8000 reachable", file=sys.stderr)
        return 1

    outdir.mkdir(parents=True, exist_ok=True)

    page_images = ensure_page_images(
        str(pdf_path), outdir / "pages", args.dpi, first_page, last_page,
        total_pages, args.npages_parallel, args.force)

    pages_text = ocr_pages(
        page_images, outdir / "raw_pages", width, client, args.model,
        args.temperature, args.max_tokens, args.npages_parallel,
        args.force, args.continue_on_error)
    if not pages_text:
        print("error: no pages OCRed", file=sys.stderr)
        return 1

    dump_path = outdir / f"{pdf_path.stem}_dsocr2.md"
    dump_path.write_text(build_raw_dump(
        pdf_path.name, pages_text, total_pages, args.dpi, args.model,
        args.max_tokens, args.temperature), encoding="utf-8")
    print(f"wrote raw dump {dump_path}")

    figures_dir = outdir / "figures"
    manifest_path = outdir / "figures_manifest.json"
    if args.skip_figures:
        manifest: list[dict] = []
        print("skipping figure crops (--skip-figures)")
    else:
        manifest = crop_figures(pages_text, page_images, width, figures_dir)
        print(f"wrote {len(manifest)} figure crop(s) to {figures_dir}")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    if args.no_postprocess:
        if args.no_cache_dir:
            shutil.rmtree(outdir / "pages", ignore_errors=True)
        print("done (raw dump + figures; --no-postprocess).")
        return 0

    final_md = outdir / "md" / f"{pdf_path.stem}_dsocr2.md"
    result = dsocr2_postprocess.postprocess(
        str(dump_path), str(figures_dir), str(manifest_path), str(final_md))
    if result["problems"]:
        print("PROBLEMS:", file=sys.stderr)
        for prob in result["problems"]:
            print("  -", prob, file=sys.stderr)
        return 1
    print(f"wrote {result['output']}")
    print(f"  slides: {result['slides']}  images: {result['images']}  "
          f"tables: {result['tables']}  equations: {result['equations']}")

    if args.no_cache_dir:
        shutil.rmtree(outdir / "pages", ignore_errors=True)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
