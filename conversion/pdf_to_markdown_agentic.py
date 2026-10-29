#!/usr/bin/env python3
"""Convert PDF to Markdown using pdftoppm + vision-language model (OpenAI API)
with agentic syntax validation and correction via LangChain DeepAgents."""

import argparse
import base64
import glob
import os
import re
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import markdown
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deepagents import create_deep_agent

from list_pdf_figures import analyze_pdf

# Ensure pixi environment's bin is on PATH so subprocess calls find pixi-installed tools
_pixi_bin = str(Path(__file__).resolve().parent / ".pixi" / "envs" / "default" / "bin")
if os.path.isdir(_pixi_bin):
    import os as _os
    _current_path = os.environ.get("PATH", "")
    if _pixi_bin not in _current_path:
        os.environ["PATH"] = _pixi_bin + os.pathsep + _current_path


# ---------------------------------------------------------------------------
# PDF → image helpers
# ---------------------------------------------------------------------------

def get_pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF using pdfinfo."""
    cmd = ["pdfinfo", pdf_path]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not determine page count for {pdf_path}")



def render_pdf_page(pdf_path: str, output_prefix: str, dpi: int, page_num_1indexed: int) -> str:
    """Render a single PDF page to PNG using pdftoppm."""
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        "-f",
        str(page_num_1indexed),
        "-l",
        str(page_num_1indexed),
        pdf_path,
        output_prefix,
    ]
    subprocess.run(cmd, check=True)
    matches = sorted(glob.glob(f"{output_prefix}-*.png"))
    page_suffixes = {
        f"{page_num_1indexed}.png",
        f"{page_num_1indexed:02d}.png",
        f"{page_num_1indexed:03d}.png",
        f"{page_num_1indexed:04d}.png",
    }
    page_matches = [
        match for match in matches
        if any(match.endswith(f"-{suffix}") for suffix in page_suffixes)
    ]
    if len(page_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one rendered PNG for page {page_num_1indexed}, found {len(page_matches)}: {page_matches}"
        )
    return page_matches[0]



def convert_pdf_to_images(
    pdf_path: str,
    output_prefix: str,
    dpi: int = 300,
    start_page: int = 0,
    end_page: int = None,
    npages_parallel: int = 8,
) -> list[str]:
    """Convert PDF pages to PNG images using parallel per-page pdftoppm calls."""
    total_pages = get_pdf_page_count(pdf_path)
    page_start_1indexed = start_page + 1
    page_end_1indexed = end_page if end_page is not None else total_pages

    if page_start_1indexed < 1 or page_start_1indexed > total_pages:
        raise ValueError(f"start_page {start_page} is out of range for PDF with {total_pages} page(s)")
    if page_end_1indexed < page_start_1indexed or page_end_1indexed > total_pages:
        raise ValueError(
            f"end_page {end_page} is out of range for requested start_page {start_page} and PDF with {total_pages} page(s)"
        )

    page_numbers = list(range(page_start_1indexed, page_end_1indexed + 1))
    print(f"  Rendering {len(page_numbers)} page image(s) with up to {npages_parallel} parallel pdftoppm workers")

    rendered_images: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=npages_parallel) as executor:
        future_to_page = {
            executor.submit(render_pdf_page, pdf_path, output_prefix, dpi, page_num): page_num
            for page_num in page_numbers
        }
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            rendered_images[page_num] = future.result()
            print(f"  Rendered page {page_num}")

    images = [rendered_images[page_num] for page_num in page_numbers]
    print(f"  Generated {len(images)} page images for requested range")
    return images


def encode_image(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_bytes(data: bytes) -> str:
    """Encode raw bytes to base64 string for use in API requests."""
    return base64.b64encode(data).decode("utf-8")


# ---------------------------------------------------------------------------
# Vision-model page conversion
# ---------------------------------------------------------------------------

def page_to_markdown(
    page_image_path: str,
    llm,
    prompt: str = (
        "Convert the content of this page into clean, well-structured Markdown. "
        "Include all body text, tables, and equations. Preserve headings, lists, and formatting. "
        "For each figure, plot, chart, diagram, logo, or other non-text visual element, insert a brief comment like "
        "<!-- Figure: description --> at the correct location instead of transcribing the figure's internal text, axes, legend, or labels. "
        "Do NOT reproduce the contents of figures as plain text; use only the figure comment placeholder for them. "
        "Do NOT include any conversational text or explanations — output ONLY the Markdown content.\n\n"
        "Output the markdown inside a markdown code block:"
    ),
) -> str:
    """Send a page image to the vision model and get back Markdown."""
    b64 = encode_image(page_image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
    )
    response = llm.invoke([message])
    return response.content


def normalize_markdown_response(md_text: str) -> str:
    """Strip wrapper whitespace and markdown code fences from model output."""
    md_text = md_text.strip()
    if md_text.startswith("```") and md_text.endswith("```"):
        lines = md_text.splitlines()
        md_text = "\n".join(lines[1:-1]).strip()
    return md_text


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


# ---------------------------------------------------------------------------
# Markdown validation
# ---------------------------------------------------------------------------

def _validate_markdown_text(text: str) -> list[str]:
    """Validate markdown syntax by attempting to convert it to HTML.

    Returns a list of error strings; empty list means the markdown is valid.
    """
    errors: list[str] = []

    md = markdown.Markdown(
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


# ---------------------------------------------------------------------------
# Markdown validation tool (for the review agent)
# ---------------------------------------------------------------------------

@tool
def validate_markdown(file_path: str) -> str:
    """Validate markdown syntax in a file using the Python markdown library.

    Returns the full validation result (errors or success message).
    """
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"ERROR reading file: {exc}"

    errors = _validate_markdown_text(text)

    if errors:
        return "VALIDATION ERRORS:\n" + "\n".join(errors)
    return "VALIDATION PASSED: file is valid markdown."


def create_review_agent(llm, system_prompt: str):
    """Create a DeepAgents reviewer with three tools: read_file, edit_file, validate_markdown."""
    review_model = ChatOpenAI(
        model=llm.model,
        api_key=str(llm.openai_api_key) if hasattr(llm, "openai_api_key") else llm.api_key,
        base_url=llm.openai_api_base if hasattr(llm, "openai_api_base") else llm.base_url,
        use_responses_api=False,
    )

    agent = create_deep_agent(
        model=review_model,
        tools=[validate_markdown],
        system_prompt=system_prompt,
    )
    return agent


def review_and_fix_md(file_path: str, llm, max_iterations: int = 10) -> str:
    """Agentic review loop: validate markdown, fix errors until clean."""
    abs_path = str(Path(file_path).resolve())
    file_name = Path(abs_path).name

    print(f"\n=== Agentic Markdown Review ({file_name}) ===")

    agent = create_review_agent(
        llm=llm,
        system_prompt=(
            "You are a Markdown syntax auditor and fixer. Your job is to read a "
            "Markdown file, validate its syntax, fix any issues you find, and "
            "re-validate until the file passes.\n\n"
            "You have three tools available:\n"
            "- read_file: read the current contents of a file\n"
            "- edit_file: make targeted fixes to a file\n"
            "- validate_markdown: run the Python markdown parser on a file and get "
            "structured error output (or a PASS message)\n\n"
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
        ),
    )

    task_desc = (
        f"Review and fix markdown syntax in the file '{abs_path}'.\n\n"
        "Use your validate_markdown tool to check the file, then read_file to see "
        "the content, and edit_file to fix any errors. Re-validate after each fix "
        "cycle. Continue until validate_markdown returns 'VALIDATION PASSED' or you "
        "determine no more fixes are possible."
    )

    output = None
    for iteration in range(1, max_iterations + 1):
        print(f"  Review iteration {iteration}/{max_iterations}...")
        try:
            result = agent.invoke({"messages": [HumanMessage(content=task_desc)]})
            output = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            print(f"  Agent error on iteration {iteration}: {exc}")
            break

        if output is None:
            print(f"  No output from agent on iteration {iteration}.")
            break

        if "VALIDATION PASSED" in output:
            print(f"  Markdown validation passed on iteration {iteration}.")
            return output

    print("  Review completed (max iterations reached).")
    return output


# ---------------------------------------------------------------------------
# Figure extraction helpers
# ---------------------------------------------------------------------------

def write_figure_debug(output_dir: Path, page_num: int, name: str, content: str) -> None:
    """Write per-page figure-debug text artifacts."""
    debug_dir = output_dir / "figure_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"page_{page_num:03d}_{name}.txt"
    debug_path.write_text(content, encoding="utf-8")


def scale_bbox_to_pixels(bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
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


def normalize_bbox_to_image_space(bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return top-left and bottom-left interpretations of a reported bbox."""
    x1, y1, x2, y2 = scale_bbox_to_pixels(bbox, image_size)
    width, height = image_size
    top_left = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    bottom_left = (max(0, x1), max(0, height - y2), min(width, x2), min(height, height - y1))
    return top_left, bottom_left


def clamp_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Clamp bbox to image bounds."""
    x1, y1, x2, y2 = bbox
    width, height = image_size
    return (
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )



def choose_best_bbox(img, bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], dict]:
    """Choose between top-left and bottom-left bbox interpretations."""
    import PIL.ImageStat as ImageStat

    top_left, bottom_left = normalize_bbox_to_image_space(bbox, img.size)

    def _score(candidate):
        cx1, cy1, cx2, cy2 = candidate
        if cx2 <= cx1 or cy2 <= cy1:
            return -1.0
        crop = img.crop(candidate).convert('L')
        stat = ImageStat.Stat(crop)
        mean = stat.mean[0] if stat.mean else 255.0
        return 255.0 - mean

    top_score = _score(top_left)
    bottom_score = _score(bottom_left)
    chosen = bottom_left if bottom_score > top_score else top_left
    return chosen, {
        'top_left': top_left,
        'bottom_left': bottom_left,
        'top_score': top_score,
        'bottom_score': bottom_score,
        'chosen': chosen,
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



def normalize_review_bbox(raw_bbox, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Normalize a review-suggested bbox into pixel coordinates."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw_bbox)
    except (TypeError, ValueError):
        return None
    scaled = scale_bbox_to_pixels((x1, y1, x2, y2), image_size)
    return clamp_bbox(scaled, image_size)



def review_figure_crop(llm, page_b64: str, crop_b64: str, description: str) -> tuple[dict | None, str]:
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
        response = llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{page_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{crop_b64}"},
                },
            ])
        ])
        response_text = str(response.content)
    except Exception as exc:
        return None, f"ERROR: {exc}"
    return parse_json_object(response_text), response_text


def detect_figure_candidates_vlm(
    page_image_path: str,
    output_dir: Path,
    llm,
    figure_markers: list[dict],
    page_num: int,
) -> list[dict]:
    """Locate described figures on the page using markdown-derived descriptions and a VLM."""
    if not figure_markers:
        return []

    b64 = encode_image(page_image_path)
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
    response = llm.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ]
        )
    ])

    write_figure_debug(
        output_dir,
        page_num,
        "coordinate_extraction_vlm",
        "PROMPT:\n" + prompt + "\n\nRESPONSE:\n" + str(response.content),
    )

    by_index = {fig["marker_index"]: fig for fig in figure_markers}
    grouped_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    grouped_desc: dict[int, str] = {}
    for raw_line in response.content.splitlines():
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
            grouped_desc[marker_index] = desc_part.replace("DESC:", "").strip() if desc_part else original["description"]
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
        output_dir,
        page_num,
        "coordinate_candidates_vlm",
        "\n".join(str(candidate) for candidate in candidates) or "NO_CANDIDATES",
    )
    return candidates


def normalized_pdf_box_to_pixel_bbox(box: dict, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Convert qpdf normalized boxes (bottom-left origin) to image pixel bbox (top-left origin)."""
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
    """Analyze the PDF once with qpdf and return normalized figure placements by 1-indexed page."""
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



def detect_figure_candidates_heuristic(
    page_image_path: str,
    output_dir: Path,
    figure_markers: list[dict],
    page_num: int,
    heuristic_page_map: dict[int, list[dict]] | None,
) -> list[dict]:
    """Pair qpdf-derived figure boxes with markdown figure markers in reading order."""
    if not figure_markers or heuristic_page_map is None:
        return []

    from PIL import Image

    image = Image.open(page_image_path)
    image_size = image.size
    raw_boxes = heuristic_page_map.get(page_num, [])
    pixel_boxes: list[dict] = []
    for raw_box in raw_boxes:
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
        f"mode=heuristic page={page_num} markers={len(figure_markers)} boxes={len(pixel_boxes)} paired={pair_count}"
    ]

    for idx in range(pair_count):
        marker = figure_markers[idx]
        box_info = pixel_boxes[idx]
        pairing_logs.append(
            f"marker_index={marker['marker_index']} desc={marker['description']!r} -> bbox={box_info['bbox']} normalized={box_info['normalized_box']} object={box_info['object']} name={box_info['name']} reasons={box_info['figure_reasons']}"
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

    if len(figure_markers) > pair_count:
        for marker in figure_markers[pair_count:]:
            pairing_logs.append(
                f"marker_index={marker['marker_index']} desc={marker['description']!r} -> NO_HEURISTIC_BOX"
            )
    if len(pixel_boxes) > pair_count:
        for box_info in pixel_boxes[pair_count:]:
            pairing_logs.append(
                f"UNUSED_HEURISTIC_BOX bbox={box_info['bbox']} normalized={box_info['normalized_box']} object={box_info['object']} name={box_info['name']} reasons={box_info['figure_reasons']}"
            )

    write_figure_debug(
        output_dir,
        page_num,
        "coordinate_candidates_heuristic",
        "\n".join(pairing_logs),
    )
    return candidates



def detect_figure_candidates(
    page_image_path: str,
    output_dir: Path,
    llm,
    figure_markers: list[dict],
    page_num: int,
    figure_extraction_mode: str,
    heuristic_page_map: dict[int, list[dict]] | None,
) -> list[dict]:
    """Dispatch figure candidate detection to the requested backend."""
    if figure_extraction_mode == "vlm":
        return detect_figure_candidates_vlm(page_image_path, output_dir, llm, figure_markers, page_num)
    if figure_extraction_mode == "heuristic":
        return detect_figure_candidates_heuristic(page_image_path, output_dir, figure_markers, page_num, heuristic_page_map)
    raise ValueError(f"Unsupported figure extraction mode: {figure_extraction_mode}")


def validate_and_extract_figures(
    img_path: str,
    output_dir: Path,
    llm,
    page_num: int,
    candidates: list[dict],
) -> list[dict]:
    """Validate candidate figure crops, retry once on bad reviews, and save figures."""
    from io import BytesIO
    import PIL.Image as Image

    if not candidates:
        return []

    img_out_dir = output_dir / "figures"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path)
    page_b64 = encode_image(img_path)
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

            buf = BytesIO()
            crop.save(buf, format="PNG")
            crop_b64 = encode_image_bytes(buf.getvalue())

            review_data, review_raw = review_figure_crop(
                llm=llm,
                page_b64=page_b64,
                crop_b64=crop_b64,
                description=candidate["description"],
            )

            attempt_record = {
                "attempt": attempt_num,
                "bbox": current_bbox,
                "crop_size": crop.size,
                "review": review_data,
                "review_raw": review_raw,
            }
            attempts.append(attempt_record)

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

        fig_name = f"fig_{page_num}_{candidate['marker_index']}.png"
        fig_path = img_out_dir / fig_name
        selected_crop.save(str(fig_path))
        figure_record = {
            "page_num": page_num,
            "marker_index": candidate["marker_index"],
            "description": candidate["description"],
            "reported_description": candidate.get("reported_description", candidate["description"]),
            "saved_path": str(fig_path.relative_to(output_dir)),
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
        output_dir,
        page_num,
        "validation",
        "\n\n".join(validation_logs) if validation_logs else "NO_VALIDATION_LOGS",
    )
    return figures


def replace_figure_markers(md_text: str, figures: list[dict], output_dir: Path, page_num: int) -> str:
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
            replacement_logs.append(
                f"marker_index={current_index} desc={description!r} -> NO_MATCH"
            )
            return match.group(0)
        replacement_logs.append(
            f"marker_index={current_index} desc={description!r} -> {fig['saved_path']} bbox={fig['bbox']}"
        )
        return f"![{description}]({fig['saved_path']})"

    replaced = re.sub(r"<!--\s*Figure:\s*(.*?)\s*-->", _replace, md_text)
    write_figure_debug(
        output_dir,
        page_num,
        "replacement",
        "\n".join(replacement_logs) if replacement_logs else "NO_REPLACEMENTS",
    )
    return replaced


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def create_parser():
    """Create the shared argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown via vision model, with agentic syntax validation."
    )
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: same dir as PDF)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for page images (default: 300)")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8",
                        help="OpenAI-compatible model name")
    parser.add_argument("--api-key", default="EMPTY",
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8008/v1",
                        help="OpenAI API base URL")
    parser.add_argument("--start-page", type=int, default=0,
                        help="0-indexed page to start from (default: 0)")
    parser.add_argument("--end-page", type=int, default=None,
                        help="0-indexed page to stop before (default: all pages)")
    parser.add_argument("--continue-on-error", action="store_true", default=True,
                        help="Continue on API errors (default: True)")
    parser.add_argument("--no-cache-dir", action="store_true",
                        help="Skip saving intermediate Markdown files")
    parser.add_argument("--max-review-iterations", type=int, default=5,
                        help="Max agentic review→fix cycles per file (default: 5)")
    parser.add_argument("--skip-review", action="store_true",
                        help="Skip agentic markdown review step")
    parser.add_argument("--skip-figure-extraction", action="store_true",
                        help="Skip figure detection and extraction")
    parser.add_argument("--figure-extraction-mode", choices=["vlm", "heuristic"], default="vlm",
                        help="Figure detection backend to use when extraction is enabled (default: vlm)")
    parser.add_argument("--npages-parallel", type=int, default=8,
                        help="Number of pages to process in parallel per batch (default: 8)")
    return parser


def load_existing_pages(output_md: Path, pdf_path: Path) -> tuple[str, list[str]]:
    """Load existing markdown file and recover page blocks for resume support."""
    existing_pages: list[str] = []
    header = "# " + pdf_path.name
    if output_md.exists() and output_md.stat().st_size > 0:
        print(f"\n  Found existing output file ({output_md.stat().st_size} bytes). Preserving previous pages.")
        content = output_md.read_text(encoding="utf-8")
        header_line = content.splitlines()[0] if content.strip() else header
        prefix = header_line + "\n\n"
        if content.startswith(prefix):
            body = content[len(prefix):]
        else:
            body = content[len(header_line):]
        page_blocks = re.split(r'^---$', body, flags=re.MULTILINE)
        existing_pages = [block.strip() for block in page_blocks if block.strip()]
        print(f"  Preserved {len(existing_pages)} existing page block(s).")
    return header, existing_pages


def write_output_markdown(output_md: Path, header: str, pages: list[str]) -> None:
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        for page in pages:
            f.write(page)
            f.write("\n\n---\n\n")


def create_llm_client(args):
    """Create a fresh LLM client instance."""
    return ChatOpenAI(
        model=args.model,
        api_key=args.api_key,
        base_url=args.llm_base_url,
    )



def process_page(
    args,
    img_path: str,
    out_dir: Path,
    page_num: int,
    heuristic_page_map: dict[int, list[dict]] | None = None,
) -> str:
    """Run the full markdown-first pipeline for a single page."""
    llm = create_llm_client(args)
    md_text = normalize_markdown_response(page_to_markdown(img_path, llm))

    if args.skip_figure_extraction:
        return md_text

    figure_markers = extract_figure_descriptions_from_markdown(md_text)
    if not figure_markers:
        return md_text

    write_figure_debug(
        out_dir,
        page_num,
        "markers",
        "mode=" + args.figure_extraction_mode + "\n" + "\n".join(str(marker) for marker in figure_markers),
    )

    candidates = detect_figure_candidates(
        img_path,
        out_dir,
        llm,
        figure_markers,
        page_num,
        args.figure_extraction_mode,
        heuristic_page_map,
    )
    extracted_figures = validate_and_extract_figures(
        img_path=img_path,
        output_dir=out_dir,
        llm=llm,
        page_num=page_num,
        candidates=candidates,
    )
    if not extracted_figures:
        return md_text

    return replace_figure_markers(md_text, extracted_figures, out_dir, page_num)


def run_pipeline(args, review_llm, image_paths, image_prefix, output_md, pdf_path, heuristic_page_map=None):
    """Core pipeline logic using batched parallel page processing."""
    print(f"\n=== Step 3: Converting pages to Markdown ({args.npages_parallel} pages in parallel) ===\n")

    header, existing_pages = load_existing_pages(output_md, pdf_path)
    full_md_lines = existing_pages[:]
    out_dir = output_md.parent

    start_index = len(existing_pages)
    remaining = list(enumerate(image_paths[start_index:], start=start_index))

    for batch_start in range(0, len(remaining), args.npages_parallel):
        batch = remaining[batch_start:batch_start + args.npages_parallel]
        if not batch:
            continue
        batch_page_numbers = [idx + args.start_page + 1 for idx, _ in batch]
        print(f"Processing batch pages {batch_page_numbers[0]}-{batch_page_numbers[-1]}...")

        batch_results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=args.npages_parallel) as executor:
            future_to_page = {
                executor.submit(
                    process_page,
                    args,
                    img_path,
                    out_dir,
                    idx + args.start_page + 1,
                    heuristic_page_map,
                ): (idx, img_path)
                for idx, img_path in batch
            }

            for future in as_completed(future_to_page):
                idx, _img_path = future_to_page[future]
                page_num = idx + args.start_page + 1
                print(f"[Page {page_num}]", end=" ", flush=True)
                try:
                    batch_results[idx] = future.result()
                    print("OK")
                except Exception as e:
                    print(f"ERROR: {e}")
                    if not args.continue_on_error:
                        print("\nAborting due to error.", file=sys.stderr)
                        sys.exit(1)

        for idx, _img_path in batch:
            if idx in batch_results:
                full_md_lines.append(batch_results[idx])

        write_output_markdown(output_md, header, full_md_lines)

    final_path = output_md
    write_output_markdown(final_path, header, full_md_lines)
    print(f"\n=== Step 3 complete. Output: {final_path} ===")

    if not args.skip_review:
        print("\n=== Step 4: Agentic Markdown Review ===")
        review_result = review_and_fix_md(
            str(final_path), review_llm, max_iterations=args.max_review_iterations
        )
        print(f"\n  Review result: {review_result}")
    else:
        print("\n=== Step 4: Skipped (--skip-review) ===")

    if args.no_cache_dir:
        print("Cleaning up intermediate images...")
        for img in glob.glob(image_prefix + "*.png"):
            os.remove(img)
        print("Done.")

    print(f"\n=== Done! Output: {final_path} ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = create_parser()
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else pdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    image_prefix = str(out_dir / f"{stem}_page_")

    heuristic_page_map = None
    if not args.skip_figure_extraction and args.figure_extraction_mode == "heuristic":
        print("=== Step 0: Precomputing heuristic figure boxes with qpdf ===")
        heuristic_page_map = build_heuristic_page_map(str(pdf_path))
        total_boxes = sum(len(items) for items in heuristic_page_map.values())
        print(f"  Heuristic analysis found {total_boxes} figure-like placement(s) across {len(heuristic_page_map)} page(s)")

    print(f"=== Step 1: Converting {pdf_path.name} to PNG images (dpi={args.dpi}, up to {args.npages_parallel} parallel workers) ===")
    image_paths = convert_pdf_to_images(
        str(pdf_path),
        image_prefix,
        args.dpi,
        start_page=args.start_page,
        end_page=args.end_page,
        npages_parallel=args.npages_parallel,
    )
    print(f"  Processing pages {args.start_page}-{args.start_page + len(image_paths) - 1} ({len(image_paths)} pages)")

    print(f"\n=== Step 2: Initializing LLM ({args.model}) ===")
    review_llm = create_llm_client(args)

    output_md = out_dir / f"{pdf_path.stem}.md"
    run_pipeline(args, review_llm, image_paths, image_prefix, output_md, pdf_path, heuristic_page_map=heuristic_page_map)


if __name__ == "__main__":
    main()
