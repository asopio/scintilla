#!/usr/bin/env python3
"""List figure-like PDF XObjects using per-page qpdf extraction.

This version avoids cross-page object confusion by extracting each page into a
single-page PDF with qpdf, then inspecting that isolated page's QDF/JSON
structure. It outputs normalized page coordinates x1, y1, x2, y2 in [0,1].
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

_pixi_bin = str(Path(__file__).resolve().parent / ".pixi" / "envs" / "default" / "bin")
if os.path.isdir(_pixi_bin):
    _current_path = os.environ.get("PATH", "")
    if _pixi_bin not in _current_path:
        os.environ["PATH"] = _pixi_bin + os.pathsep + _current_path

OBJ_RE = re.compile(r"(\d+) (\d+) obj\b(.*?)endobj", re.S)
REF_RE = re.compile(r"/(\S+)\s+(\d+)\s+(\d+)\s+R")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class Placement:
    name: str
    bbox: list[float] | None
    ctm: list[float] | None
    effective_ctm: list[float] | None
    page_rect: list[float] | None
    normalized_box: dict[str, float] | None
    source_stream_object: str


def run_qpdf_json(pdf_path: str) -> dict[str, Any]:
    result = subprocess.run(["qpdf", "--json", pdf_path], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def run_qpdf_qdf(pdf_path: str) -> str:
    result = subprocess.run(
        ["qpdf", "--qdf", "--object-streams=disable", pdf_path, "-"], check=True, capture_output=True
    )
    return result.stdout.decode("latin1", errors="ignore")


def parse_objects(qdf_text: str) -> dict[str, str]:
    return {f"{m.group(1)} {m.group(2)}": m.group(3) for m in OBJ_RE.finditer(qdf_text)}


def parse_number_array(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    nums = [float(x) for x in NUMBER_RE.findall(raw)]
    return nums if nums else None


def parse_matrix(obj_body: str) -> list[float] | None:
    m = re.search(r"/Matrix\s*\[(.*?)\]", obj_body, re.S)
    nums = parse_number_array(m.group(1) if m else None)
    return nums if nums and len(nums) == 6 else None


def parse_bbox(obj_body: str) -> list[float] | None:
    m = re.search(r"/BBox\s*\[(.*?)\]", obj_body, re.S)
    nums = parse_number_array(m.group(1) if m else None)
    return nums if nums and len(nums) == 4 else None


def parse_media_box(obj_body: str, objects: dict[str, str] | None = None) -> list[float] | None:
    m = re.search(r"/MediaBox\s*\[(.*?)\]", obj_body, re.S)
    nums = parse_number_array(m.group(1) if m else None)
    if nums and len(nums) == 4:
        return nums
    if objects is not None:
        ref = re.search(r"/MediaBox\s+(\d+)\s+(\d+)\s+R", obj_body)
        if ref:
            ref_oid = f"{ref.group(1)} {ref.group(2)}"
            ref_body = objects.get(ref_oid, "")
            ref_nums = parse_number_array(ref_body)
            if ref_nums and len(ref_nums) == 4:
                return ref_nums
    return None


def parse_subtype(obj_body: str) -> str | None:
    m = re.search(r"/Subtype\s+(/\S+)", obj_body)
    return m.group(1) if m else None


def parse_resource_xobjects(resource_body: str) -> dict[str, str]:
    m = re.search(r"/XObject\s*<<(.+?)>>", resource_body, re.S)
    if not m:
        return {}
    return {name: f"{a} {b}" for name, a, b in REF_RE.findall(m.group(1))}


def extract_stream(obj_body: str) -> str:
    m = re.search(r"stream\r?\n(.*?)\r?\nendstream", obj_body, re.S)
    return m.group(1) if m else ""


def multiply_matrices(m1: list[float], m2: list[float]) -> list[float]:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return [
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    ]


def apply_matrix(m: list[float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def bbox_to_page_rect(bbox: list[float], ctm: list[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    pts = [apply_matrix(ctm, x0, y0), apply_matrix(ctm, x0, y1), apply_matrix(ctm, x1, y0), apply_matrix(ctm, x1, y1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def normalize_rect(rect: tuple[float, float, float, float], media_box: list[float]) -> dict[str, float]:
    mx0, my0, mx1, my1 = media_box
    pw = mx1 - mx0
    ph = my1 - my0
    x0, y0, x1, y1 = rect

    def clip(v: float) -> float:
        return max(0.0, min(1.0, v))

    return {
        "x1": clip((x0 - mx0) / pw) if pw else 0.0,
        "y1": clip((y0 - my0) / ph) if ph else 0.0,
        "x2": clip((x1 - mx0) / pw) if pw else 0.0,
        "y2": clip((y1 - my0) / ph) if ph else 0.0,
    }


def classify_figure_like(subtype: str | None, obj_body: str, bbox: list[float] | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if subtype == "/Image":
        reasons.append("image_xobject")
        return True, reasons
    if subtype != "/Form":
        return False, reasons
    if bbox:
        x0, y0, x1, y1 = bbox
        if abs(x1 - x0) > 10 and abs(y1 - y0) > 10:
            reasons.append("nontrivial_bbox")
    if "/PTEX.FileName" in obj_body:
        reasons.append("imported_pdf")
    stream = extract_stream(obj_body)
    op_counts = {
        "BT": len(re.findall(r"(?<!\S)BT(?!\S)", stream)),
        "m": len(re.findall(r"(?<!\S)m(?!\S)", stream)),
        "l": len(re.findall(r"(?<!\S)l(?!\S)", stream)),
        "c": len(re.findall(r"(?<!\S)c(?!\S)", stream)),
        "re": len(re.findall(r"(?<!\S)re(?!\S)", stream)),
        "S": len(re.findall(r"(?<!\S)S(?!\S)", stream)),
        "f": len(re.findall(r"(?<!\S)f(?!\S)", stream)),
        "Do": len(re.findall(r"(?<!\S)/\S+\s+Do(?!\S)", stream)),
    }
    if op_counts["BT"] > 0:
        reasons.append("contains_text")
    if sum(op_counts[k] for k in ["m", "l", "c", "re", "S", "f"]) >= 10:
        reasons.append("vector_ops")
    if op_counts["Do"] > 0:
        reasons.append("nested_xobject")
    if "/Resources" in obj_body:
        reasons.append("has_resources")
    return bool(reasons), reasons


def parse_placements(stream_text: str, xobjects: dict[str, str]) -> list[tuple[str, list[float]]]:
    tokens = re.findall(r"/\S+|\[[^\]]*\]|\([^)]*\)|\S+", stream_text)
    ctm_stack: list[list[float]] = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    placements: list[tuple[str, list[float]]] = []
    operands: list[str] = []

    def current() -> list[float]:
        return ctm_stack[-1][:]

    for tok in tokens:
        if tok == "q":
            ctm_stack.append(current())
            operands.clear()
            continue
        if tok == "Q":
            if len(ctm_stack) > 1:
                ctm_stack.pop()
            operands.clear()
            continue
        if tok == "cm":
            if len(operands) >= 6:
                vals = [float(x) for x in operands[-6:]]
                ctm_stack[-1] = multiply_matrices(current(), vals)
            operands.clear()
            continue
        if tok == "Do":
            if operands:
                name = operands[-1]
                if name.startswith("/"):
                    xname = name[1:]
                    if xname in xobjects:
                        placements.append((xname, current()))
            operands.clear()
            continue
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", tok) or tok.startswith("/"):
            operands.append(tok)
            continue
        operands.clear()
    return placements


def render_pdf_page(pdf_path: str, output_prefix: str, dpi: int, page_num_1indexed: int) -> Path:
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", str(page_num_1indexed), "-l", str(page_num_1indexed), pdf_path, output_prefix],
        check=True,
        capture_output=True,
    )
    prefix_path = Path(output_prefix)
    candidates = sorted(prefix_path.parent.glob(f"{prefix_path.name}-*.png"))
    suffixes = {f"{page_num_1indexed}.png", f"{page_num_1indexed:02d}.png", f"{page_num_1indexed:03d}.png", f"{page_num_1indexed:04d}.png"}
    matches = [p for p in candidates if any(str(p).endswith(f"-{suffix}") for suffix in suffixes)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rendered PNG for page {page_num_1indexed}, found {len(matches)}")
    return matches[0]


def extract_single_page_pdf(src_pdf: str, page_num_1indexed: int, output_pdf: str) -> None:
    subprocess.run(
        ["qpdf", src_pdf, "--pages", src_pdf, str(page_num_1indexed), "--", output_pdf],
        check=True,
        capture_output=True,
    )


def analyze_single_page_pdf(page_pdf_path: str, physical_page_num: int, page_label: Any, debug: bool) -> dict[str, Any]:
    qjson = run_qpdf_json(page_pdf_path)
    qdf = run_qpdf_qdf(page_pdf_path)
    objects = parse_objects(qdf)

    page_oid = next(
        oid
        for oid, body in objects.items()
        if re.search(r"/Type\s+/Page\b", body) and not re.search(r"/Type\s+/Pages\b", body)
    )
    page_body = objects[page_oid]
    media_box = parse_media_box(page_body, objects)
    if not media_box:
        raise RuntimeError(f"Could not determine MediaBox for extracted page {page_pdf_path}")

    contents: list[str] = []
    contents_array = re.search(r"/Contents\s*\[(.*?)\]", page_body, re.S)
    if contents_array:
        contents.extend(f"{a} {b}" for a, b in re.findall(r"(\d+)\s+(\d+)\s+R", contents_array.group(1)))
    else:
        contents_single = re.search(r"/Contents\s+(\d+)\s+(\d+)\s+R", page_body)
        if contents_single:
            contents.append(f"{contents_single.group(1)} {contents_single.group(2)}")

    resource_xobjects: dict[str, str] = parse_resource_xobjects(page_body)
    resref = re.search(r"/Resources\s+(\d+)\s+(\d+)\s+R", page_body)
    if not resource_xobjects and resref:
        resource_oid = f"{resref.group(1)} {resref.group(2)}"
        resource_xobjects = parse_resource_xobjects(objects.get(resource_oid, ""))

    placements_by_name: dict[str, list[Placement]] = {}
    debug_streams: list[dict[str, Any]] = []
    for content_oid in contents:
        stream_text = extract_stream(objects.get(content_oid, ""))
        if debug:
            debug_streams.append({"content_object": content_oid, "do_names": re.findall(r"/(\S+)\s+Do\b", stream_text)})
        for xname, ctm in parse_placements(stream_text, resource_xobjects):
            xoid = resource_xobjects[xname]
            obj_body = objects.get(xoid, "")
            subtype = parse_subtype(obj_body)
            bbox = parse_bbox(obj_body)
            if bbox is None and subtype == "/Image":
                bbox = [0.0, 0.0, 1.0, 1.0]
            effective_ctm = None
            page_rect = None
            normalized_box = None
            if bbox is not None:
                xobject_matrix = parse_matrix(obj_body) or [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                effective_ctm = multiply_matrices(ctm, xobject_matrix)
                rect = bbox_to_page_rect(bbox, effective_ctm)
                page_rect = [rect[0], rect[1], rect[2], rect[3]]
                normalized_box = normalize_rect(rect, media_box)
            placements_by_name.setdefault(xname, []).append(
                Placement(
                    name=xname,
                    bbox=bbox,
                    ctm=ctm,
                    effective_ctm=effective_ctm,
                    page_rect=page_rect,
                    normalized_box=normalized_box,
                    source_stream_object=content_oid,
                )
            )

    page_items: list[dict[str, Any]] = []
    for xname, xoid in resource_xobjects.items():
        obj_body = objects.get(xoid, "")
        subtype = parse_subtype(obj_body)
        bbox = parse_bbox(obj_body)
        matrix = parse_matrix(obj_body)
        figure_like, reasons = classify_figure_like(subtype, obj_body, bbox)
        nested_xobjects = parse_resource_xobjects(obj_body)
        item: dict[str, Any] = {
            "name": xname,
            "object": xoid,
            "subtype": subtype,
            "figure_like": figure_like,
            "figure_reasons": reasons,
            "bbox": bbox,
            "matrix": matrix,
            "placements": [
                {
                    "source_stream_object": p.source_stream_object,
                    "ctm": p.ctm,
                    "effective_ctm": p.effective_ctm,
                    "page_rect": p.page_rect,
                    "normalized_box": p.normalized_box,
                }
                for p in placements_by_name.get(xname, [])
            ],
        }
        if subtype == "/Image":
            for key, rx in {
                "width": r"/Width\s+(\d+)",
                "height": r"/Height\s+(\d+)",
                "bits_per_component": r"/BitsPerComponent\s+(\d+)",
                "color_space": r"/ColorSpace\s+(/\S+|\[[^\]]+\])",
                "filter": r"/Filter\s+(/\S+|\[[^\]]+\])",
            }.items():
                m = re.search(rx, obj_body, re.S)
                item[key] = m.group(1).strip() if m else None
        if nested_xobjects:
            item["nested_xobjects"] = [
                {"name": k, "object": v, "subtype": parse_subtype(objects.get(v, ""))}
                for k, v in nested_xobjects.items()
            ]
        if figure_like:
            page_items.append(item)

    page_record = {
        "page": physical_page_num,
        "page_label": page_label,
        "pageposfrom1": physical_page_num,
        "page_object": page_oid,
        "media_box": media_box,
        "figures": page_items,
    }
    if debug:
        page_record["debug"] = {"content_objects": contents, "resource_xobjects": resource_xobjects, "streams": debug_streams}
    return page_record


def analyze_pdf(pdf_path: str, debug_page: int | None = None) -> dict[str, Any]:
    qjson = run_qpdf_json(pdf_path)
    pages_out: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for page_index, page in enumerate(qjson["pages"], start=1):
            if debug_page is not None and page_index != debug_page:
                continue
            single_page_pdf = str(Path(tmpdir) / f"page-{page_index:04d}.pdf")
            extract_single_page_pdf(pdf_path, page_index, single_page_pdf)
            pages_out.append(analyze_single_page_pdf(single_page_pdf, page_index, page.get("label"), debug_page is not None))
    return {"pdf": str(Path(pdf_path)), "pages": pages_out}


def draw_boxes_on_pages(pdf_path: str, result: dict[str, Any], output_dir: str, dpi: int) -> None:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for --render-boxes-dir but is not installed in the current environment")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    with tempfile.TemporaryDirectory() as tmpdir:
        for page in result["pages"]:
            figures = [f for f in page["figures"] if f.get("placements")]
            if not figures:
                continue
            page_num = page["page"]
            page_label = page.get("page_label")
            label_suffix = ""
            if isinstance(page_label, dict) and page_label.get("/St") is not None:
                label_suffix = f"-label-{page_label['/St']}"
            rendered = render_pdf_page(pdf_path, str(Path(tmpdir) / f"page-{page_num}"), dpi, page_num)
            image = Image.open(rendered).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for figure in figures:
                for placement in figure["placements"]:
                    box = placement.get("normalized_box")
                    if not box:
                        continue
                    x1 = int(round(box["x1"] * width))
                    y1 = int(round((1.0 - box["y2"]) * height))
                    x2 = int(round(box["x2"] * width))
                    y2 = int(round((1.0 - box["y1"]) * height))
                    draw.rectangle((x1, y1, x2, y2), outline=(0, 128, 255), width=4)
                    label = f"{figure['name']}:{figure['object']}"
                    tx1 = max(0, x1)
                    ty1 = max(0, y1 - 18)
                    tx2 = min(width, tx1 + max(80, 7 * len(label)))
                    ty2 = min(height, ty1 + 18)
                    draw.rectangle((tx1, ty1, tx2, ty2), fill=(255, 255, 255))
                    draw.text((tx1 + 2, ty1 + 2), label, fill=(0, 128, 255))
            out_path = outdir / f"{stem}-page-{page_num:03d}{label_suffix}-figures.png"
            image.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="List figure-like PDF XObjects and page-relative boxes")
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("-o", "--output", help="Write JSON output to file")
    parser.add_argument("--render-boxes-dir", help="Render PDF pages to PNG and draw detected figure boxes into this directory")
    parser.add_argument("--dpi", type=int, default=150, help="DPI to use for optional page rendering")
    parser.add_argument("--debug-page", type=int, help="Analyze only one 1-indexed page and include detailed geometry/debug metadata")
    args = parser.parse_args()

    result = analyze(args.pdf, debug_page=args.debug_page)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.render_boxes_dir:
        draw_boxes_on_pages(args.pdf, result, args.render_boxes_dir, args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
