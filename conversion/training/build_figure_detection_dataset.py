#!/usr/bin/env python3
"""Build a multi-PDF training set for train_qwen_figure_detector.py.

Scans a directory of PDFs, runs page-isolated figure detection from
list_pdf_figures.py, renders page PNGs, and writes a combined manifest with one
record per page image.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def _manifest_path(path: Path) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return str(path)


def _manifest_dir_path(path: Path) -> str:
    return _manifest_path(path)


def _manifest_file_path(path: Path) -> str:
    return _manifest_path(path)


from PIL import Image, ImageDraw

from list_pdf_figures import analyze_pdf, render_pdf_page


def find_existing_rendered_page(image_dir: Path, page_num: int) -> Path | None:
    prefix = f"page-{page_num}"
    candidates = sorted(image_dir.glob(f"{prefix}-*.png"))
    if len(candidates) == 1:
        return candidates[0]
    suffixes = {f"{page_num}.png", f"{page_num:02d}.png", f"{page_num:03d}.png", f"{page_num:04d}.png"}
    matches = [p for p in candidates if any(str(p).endswith(f"-{suffix}") for suffix in suffixes)]
    if len(matches) == 1:
        return matches[0]
    return None


def normalize_figures(page_record: dict[str, Any]) -> list[dict[str, float]]:
    figures: list[dict[str, float]] = []
    for fig in page_record.get("figures", []):
        for placement in fig.get("placements", []):
            box = placement.get("normalized_box")
            if not box:
                continue
            figures.append(
                {
                    "x1": round(float(box["x1"]), 6),
                    "y1": round(float(box["y1"]), 6),
                    "x2": round(float(box["x2"]), 6),
                    "y2": round(float(box["y2"]), 6),
                }
            )
    figures.sort(key=lambda b: (b["y1"], b["x1"], b["y2"], b["x2"]))
    deduped: list[dict[str, float]] = []
    seen = set()
    for box in figures:
        key = (box["x1"], box["y1"], box["x2"], box["y2"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


def write_bbox_overlay(image_path: Path, page_record: dict[str, Any], overlay_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for fig in page_record.get("figures", []):
        for placement in fig.get("placements", []):
            box = placement.get("normalized_box")
            if not box:
                continue
            x1 = int(round(float(box["x1"]) * width))
            y1 = int(round((1.0 - float(box["y2"])) * height))
            x2 = int(round(float(box["x2"]) * width))
            y2 = int(round((1.0 - float(box["y1"])) * height))
            draw.rectangle((x1, y1, x2, y2), outline=(0, 128, 255), width=4)
            label = f"{fig.get('name', 'fig')}:{fig.get('object', '')}"
            tx1 = max(0, x1)
            ty1 = max(0, y1 - 18)
            tx2 = min(width, tx1 + max(80, 7 * len(label)))
            ty2 = min(height, ty1 + 18)
            draw.rectangle((tx1, ty1, tx2, ty2), fill=(255, 255, 255))
            draw.text((tx1 + 2, ty1 + 2), label, fill=(0, 128, 255))
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(overlay_path)


def build_dataset(
    pdf_dir: str,
    output_dir: str,
    dpi: int,
    limit: int | None,
    write_bbox_images: bool,
    skip_existing_images: bool,
    skip_existing_bbox_images: bool,
    skip_failed_pdfs: bool,
) -> dict[str, Any]:
    pdf_root = Path(pdf_dir)
    out_root = Path(output_dir)
    images_root = out_root / "images"
    annotations_root = out_root / "annotations"
    bbox_root = out_root / "bbox_images"
    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)
    if write_bbox_images:
        bbox_root.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(pdf_root.glob("*.pdf"))
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    records: list[dict[str, Any]] = []
    skipped_pdfs: list[dict[str, str]] = []
    for pdf_index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{pdf_index}/{len(pdf_paths)}] {pdf_path.name}")
        try:
            analysis = analyze_pdf(str(pdf_path))
        except subprocess.CalledProcessError as exc:
            if not skip_failed_pdfs:
                raise
            message = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or str(exc))
            print(f"  skipping {pdf_path.name}: qpdf failed with exit code {exc.returncode}")
            skipped_pdfs.append({"pdf_name": pdf_path.name, "pdf_path": _manifest_file_path(pdf_path), "error": message.strip()})
            continue
        ann_path = annotations_root / f"{pdf_path.stem}.json"
        ann_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

        pdf_image_dir = images_root / pdf_path.stem
        pdf_image_dir.mkdir(parents=True, exist_ok=True)
        for page_record in analysis["pages"]:
            page_num = int(page_record["page"])
            image_path = None
            if skip_existing_images:
                image_path = find_existing_rendered_page(pdf_image_dir, page_num)
                if image_path is not None:
                    print(f"  reusing image for page {page_num}: {image_path.name}")
            if image_path is None:
                image_path = render_pdf_page(str(pdf_path), str(pdf_image_dir / f"page-{page_num}"), dpi, page_num)
            bbox_image_path = None
            if write_bbox_images:
                bbox_image_path = bbox_root / pdf_path.stem / image_path.name
                if skip_existing_bbox_images and bbox_image_path.exists():
                    print(f"  reusing bbox image for page {page_num}: {bbox_image_path.name}")
                else:
                    write_bbox_overlay(image_path, page_record, bbox_image_path)
            record = {
                "pdf_path": _manifest_file_path(pdf_path),
                "pdf_name": pdf_path.name,
                "page": page_num,
                "page_label": page_record.get("page_label"),
                "image_path": _manifest_file_path(image_path),
                "bbox_image_path": _manifest_file_path(bbox_image_path) if bbox_image_path is not None else None,
                "target": {"figures": normalize_figures(page_record)},
                "annotation_path": _manifest_file_path(ann_path),
            }
            records.append(record)

    manifest = {
        "pdf_dir": _manifest_dir_path(pdf_root),
        "output_dir": _manifest_dir_path(out_root),
        "dpi": dpi,
        "records": records,
        "skipped_pdfs": skipped_pdfs,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "skipped_pdfs.json").write_text(json.dumps(skipped_pdfs, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figure-detection training data from a directory of PDFs")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing input PDFs")
    parser.add_argument("--output-dir", required=True, help="Directory to write rendered images and manifest")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rendered page PNGs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of PDFs to process")
    parser.add_argument("--write-bbox-images", action="store_true", help="Write per-page validation images with detected bounding boxes")
    parser.add_argument(
        "--skip-existing-images",
        action="store_true",
        help="Reuse already-rendered page PNGs when present and only refresh annotations, overlays, and manifest",
    )
    parser.add_argument(
        "--skip-bbox",
        action="store_true",
        help="Reuse already-written bbox overlay images when present and only refresh annotations and manifest",
    )
    parser.add_argument(
        "--skip-failed-pdfs",
        action="store_true",
        help="Skip PDFs that qpdf cannot analyze and continue building the dataset",
    )
    args = parser.parse_args()

    manifest = build_dataset(
        args.pdf_dir,
        args.output_dir,
        args.dpi,
        args.limit,
        args.write_bbox_images,
        args.skip_existing_images,
        args.skip_bbox,
        args.skip_failed_pdfs,
    )
    print(f"Wrote {len(manifest['records'])} page records to {Path(args.output_dir) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
