#!/usr/bin/env python3
"""Build a Vortex dataset for figure-detection training data.

Reads a dataset directory produced by build_figure_detection_dataset.py, loads the
manifest and associated page PNGs, and writes a single .vortex file containing
image bytes plus labels/metadata in a columnar format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import vortex as vx


TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_manifest_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def iter_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest.get("records", []))


def build_arrow_table(manifest: dict[str, Any], split_name: str | None) -> pa.Table:
    rows: list[dict[str, Any]] = []
    dataset_output_dir = manifest.get("output_dir")
    dpi = manifest.get("dpi")

    for idx, record in enumerate(iter_records(manifest)):
        image_path = resolve_manifest_path(record["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found for record {idx}: {image_path}")

        bbox_image_path = record.get("bbox_image_path")
        resolved_bbox_image_path = resolve_manifest_path(bbox_image_path) if bbox_image_path else None
        if resolved_bbox_image_path is not None and not resolved_bbox_image_path.exists():
            resolved_bbox_image_path = None

        target = record.get("target", {"figures": []})
        figures = target.get("figures", [])

        rows.append(
            {
                "record_id": idx,
                "split": split_name,
                "pdf_path": record.get("pdf_path"),
                "pdf_name": record.get("pdf_name"),
                "page": int(record["page"]),
                "page_label": record.get("page_label"),
                "image_path": str(image_path),
                "image_bytes": image_path.read_bytes(),
                "bbox_image_path": str(resolved_bbox_image_path) if resolved_bbox_image_path else None,
                "bbox_image_bytes": resolved_bbox_image_path.read_bytes() if resolved_bbox_image_path else None,
                "annotation_path": record.get("annotation_path"),
                "target_json": json.dumps(target, separators=(",", ":")),
                "figures_json": json.dumps(figures, separators=(",", ":")),
                "figure_count": len(figures),
                "dataset_output_dir": dataset_output_dir,
                "dpi": dpi,
            }
        )

    return pa.Table.from_pylist(rows)


def write_vortex(table: pa.Table, output_path: Path, compact: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        vx.io.VortexWriteOptions.compact().write(table, str(output_path))
    else:
        vx.io.write(table, str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a .vortex file from a figure-detection dataset manifest")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory produced by build_figure_detection_dataset.py")
    parser.add_argument("--output", required=True, help="Output .vortex file path")
    parser.add_argument("--manifest", default=None, help="Optional explicit manifest path (defaults to <dataset-dir>/manifest.json)")
    parser.add_argument("--split-name", default=None, help="Optional split label to store in the file, e.g. train or val")
    parser.add_argument("--compact", action="store_true", help="Prefer smaller Vortex files over read throughput")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else Path(args.dataset_dir) / "manifest.json"
    manifest = load_manifest(manifest_path)
    table = build_arrow_table(manifest, args.split_name)
    write_vortex(table, Path(args.output), args.compact)

    print(f"Wrote {table.num_rows} records to {args.output}")
    print(f"Columns: {table.column_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
