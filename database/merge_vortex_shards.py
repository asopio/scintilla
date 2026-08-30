#!/usr/bin/env python3
"""Merge a directory of Vortex shard files into one Vortex file.

The merge is streamed shard-by-shard and batch-by-batch, so the full dataset is
not loaded into memory at once.

Examples:
  python merge_vortex_shards.py \
    --input db/jetm_202326_30p.vortex \
    --output db/jetm_202326_30p_merged.vortex

  python merge_vortex_shards.py \
    --input db/slides_db_vortex/slides \
    --output db/slides_merged.vortex
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import vortex as vx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing .vortex shard files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the merged .vortex file to create",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Rows per Arrow batch to stream from each shard (default: 2048)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    return parser.parse_args()


def find_shards(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    shards = sorted(input_dir.glob("*.vortex"))
    if not shards:
        raise ValueError(f"No .vortex shard files found in {input_dir}")
    return shards


def iter_shard_batches(
    shard_readers: list[tuple[Path, pa.RecordBatchReader]],
    schema: pa.Schema,
) -> Iterator[pa.RecordBatch]:
    for shard_path, reader in shard_readers:
        if not reader.schema.equals(schema):
            raise ValueError(f"Shard schema mismatch for {shard_path}:\nexpected {schema}\ngot {reader.schema}")
        print(f"Streaming {shard_path.name}")
        yield from reader


def merge_shards(input_dir: Path, output_path: Path, batch_size: int, force: bool) -> None:
    shard_paths = find_shards(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if not force:
            raise ValueError(f"Output already exists: {output_path}. Use --force to overwrite.")
        output_path.unlink()

    shard_readers = [
        (shard_path, vx.open(str(shard_path)).to_arrow(batch_size=batch_size)) for shard_path in shard_paths
    ]
    schema = shard_readers[0][1].schema

    def all_batches() -> Iterator[pa.RecordBatch]:
        yield from iter_shard_batches(shard_readers, schema=schema)

    vx.io.write(pa.RecordBatchReader.from_batches(schema, all_batches()), str(output_path))
    print(f"Merged {len(shard_paths)} shards into {output_path}")
    print(f"Written: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    args = parse_args()
    merge_shards(
        input_dir=Path(args.input),
        output_path=Path(args.output),
        batch_size=args.batch_size,
        force=args.force,
    )
