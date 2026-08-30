#!/usr/bin/env python3
"""Build a database from egam_meetings metadata, markdown, and PDFs.

Each row represents one slide. Presentation-level metadata is duplicated across rows.
By default, PDF pages are rendered to JPEG bytes in memory via pdfplumber.
With --store-djvu, PDF pages are rendered to color DjVu bytes, one per slide row,
preferring djvudigital when available and falling back to c44 otherwise.

With --store-pdfs, the output becomes a database directory containing:
  - slides dataset     : one row per slide with slide_number and extracted content
  - documents dataset  : one row per document with pdf_binary stored once

Output formats (choose with --format):
  - parquet  : slides.parquet
  - vortex   : slides.vortex   (default, faster random access, similar compression)

Memory strategy:
  - Rows are accumulated in a columnar dict-of-lists buffer (3-4x faster than
    list-of-dicts DataFrame construction, per polars benchmarks).
  - Every --chunk-size documents the buffer is flushed:
      Parquet : written as a row group via pyarrow.parquet.ParquetWriter.
      Vortex  : streamed through a pyarrow.RecordBatchReader into vx.io.write().
  - PDF pages are rendered and appended one at a time instead of building a
    whole-document JPEG list first.
  - With --store-pdfs, slides are written without inline JPEGs and PDFs are
    stored once in a separate documents dataset.
  - With --store-djvu, slides are written with inline color DjVu binaries
    instead of JPEGs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pdfplumber
import pyarrow as pa
import pyarrow.parquet as pq
from PyPDF2 import PdfReader, PdfWriter

# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


def parse_slides_md(md_text: str) -> dict[int, dict]:
    """Parse markdown into a mapping of slide_number -> {extracted_text, visual_description}."""
    slides: dict[int, dict] = {}
    slide_blocks = re.split(r"(?=^## Slide \d+)", md_text, flags=re.MULTILINE)
    for block in slide_blocks:
        m = re.match(r"^## Slide (\d+)\s*\n", block)
        if not m:
            continue
        slide_number = int(m.group(1))

        text_match = re.search(
            r"### Extracted Text\s*\n(.*?)(?=### Visual Description|\Z)",
            block,
            flags=re.DOTALL,
        )
        desc_match = re.search(
            r"### Visual Description\s*\n(.*?)(?=^## Slide |\Z)",
            block,
            flags=re.DOTALL,
        )

        slides[slide_number] = {
            "extracted_text": text_match.group(1).strip() if text_match else "",
            "visual_description": desc_match.group(1).strip() if desc_match else "",
        }
    return slides


# ---------------------------------------------------------------------------
# PDF -> JPEG bytes
# ---------------------------------------------------------------------------

_DJVU_BACKEND: tuple[str, str] | None = None
_DJVU_FALLBACK_WARNED = False


def _find_tool_path(*names: str) -> str | None:
    pixi_bin = Path(__file__).resolve().parent / ".pixi" / "envs" / "default" / "bin"
    for name in names:
        candidate = shutil.which(name)
        if candidate and Path(candidate).exists():
            return candidate
        pixi_candidate = pixi_bin / name
        if pixi_candidate.exists():
            return str(pixi_candidate)
    return None


def choose_djvu_backend() -> tuple[str, str]:
    global _DJVU_BACKEND, _DJVU_FALLBACK_WARNED
    if _DJVU_BACKEND is not None:
        return _DJVU_BACKEND

    djvudigital = _find_tool_path("djvudigital")
    if djvudigital is not None:
        check = subprocess.run(
            [djvudigital, "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode == 0:
            _DJVU_BACKEND = ("djvudigital", djvudigital)
            return _DJVU_BACKEND
        if not _DJVU_FALLBACK_WARNED:
            print(
                "  WARNING: djvudigital is installed but unusable in this environment; "
                "falling back to c44. It requires a Ghostscript build with the djvusep device."
            )
            _DJVU_FALLBACK_WARNED = True

    c44 = _find_tool_path("c44")
    if c44 is not None:
        _DJVU_BACKEND = ("c44", c44)
        return _DJVU_BACKEND

    raise RuntimeError(
        "Could not find a usable DjVu encoder. Install djvudigital with djvusep-enabled "
        "Ghostscript or install c44 from djvulibre."
    )


def page_to_jpeg_bytes(page: Any, resolution: int = 100) -> bytes:
    """Render one PDF page to JPEG bytes in memory."""
    img = page.to_image(resolution=resolution)
    buf = BytesIO()
    img.original.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def page_to_djvu_bytes(page: Any, resolution: int = 100) -> bytes:
    """Render one PDF page to color DjVu bytes via c44."""
    img = page.to_image(resolution=resolution)
    backend, c44 = choose_djvu_backend()
    if backend != "c44":
        raise RuntimeError("page_to_djvu_bytes requires c44 backend")
    with tempfile.TemporaryDirectory(prefix="build-db-djvu-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        ppm_path = tmpdir_path / "slide.ppm"
        djvu_path = tmpdir_path / "slide.djvu"
        img.original.save(ppm_path, format="PPM")
        subprocess.run(
            [
                c44,
                "-dpi",
                str(resolution),
                "-crcbfull",
                str(ppm_path),
                str(djvu_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return djvu_path.read_bytes()


def pdf_page_to_djvu_bytes(pdf_path: Path, slide_num: int, resolution: int = 100) -> bytes:
    """Convert one PDF page directly to DjVu bytes via djvudigital when available."""
    backend, tool_path = choose_djvu_backend()
    if backend != "djvudigital":
        raise RuntimeError("pdf_page_to_djvu_bytes requires djvudigital backend")

    with tempfile.TemporaryDirectory(prefix="build-db-djvudigital-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        page_pdf_path = tmpdir_path / "slide.pdf"
        djvu_path = tmpdir_path / "slide.djvu"

        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        writer.add_page(reader.pages[slide_num - 1])
        with open(page_pdf_path, "wb") as f:
            writer.write(f)

        subprocess.run(
            [
                tool_path,
                f"--dpi={resolution}",
                "--exact-color",
                "--words",
                str(page_pdf_path),
                str(djvu_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return djvu_path.read_bytes()


# ---------------------------------------------------------------------------
# Columnar buffer (dict-of-lists)
#
# Building a DataFrame from a dict-of-lists is ~3.5x faster than from a
# list-of-dicts because polars can hand the pre-arranged column arrays
# directly to Arrow without transposing row-by-row.
# ---------------------------------------------------------------------------


def _empty_slide_buffer(asset_kind: str | None) -> dict[str, list[Any]]:
    buf = {
        "stem": [],
        "year_month": [],
        "category_id": [],
        "meeting_id": [],
        "meeting_title": [],
        "meeting_url": [],
        "date": [],
        "contribution_id": [],
        "contribution_title": [],
        "contribution_url": [],
        "author": [],
        "authors_json": [],
        "attachment_title": [],
        "attachment_id": [],
        "attachment_is_protected": [],
        "attachment_size": [],
        "download_url": [],
        "pdf_file": [],
        "pdf_path": [],
        "slide_number": [],
        "extracted_text": [],
        "visual_description": [],
    }
    if asset_kind == "jpeg":
        buf["slide_jpeg"] = []
    elif asset_kind == "djvu":
        buf["slide_djvu"] = []
    return buf


_SLIDES_SCHEMA = pa.schema(
    [
        ("stem", pa.string()),
        ("year_month", pa.string()),
        ("category_id", pa.int64()),
        ("meeting_id", pa.string()),
        ("meeting_title", pa.string()),
        ("meeting_url", pa.string()),
        ("date", pa.string()),
        ("contribution_id", pa.string()),
        ("contribution_title", pa.string()),
        ("contribution_url", pa.string()),
        ("author", pa.string()),
        ("authors_json", pa.string()),
        ("attachment_title", pa.string()),
        ("attachment_id", pa.int64()),
        ("attachment_is_protected", pa.bool_()),
        ("attachment_size", pa.int64()),
        ("download_url", pa.string()),
        ("pdf_file", pa.string()),
        ("pdf_path", pa.string()),
        ("slide_number", pa.int64()),
        ("extracted_text", pa.string()),
        ("visual_description", pa.string()),
    ]
)

_SLIDES_WITH_JPEG_SCHEMA = pa.schema(
    [
        *_SLIDES_SCHEMA,
        ("slide_jpeg", pa.binary()),
    ]
)

_SLIDES_WITH_DJVU_SCHEMA = pa.schema(
    [
        *_SLIDES_SCHEMA,
        ("slide_djvu", pa.binary()),
    ]
)

_DOCUMENTS_SCHEMA = pa.schema(
    [
        ("stem", pa.string()),
        ("year_month", pa.string()),
        ("category_id", pa.int64()),
        ("meeting_id", pa.string()),
        ("meeting_title", pa.string()),
        ("meeting_url", pa.string()),
        ("date", pa.string()),
        ("contribution_id", pa.string()),
        ("contribution_title", pa.string()),
        ("contribution_url", pa.string()),
        ("author", pa.string()),
        ("authors_json", pa.string()),
        ("attachment_title", pa.string()),
        ("attachment_id", pa.int64()),
        ("attachment_is_protected", pa.bool_()),
        ("attachment_size", pa.int64()),
        ("download_url", pa.string()),
        ("pdf_file", pa.string()),
        ("pdf_path", pa.string()),
        ("pdf_page_count", pa.int64()),
        ("pdf_binary", pa.binary()),
    ]
)


def _buf_to_arrow(buf: dict[str, list[Any]], schema: pa.Schema) -> pa.Table:
    return pa.table(buf, schema=schema)


def _empty_documents_buffer() -> dict[str, list[Any]]:
    return {
        "stem": [],
        "year_month": [],
        "category_id": [],
        "meeting_id": [],
        "meeting_title": [],
        "meeting_url": [],
        "date": [],
        "contribution_id": [],
        "contribution_title": [],
        "contribution_url": [],
        "author": [],
        "authors_json": [],
        "attachment_title": [],
        "attachment_id": [],
        "attachment_is_protected": [],
        "attachment_size": [],
        "download_url": [],
        "pdf_file": [],
        "pdf_path": [],
        "pdf_page_count": [],
        "pdf_binary": [],
    }


# ---------------------------------------------------------------------------
# Per-document: append slides into the columnar buffer
# ---------------------------------------------------------------------------

_BASE_KEYS = (
    "stem",
    "year_month",
    "category_id",
    "meeting_id",
    "meeting_title",
    "meeting_url",
    "date",
    "contribution_id",
    "contribution_title",
    "contribution_url",
    "author",
    "authors_json",
    "attachment_title",
    "attachment_id",
    "attachment_is_protected",
    "attachment_size",
    "download_url",
    "pdf_file",
    "pdf_path",
)


def _append_slide_row(
    buf: dict[str, list[Any]],
    base_vals: tuple[Any, ...],
    slide_num: int,
    md_slide: dict[str, Any],
    slide_asset: bytes | None,
) -> None:
    for key, val in zip(_BASE_KEYS, base_vals, strict=False):
        buf[key].append(val)
    buf["slide_number"].append(slide_num)
    buf["extracted_text"].append(md_slide.get("extracted_text", ""))
    buf["visual_description"].append(md_slide.get("visual_description", ""))
    if "slide_jpeg" in buf:
        buf["slide_jpeg"].append(slide_asset)
    if "slide_djvu" in buf:
        buf["slide_djvu"].append(slide_asset)


def _append_document_row(
    buf: dict[str, list[Any]],
    base_vals: tuple[Any, ...],
    pdf_page_count: int,
    pdf_binary: bytes | None,
) -> None:
    for key, val in zip(_BASE_KEYS, base_vals, strict=False):
        buf[key].append(val)
    buf["pdf_page_count"].append(pdf_page_count)
    buf["pdf_binary"].append(pdf_binary)


def _load_base_values(json_path: Path) -> tuple[tuple[Any, ...], dict[str, Any], Path, Path]:
    year_month = json_path.parent.parent.name
    stem = json_path.stem

    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    authors_json = json.dumps(meta.get("authors", []))

    md_path = json_path.parent.parent / "md" / f"{stem}.md"
    pdf_path = json_path.parent.parent / meta.get("pdf_file", f"{stem}.pdf")

    base_vals = (
        stem,
        year_month,
        meta.get("category_id"),
        str(meta.get("meeting_id", "")),
        meta.get("meeting_title", ""),
        meta.get("meeting_url", ""),
        meta.get("date", ""),
        str(meta.get("contribution_id", "")),
        meta.get("contribution_title", ""),
        meta.get("contribution_url", ""),
        meta.get("author", ""),
        authors_json,
        meta.get("attachment_title", ""),
        meta.get("attachment_id"),
        meta.get("attachment_is_protected", False),
        meta.get("attachment_size"),
        meta.get("download_url", ""),
        meta.get("pdf_file", ""),
        meta.get("pdf_path", ""),
    )
    return base_vals, meta, md_path, pdf_path


def append_doc_to_slide_buffer(
    json_path: Path,
    resolution: int,
    max_pages: int,
    buf: dict[str, list[Any]],
    asset_kind: str | None,
) -> int:
    """Parse one document and append its slides to buf. Returns number of slides added."""
    base_vals, _meta, md_path, pdf_path = _load_base_values(json_path)
    md_slides = parse_slides_md(md_path.read_text(encoding="utf-8")) if md_path.exists() else {}
    pdf_page_count = 0
    djvu_backend: str | None = None
    if asset_kind == "djvu":
        djvu_backend, _ = choose_djvu_backend()

    # Track the number of rows appended before processing, so we can detect failures
    rows_before = len(buf["stem"])

    if pdf_path.exists():
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pdf_page_count = len(pdf.pages)
                if asset_kind is None:
                    pass
                elif pdf_page_count > max_pages:
                    print(
                        f"  WARNING: {pdf_path.name} has {pdf_page_count} pages, "
                        f"skipping slide assets (exceeds --max-pages={max_pages})"
                    )
                elif asset_kind == "djvu" and djvu_backend == "djvudigital":
                    for slide_num in range(1, pdf_page_count + 1):
                        slide_asset: bytes | None
                        try:
                            slide_asset = pdf_page_to_djvu_bytes(
                                pdf_path,
                                slide_num,
                                resolution=resolution,
                            )
                        except Exception as exc:
                            print(
                                f"  WARNING: could not convert page {slide_num} "
                                f"of {pdf_path.name} with djvudigital: {exc}"
                            )
                            slide_asset = None
                        _append_slide_row(
                            buf,
                            base_vals,
                            slide_num,
                            md_slides.get(slide_num, {}),
                            slide_asset,
                        )
                else:
                    for slide_num, page in enumerate(pdf.pages, start=1):
                        slide_asset: bytes | None
                        try:
                            if asset_kind == "jpeg":
                                slide_asset = page_to_jpeg_bytes(page, resolution=resolution)
                            elif asset_kind == "djvu":
                                slide_asset = page_to_djvu_bytes(page, resolution=resolution)
                            else:
                                raise ValueError(f"Unsupported slide asset kind: {asset_kind}")
                        except Exception as exc:
                            print(f"  WARNING: could not render page {slide_num} of {pdf_path.name}: {exc}")
                            slide_asset = None
                        _append_slide_row(
                            buf,
                            base_vals,
                            slide_num,
                            md_slides.get(slide_num, {}),
                            slide_asset,
                        )
        except Exception as exc:
            print(
                f"  WARNING: could not process {pdf_path.name} (attempted to process "
                f"{len(buf['stem']) - rows_before} slides before error): {exc}"
            )

    n_slides = max(pdf_page_count, len(md_slides))
    # Calculate which slides were actually rendered by checking the buffer growth
    rows_appended = len(buf["stem"]) - rows_before
    rendered_up_to = rows_appended

    # Fill in any remaining slides with None assets
    for slide_num in range(rendered_up_to + 1, n_slides + 1):
        _append_slide_row(
            buf,
            base_vals,
            slide_num,
            md_slides.get(slide_num, {}),
            None,
        )

    return n_slides


def append_doc_to_documents_buffer(json_path: Path, buf: dict[str, list[Any]]) -> int:
    """Append one document row with PDF bytes stored once."""
    base_vals, _meta, _md_path, pdf_path = _load_base_values(json_path)
    pdf_page_count = 0
    pdf_binary: bytes | None = None
    if pdf_path.exists():
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pdf_page_count = len(pdf.pages)
        except Exception as exc:
            print(f"  WARNING: could not inspect {pdf_path.name}: {exc}")
        try:
            pdf_binary = pdf_path.read_bytes()
        except Exception as exc:
            print(f"  WARNING: could not read {pdf_path.name}: {exc}")
    _append_document_row(buf, base_vals, pdf_page_count, pdf_binary)
    return 1


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetWriteResult:
    output_path: str
    rows_written: int


@dataclass(frozen=True)
class ShardBuildResult:
    slides: DatasetWriteResult
    documents: DatasetWriteResult | None = None


def iter_slide_tables(
    json_files: list[Path],
    resolution: int,
    max_pages: int,
    chunk_size: int,
    asset_kind: str | None,
    progress_label: str = "",
) -> tuple[Iterator[pa.Table], dict[str, int]]:
    stats = {"rows_written": 0}
    n_files = len(json_files)
    label = f"{progress_label} " if progress_label else ""
    if asset_kind == "jpeg":
        schema = _SLIDES_WITH_JPEG_SCHEMA
    elif asset_kind == "djvu":
        schema = _SLIDES_WITH_DJVU_SCHEMA
    else:
        schema = _SLIDES_SCHEMA

    def _iter() -> Iterator[pa.Table]:
        buf = _empty_slide_buffer(asset_kind=asset_kind)
        chunk_docs = 0

        for i, json_path in enumerate(json_files, 1):
            n = append_doc_to_slide_buffer(
                json_path,
                resolution=resolution,
                max_pages=max_pages,
                buf=buf,
                asset_kind=asset_kind,
            )
            stats["rows_written"] += n
            chunk_docs += 1

            if i % 100 == 0:
                print(f"  {label}Processing {i}/{n_files} ({stats['rows_written']} slides so far) ...")

            if chunk_docs >= chunk_size and buf["stem"]:
                yield _buf_to_arrow(buf, schema)
                buf = _empty_slide_buffer(asset_kind=asset_kind)
                chunk_docs = 0

        if buf["stem"]:
            yield _buf_to_arrow(buf, schema)

    return _iter(), stats


def iter_document_tables(
    json_files: list[Path],
    chunk_size: int,
    progress_label: str = "",
) -> tuple[Iterator[pa.Table], dict[str, int]]:
    stats = {"rows_written": 0}
    n_files = len(json_files)
    label = f"{progress_label} " if progress_label else ""

    def _iter() -> Iterator[pa.Table]:
        buf = _empty_documents_buffer()
        chunk_docs = 0

        for i, json_path in enumerate(json_files, 1):
            n = append_doc_to_documents_buffer(json_path, buf=buf)
            stats["rows_written"] += n
            chunk_docs += 1

            if i % 100 == 0:
                print(f"  {label}Processing {i}/{n_files} ({stats['rows_written']} documents so far) ...")

            if chunk_docs >= chunk_size and buf["stem"]:
                yield _buf_to_arrow(buf, _DOCUMENTS_SCHEMA)
                buf = _empty_documents_buffer()
                chunk_docs = 0

        if buf["stem"]:
            yield _buf_to_arrow(buf, _DOCUMENTS_SCHEMA)

    return _iter(), stats


def write_table_stream(
    output_path: Path,
    schema: pa.Schema,
    table_iter: Iterator[pa.Table],
    fmt: str,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        parquet_writer: pq.ParquetWriter | None = None
        try:
            for table in table_iter:
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(str(output_path), schema)
                parquet_writer.write_table(table)
        finally:
            if parquet_writer is not None:
                parquet_writer.close()
        return parquet_writer is not None
    else:
        import vortex as vx

        batches = (batch for table in table_iter for batch in table.to_batches())
        first_batch = next(batches, None)
        if first_batch is None:
            return False
        reader = pa.RecordBatchReader.from_batches(schema, itertools.chain([first_batch], batches))
        vx.io.write(reader, str(output_path))
        return True


def build_slides_shard(
    json_files: list[Path],
    output_path: Path,
    resolution: int,
    max_pages: int,
    fmt: str,
    chunk_size: int,
    asset_kind: str | None,
) -> DatasetWriteResult:
    table_iter, stats = iter_slide_tables(
        json_files,
        resolution=resolution,
        max_pages=max_pages,
        chunk_size=chunk_size,
        asset_kind=asset_kind,
        progress_label=output_path.stem,
    )
    if asset_kind == "jpeg":
        schema = _SLIDES_WITH_JPEG_SCHEMA
    elif asset_kind == "djvu":
        schema = _SLIDES_WITH_DJVU_SCHEMA
    else:
        schema = _SLIDES_SCHEMA
    write_table_stream(output_path, schema, table_iter, fmt)
    return DatasetWriteResult(str(output_path), stats["rows_written"])


def build_documents_shard(
    json_files: list[Path],
    output_path: Path,
    fmt: str,
    chunk_size: int,
) -> DatasetWriteResult:
    table_iter, stats = iter_document_tables(
        json_files,
        chunk_size=chunk_size,
        progress_label=output_path.stem,
    )
    write_table_stream(output_path, _DOCUMENTS_SCHEMA, table_iter, fmt)
    return DatasetWriteResult(str(output_path), stats["rows_written"])


def build_shard(
    json_files: list[Path],
    slide_output_path: Path,
    resolution: int,
    max_pages: int,
    fmt: str,
    chunk_size: int,
    store_pdfs: bool,
    store_djvu: bool,
    documents_output_path: Path | None = None,
) -> ShardBuildResult:
    slide_asset_kind = "djvu" if store_djvu else (None if store_pdfs else "jpeg")
    slides = build_slides_shard(
        json_files,
        slide_output_path,
        resolution=resolution,
        max_pages=max_pages,
        fmt=fmt,
        chunk_size=chunk_size,
        asset_kind=slide_asset_kind,
    )
    documents: DatasetWriteResult | None = None
    if store_pdfs:
        if documents_output_path is None:
            raise ValueError("documents_output_path is required when --store-pdfs is enabled")
        documents = build_documents_shard(
            json_files,
            documents_output_path,
            fmt=fmt,
            chunk_size=chunk_size,
        )
    return ShardBuildResult(slides=slides, documents=documents)


def _split_json_files(json_files: list[Path], workers: int) -> list[list[Path]]:
    return [json_files[i::workers] for i in range(workers) if json_files[i::workers]]


def _worker_build_shard(
    json_files: list[str],
    slide_output_path: str,
    documents_output_path: str | None,
    resolution: int,
    max_pages: int,
    fmt: str,
    chunk_size: int,
    store_pdfs: bool,
    store_djvu: bool,
) -> ShardBuildResult:
    return build_shard(
        [Path(p) for p in json_files],
        Path(slide_output_path),
        resolution=resolution,
        max_pages=max_pages,
        fmt=fmt,
        chunk_size=chunk_size,
        store_pdfs=store_pdfs,
        store_djvu=store_djvu,
        documents_output_path=Path(documents_output_path) if documents_output_path else None,
    )


def build_database(
    meetings_dir: Path,
    output_path: Path,
    resolution: int,
    max_pages: int,
    fmt: str,
    chunk_size: int,
    workers: int,
    store_pdfs: bool,
    store_djvu: bool,
) -> None:
    json_files = sorted(meetings_dir.rglob("metadata/*.json"))
    print(f"Found {len(json_files)} metadata files")
    print(
        f"Chunk size: {chunk_size} documents | Format: {fmt} | "
        f"Workers: {workers} | Store PDFs: {store_pdfs} | Store DjVu: {store_djvu}"
    )

    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if store_pdfs and store_djvu:
        raise ValueError("--store-pdfs and --store-djvu cannot be used together")

    if store_pdfs:
        output_path.mkdir(parents=True, exist_ok=True)
        if any(output_path.iterdir()):
            raise ValueError(f"PDF-storage output directory must be empty: {output_path}")
        slides_target = output_path / f"slides.{fmt}"
        documents_target = output_path / f"documents.{fmt}"
    else:
        slides_target = output_path
        documents_target = None

    if workers == 1:
        result = build_shard(
            json_files,
            slides_target,
            resolution=resolution,
            max_pages=max_pages,
            fmt=fmt,
            chunk_size=chunk_size,
            store_pdfs=store_pdfs,
            store_djvu=store_djvu,
            documents_output_path=documents_target,
        )
        if result.slides.rows_written == 0:
            print("No slides found — nothing written.")
            return
        written = Path(result.slides.output_path)
        print(f"Total slide rows: {result.slides.rows_written}")
        print(f"Written slides: {written}  ({written.stat().st_size / 1e6:.1f} MB)")
        if result.documents is not None:
            docs_written = Path(result.documents.output_path)
            print(f"Written documents: {docs_written}  ({docs_written.stat().st_size / 1e6:.1f} MB)")
        return

    if not store_pdfs:
        output_path.mkdir(parents=True, exist_ok=True)
    if any(output_path.iterdir()) and not store_pdfs:
        raise ValueError(f"Parallel output directory must be empty: {output_path}")

    if not json_files:
        print("No slides found — nothing written.")
        return

    shard_groups = _split_json_files(json_files, min(workers, len(json_files)))
    shard_width = max(4, len(str(len(shard_groups) - 1)))
    slides_output_dir = output_path / "slides" if store_pdfs else output_path
    documents_output_dir = output_path / "documents" if store_pdfs else None
    slides_output_dir.mkdir(parents=True, exist_ok=True)
    if documents_output_dir is not None:
        documents_output_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    results: list[ShardBuildResult] = []
    with ProcessPoolExecutor(max_workers=len(shard_groups)) as executor:
        for shard_idx, shard_json_files in enumerate(shard_groups):
            slides_shard_path = slides_output_dir / f"part-{shard_idx:0{shard_width}d}.{fmt}"
            documents_shard_path = (
                documents_output_dir / f"part-{shard_idx:0{shard_width}d}.{fmt}"
                if documents_output_dir is not None
                else None
            )
            futures.append(
                executor.submit(
                    _worker_build_shard,
                    [str(p) for p in shard_json_files],
                    str(slides_shard_path),
                    str(documents_shard_path) if documents_shard_path else None,
                    resolution,
                    max_pages,
                    fmt,
                    chunk_size,
                    store_pdfs,
                    store_djvu,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            written = Path(result.slides.output_path)
            size_mb = written.stat().st_size / 1e6 if written.exists() else 0.0
            print(f"Completed shard: {written.name} ({result.slides.rows_written} rows, {size_mb:.1f} MB)")
            if result.documents is not None:
                docs_written = Path(result.documents.output_path)
                docs_size_mb = docs_written.stat().st_size / 1e6 if docs_written.exists() else 0.0
                print(
                    f"Completed document shard: {docs_written.name} "
                    f"({result.documents.rows_written} rows, {docs_size_mb:.1f} MB)"
                )

    total_slides = sum(result.slides.rows_written for result in results)
    if total_slides == 0:
        print("No slides found — nothing written.")
        return

    print(f"Total slide rows: {total_slides}")
    print(f"Written slide shards: {len(results)} in {slides_output_dir}")
    if store_pdfs:
        total_documents = sum(result.documents.rows_written for result in results if result.documents is not None)
        print(f"Total document rows: {total_documents}")
        print(f"Written document shards: {len(results)} in {documents_output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meetings-dir",
        default="./egam_meetings",
        help="Root directory containing YYYY-MM meeting subdirectories",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file path for single-worker runs, or output directory for "
            "multi-worker shard runs. With --store-pdfs, always treated as an "
            "empty database directory."
        ),
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["parquet", "vortex"],
        default="vortex",
        help="Output file format (default: vortex)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=200,
        help="JPEG render resolution in DPI (default: 200)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Skip PDF image rendering for documents with more pages than this (default: 200)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Flush to disk every N documents to cap peak memory (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=("Number of worker processes. When >1, --output is treated as an empty shard directory (default: 1)"),
    )
    storage_group = parser.add_mutually_exclusive_group()
    storage_group.add_argument(
        "--store-pdfs",
        action="store_true",
        help=(
            "Store each source PDF once in a companion documents dataset and "
            "write slides without inline JPEGs. Output becomes a database "
            "directory containing slides and documents datasets."
        ),
    )
    storage_group.add_argument(
        "--store-djvu",
        action="store_true",
        help=(
            "Store each slide as an inline color DjVu binary in the slides "
            "dataset instead of inline JPEGs. Prefers djvudigital when its "
            "Ghostscript backend is available, otherwise falls back to c44."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ext = "vortex" if args.fmt == "vortex" else "parquet"
    if args.output:
        output = Path(args.output)
    elif args.store_pdfs:
        output = Path(f"./db/slides_db_{ext}")
    elif args.workers > 1:
        output = Path(f"./db/slides_{ext}_shards")
    else:
        output = Path(f"./db/slides.{ext}")
    build_database(
        Path(args.meetings_dir),
        output,
        resolution=args.resolution,
        max_pages=args.max_pages,
        fmt=args.fmt,
        chunk_size=args.chunk_size,
        workers=args.workers,
        store_pdfs=args.store_pdfs,
        store_djvu=args.store_djvu,
    )
