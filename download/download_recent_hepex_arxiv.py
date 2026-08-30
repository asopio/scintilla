#!/usr/bin/env python3
"""Download recent arXiv papers, optionally excluding another category.

Fetches metadata from the arXiv API and downloads PDFs for the N most recent
papers in a selected category, with optional exclusion of papers cross-listed
in another category.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_NS = {"arxiv": "http://arxiv.org/schemas/atom"}


def fetch_feed(
    category: str,
    max_results: int,
    start: int = 0,
    sort_by: str = "submittedDate",
    exclude_category: str | None = None,
) -> str:
    search_query = f"cat:{category}"
    if exclude_category:
        search_query = f"{search_query} ANDNOT cat:{exclude_category}"
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def parse_entries(feed_xml: str) -> list[dict[str, str]]:
    root = ET.fromstring(feed_xml)
    entries: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        title = entry.findtext("atom:title", default="", namespaces=ATOM_NS).strip()
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
        updated = entry.findtext("atom:updated", default="", namespaces=ATOM_NS).strip()
        primary_category = ""
        primary_category_elem = entry.find("arxiv:primary_category", ARXIV_NS)
        if primary_category_elem is not None:
            primary_category = primary_category_elem.attrib.get("term", "").strip()
        categories = [
            category.attrib.get("term", "").strip()
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term", "").strip()
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "").strip()
                break
        if not pdf_url and entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"
        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else ""
        entries.append(
            {
                "id": arxiv_id,
                "title": title,
                "published": published,
                "updated": updated,
                "pdf_url": pdf_url,
                "entry_id": entry_id,
                "primary_category": primary_category,
                "categories": categories,
            }
        )
    return entries


def filter_excluded_category(entries: list[dict[str, str]], exclude_category: str | None) -> list[dict[str, str]]:
    if not exclude_category:
        return entries
    return [entry for entry in entries if exclude_category not in entry.get("categories", [])]


def sanitize_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_.") or "paper"


def download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-doc-converter/1.0"})
    with urllib.request.urlopen(req) as response, dest.open("wb") as f:
        f.write(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Download recent arXiv PDFs")
    parser.add_argument("--category", default="hep-ex", help="Primary arXiv category to download")
    parser.add_argument(
        "--exclude-category",
        help="Exclude papers that are also listed in this arXiv category",
    )
    parser.add_argument("--max-results", type=int, default=200, help="Number of recent papers to download")
    parser.add_argument("--output-dir", default="downloads/arxiv_hep_ex_recent", help="Download directory")
    parser.add_argument("--metadata-file", default="metadata.json", help="Metadata JSON file name inside output dir")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between PDF downloads")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs that already exist")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    feed_xml = fetch_feed(args.category, args.max_results, exclude_category=args.exclude_category)
    entries = parse_entries(feed_xml)
    entries = filter_excluded_category(entries, args.exclude_category)
    entries = entries[: args.max_results]

    downloaded: list[dict[str, str]] = []
    for idx, entry in enumerate(entries, start=1):
        arxiv_id = entry["id"]
        title = entry["title"]
        pdf_url = entry["pdf_url"]
        filename = sanitize_filename(f"{idx:03d}_{arxiv_id}.pdf")
        pdf_path = outdir / filename
        if args.skip_existing and pdf_path.exists():
            entry["pdf_path"] = str(pdf_path)
            downloaded.append(entry)
            continue
        print(f"[{idx}/{len(entries)}] {arxiv_id} -> {pdf_path.name}")
        print(f"  {title}")
        download_file(pdf_url, pdf_path)
        entry["pdf_path"] = str(pdf_path)
        downloaded.append(entry)
        if idx < len(entries) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    metadata_path = outdir / args.metadata_file
    metadata_path.write_text(json.dumps(downloaded, indent=2), encoding="utf-8")
    print(f"Saved {len(downloaded)} metadata records to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
