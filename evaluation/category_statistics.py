#!/usr/bin/env python3
"""Compile statistics about Indico categories.

For each category ID in atlas_physics_subcategories.txt, queries the Indico
search API to count PDF-attachment presentations per year.  One API call is
made per (category, year) pair — each call returns only the total count, so
the whole script completes in minutes rather than hours.

Results are saved to category_statistics.json (raw data) and
category_statistics.md (formatted report).

Usage:
    python category_statistics.py [--start-year YYYY] [--resume]

    --start-year  Earliest year to include (default: 2010)
    --resume      Load any previously saved JSON and skip already-fetched
                  categories
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

INDICO_SITE = "https://indico.cern.ch"
REPO_ROOT = Path(__file__).parent
CATEGORIES_FILE = REPO_ROOT / "atlas_physics_subcategories.txt"
OUTPUT_JSON = REPO_ROOT / "category_statistics.json"
OUTPUT_MD = REPO_ROOT / "category_statistics.md"
DEFAULT_START_YEAR = 2010


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def load_api_token() -> str:
    token = os.getenv("INDICO_API_TOKEN", "").strip()
    if token:
        return token
    batch_script = REPO_ROOT / "batch_dl_convert.sh"
    if batch_script.exists():
        for line in batch_script.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("INDICO_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise SystemExit(
        "No INDICO_API_TOKEN found. Set the environment variable or ensure "
        "batch_dl_convert.sh contains INDICO_API_TOKEN=..."
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def fetch_json(path: str, params: dict, api_token: str) -> dict:
    url = INDICO_SITE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Search-API helpers
# ---------------------------------------------------------------------------


def search_pdf_count(category_id: int, year_start: str, year_end: str, api_token: str) -> int:
    """Return count of PDF attachments in a category within [year_start, year_end)."""
    data = fetch_json(
        "/search/api/search",
        {
            "q": ".pdf",
            "type": "attachment",
            "category_id": category_id,
            "start_range": f"[{year_start} TO {year_end}]",
        },
        api_token,
    )
    return data.get("total", 0)


def search_category_name(category_id: int, api_token: str) -> str:
    """Resolve category display name via a single-result search query."""
    try:
        data = fetch_json(
            "/search/api/search",
            {"q": ".pdf", "type": "attachment", "category_id": category_id},
            api_token,
        )
        for result in data.get("results", []):
            for entry in result.get("category_path", []):
                if entry.get("id") == category_id:
                    return entry["title"]
    except (urllib.error.URLError, OSError, KeyError):
        pass
    return "Unknown"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def cumulative_since(by_year: dict, since_year: int) -> int:
    return sum(v for y, v in by_year.items() if y.isdigit() and int(y) >= since_year)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(all_stats: list[dict], output_file: Path, start_year: int, end_year: int) -> None:
    years = [str(y) for y in range(start_year, end_year + 1)]

    lines: list[str] = [
        "# ATLAS Indico Category Statistics",
        "",
        "Counts PDF-attachment presentations via the Indico search API (`q=.pdf&type=attachment`).",
        f"Date range: **{start_year}** to **{end_year}** | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Per-year PDF presentation counts",
        "",
    ]

    col_names = ["Category ID", "Category Name", "Total", *years]
    lines.append("| " + " | ".join(col_names) + " |")
    lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")

    for s in all_stats:
        if "error" in s:
            row = [str(s["id"]), s.get("name", "ERROR"), "ERR"] + ["-"] * len(years)
        else:
            row = [
                str(s["id"]),
                s.get("name", "Unknown"),
                str(s.get("total", 0)),
            ]
            for y in years:
                row.append(str(s.get("by_year", {}).get(y, 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Cumulative counts since cutoff year",
        "",
        "| Category ID | Category Name | Since 2016 | Since 2021 |",
        "| --- | --- | --- | --- |",
    ]
    for s in all_stats:
        if "error" in s:
            lines.append(f"| {s['id']} | {s.get('name', 'ERROR')} | ERR | ERR |")
            continue
        c2016 = cumulative_since(s.get("by_year", {}), 2016)
        c2021 = cumulative_since(s.get("by_year", {}), 2021)
        lines.append(f"| {s['id']} | {s['name']} | {c2016} | {c2021} |")

    ok = [s for s in all_stats if "error" not in s]
    grand_total = sum(s.get("total", 0) for s in ok)
    grand_2016 = sum(cumulative_since(s.get("by_year", {}), 2016) for s in ok)
    grand_2021 = sum(cumulative_since(s.get("by_year", {}), 2021) for s in ok)

    lines += [
        "",
        "## Grand totals across all categories",
        "",
        f"- **Total PDF presentations** ({start_year}–{end_year}): {grand_total}",
        f"- **Since 2016**: {grand_2016}",
        f"- **Since 2021**: {grand_2021}",
        "",
    ]

    output_file.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="Earliest year to include (default: %(default)s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load previously saved JSON and skip already-fetched categories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_token = load_api_token()
    end_year = datetime.now().year
    years = list(range(args.start_year, end_year + 1))

    category_ids = [
        int(line.strip()) for line in CATEGORIES_FILE.read_text(encoding="utf-8").splitlines() if line.strip().isdigit()
    ]
    print(
        f"Found {len(category_ids)} categories; querying {len(years)} years each "
        f"({len(category_ids) * len(years)} API calls total)"
    )

    # Resume support
    all_stats: list[dict] = []
    done_ids: set[int] = set()
    if args.resume and OUTPUT_JSON.exists():
        all_stats = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        done_ids = {s["id"] for s in all_stats}
        print(f"Resuming: {len(done_ids)} categories already fetched")

    for cat_id in category_ids:
        if cat_id in done_ids:
            print(f"  [{cat_id}] skipped (already fetched)")
            continue

        print(f"  [{cat_id}] ", end="", flush=True)

        # Resolve category name from a quick first-page query
        cat_name = search_category_name(cat_id, api_token)

        by_year: dict[str, int] = {}
        error: str | None = None
        for year in years:
            y_start = f"{year}-01-01"
            y_end = f"{year + 1}-01-01"
            try:
                count = search_pdf_count(cat_id, y_start, y_end, api_token)
                by_year[str(year)] = count
                print(".", end="", flush=True)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
                error = str(exc)
                print(f"ERROR({year}:{exc})", end="", flush=True)
                by_year[str(year)] = -1
            time.sleep(0.1)

        total = sum(v for v in by_year.values() if v >= 0)
        print(f" → {cat_name!r}: {total} total")

        entry: dict = {"id": cat_id, "name": cat_name, "total": total, "by_year": by_year}
        if error:
            entry["error"] = error
        all_stats.append(entry)
        OUTPUT_JSON.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")

    write_report(all_stats, OUTPUT_MD, args.start_year, end_year)
    print(f"\nDone. Report → {OUTPUT_MD}")
    print(f"Raw data  → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
