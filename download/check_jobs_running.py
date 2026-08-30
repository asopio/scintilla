#!/usr/bin/env python3
"""Monitor markdown conversion jobs with rich tables and ASCII time-series charts."""

import argparse
import math
import socket
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import asciichartpy
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIRS = [
    # "./higp_meetings",
    # "./higg_meetings",
    "./jetm_meetings",
    "./ichep_2026",
    # "./egam_meetings",
]

EMAIL = "asopio@nersc.gov"
CHECKPOINT_DIR = Path("/tmp")
STALE_THRESHOLD_SECONDS = 600  # 10 minutes

# ── Helpers ────────────────────────────────────────────────────────────────────

console = Console()


def checkpoint_path(output_dir: Path) -> Path:
    safe = output_dir.resolve().as_posix().replace("/", "_")
    return CHECKPOINT_DIR / f"markdown_count_checkpoint{safe}.txt"


def count_files(directory: Path, glob: str) -> int:
    return sum(1 for f in directory.rglob(glob) if f.is_file())


def md_counts_by_period(output_dir: Path, bucket: str = "hour") -> tuple[dict[str, int], str]:
    """Return ({period_key: count}, bucket) bucketed by each file's mtime.

    bucket is one of "hour" (default), "minute", or "second".
    The range shown is the most recent N buckets, where N = min(300, chart_width),
    so each bucket maps 1:1 to one terminal character with no downsampling.
    The right edge is always "now".
    """
    fmt_map = {
        "hour": ("%Y-%m-%d %H:00:00", timedelta(hours=1)),
        "minute": ("%Y-%m-%d %H:%M:00", timedelta(minutes=1)),
        "second": ("%Y-%m-%d %H:%M:%S", timedelta(seconds=1)),
    }
    fmt, step = fmt_map[bucket]

    mtimes: list[datetime] = []
    for md_file in output_dir.rglob("*.md"):
        if md_file.is_file():
            mtimes.append(datetime.fromtimestamp(md_file.stat().st_mtime))

    if not mtimes:
        return {}, bucket

    buckets: dict[str, int] = {}
    for t in mtimes:
        key = t.strftime(fmt)
        buckets[key] = buckets.get(key, 0) + 1

    # Truncate to the chosen bucket boundary
    def truncate(dt: datetime) -> datetime:
        if bucket == "hour":
            return dt.replace(minute=0, second=0, microsecond=0)
        if bucket == "minute":
            return dt.replace(second=0, microsecond=0)
        return dt.replace(microsecond=0)

    first_dt = truncate(min(mtimes))
    last_dt = truncate(datetime.now())

    # Cap the window to min(300, chart_width) buckets so every bucket maps
    # 1:1 to a terminal character — no downsampling needed in the chart.
    chart_width = max(40, (console.width or 120) - 16)
    max_bins = min(300, chart_width)
    start_dt = max(first_dt, last_dt - max_bins * step)

    result: dict[str, int] = {}
    cur = start_dt
    while cur <= last_dt:
        key = cur.strftime(fmt)
        result[key] = buckets.get(key, 0)
        cur += step

    return result, bucket


def send_email(subject: str, body: str) -> None:
    try:
        subprocess.run(
            ["mailx", "-s", subject, EMAIL],
            input=body.encode(),
            check=True,
            capture_output=True,
        )
        console.print(f"  [green]✉  Notified {EMAIL}[/green]")
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print(f"  [yellow]⚠  mailx unavailable — would have sent: '{subject}' to {EMAIL}[/yellow]")


def check_and_update_checkpoint(cp: Path, current_count: int, dir_name: str, hostname: str) -> None:
    if cp.exists():
        prev = int(cp.read_text().strip())
        time_diff = datetime.now().timestamp() - cp.stat().st_mtime
        if current_count > prev:
            cp.write_text(str(current_count))
        elif current_count == prev and time_diff >= STALE_THRESHOLD_SECONDS:
            send_email(
                "Markdown Conversion Job Stopped",
                f"Markdown conversion job is no longer running on {hostname} in {dir_name}",
            )
    else:
        cp.write_text(str(current_count))


# ── Display ────────────────────────────────────────────────────────────────────


def progress_bar(pct: float, width: int = 10) -> str:
    filled = int(pct / 100 * width)
    color = "green" if pct >= 90 else "yellow" if pct >= 50 else "red"
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] [{color}]{pct:.1f}%[/{color}]"


def render_table(stats: list[tuple[str, int, int, float, str]], hostname: str) -> None:
    console.print()
    console.rule("[bold cyan]Statistics[/bold cyan]")

    total_md = sum(s[1] for s in stats)
    total_pdf = sum(s[2] for s in stats)
    total_pct = total_md / total_pdf * 100 if total_pdf > 0 else 0.0

    table = Table(
        title=(
            f"Markdown Conversion Status  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  [cyan]{hostname}[/cyan]"
        ),
        box=box.ROUNDED,
        header_style="bold white on dark_blue",
        border_style="bright_blue",
        show_footer=True,
        padding=(0, 1),
    )
    table.add_column("Category", style="bold cyan", footer="[bold]Total[/bold]")
    table.add_column("MD Files", justify="right", style="green", footer=f"[bold]{total_md:,}[/bold]")
    table.add_column("PDF Files", justify="right", style="blue", footer=f"[bold]{total_pdf:,}[/bold]")
    table.add_column("Converted", justify="left", footer=progress_bar(total_pct))
    table.add_column("Status", justify="center", footer="")

    for cat, md_count, pdf_count, pct, status in stats:
        table.add_row(
            cat,
            f"{md_count:,}",
            f"{pdf_count:,}",
            progress_bar(pct),
            status,
        )

    console.print(table)


def render_charts(category_monthly: dict[str, tuple[dict[str, int], str]]) -> None:
    console.print()
    console.rule("[bold cyan]Markdown Conversion Rate over Time (by file mtime)[/bold cyan]")

    # Reserve space for y-axis labels (~12 chars) and panel borders (~4 chars)
    max(40, (console.width or 120) - 16)

    for cat, (monthly, granularity) in category_monthly.items():
        if not monthly:
            continue

        periods = sorted(monthly.keys())
        values = [float(monthly[p]) for p in periods]

        if not any(v > 0 for v in values):
            console.print(f"\n  [dim]{cat}: no converted files yet[/dim]")
            continue

        # Each bucket maps 1:1 to one character — no downsampling needed.
        fmt = (
            "%Y-%m-%d %H:00:00"
            if granularity == "hour"
            else "%Y-%m-%d %H:%M:00"
            if granularity == "minute"
            else "%Y-%m-%d %H:%M:%S"
        )

        chart = asciichartpy.plot(values, {"height": 10})

        # X-axis: ticks labelled as negative day offset from now (-N … 0).
        # Compute how many terminal characters each day occupies after
        # downsampling, then choose the smallest tick interval (in whole days)
        # that keeps labels from overlapping (allowing ~4 chars per label).
        Y_AXIS_WIDTH = 10
        now_dt = datetime.now()
        span_days = (datetime.strptime(periods[-1], fmt) - datetime.strptime(periods[0], fmt)).total_seconds() / 86400
        chars_per_day = len(periods) / span_days if span_days > 0 else len(periods)
        tick_every = max(1, math.ceil(4 / chars_per_day))

        x_axis_chars = [" "] * len(periods)
        prev_ticked: int | None = None
        for i, p in enumerate(periods):
            dt = datetime.strptime(p, fmt)
            day_num = -int((now_dt - dt).total_seconds() / 86400)
            if day_num % tick_every == 0 and day_num != prev_ticked:
                label = str(day_num)
                for j, ch in enumerate(label):
                    if i + j < len(x_axis_chars):
                        x_axis_chars[i + j] = ch
                prev_ticked = day_num

        x_axis = " " * Y_AXIS_WIDTH + "".join(x_axis_chars)

        console.print()
        console.print(
            Panel(
                f"{chart}\n[dim]{x_axis}[/dim]",
                title=f"[bold magenta]{cat}[/bold magenta]",
                subtitle=(
                    f"[dim]{periods[0]} → {periods[-1]}  "
                    f"({len(monthly)} {granularity} buckets, "
                    f"{int(sum(monthly.values())):,} MD files total)[/dim]"
                ),
                border_style="magenta",
                padding=(0, 1),
            )
        )


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor markdown conversion jobs.")
    parser.add_argument(
        "--bucket",
        "-b",
        choices=["hour", "minute", "second"],
        default="hour",
        help="Time bucket size for the conversion-rate chart (default: hour).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hostname = socket.gethostname()
    stats: list[tuple[str, int, int, float, str]] = []
    category_monthly: dict[str, tuple[dict[str, int], str]] = {}

    console.print()
    console.rule(f"[bold]Markdown Conversion Monitor  ·  {hostname}[/bold]")

    for dir_str in OUTPUT_DIRS:
        output_dir = Path(dir_str)
        if not output_dir.exists():
            console.print(f"\n[red]✗ Directory not found: {output_dir}[/red]")
            continue

        cat = output_dir.name
        console.print(f"\nChecking [bold cyan]{cat}[/bold cyan]...")

        md_count = count_files(output_dir, "*.md")
        pdf_count = count_files(output_dir, "*.pdf")
        pct = md_count / pdf_count * 100 if pdf_count > 0 else 0.0
        complete = md_count == pdf_count and md_count > 0

        cp = checkpoint_path(output_dir)
        check_and_update_checkpoint(cp, md_count, dir_str, hostname)

        if complete:
            send_email(
                "Markdown Conversion Complete",
                f"Markdown conversion complete on {hostname}: {md_count} files in {dir_str}",
            )

        status = "[bold green]✔ Complete[/bold green]" if complete else "[yellow]⏳ In Progress[/yellow]"
        stats.append((cat, md_count, pdf_count, pct, status))

        console.print(
            f"  {datetime.now().strftime('%H:%M:%S')}  "
            f"[green]{md_count:,}[/green] MD  /  [blue]{pdf_count:,}[/blue] PDF"
        )

        category_monthly[cat] = md_counts_by_period(output_dir, args.bucket)

    render_table(stats, hostname)
    render_charts(category_monthly)
    console.print()


if __name__ == "__main__":
    main()
