#!/usr/bin/env python3
"""Convert preprocessed Markdown to HTML, EPUB, or zipped HTML via configurable backends."""

import argparse
import hashlib
import html
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin


PIXI_PANDOC = (
    Path(__file__).resolve().parent.parent
    / ".pixi"
    / "envs"
    / "default"
    / "bin"
    / "pandoc"
)
PNPM = Path(__file__).resolve().parent.parent / ".pixi" / "envs" / "default" / "bin" / "pnpm"
DEFAULT_BOOK_CSS = Path(__file__).resolve().parent / "default_book.css"
KATEX_DIST_DIR = Path(__file__).resolve().parent.parent / "node_modules" / "katex" / "dist"

HTML_FORMAT = "html"
EPUB_FORMAT = "epub"
ZIP_FORMAT = "zip"
PANDOC_BACKEND = "pandoc"
MARKDOWN_IT_BACKEND = "markdown-it"
MATHJAX_BACKEND = "mathjax"
KATEX_BACKEND = "katex"
KATEX_UNSUPPORTED_MACRO_REPLACEMENTS = {
    r"\textmu": r"\mu",
}


class KatexRenderer:
    def __init__(self) -> None:
        self.cache: dict[tuple[str, bool], str] = {}

    def __call__(self, content: str, options: dict[str, object]) -> str:
        display_mode = bool(options.get("display_mode")) or "\\tag{" in content
        cache_key = (content, display_mode)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        rendered = render_katex_server_side(content, display_mode=display_mode)
        self.cache[cache_key] = rendered
        return rendered


def default_title_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_clean"):
        stem = stem[:-6]
    return stem.replace("_", " ").replace("-", " ").strip() or path.stem


def resolve_cover_image(cover_image: str | None, base_dir: Path) -> Path | None:
    if not cover_image:
        return None
    cover_p = Path(cover_image)
    if not cover_p.is_absolute():
        cover_p = base_dir / cover_p
    cover_p = cover_p.resolve()
    if not cover_p.exists():
        print(f"WARNING: cover image not found: {cover_p}", file=sys.stderr)
        return None
    return cover_p


def resolve_doc_backend(output_format: str, doc_backend: str | None) -> str:
    if doc_backend is not None:
        if output_format == EPUB_FORMAT and doc_backend != PANDOC_BACKEND:
            raise ValueError("EPUB output currently requires --doc-backend pandoc")
        return doc_backend

    if output_format in {HTML_FORMAT, EPUB_FORMAT}:
        return PANDOC_BACKEND
    if output_format == ZIP_FORMAT:
        return MARKDOWN_IT_BACKEND
    raise ValueError(f"Unsupported format: {output_format}")


def resolve_math_backend(doc_backend: str, math_backend: str | None) -> str:
    if math_backend is not None:
        if doc_backend == PANDOC_BACKEND and math_backend != MATHJAX_BACKEND:
            raise ValueError("Pandoc output currently requires --math-backend mathjax")
        return math_backend

    if doc_backend == PANDOC_BACKEND:
        return MATHJAX_BACKEND
    if doc_backend == MARKDOWN_IT_BACKEND:
        return KATEX_BACKEND
    raise ValueError(f"Unsupported doc backend: {doc_backend}")


def pandoc_target_format(output_format: str) -> str:
    if output_format == HTML_FORMAT:
        return "html5"
    if output_format == EPUB_FORMAT:
        return EPUB_FORMAT
    raise ValueError(f"Unsupported pandoc output format: {output_format}")


def output_file_for(output_dir: Path, output_format: str) -> Path:
    if output_format == HTML_FORMAT:
        return output_dir / "index.html"
    if output_format == EPUB_FORMAT:
        return output_dir / "book.epub"
    if output_format == ZIP_FORMAT:
        return output_dir / "book_html.zip"
    raise ValueError(f"Unsupported format: {output_format}")


def output_label_for(output_format: str) -> str:
    return output_format.upper()


def build_pandoc_command(
    md_p: Path,
    output_file: Path,
    *,
    output_format: str,
    math_backend: str,
    title: str,
    author: str | None,
    cover_p: Path | None,
    css_p: Path | None,
) -> list[str]:
    fmt = pandoc_target_format(output_format)
    cmd = [
        str(PIXI_PANDOC),
        str(md_p),
        "--standalone",
        "--from", "markdown+tex_math_dollars+tex_math_single_backslash",
        "--to", fmt,
        "--metadata", f"title={title}",
        "--toc",
        "--toc-depth=3",
        "-o", str(output_file),
    ]

    if author:
        cmd.extend(["--metadata", f"author={author}"])

    if output_format == HTML_FORMAT:
        cmd.extend(["--highlight-style=pygments", "--self-contained"])
        if math_backend == MATHJAX_BACKEND:
            cmd.append("--mathjax")
        else:
            raise ValueError(f"Unsupported pandoc math backend: {math_backend}")
        if css_p:
            cmd.extend(["--css", str(css_p)])
    elif output_format == EPUB_FORMAT:
        if math_backend == MATHJAX_BACKEND:
            cmd.extend(["--webtex"])
        else:
            raise ValueError(f"Unsupported pandoc math backend: {math_backend}")
        if cover_p:
            cmd.extend(["--epub-cover-image", str(cover_p)])

    return cmd


def render_math_mathjax(content: str, options: dict[str, object]) -> str:
    if options.get("display_mode"):
        return f"$$\n{content}\n$$"
    return f"\\({content}\\)"


def normalize_katex_input(content: str) -> str:
    normalized = content
    for old, new in KATEX_UNSUPPORTED_MACRO_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)
    return normalized


def render_katex_server_side(content: str, *, display_mode: bool) -> str:
    if not PNPM.exists():
        raise FileNotFoundError(f"pnpm not found in pixi env: {PNPM}")

    normalized_content = normalize_katex_input(content)

    cmd = [str(PNPM), "exec", "katex", "-F", "htmlAndMathml"]
    if display_mode:
        cmd.append("--display-mode")

    result = subprocess.run(
        cmd,
        input=normalized_content,
        capture_output=True,
        text=True,
        cwd=KATEX_DIST_DIR.parent.parent.parent if KATEX_DIST_DIR.exists() else Path(__file__).resolve().parent.parent,
    )
    if result.returncode != 0:
        raise RuntimeError(f"KaTeX render failed: {result.stderr.strip() or result.stdout.strip()}")

    return result.stdout.strip()


def build_markdown_it(math_backend: str) -> MarkdownIt:
    if math_backend == KATEX_BACKEND:
        math_renderer = KatexRenderer()
    elif math_backend == MATHJAX_BACKEND:
        math_renderer = render_math_mathjax
    else:
        raise ValueError(f"Unsupported markdown-it math backend: {math_backend}")

    return (
        MarkdownIt("commonmark", {"html": True, "linkify": True})
        .enable("table")
        .use(footnote_plugin)
        .use(dollarmath_plugin, renderer=math_renderer)
    )


def bundle_katex_assets(bundle_dir: Path) -> str | None:
    if not KATEX_DIST_DIR.exists():
        print(f"WARNING: KaTeX dist directory not found: {KATEX_DIST_DIR}", file=sys.stderr)
        return None

    katex_bundle_dir = bundle_dir / "katex"
    katex_bundle_dir.mkdir(parents=True, exist_ok=True)

    asset_src = KATEX_DIST_DIR / "katex.min.css"
    if not asset_src.exists():
        print(f"WARNING: KaTeX asset not found: {asset_src}", file=sys.stderr)
        return None
    asset_dest = katex_bundle_dir / "katex.min.css"
    shutil.copy2(asset_src, asset_dest)

    fonts_src = KATEX_DIST_DIR / "fonts"
    fonts_dest = katex_bundle_dir / "fonts"
    if fonts_dest.exists():
        shutil.rmtree(fonts_dest)
    shutil.copytree(fonts_src, fonts_dest)

    return "katex/katex.min.css"


def render_math_assets(
    *,
    math_backend: str,
    bundle_dir: Path,
) -> tuple[str, str]:
    if math_backend == KATEX_BACKEND:
        katex_css_href = bundle_katex_assets(bundle_dir)
        head_html = ""
        if katex_css_href:
            head_html += f'  <link rel="stylesheet" href="{html.escape(katex_css_href, quote=True)}">\n'
        return head_html, ""

    if math_backend == MATHJAX_BACKEND:
        return (
            "  <script>window.MathJax = {tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]}};</script>\n"
            "  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>\n",
            "",
        )

    raise ValueError(f"Unsupported markdown-it math backend: {math_backend}")


def wrap_html_document(
    body_html: str,
    *,
    title: str,
    css_href: str | None,
    extra_head_html: str,
    extra_body_end_html: str,
) -> str:
    css_tag = f'  <link rel="stylesheet" href="{html.escape(css_href, quote=True)}">\n' if css_href else ""
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>{html.escape(title)}</title>\n"
        f"{css_tag}"
        f"{extra_head_html}"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        f"{extra_body_end_html}"
        "</body>\n"
        "</html>\n"
    )


def iter_markdown_asset_paths(markdown_text: str) -> list[Path]:
    matches = re.findall(r'!\[[^\]]*\]\(([^)\s]+(?:\s+"[^"]*")?)\)', markdown_text)
    asset_paths: list[Path] = []
    for match in matches:
        target = match.strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1].strip()
        if ' "' in target:
            target = target.split(' "', 1)[0]
        if target:
            asset_paths.append(Path(target))
    return asset_paths


def copy_markdown_assets(markdown_text: str, *, source_dir: Path, bundle_dir: Path) -> None:
    for asset_rel in iter_markdown_asset_paths(markdown_text):
        if asset_rel.is_absolute():
            asset_src = asset_rel
            asset_dest = bundle_dir / asset_rel.name
        else:
            asset_src = (source_dir / asset_rel).resolve()
            asset_dest = (bundle_dir / asset_rel).resolve()

        try:
            asset_dest.relative_to(bundle_dir.resolve())
        except ValueError:
            print(f"WARNING: skipping asset outside bundle path: {asset_rel}", file=sys.stderr)
            continue

        if not asset_src.exists() or not asset_src.is_file():
            print(f"WARNING: markdown asset not found: {asset_src}", file=sys.stderr)
            continue

        asset_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset_src, asset_dest)


def bundle_id(md_p: Path, math_backend: str) -> str:
    digest = hashlib.sha256(str(md_p).encode("utf-8")).hexdigest()[:8]
    return f"book_html_{math_backend}_{digest}"


def write_zip_html_output(
    md_p: Path,
    output_dir: Path,
    *,
    title: str,
    css_p: Path | None,
    math_backend: str,
) -> Path:
    markdown_text = md_p.read_text(encoding="utf-8")
    renderer = build_markdown_it(math_backend)
    rendered_html = renderer.render(markdown_text)

    bundle_dir = output_dir / bundle_id(md_p, math_backend)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copy_markdown_assets(markdown_text, source_dir=md_p.parent, bundle_dir=bundle_dir)

    css_href = None
    if css_p:
        css_target = bundle_dir / css_p.name
        shutil.copy2(css_p, css_target)
        css_href = css_target.name

    extra_head_html, extra_body_end_html = render_math_assets(
        math_backend=math_backend,
        bundle_dir=bundle_dir,
    )

    html_doc = wrap_html_document(
        rendered_html,
        title=title,
        css_href=css_href,
        extra_head_html=extra_head_html,
        extra_body_end_html=extra_body_end_html,
    )
    index_path = bundle_dir / "index.html"
    index_path.write_text(html_doc, encoding="utf-8")

    zip_path = output_file_for(output_dir, ZIP_FORMAT)
    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(bundle_dir.parent))

    return zip_path


def convert_with_pandoc(
    md_p: Path,
    output_dir: Path,
    *,
    output_format: str,
    math_backend: str,
    title: str,
    author: str | None,
    cover_p: Path | None,
    css_p: Path | None,
) -> Path:
    output_file = output_file_for(output_dir, output_format)
    cmd = build_pandoc_command(
        md_p,
        output_file,
        output_format=output_format,
        math_backend=math_backend,
        title=title,
        author=author,
        cover_p=cover_p,
        css_p=css_p,
    )

    print("Running pandoc...")
    print("  " + " ".join(shlex.quote(part) for part in cmd) + "\n")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=md_p.parent)
    if result.returncode != 0:
        print(f"pandoc stdout:\n{result.stdout}", file=sys.stderr)
        print(f"pandoc stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return output_file


def convert_with_markdown_it(
    md_p: Path,
    output_dir: Path,
    *,
    output_format: str,
    math_backend: str,
    title: str,
    css_p: Path | None,
) -> Path:
    if output_format != ZIP_FORMAT:
        raise ValueError("markdown-it backend currently supports only --format zip")
    return write_zip_html_output(
        md_p,
        output_dir,
        title=title,
        css_p=css_p,
        math_backend=math_backend,
    )


def convert(
    md_path: str,
    output_dir: str,
    *,
    output_format: str = HTML_FORMAT,
    doc_backend: str | None = None,
    math_backend: str | None = None,
    title: str,
    author: str | None = None,
    cover_image: str | None = None,
) -> Path:
    """Produce HTML, EPUB, or zipped HTML output."""
    md_p = Path(md_path).resolve()
    if not md_p.exists():
        print(f"ERROR: markdown not found: {md_p}", file=sys.stderr)
        sys.exit(1)

    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    resolved_doc_backend = resolve_doc_backend(output_format, doc_backend)
    resolved_math_backend = resolve_math_backend(resolved_doc_backend, math_backend)
    cover_p = resolve_cover_image(cover_image, md_p.parent)
    css_p: Path | None = DEFAULT_BOOK_CSS if DEFAULT_BOOK_CSS.exists() else None

    if resolved_doc_backend == PANDOC_BACKEND:
        output_file = convert_with_pandoc(
            md_p,
            out_p,
            output_format=output_format,
            math_backend=resolved_math_backend,
            title=title,
            author=author,
            cover_p=cover_p,
            css_p=css_p,
        )
    elif resolved_doc_backend == MARKDOWN_IT_BACKEND:
        output_file = convert_with_markdown_it(
            md_p,
            out_p,
            output_format=output_format,
            math_backend=resolved_math_backend,
            title=title,
            css_p=css_p,
        )
    else:
        raise ValueError(f"Unsupported doc backend: {resolved_doc_backend}")

    size_kb = output_file.stat().st_size / 1024
    print(f"{output_label_for(output_format)} output: {output_file} ({size_kb:.0f} KB)")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert preprocessed markdown to HTML, EPUB, or zipped HTML."
    )
    parser.add_argument(
        "md",
        help="Path to the preprocessed markdown file (e.g. output_clean.md)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output_html",
        help="Output directory (default: output_html)",
    )
    parser.add_argument(
        "--format",
        choices=[HTML_FORMAT, EPUB_FORMAT, ZIP_FORMAT],
        default=HTML_FORMAT,
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--doc-backend",
        choices=[PANDOC_BACKEND, MARKDOWN_IT_BACKEND],
        default=None,
        help="Document rendering backend (default depends on --format)",
    )
    parser.add_argument(
        "--math-backend",
        choices=[MATHJAX_BACKEND, KATEX_BACKEND],
        default=None,
        help="Math rendering backend (default depends on --doc-backend)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Book title (default: derived from markdown filename)",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="Author name",
    )
    parser.add_argument(
        "--cover",
        default=None,
        help="Path to cover image, absolute or relative to the markdown file",
    )
    args = parser.parse_args()

    md_path = Path(args.md).resolve()
    title = args.title or default_title_from_path(md_path)
    convert(
        str(md_path),
        args.output_dir,
        output_format=args.format,
        doc_backend=args.doc_backend,
        math_backend=args.math_backend,
        title=title,
        author=args.author,
        cover_image=args.cover,
    )


if __name__ == "__main__":
    main()
