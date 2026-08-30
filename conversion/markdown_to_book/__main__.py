#!/usr/bin/env python3
"""Run: preprocess → conversion → HTML/EPUB/zipped HTML output."""

import argparse
from pathlib import Path
from markdown_to_book.preprocess import preprocess, default_title_from_path
from markdown_to_book.convert import convert

DEFAULT_OUTPUT_DIR = Path("output_html")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess markdown and convert it to standalone HTML, EPUB, or zipped HTML."
    )
    parser.add_argument("md", help="Path to the input markdown file")
    parser.add_argument(
        "-o", "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory (default: output_html)",
    )
    parser.add_argument(
        "--format",
        choices=["html", "epub", "zip"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--doc-backend",
        choices=["pandoc", "markdown-it"],
        default=None,
        help="Document rendering backend (default depends on --format)",
    )
    parser.add_argument(
        "--math-backend",
        choices=["mathjax", "katex"],
        default=None,
        help="Math rendering backend (default depends on --doc-backend)",
    )
    parser.add_argument(
        "--clean-output",
        default=None,
        help="Path to write the preprocessed markdown (default: <input>_clean.md)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Book title (default: derived from input filename)",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="Author name",
    )
    parser.add_argument(
        "--cover",
        default=None,
        help="Path to cover image, absolute or relative to the input markdown",
    )
    parser.add_argument(
        "--alt-text-treatment",
        choices=["clean", "keep"],
        default="clean",
        help="How to handle image alt text in cleaned markdown (default: clean)",
    )
    parser.add_argument(
        "--equation-number-tags",
        choices=["clean", "keep"],
        default="keep",
        help="How to handle LaTeX \\tag{...} equation numbers in cleaned markdown (default: clean)",
    )
    args = parser.parse_args()

    md_in = Path(args.md).resolve()
    out_dir = Path(args.output_dir).resolve()
    cleaned_md = Path(args.clean_output).resolve() if args.clean_output else md_in.parent / f"{md_in.stem}_clean.md"
    title = args.title or default_title_from_path(md_in)

    print(f"[1/2] Preprocessing {md_in.name}")
    raw = md_in.read_text(encoding="utf-8")
    cleaned = preprocess(
        raw,
        title=title,
        author=args.author,
        alt_text_treatment=args.alt_text_treatment,
        equation_number_tags=args.equation_number_tags,
    )
    cleaned_md.write_text(cleaned, encoding="utf-8")
    print(f"       → {cleaned_md.name} ({len(cleaned.splitlines())} lines)\n")

    backend_parts = []
    if args.doc_backend:
        backend_parts.append(args.doc_backend)
    if args.math_backend:
        backend_parts.append(args.math_backend)
    backend_label = ", ".join(backend_parts) if backend_parts else "auto"
    print(f"[2/2] Converting to {args.format.upper()} ({backend_label})")
    convert(
        str(cleaned_md),
        str(out_dir),
        output_format=args.format,
        doc_backend=args.doc_backend,
        math_backend=args.math_backend,
        title=title,
        author=args.author,
        cover_image=args.cover,
    )


if __name__ == "__main__":
    main()
