#!/usr/bin/env python3
"""Preprocess raw PDF-to-markdown output into clean, conversion-ready markdown."""

import argparse
import re
from pathlib import Path

_ROMAN_RE = re.compile(r'^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii|xxiv|xxv|xxvi|xxvii|xxviii|xxix|xxx)$')
_DIGIT_RE = re.compile(r'^\d+$')


def is_orphan_page_number(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    return bool(_ROMAN_RE.match(s)) or (_DIGIT_RE.match(s) and len(s) <= 3)


def default_title_from_path(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return stem or path.stem


def preprocess(
    raw: str,
    *,
    title: str | None = None,
    author: str | None = None,
    alt_text_treatment: str = "clean",
    equation_number_tags: str = "clean",
) -> str:
    if alt_text_treatment not in {"clean", "keep"}:
        raise ValueError(f"Unsupported alt_text_treatment: {alt_text_treatment}")
    if equation_number_tags not in {"clean", "keep"}:
        raise ValueError(f"Unsupported equation_number_tags: {equation_number_tags}")

    blocks = re.split(r'^---+$', raw, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip()]
    if not blocks:
        return ""

    cleaned_blocks: list[str] = []

    for idx, block in enumerate(blocks):
        lines = block.split("\n")

        if idx == 0:
            if lines and re.match(r'^#\s+.*\.pdf', lines[0]):
                lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
            if lines and re.match(r'^#\s+', lines[0]):
                lines.pop(0)
                while lines and not lines[0].strip():
                    lines.pop(0)
                if lines and lines[0].strip() and not lines[0].strip().startswith('!') and not lines[0].strip().startswith('#'):
                    lines.pop(0)

        lines = _clean_block(
            lines,
            False,
            alt_text_treatment=alt_text_treatment,
            equation_number_tags=equation_number_tags,
        )
        cleaned_blocks.append("\n".join(lines))

    result = "\n\n<div class=\"pagebreak\"></div>\n\n".join(cleaned_blocks)

    prefix_parts: list[str] = []
    if title:
        prefix_parts.append(f"# {title}")
    if author:
        prefix_parts.append(author)
    if prefix_parts:
        result = "\n\n".join(prefix_parts) + "\n\n---\n\n" + result

    result = re.sub(r'\n{2,}-+\n{2,}\n{2,}\n', '\n\n---\n\n', result)
    return _normalize_ws(result)


def _clean_block(
    lines: list[str],
    is_first: bool,
    *,
    alt_text_treatment: str,
    equation_number_tags: str,
) -> list[str]:
    result: list[str] = []
    i = 0
    n = len(lines)

    if is_first and lines and re.match(r'^#\s+.*\.pdf', lines[0]):
        i = 1

    while i < n:
        line = lines[i]
        s = line.strip()

        if alt_text_treatment == "clean" and s.startswith("![") and "](" in s and s.endswith(")"):
            line = re.sub(r'^!\[[^\]]*\]\(', '![](', line)
            s = line.strip()

        if equation_number_tags == "clean":
            single_line_tag_match = re.fullmatch(r'\$\$\s*(.*?)\s*\\tag\{([^}]+)\}\s*\$\$', s)
            if single_line_tag_match:
                expr, eq_no = single_line_tag_match.groups()
                expr = re.sub(r'\s*,?\s*$', '', expr.rstrip())
                result.append("$$")
                result.append(expr)
                result.append("$$")
                result.append("")
                result.append(f"({eq_no})")
                i += 1
                continue

            if s == "$$":
                j = i + 1
                while j < n and lines[j].strip() != "$$":
                    j += 1
                if j < n:
                    inner_lines = lines[i + 1:j]
                    tag_value = None
                    for inner_line in inner_lines:
                        match = re.search(r'\\tag\{([^}]+)\}', inner_line)
                        if match and tag_value is None:
                            tag_value = match.group(1)
                    if tag_value is not None:
                        cleaned_inner_lines = [
                            re.sub(r'\s*,?\s*\\tag\{[^}]+\}', '', inner_line).rstrip()
                            for inner_line in inner_lines
                        ]
                        result.append(line)
                        result.extend(cleaned_inner_lines)
                        result.append(lines[j])
                        result.append("")
                        result.append(f"({tag_value})")
                        i = j + 1
                        continue

        if s.startswith("<!-- Figure:"):
            content = s.replace("<!-- Figure: ", "").rstrip(" -->")
            result.append(f"*Figure: {content}*\n")
            i += 1
            continue

        if s in ("<!-- Table: start -->", "<!-- Table: end -->"):
            i += 1
            continue

        if is_orphan_page_number(line):
            if not result:
                i += 1
                continue
            if i == n - 1:
                i += 1
                continue

        result.append(line)
        i += 1

    return result


def _normalize_ws(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n'


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw PDF-to-markdown output into clean, conversion-ready markdown."
    )
    parser.add_argument("input", help="Path to the input markdown file")
    parser.add_argument("output", nargs="?", help="Path to the output markdown file")
    parser.add_argument("--title", default=None, help="Book title to prepend")
    parser.add_argument("--author", default=None, help="Author name to prepend")
    parser.add_argument(
        "--alt-text-treatment",
        choices=["clean", "keep"],
        default="clean",
        help="How to handle image alt text in cleaned markdown (default: clean)",
    )
    parser.add_argument(
        "--equation-number-tags",
        choices=["clean", "keep"],
        default="clean",
        help="How to handle LaTeX \\tag{...} equation numbers in cleaned markdown (default: clean)",
    )
    args = parser.parse_args()

    inp = Path(args.input).resolve()
    outp = Path(args.output).resolve() if args.output else inp.parent / f"{inp.stem}_clean.md"
    title = args.title or default_title_from_path(inp)

    raw = inp.read_text(encoding="utf-8")
    cleaned = preprocess(
        raw,
        title=title,
        author=args.author,
        alt_text_treatment=args.alt_text_treatment,
        equation_number_tags=args.equation_number_tags,
    )
    outp.write_text(cleaned, encoding="utf-8")

    print(f"Preprocessed {inp.name} -> {outp.name} ({len(cleaned.splitlines())} lines)")


if __name__ == "__main__":
    main()
