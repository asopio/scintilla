#!/usr/bin/env python3
"""Post-process a DeepSeek-OCR2 grounding-mode dump into the standard
scintilla markdown format (same conventions as the other conversion
outputs for this document family: `# <name>` title, `## Slide N` per
page, `---` between pages, pipe tables, `![...](...)` figures).

Input format: page markers plus located blocks

    <!-- ==================== PAGE N ==================== -->
    <|ref|>TYPE<|/ref|><|det|>[[x1, y1, x2, y2][, [..]]]<|/det|>
    <block content lines>

Block types seen in the dump: title, sub_title, text, table, equation,
image.  The `<|det|>` boxes are page-location metadata; the converted
output is single-column, so the marker lines are dropped and only the
content is kept.

Transformations:

  * page markers          -> `## Slide N` sections, `---` between/after
  * location marker lines -> dropped
  * image blocks          -> `![Figure k](<relpath>)`, k from the
                             already-extracted crop
                             `figures/page-NN-fKK.png`; crops are matched
                             to the det boxes per page (identity, then a
                             90-deg transform for the re-OCRed rotated
                             slides of the legacy Evian manifest, then
                             document order as fallback)
  * HTML `<table>`        -> pipe table (first row = header,
                             `| --- |` separator, pipes escaped)
  * equation blocks       -> `$$ ... $$` display math
  * inline math `\\( \\)` -> `$ ... $`
  * content headings      -> demoted one level, floor `###`
                             (slide headings occupy `##`)
  * `•` bullets           -> `-`
  * blank pages           -> `&nbsp;`

Usage:

    python3 dsocr2_postprocess.py [input.md] [-o OUTPUT]
                                  [--figures-dir DIR] [--manifest FILE]

With no arguments the Evian1992LHC sample defaults are used (paths are
relative to the repo root, i.e. this script's parent directory).

``postprocess()`` is importable: ``pdf_to_markdown_dsocr2.py`` calls it
to finish its pipeline.
"""

import argparse
import html as htmllib
import json
import os
import re
import sys

PAGE_RE = re.compile(r"^<!--\s*=+\s*PAGE (\d+)\s*=+\s*-->")
MARKER_RE = re.compile(r"^<\|ref\|>(\w+)<\|/ref\|><\|det\|>(.*?)<\|/det\|>\s*$")
BOX_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
FIG_INDEX_RE = re.compile(r"page-\d+-f(\d+)\.")


def parse_boxes(payload: str) -> list[tuple[int, int, int, int]]:
    return [tuple(map(int, m.groups())) for m in BOX_RE.finditer(payload)]


def rot90cw(box: tuple[int, int, int, int], size: int = 1000):
    """Landscape-model box -> portrait-model box (90 deg clockwise).

    The rotated slides (ROTATE_270 re-OCR) keep their detection boxes in
    the landscape frame; in the portrait dump the same box appears as
    (1000-y2, x1, 1000-y1, x2).
    """
    x1, y1, x2, y2 = box
    return (size - y2, x1, size - y1, x2)


def boxes_close(a, b, tol: int = 10) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def match_images(md_boxes, entries):
    """Map each image-block box to a manifest entry.

    ``entries`` is a list of ``(box_model, file)`` in figure order.
    Returns ``(mapping, note)`` where ``mapping[i]`` is the file for the
    i-th image block, or ``None`` with a warning note for the order
    fallback.
    """
    if len(md_boxes) != len(entries):
        return None, (f"image count mismatch: {len(md_boxes)} block(s) in "
                      f"markdown vs {len(entries)} extracted figure(s)")
    for transform in (lambda b: b, rot90cw):
        used = [False] * len(entries)
        mapping = []
        for mb in md_boxes:
            for j, (ebox, _) in enumerate(entries):
                if not used[j] and boxes_close(mb, transform(ebox)):
                    used[j] = True
                    mapping.append(entries[j][1])
                    break
            else:
                mapping = None
                break
        if mapping is not None:
            return mapping, None
    order = [f for _, f in entries]
    return order, "no box match (identity or rotated); used document order"


def parse_blocks(lines: list[str]) -> list[tuple[str, list, list[str]]]:
    """Split a page's lines into (type, boxes, content) blocks."""
    blocks = []
    i = 0
    while i < len(lines):
        m = MARKER_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        btype, boxes = m.group(1), parse_boxes(m.group(2))
        j = i + 1
        content = []
        while j < len(lines) and not MARKER_RE.match(lines[j].strip()):
            content.append(lines[j])
            j += 1
        while content and not content[0].strip():
            content.pop(0)
        while content and not content[-1].strip():
            content.pop()
        blocks.append((btype, boxes, content))
        i = j
    return blocks


def convert_math(line: str) -> str:
    """Convert ``\\( \\)`` / ``\\[ \\]`` delimiters to ``$`` / ``$$``.

    Tolerates degenerate model output seen on dense arXiv pages:

    * nested ``\\( ... \\( ... \\) ... \\)`` — innermost pairs are
      converted first (looping to fixpoint), and any ``$`` markers left
      over from an inner conversion are stripped before the outer span
      is wrapped, so the result stays well-formed;
    * unpaired delimiters (the model emitted them around plain prose) —
      whatever is left over after pairing is unescaped to literal
      characters;
    * an unclosed ``\\[`` (generation truncated inside a display block)
      — the block is closed at the end of the line.
    """
    # Inline math: innermost pairs first.  The guard unit is
    # (negative-lookahead + ONE character) repeated, so the span cannot
    # cross a nested delimiter.
    while True:
        new = re.sub(
            r"\\\((?:(?!(\\\(|\\\))).)*?\\\)",
            lambda m: "$" + m.group(0)[2:-2].strip().replace("$", "") + "$",
            line)
        if new == line:
            break
        line = new
    # Display math: closed pairs, innermost first.
    while True:
        new = re.sub(
            r"\\\[(?:(?!(\\\[|\\\))).)*?\\\]",
            lambda m: "$$ " + m.group(0)[2:-2].strip() + " $$",
            line)
        if new == line:
            break
        line = new
    # Display math: unclosed block (truncated generation) -> close it.
    while "\\[" in line:
        line = re.sub(r"\\\[\s*(.*)$", r"$$ \1 $$", line, count=1)
    # Dangling delimiters that never paired -> literal characters.
    line = line.replace("\\(", "(").replace("\\)", ")")
    line = line.replace("\\[", "[").replace("\\]", "]")
    return line


def demote_heading(line: str) -> str:
    m = re.match(r"^(#{1,6})\s+(.*)$", line)
    if m:
        level = min(len(m.group(1)) + 1, 6)
        level = max(level, 3)  # slide content starts at ###
        return "#" * level + " " + re.sub(r"^•\s*", "", m.group(2))
    return line


def to_bullet(line: str) -> str:
    return re.sub(r"^(\s*)•\s*", r"\1- ", line)


def collapse_spaces(line: str) -> str:
    """Collapse runs of spaces outside math spans."""
    parts = re.split(r"(\$\$.*?\$\$|\$[^$]*\$)", line)
    return "".join(p if p.startswith("$") else re.sub(r" {2,}", " ", p)
                   for p in parts)


def transform_text_line(line: str) -> str:
    line = line.rstrip()
    if not line:
        return ""
    line = demote_heading(line)
    line = to_bullet(line)
    line = convert_math(line)
    line = collapse_spaces(line)
    return line


def html_table_to_md(table_html: str) -> list[str] | None:
    rows = []
    for row in re.findall(r"<tr>(.*?)</tr>", table_html, re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        cells = [convert_math(htmllib.unescape(re.sub(r"\s+", " ", c)).strip())
                 .replace("|", "\\|") for c in cells]
        rows.append(cells)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + " --- |" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return out


def render_page(page: int, blocks, image_files: list[str] | None,
                figures_dir: str, out_dir: str) -> list[str]:
    lines = [f"## Slide {page}", ""]
    if not blocks:
        lines.append("&nbsp;")
        return lines
    img_idx = 0
    for btype, boxes, content in blocks:
        if btype == "image":
            fname = image_files[img_idx] if image_files else None
            img_idx += 1
            if fname is None:
                lines += [f"<!-- image (no extracted figure found): "
                          f"{boxes[0] if boxes else '?'} -->", ""]
                continue
            m = FIG_INDEX_RE.search(fname)
            k = int(m.group(1)) if m else img_idx
            rel = os.path.relpath(os.path.join(figures_dir, fname),
                                  out_dir).replace(os.sep, "/")
            lines += [f"![Figure {k}]({rel})", ""]
        elif btype == "table":
            tbl = html_table_to_md("\n".join(content))
            if tbl is None:
                lines += [f"<!-- unparseable table -->", ""]
                continue
            lines += tbl + [""]
        elif btype == "equation":
            for cl in content:
                t = convert_math(cl.strip())
                if t:
                    lines.append(t)
            lines.append("")
        else:  # title, sub_title, text
            for cl in content:
                t = transform_text_line(cl)
                if t:
                    lines.append(t)
                else:
                    lines.append("")
            lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return lines


def postprocess(input_path: str, figures_dir: str, manifest_path: str,
                output_path: str) -> dict:
    """Convert a raw grounding dump to the standard markdown format.

    Writes ``output_path`` (creating its parent directory) and returns
    ``{"output", "slides", "images", "tables", "equations",
    "problems"}``; ``problems`` is a list of validation error strings
    (empty on success).  The file is written even when validation
    fails, so the caller can inspect it.

    Raises ``ValueError`` if the dump has no PAGE markers.
    """
    figures_dir = os.path.abspath(figures_dir)
    out_dir = os.path.dirname(os.path.abspath(output_path))

    lines = open(input_path, encoding="utf-8").read().splitlines()
    preamble: list[str] = []
    pages: dict[int, list[str]] = {}
    cur = None
    for ln in lines:
        m = PAGE_RE.match(ln.strip())
        if m:
            cur = int(m.group(1))
            pages[cur] = []
        elif cur is None:
            preamble.append(ln)
        else:
            pages[cur].append(ln)
    if not pages:
        raise ValueError("no PAGE markers found")

    manifest = json.load(open(manifest_path, encoding="utf-8"))
    man_by_page: dict[int, list] = {}
    for e in sorted(manifest, key=lambda e: (e["page"], e["file"])):
        man_by_page.setdefault(e["page"], []).append(
            (tuple(e["box_model"]), os.path.basename(e["file"])))

    n_img = n_tbl = n_eq = 0
    chunks = []
    for page in sorted(pages):
        blocks = parse_blocks(pages[page])
        image_files = None
        img_blocks = [b for b in blocks if b[0] == "image"]
        if img_blocks:
            md_boxes = [b[1][0] for b in img_blocks if b[1]]
            entries = man_by_page.get(page, [])
            if len(md_boxes) == len(entries):
                mapping, note = match_images(md_boxes, entries)
                if note:
                    print(f"warning: page {page}: {note}", file=sys.stderr)
                image_files = mapping
            else:
                print(f"warning: page {page}: {len(md_boxes)} image block(s) "
                      f"vs {len(entries)} extracted figure(s); "
                      f"using document order", file=sys.stderr)
                image_files = [f for _, f in entries][:len(img_blocks)]
            n_img += len(img_blocks)
        n_tbl += sum(1 for b in blocks if b[0] == "table")
        n_eq += sum(1 for b in blocks if b[0] == "equation")
        chunks.append("\n".join(
            render_page(page, blocks, image_files, figures_dir, out_dir)))

    doc = ("\n".join(preamble).rstrip("\n") + "\n\n"
           + "\n\n---\n\n".join(chunks) + "\n\n---\n")

    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc)

    # ---- validation ----
    problems = []
    if "<|ref|>" in doc or "<|det|>" in doc:
        problems.append("leftover <|ref|>/<|det|> block markers")
    if re.search(r"\\\(|\\\[", doc):
        problems.append("leftover \\( / \\[ math delimiters")
    if "<table>" in doc:
        problems.append("leftover HTML table")
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", doc):
        p = os.path.normpath(os.path.join(out_dir, m.group(1)))
        if not os.path.isfile(p):
            problems.append(f"missing figure file: {m.group(1)}")

    return {
        "output": output_path,
        "slides": len(chunks),
        "images": n_img,
        "tables": n_tbl,
        "equations": n_eq,
        "problems": problems,
    }


def main() -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?",
                    default=os.path.join(repo_root, "Evian1992LHC_sample_dsocr2.md"))
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    input_path = args.input
    figures_dir = os.path.abspath(
        args.figures_dir or os.path.join(repo_root, "example_docs/evian/figures"))
    manifest_path = args.manifest or os.path.join(
        os.path.dirname(figures_dir), "figures_manifest.json")
    output_path = args.output or os.path.join(
        os.path.dirname(figures_dir), "md",
        os.path.splitext(os.path.basename(input_path))[0] + ".md")

    if not os.path.isfile(input_path):
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1
    try:
        result = postprocess(input_path, figures_dir, manifest_path, output_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result["problems"]:
        print("PROBLEMS:", file=sys.stderr)
        for p in result["problems"]:
            print("  -", p, file=sys.stderr)
        return 1

    print(f"wrote {result['output']}")
    print(f"  slides: {result['slides']}  images: {result['images']}  "
          f"tables: {result['tables']}  equations: {result['equations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
