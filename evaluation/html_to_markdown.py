#!/usr/bin/env python3
"""Build ground-truth markdown for the arXiv testing set.

Converts the LaTeXML (ar5iv-style) HTML of each paper under
``html/<id>/`` into a markdown file that mimics the output conventions of
our PDF->markdown converter (see ``conversion/AGENTS.md``):

* top-level ``# <id>.pdf`` header (same synthetic header the pipeline adds)
* ``##`` for the paper title, ``###`` for sections, ``####`` for
  subsections, and so on
* ``$...$`` / ``$$...$$`` LaTeX math, taken from LaTeXML ``alttext``
  attributes (i.e. the original LaTeX source of each equation)
* ``![caption](figures/<id>/<image>)`` for figures; the referenced images
  are copied next to the markdown
* pipe tables (with colspan/rowspan expanded), ``-`` / ``1.`` lists,
  ``> `` quotations, fenced code blocks, footnote syntax where present

Output layout::

    testing_set/md/<id>.md
    testing_set/md/figures/<id>/<image files>
    testing_set/md/manifest.json      (per-paper stats + image list)

Usage (from the repo root or the testing_set directory)::

    pixi exec -- python example_docs/testing_set/html_to_markdown.py
    pixi exec -- python example_docs/testing_set/html_to_markdown.py \
        --only 0707.3088 2412.14046 --workers 8
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

TESTING_SET = Path(__file__).resolve().parent
HTML_DIR = TESTING_SET / "html"
OUT_DIR = TESTING_SET / "md"

BULLET_TEX = {"\\bullet", "*", "\\@", "\\ast", "-"}
BULLET_TEXT = {"", "\u2022", "\u2219", "\u00b7", "-", "\u25e6", "\u2023"}

STATS_KEYS = (
    "math_inline",
    "math_block",
    "figures",
    "images",
    "tables",
    "list_items",
    "bibitems",
    "headings",
    "paragraphs",
)


class PaperContext:
    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        self.stats = {k: 0 for k in STATS_KEYS}
        self.images: set[str] = set()
        self.footnotes: list[list[str]] = []  # [mark, text] in document order
        self.src_dir: Path | None = None
        # extracted inline-SVG figures: filename -> SVG markup
        self.svg_files: dict[str, str] = {}

    def count(self, key: str, n: int = 1) -> None:
        self.stats[key] += n


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------


def classes(node: Tag) -> set[str]:
    return set(node.get("class") or [])


def math_to_tex(node: Tag) -> str:
    """Original LaTeX source of a LaTeXML math element."""
    tex = node.get("alttext")
    if tex is None:
        ann = node.find("annotation")
        tex = ann.get_text() if ann else node.get_text()
    tex = tex.strip()
    tex = tex.replace("\u200b", "")
    # Invisible layout glue that never shows up in the rendered page:
    #   \allowbreak   -> \penalty 10000\  (line-break point, renders as a space)
    #   \penalty <n>  -> break glue
    #   \kern <l>pt   -> sub-point spacing correction (e.g. in "TeV")
    #   \relax        -> no-op
    tex = re.sub(r"\\penalty\s*10000\\ ", " ", tex)
    tex = re.sub(r"\\penalty\s*-?\d+", "", tex)
    tex = re.sub(r"\\kern\s*-?\d*\.?\d+pt", "", tex)
    tex = re.sub(r"\\relax\b", "", tex)
    # Some authors nest $...$ inside a math element (e.g. \mbox{$\;$(stat.)}
    # or "100 GeV $\leq$ ...").  The delimiters never appear in the rendered
    # page, and inside our $...$ output they would break the markdown, so
    # they are dropped.  (No alttext in the corpus uses \$ for a literal $,
    # and all counts are even, so this is lossless here.)
    tex = tex.replace("$", "")
    # Normalise line breaks in the source to single spaces (safe for TeX:
    # row separators in matrix/array environments are '\\' or '&', not newlines).
    tex = re.sub(r"[ \t]*\n[ \t]*", " ", tex).strip()
    # A trailing "\ " (TeX space command) at the end of the alttext would
    # leave a lone backslash that markdown reads as an escaped "$".
    while len(tex) >= 2 and tex[-1] == "\\" and tex[-2] != "\\":
        tex = tex[:-1]
    return tex


def inline_parts(node, emph: bool = True) -> list[str]:
    """Flatten a node to a list of raw text parts (whitespace preserved).

    Math becomes ``$...$``, emphasis markers are inserted tight (no spaces
    added, so the surrounding HTML whitespace is authoritative), ``<br>``
    becomes a newline, and images (ORCID icons, logos) are dropped.
    Inside ``<sup>``/``<sub>`` emphasis is suppressed and the text is
    attached tight to the base, mirroring the rendered page (e.g. ``fb-1``).
    """
    out: list[str] = []
    if isinstance(node, NavigableString):
        # a literal "$" in text (e.g. in a description of Pythia cards)
        # would otherwise be read as an inline-math delimiter
        out.append(str(node).replace("$", "\\$"))
    elif isinstance(node, Tag):
        name = node.name
        cl = classes(node)
        if name == "math":
            tex = math_to_tex(node)
            if tex:
                out.append(f"${tex}$")
        elif name == "br":
            out.append("\n")
        elif name == "img":
            pass  # ORCID icons / small logos: not part of the document text
        elif "ltx_phantom" in cl or "visibility:hidden" in (node.get("style") or ""):
            pass  # \phantom{...}: invisible alignment spacer, not shown on the page
        elif name in ("sup", "sub"):
            t = norm_ws("".join(inline_parts_children(node, emph=False)))
            if t:
                out.append(t)
        elif emph and (name in ("em", "i") or "ltx_font_italic" in cl):
            t = norm_ws("".join(inline_parts_children(node, emph)))
            if t:
                out.append(f"*{t}*")
        elif emph and (name in ("strong", "b") or "ltx_font_bold" in cl):
            t = norm_ws("".join(inline_parts_children(node, emph)))
            if t:
                out.append(f"**{t}**")
        elif emph and "ltx_font_typewriter" in cl:
            t = norm_ws("".join(inline_parts_children(node, emph)))
            if t:
                out.append(f"`{t}`")
        else:
            out.extend(inline_parts_children(node, emph))
    return out


def inline_parts_children(node: Tag, emph: bool = True) -> list[str]:
    return _inline_parts_from(node.children, emph)


def _inline_parts_from(children, emph: bool = True) -> list[str]:
    """Flatten children, merging adjacent inline math elements.

    LaTeXML sometimes emits neighbouring ``<math>`` elements with no
    whitespace between them (e.g. ``pp`` as two upright-p groups).  Left
    as-is that would render as ``$a$$b$``, which markdown parsers read as
    the start of a display equation, so the fragments are joined into a
    single ``$ab$``.
    """
    parts: list[str] = []
    last_math_idx = None
    for child in children:
        if (isinstance(child, Tag) and child.name == "math"
                and child.get("display") != "block"):
            tex = math_to_tex(child)
            if not tex:
                last_math_idx = None
                continue
            seg = f"${tex}$"
            if (last_math_idx is not None
                    and parts[last_math_idx].count("$") == 2
                    and seg.count("$") == 2):
                parts[last_math_idx] = parts[last_math_idx][:-1] + seg[1:]
            else:
                parts.append(seg)
                last_math_idx = (len(parts) - 1
                                 if seg.count("$") == 2 else None)
            continue
        last_math_idx = None
        parts.extend(inline_parts(child, emph))
    return parts


def norm_ws(s: str) -> str:
    """Collapse all whitespace runs (incl. zero-width spaces) to single spaces."""
    return re.sub(r"[ \t\r\n\u200b]+", " ", s).strip()


def cell_norm(s: str) -> str:
    """Normalise table-cell text: keep line breaks as <br>."""
    lines = [norm_ws(line) for line in s.split("\n")]
    lines = [line for line in lines if line]
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# block renderers
# ---------------------------------------------------------------------------


def render_math(tag: Tag, ctx: PaperContext) -> list[str]:
    tex = math_to_tex(tag)
    if not tex:
        return []
    if tag.get("display") == "block":
        ctx.count("math_block")
        tex = re.sub(r"^\\displaystyle\s*", "", tex)
        return [f"$$ {tex} $$"]
    ctx.count("math_inline")
    return [f"${tex}$"]


def render_equation_group(tag: Tag, ctx: PaperContext) -> list[str]:
    """Merge the rows of an equation group into one ``$$ a \\ b $$`` block."""
    parts: list[str] = []
    for tr in tag.find_all("tr"):
        row = []
        for m in tr.find_all("math"):
            t = re.sub(r"^\\displaystyle\s*", "", math_to_tex(m))
            if t:
                row.append(t)
        if row:
            parts.append(" ".join(row))
    if not parts:
        return []
    ctx.count("math_block")
    return [f"$$ {' \\\\ '.join(parts)} $$"]


def render_heading(tag: Tag, ctx: PaperContext) -> list[str]:
    cl = classes(tag)
    if "ltx_title_abstract" in cl:
        return []  # label is emitted by the ltx_abstract wrapper
    text = norm_ws("".join(inline_parts_children(tag)))
    if not text:
        return []
    ctx.count("headings")
    if "ltx_theorem" in cl or "ltx_title_theorem" in cl:
        return [f"**{text}**"]
    if tag.name == "h1":
        level = 2 if "ltx_title_document" in cl else 3
    elif tag.name == "h2":
        level = 3
    elif tag.name == "h3":
        level = 4
    else:
        level = min(int(tag.name[1]) + 1, 6)
    return [f"{'#' * level} {text}"]


def find_email_in_tex(tex_text: str, name: str) -> str | None:
    """First e-mail address within ~400 chars after the author name in .tex."""
    if not tex_text:
        return None
    i = tex_text.find(name)
    if i == -1 and " " in name:
        i = tex_text.find(name.split()[-1])
    if i == -1:
        return None
    m = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", tex_text[i:i + 400])
    return m.group(0) if m else None


def render_authors(tag: Tag, ctx: PaperContext) -> list[str]:
    names = []
    creators = tag.find_all("span", class_="ltx_creator")
    for creator in (creators if creators else
                    tag.find_all("span", class_="ltx_personname")):
        pn = creator if "ltx_personname" in classes(creator) else \
            creator.find("span", class_="ltx_personname")
        raw = "".join(inline_parts_children(pn if pn else creator))
        lines = [norm_ws(l) for l in raw.split("\n")]
        lines = [l for l in lines if l]
        if not lines:
            continue
        # dangling "E-mail" label: the address was lost in the HTML;
        # recover it from the LaTeX source (name, then first e-mail nearby)
        if re.fullmatch(r"E-mail:?", lines[-1]):
            lines.pop()
            addr = None
            if ctx.src_dir:
                for tf in sorted(ctx.src_dir.glob("*.tex")):
                    addr = find_email_in_tex(
                        tf.read_text(encoding="utf-8", errors="replace"), lines[0])
                    if addr:
                        break
            if addr:
                lines.append(f"E-mail: {addr}")
        # author notes: affiliations / e-mails / author footnotes
        note_span = creator.find("span", class_="ltx_author_notes")
        if note_span is not None:
            parts = []
            for ch in note_span.children:
                if isinstance(ch, Tag) and "ltx_role_email" in classes(ch):
                    email = norm_ws("".join(inline_parts_children(ch)))
                    if email:
                        parts.append(f"Electronic address: {email}")
                else:
                    t = (norm_ws("".join(inline_parts_children(ch)))
                         if isinstance(ch, Tag) else norm_ws(str(ch)))
                    if t:
                        parts.append(t)
            if parts:
                lines.append("(" + "; ".join(parts) + ")")
        t = " ".join(lines)
        if t and (not names or names[-1] != t):
            names.append(t)
    if not names:
        t = norm_ws("".join(inline_parts_children(tag)))
        return [t] if t else []
    return ["; ".join(names)]


def render_abstract(tag: Tag, ctx: PaperContext) -> list[str]:
    blocks = [b for b in (emit_blocks(c, ctx) for c in tag.children) for b in b]
    return ["**Abstract:**"] + blocks


def render_note(tag: Tag) -> list[str]:
    content = tag.find("span", class_="ltx_note_content") or tag
    t = norm_ws("".join(inline_parts_children(content)))
    t = re.sub(r"^[^\w]*", "", t)  # drop leading note marks (dagger etc.)
    return [t] if t else []


def build_grid(rows: list[Tag]) -> list[list[str | None]]:
    """Expand colspan/rowspan cells into a rectangular grid of raw texts.

    Returns a list of rows; each cell is ``None`` for a continuation of a
    rowspan (rendered as empty) or the raw inline text of the cell.
    """
    grid: list[list[str | None]] = []
    active: dict[int, list] = {}  # col -> [rows_remaining, raw_text]

    for tr in rows:
        row: list[str | None] = []
        col = 0
        tds = tr.find_all(["td", "th"], recursive=False)
        for td in tds:
            while col in active:
                row.append(None)
                active[col][0] -= 1
                if active[col][0] <= 0:
                    del active[col]
                col += 1
            colspan = max(1, int(td.get("colspan", 1) or 1))
            rowspan = max(1, int(td.get("rowspan", 1) or 1))
            raw = "".join(inline_parts(td))
            for k in range(colspan):
                row.append(raw if k == 0 else None)
                if rowspan > 1:
                    active[col] = [rowspan - 1, raw]
                col += 1
        while col in active:
            row.append(None)
            active[col][0] -= 1
            if active[col][0] <= 0:
                del active[col]
            col += 1
        grid.append(row)
    return grid


def materialize_span_tables(article: Tag) -> int:
    """Convert span-based tables into real ``table``/``tr``/``td`` markup.

    LaTeXML renders some tables (usually nested in paragraphs or
    transformed boxes) as ``span.ltx_tabular`` with ``span.ltx_tr`` /
    ``span.ltx_td`` children.  Materialising them lets ``render_tabular``
    handle them uniformly.  Wrapper boxes containing nothing but the
    table are unwrapped so the table becomes a direct child of its
    paragraph.
    """
    n = 0
    for st in article.find_all("span", class_="ltx_tabular"):
        if st.find_parent("span", class_="ltx_tabular") is not None:
            continue  # outermost only
        table = st.new_tag("table")
        table["class"] = sorted(classes(st))
        trs = [t for t in st.find_all("span", class_="ltx_tr")
               if t.find_parent("span", class_="ltx_tr") is None]
        for tr in trs:
            trow = st.new_tag("tr")
            for td in tr.find_all("span", class_="ltx_td", recursive=False):
                cell = st.new_tag("th" if "ltx_th" in classes(td) else "td")
                for attr in ("colspan", "rowspan"):
                    if td.get(attr):
                        cell[attr] = td[attr]
                for ch in list(td.children):
                    cell.append(ch.extract())
                trow.append(cell)
            table.append(trow)
        st.replace_with(table)
        # unwrap single-table wrapper boxes (span/div with no other content)
        node = table
        while node.parent is not None and node.parent.name in ("span", "div"):
            w = node.parent
            if any(isinstance(c, Tag) or str(c).strip()
                   for c in w.children if c is not node):
                break
            w.replace_with(node)
        n += 1
    return n


def render_tabular(tag: Tag, ctx: PaperContext) -> str:
    ctx.count("tables")
    rows = tag.find_all("tr")
    grid = build_grid(rows)
    ncols = max((len(r) for r in grid), default=1)
    lines: list[str] = []
    for i, row in enumerate(grid):
        cells = []
        for j in range(ncols):
            raw = row[j] if j < len(row) else None
            if raw is None:
                cells.append("")
            else:
                cell = cell_norm(raw).replace("|", "\\|")
                cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + " --- |" * ncols)
    return "\n".join(lines)


def _svg_is_decorative(sv: Tag) -> bool:
    """True for SVGs that are not figure content.

    Skips table decorations (e.g. diagonal header-split lines), empty
    placeholders, and rows of bare subfigure labels such as
    ``(a) (b) (c) (d)`` or ``a) b) c)``.
    """
    if sv.find_parent(["td", "th"]) is not None:
        return True
    text = norm_ws(sv.get_text(" "))
    if not text:
        return True
    return bool(re.fullmatch(
        r"(?:\([a-z0-9]+\)|[a-z0-9]\))(?:\s*(?:\([a-z0-9]+\)|[a-z0-9]\)))*",
        text))


def render_figure(tag: Tag, ctx: PaperContext, alt_base: str = "") -> list[str]:
    """Render a figure/table float as markdown blocks.

    Emission is phase-based (matching the original converter's output):

    figure float:

    1. every direct nested figure (panel), in document order, recursed
       with the chained caption base
    2. the cell images of each live grid, in document order
    3. the live grids, in document order, each followed by the bold
       line of the caption it claims
    4. the rest of the float in document order: own media in place,
       standalone caption bolds, paragraphs, text; equation tables and
       already-handled nodes (nested figures, grids, claimed/seeded
       captions) are skipped
    5. deferred grids (media-less, unclaimable) as bare tables, last

    table float:

    1. every own grid in document order, each followed by the bold
       line of the caption it claims (grids are hoisted ahead of any
       preceding text)
    2. the rest in document order: own media in place, equation
       tables rendered, nested figures (figure and table alike)
       recursed in place, standalone/dropped captions, text

    Claiming (non-consuming):

      * figure-float own media: one caption takes all; several
        captions take runs of media in document order (a caption
        takes the run before it, else the run after it), and each
        run's media alts come from the caption that took it
      * cell images: the single caption's text when the float has
        exactly one, else ``alt_base`` or ``Figure``
      * grids: the nearest active direct caption by document
        distance (ties prefer the preceding caption)

    A claimed figure caption seeds the media's alts; a claimed table
    caption is printed as a bold line after each claiming grid.  Table
    captions never seed image alts.

    Standalone captions: in a figure float a caption that is neither
    seeded nor claimed is printed as a bold line only when the float
    contains no media at all (outside table cells); otherwise it is
    dropped.  In a table float an unclaimed caption is printed at its
    position unless it is the last content of a grid-bearing float.

    Standalone bold lines count as figures unless the float renders a
    grid.  Media alts use ``alt_base`` when present, else the claimed
    caption, else ``Figure``.

    Grids whose cells hold SVGs (degraded diagram fragments) are
    skipped entirely inside a figure float; equation tables are
    dropped there but rendered inside table floats.
    """
    cl = classes(tag)
    is_table = "ltx_table" in cl

    def cap_text(fc: Tag) -> str:
        t = norm_ws("".join(inline_parts_children(fc)))
        # LaTeXML double-parenthesises subfigure labels: "((a))" -> "(a)"
        return re.sub(r"\(\(([^)]*)\)\)", r"(\1)", t)

    cap_nodes = tag.find_all("figcaption", recursive=False)
    cap_txt = {id(c): cap_text(c) for c in cap_nodes}
    active = [c for c in cap_nodes if cap_txt[id(c)]]

    # nested subfigures (panels with their own captions)
    nested = [f for f in tag.find_all("figure") if f is not tag]
    direct_nested = [f for f in nested
                     if all(f not in g.descendants
                            for g in nested if g is not f and g not in f.parents)]
    skip = set(id(f) for f in direct_nested)

    def in_nested(node) -> bool:
        return any(node in g.descendants for g in direct_nested)

    def media_ok(m: Tag) -> bool:
        if m.find_parent(["td", "th"]) is not None:
            return False
        if m.name == "img":
            src = m.get("src", "")
            return (bool(src)
                    and not src.startswith(("http", "data:", "ORCID")))
        return not _svg_is_decorative(m)

    own_media = [m for m in tag.find_all(["img", "svg"])
                 if media_ok(m) and not in_nested(m)]
    # top-level grids only: a ltx_tabular nested in another grid's cells
    # is flattened into the parent cell by render_tabular, not emitted
    # on its own
    own_grids = [t for t in tag.find_all("table", class_="ltx_tabular")
                 if not in_nested(t)
                 and t.find_parent("table", class_="ltx_tabular") is None]

    # document positions (preorder over the float)
    order: list = []
    pos: dict[int, int] = {}
    for d in tag.descendants:
        if id(d) not in pos:
            pos[id(d)] = len(order)
            order.append(d)

    def dpos(node) -> int:
        return pos.get(id(node), len(pos))

    def nearest_cap(node) -> Tag | None:
        """Nearest active direct caption by distance; ties -> preceding."""
        best, best_key = None, None
        for c in active:
            d = abs(dpos(node) - dpos(c))
            key = (d, 0 if dpos(c) < dpos(node) else 1)
            if best_key is None or key < best_key:
                best, best_key = c, key
        return best

    # figure floats: one caption claims all own media; several captions
    # claim runs of own media (preceding run preferred) — the original
    # converter's pairing, which keeps a diagram's media with the
    # caption that belongs to it in two-figure layout floats
    own_media_claim: dict[int, int] = {}
    if not is_table and active:
        if len(active) == 1:
            for m in own_media:
                own_media_claim[id(m)] = id(active[0])
        else:
            order = sorted(own_media + active, key=dpos)
            segs: list[list] = []
            caps_order: list = []
            cur: list = []
            for n in order:
                if n in own_media:
                    cur.append(n)
                else:
                    segs.append(cur)
                    cur = []
                    caps_order.append(n)
            segs.append(cur)
            seg_cap = [None] * len(segs)
            for j, c in enumerate(caps_order):
                if segs[j] and seg_cap[j] is None:
                    seg_cap[j] = id(c)
                elif (j + 1 < len(segs) and segs[j + 1]
                        and seg_cap[j + 1] is None):
                    seg_cap[j + 1] = id(c)
            for i, seg in enumerate(segs):
                if seg_cap[i] is not None:
                    for m in seg:
                        own_media_claim[id(m)] = seg_cap[i]

    def m_alt(m: Tag) -> str:
        if is_table or alt_base or not active:
            return alt_base or "Figure"
        cid = own_media_claim.get(id(m))
        return cap_txt[cid] if cid is not None else "Figure"

    media_alt: dict[int, str] = {id(m): m_alt(m) for m in own_media}

    # grid classification (figure floats only; table floats: all live)
    live_grids: list[Tag] = []
    deferred_grids: list[Tag] = []
    if is_table:
        live_grids = list(own_grids)
    else:
        for g in own_grids:
            if any(s.find_parent(["td", "th"]) is not None
                   for s in g.find_all("svg")):
                continue  # cells hold degraded diagram SVGs: skip entirely
            if g.find(["img", "svg"]) is None and nearest_cap(g) is None:
                deferred_grids.append(g)
            else:
                live_grids.append(g)
    for g in live_grids:
        for im in g.find_all("img"):
            if im.find_parent(["td", "th"]) is not None:
                # cell images take the single caption's text when the
                # float has exactly one, else the alt-text base
                if not is_table and not alt_base and len(active) == 1:
                    media_alt[id(im)] = cap_txt[id(active[0])]
                else:
                    media_alt[id(im)] = alt_base or "Figure"

    # claims: captions seeded by media / claimed by grids
    seeded: set[int] = set()
    grid_claimed: set[int] = set()
    grid_cap: dict[int, int] = {}
    if not is_table:
        for m in own_media:
            nc = nearest_cap(m)
            if nc is not None:
                seeded.add(id(nc))
        for g in live_grids:
            for im in g.find_all("img"):
                if im.find_parent(["td", "th"]) is not None:
                    nc = nearest_cap(im)
                    if nc is not None:
                        seeded.add(id(nc))
    for g in live_grids:
        nc = nearest_cap(g)
        if nc is not None:
            grid_claimed.add(id(nc))
            grid_cap[id(g)] = id(nc)

    # figure-float caption-drop test: any media anywhere in the float
    # (incl. nested, decorative) outside table cells
    any_media_old = False
    for m in tag.find_all(["img", "svg"]):
        if m.find_parent(["td", "th"]) is not None:
            continue
        if m.name == "svg":
            any_media_old = True
            break
        src = m.get("src", "")
        if src and not src.startswith(("http", "data:", "ORCID")):
            any_media_old = True
            break

    def content_follows(node) -> bool:
        """Non-blank, non-br/figcaption content after the node's subtree."""
        size = 1 + sum(1 for _ in node.descendants)
        for d in order[dpos(node) + size:]:
            if isinstance(d, NavigableString):
                if d.strip():
                    return True
            elif d.name not in ("br", "figcaption"):
                return True
        return False

    # alt base passed to each nested figure (chained captions)
    nested_base: dict[int, str] = {}
    for g in direct_nested:
        gfc = g.find("figcaption", recursive=False)
        fc = cap_text(gfc) if gfc is not None else ""
        if is_table or alt_base or not active:
            base = alt_base
        elif len(active) == 1:
            base = cap_txt[id(active[0])]
        else:
            nc = nearest_cap(g)
            base = cap_txt[id(nc)] if nc is not None else ""
        nested_base[id(g)] = f"{base} {fc}" if base and fc else (fc or base)

    # bold caption lines count as figures unless the float renders a
    # grid
    bold_counts = not own_grids

    blocks: list[str] = []

    def emit_image(im: Tag, alt: str) -> None:
        src = im.get("src", "")
        if not src or src.startswith(("http", "data:")) or src.startswith("ORCID"):
            return
        ctx.count("figures")
        ctx.count("images")
        ctx.images.add(src)
        blocks.append(f"![{alt or 'Figure'}](figures/{ctx.paper_id}/{src})")

    def emit_svg(sv: Tag, alt: str) -> None:
        ctx.count("figures")
        ctx.count("images")
        name = f"svg{len(ctx.svg_files) + 1}.svg"
        ctx.svg_files[name] = str(sv)
        blocks.append(f"![{alt or 'Figure'}](figures/{ctx.paper_id}/{name})")

    def contains_float_content(node: Tag) -> bool:
        if (node.find("img") is not None or node.find("svg") is not None
                or node.find("table", class_="ltx_tabular") is not None):
            return True
        return any(id(f) in skip for f in node.find_all("figure"))

    def emit_grid(g: Tag) -> None:
        blocks.append(render_tabular(g, ctx))
        cid = grid_cap.get(id(g))
        if cid is not None:
            blocks.append(f"**{cap_txt[cid]}**")

    def walk(node) -> None:
        if isinstance(node, NavigableString):
            t = norm_ws(str(node))
            if t:
                blocks.append(t)
            return
        name = node.name
        if name == "figcaption":
            if is_table:
                if id(node) in grid_claimed:
                    return
                if own_grids and not content_follows(node):
                    return
            else:
                if (id(node) in seeded or id(node) in grid_claimed
                        or any_media_old):
                    return
            if bold_counts:
                ctx.count("figures")
            blocks.append(f"**{cap_txt[id(node)]}**")
            return
        if name == "table":
            tcl = classes(node)
            if "ltx_tabular" in tcl:
                return  # emitted by its grid phase (or skipped/deferred)
            if tcl & {"ltx_equation", "ltx_equationgroup", "ltx_eqn_table"}:
                if is_table:
                    blocks.extend(emit_blocks(node, ctx))
                return  # figure float: diagram fragments, no content
            return
        if id(node) in skip:
            if is_table:
                blocks.extend(render_figure(node, ctx,
                                            alt_base=nested_base[id(node)]))
            return  # figure float: emitted by phase 1
        if name == "img":
            if id(node) in media_alt:
                emit_image(node, media_alt[id(node)])
            return
        if name == "svg":
            if id(node) in media_alt:
                emit_svg(node, media_alt[id(node)])
            return
        if name == "p":
            grids_here = [t for t in node.find_all("table",
                                                   class_="ltx_tabular")
                          if not in_nested(t)]
            if grids_here:
                # text fragments around the grid are counted paragraphs;
                # the grid itself was emitted by its grid phase
                gset = set(id(t) for t in grids_here)
                buf: list = []

                def flush_p() -> None:
                    nonlocal buf
                    t = norm_ws("".join(_inline_parts_from(buf, True)))
                    buf = []
                    if t:
                        ctx.count("paragraphs")
                        blocks.append(t)

                for ch in node.children:
                    if isinstance(ch, Tag) and (
                            id(ch) in gset
                            or gset & {id(x) for x in ch.descendants}):
                        flush_p()
                        if id(ch) not in gset:
                            walk(ch)
                    else:
                        buf.append(ch)
                flush_p()
                return
            if any(id(x) in skip for x in node.find_all("figure")):
                for c in node.children:
                    walk(c)
                return
            media = [m for m in node.find_all(["img", "svg"])
                     if media_ok(m) and not in_nested(m)]
            if media:
                for m in media:
                    if m.name == "img":
                        emit_image(m, media_alt[id(m)])
                    else:
                        emit_svg(m, media_alt[id(m)])
                # remaining inline content (e.g. bare sibling text),
                # with the emitted media's subtrees removed
                med_ids = set(id(m) for m in media)
                parts = []
                for ch in node.children:
                    if isinstance(ch, Tag) and (
                            id(ch) in med_ids
                            or med_ids & {id(x) for x in ch.descendants}):
                        continue
                    parts.extend(inline_parts(ch, True))
                t = norm_ws("".join(parts))
                if t:
                    # the old converter counted such residuals only when
                    # the paragraph held no math at all
                    if node.find("math") is None:
                        ctx.count("paragraphs")
                    blocks.append(t)
                return
            blocks.extend(emit_blocks(node, ctx))
            return
        if name in ("div", "section", "article"):
            if classes(node) & {"ltx_abstract", "ltx_authors",
                                "ltx_equationgroup"}:
                blocks.extend(emit_blocks(node, ctx))
                return
            for c in node.children:
                walk(c)
            return
        if name in ("span", "a") and contains_float_content(node):
            for c in node.children:
                walk(c)
            return
        blocks.extend(emit_blocks(node, ctx))

    if is_table:
        for g in sorted(live_grids, key=dpos):
            emit_grid(g)
    else:
        for g in sorted(direct_nested, key=dpos):
            blocks.extend(render_figure(g, ctx,
                                        alt_base=nested_base[id(g)]))
        for g in sorted(live_grids, key=dpos):
            for im in g.find_all("img"):
                if im.find_parent(["td", "th"]) is not None:
                    emit_image(im, media_alt[id(im)])
        for g in sorted(live_grids, key=dpos):
            emit_grid(g)

    for c in tag.children:
        walk(c)

    for g in sorted(deferred_grids, key=dpos):
        blocks.append(render_tabular(g, ctx))

    return blocks


def render_biblist(tag: Tag, ctx: PaperContext) -> list[str]:
    blocks = []
    for li in tag.find_all("li", class_="ltx_bibitem"):
        ctx.count("bibitems")
        parts = []
        for ch in li.children:
            if isinstance(ch, Tag) and "ltx_tag" in classes(ch):
                label = norm_ws("".join(inline_parts_children(ch)))
                if label:
                    parts.append(label)
                continue
            parts.extend(emit_blocks(ch, ctx, in_bib=True))
        text = norm_ws(" ".join(parts))
        if text:
            blocks.append(text)
    return blocks


def render_list(tag: Tag, ctx: PaperContext) -> str:
    """Render ltx_itemize / ltx_enumerate as a single markdown list block."""
    ordered = "ltx_enumerate" in classes(tag)
    items: list[str] = []
    counter = 0
    for li in tag.find_all("li", recursive=False):
        tag_span = None
        content_parts: list[str] = []
        for ch in li.children:
            if isinstance(ch, Tag) and "ltx_tag" in classes(ch):
                tag_span = ch
            else:
                content_parts.extend(emit_blocks(ch, ctx))
        content = [b for b in content_parts if norm_ws(b)]
        if not content:
            continue
        ctx.count("list_items")
        # determine bullet/label
        label = None
        if tag_span is not None:
            m = tag_span.find("math")
            if m is not None:
                tex = math_to_tex(m)
                if tex not in BULLET_TEX:
                    label = f"${tex}$"
            else:
                t = norm_ws("".join(inline_parts_children(tag_span)))
                if t not in BULLET_TEXT:
                    label = t
        if ordered and label is None:
            counter += 1
            marker = f"{counter}."
        elif label is not None:
            marker = f"- {label}"
        else:
            marker = "-"
        first = norm_ws(content[0])
        item_lines = [f"{marker} {first}"]
        for extra in content[1:]:
            extra_lines = extra.split("\n")
            item_lines.append("  " + extra_lines[0])
            item_lines.extend("  " + l for l in extra_lines[1:] if l.strip())
        items.append("\n".join(item_lines))
    return "\n".join(items)


def render_thead(tag: Tag, ctx: PaperContext) -> list[str]:
    text = norm_ws("".join(inline_parts_children(tag)))
    return [f"**{text}**"] if text else []


def emit_blocks(node, ctx: PaperContext, in_bib: bool = False) -> list[str]:
    """Convert one HTML node into a list of markdown blocks (in order)."""
    if isinstance(node, NavigableString):
        t = norm_ws(str(node))
        return [t] if t else []
    if not isinstance(node, Tag):
        return []

    name = node.name
    cl = classes(node)

    # skip list / boilerplate containers
    if cl & {"ltx_toc", "ltx_nav", "ltx_page_footer", "ltx_page_header"}:
        return []

    if name == "math":
        return render_math(node, ctx)

    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return render_heading(node, ctx)

    if name in ("html", "body", "article", "section", "div"):
        if "ltx_abstract" in cl:
            return render_abstract(node, ctx)
        if "ltx_authors" in cl:
            return render_authors(node, ctx)
        if "ltx_equationgroup" in cl:
            return render_equation_group(node, ctx)
        if "ltx_equation" in cl or "ltx_eqn_table" in cl:
            out = []
            for m in node.find_all("math"):
                out.extend(render_math(m, ctx))
            return out
        return [b for b in (emit_blocks(c, ctx) for c in node.children) for b in b]

    if name == "blockquote" or "ltx_quote" in cl:
        inner: list[str] = []
        for c in node.children:
            if isinstance(c, Tag):
                inner.extend(emit_blocks(c, ctx))
            else:
                t = norm_ws(str(c))
                if t:
                    inner.append(t)
        if not inner:
            return []
        lines = []
        for b in inner:
            for l in b.split("\n"):
                lines.append(f"> {l}" if l.strip() else ">")
        return ["\n".join(lines)]

    if name == "span" and "ltx_note" in cl:
        return render_note(node)
    if name == "span":
        t = norm_ws("".join(inline_parts_children(node)))
        return [t] if t else []

    if name == "p":
        tables = [ch for ch in node.children
                  if isinstance(ch, Tag) and ch.name == "table"]
        if not tables:
            t = norm_ws("".join(inline_parts_children(node)))
            if t:
                ctx.count("paragraphs")
            return [t] if t else []
        # paragraph wrapping a table (materialized span table): interleave
        out: list[str] = []
        buf: list = []

        def flush() -> None:
            nonlocal buf
            t = norm_ws("".join(_inline_parts_from(buf, True)))
            buf = []
            if t:
                ctx.count("paragraphs")
                out.append(t)

        tset = set(id(t) for t in tables)
        for ch in node.children:
            if isinstance(ch, Tag) and id(ch) in tset:
                flush()
                out.append(render_tabular(ch, ctx))
            else:
                buf.append(ch)
        flush()
        return out

    if name == "pre":
        text = node.get_text()
        # \begin{verbatim} consumes one leading newline in LaTeX
        if text.startswith("\n"):
            text = text[1:]
        text = text.rstrip("\n")
        return [f"```\n{text}\n```"] if text.strip() else []

    if name == "table":
        if "ltx_equationgroup" in cl:
            return render_equation_group(node, ctx)
        if "ltx_equation" in cl or "ltx_eqn_table" in cl:
            out = []
            for m in node.find_all("math"):
                out.extend(render_math(m, ctx))
            return out
        if "ltx_tabular" in cl:
            return [render_tabular(node, ctx)]
        return []

    if name == "figure":
        if "ltx_figure" in cl or "ltx_table" in cl or "ltx_float" in cl:
            return render_figure(node, ctx)
        # unclassed figure (e.g. inside a ltx_transformed wrapper):
        # render its content, it is not a float
        return [b for b in (emit_blocks(c, ctx) for c in node.children) for b in b]

    if name in ("ul", "ol"):
        if "ltx_biblist" in cl or "ltx_bibliography" in cl:
            return render_biblist(node, ctx)
        if "ltx_itemize" in cl or "ltx_enumerate" in cl:
            block = render_list(node, ctx)
            return [block] if block else []
        # plain list of something else: treat as items
        items = [f"- {norm_ws(' '.join(emit_blocks(li, ctx)))}"
                 for li in node.find_all("li", recursive=False)]
        return [b for b in items if b.strip("- ").strip()]

    if name == "li":
        if "ltx_bibitem" in cl:
            parts = []
            for ch in node.children:
                if isinstance(ch, Tag) and "ltx_tag" in classes(ch):
                    label = norm_ws("".join(inline_parts_children(ch)))
                    if label:
                        parts.append(label)
                    continue
                parts.extend(emit_blocks(ch, ctx, in_bib=True))
            t = norm_ws(" ".join(parts))
            return [t] if t else []
        return [b for b in (emit_blocks(c, ctx) for c in node.children) for b in b]

    if name == "figcaption":
        return []

    # fallback: flatten to inline text
    t = norm_ws("".join(inline_parts_children(node)))
    return [t] if t else []


# ---------------------------------------------------------------------------
# per-paper conversion
# ---------------------------------------------------------------------------


def resolve_cites(article: Tag) -> int:
    """Replace unresolved citation keys (``bib:Foo``) with their bib labels.

    Some ar5iv builds leave ``\\cite`` references as raw bib keys.  The PDF
    shows the resolved label, so we look the target bibitem up via the
    ``href`` and substitute its ``ltx_tag`` label.
    """
    fixed = 0
    for a in article.find_all("a", class_="ltx_ref"):
        href = a.get("href", "")
        if not href.startswith("#bib."):
            continue
        target = article.find(id=href[1:])
        if target is None or "ltx_bibitem" not in classes(target):
            continue
        tag = target.find("span", class_="ltx_tag")
        label = norm_ws(tag.get_text()) if tag else ""
        if label and norm_ws(a.get_text()) != label:
            a.clear()
            a.append(label)
            fixed += 1
    return fixed


def extract_footnotes(article: Tag, ctx: PaperContext) -> None:
    """Replace inline footnote spans with ``[^n]`` references.

    The note texts are stored in ``ctx.footnotes`` and appended as
    ``[^n]: text`` definitions at the end of the document.  If a paper
    reuses footnote marks (e.g. per-section numbering), every mark is
    renumbered sequentially so the definitions stay unique.
    """
    notes: list[list] = []  # [mark, span, text]
    for span in article.find_all("span", class_="ltx_role_footnote"):
        mark_el = span.find("sup", class_="ltx_note_mark")
        mark = mark_el.get_text().strip() if mark_el else str(len(notes) + 1)
        content = span.find("span", class_="ltx_note_content") or span
        for ch in content.find_all(recursive=False):
            if isinstance(ch, Tag) and classes(ch) & {"ltx_note_mark", "ltx_tag"}:
                ch.extract()
        text = norm_ws("".join(inline_parts_children(content)))
        notes.append([mark, span, text])
    if len({n[0] for n in notes}) != len(notes):
        # marks are reused (e.g. per-section numbering): renumber all of
        # them sequentially so refs and defs stay in sync
        for i, n in enumerate(notes, 1):
            n[0] = str(i)
    for mark, span, _ in notes:
        span.replace_with(NavigableString(f"[^{mark}]"))
    ctx.footnotes = [[n[0], n[2]] for n in notes]


def convert_paper(paper_id: str, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or OUT_DIR
    src_dir = HTML_DIR / paper_id
    html_path = src_dir / "index.html"
    if not html_path.exists():
        return {"paper": paper_id, "error": "no index.html"}

    ctx = PaperContext(paper_id)
    ctx.src_dir = src_dir
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")
    article = soup.find("article") or soup.body or soup

    materialize_span_tables(article)
    resolve_cites(article)
    extract_footnotes(article, ctx)

    blocks: list[str] = []
    for b in emit_blocks(article, ctx):
        b = b.strip("\n")
        if b.strip():
            blocks.append(b)

    # post-processing: drop empty/whitespace-only blocks, tidy spacing
    cleaned: list[str] = []
    for b in blocks:
        if not b.strip():
            continue
        cleaned.append(re.sub(r"[ \t]+\n", "\n", b).strip())

    # footnote definitions at the end of the document
    for mark, text in ctx.footnotes:
        if text:
            cleaned.append(f"[^{mark}]: {text}")

    md = f"# {paper_id}.pdf\n\n" + "\n\n".join(cleaned) + "\n"
    # collapse 3+ consecutive newlines
    md = re.sub(r"\n{3,}", "\n\n", md)

    out_md = out_dir / f"{paper_id}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    # copy referenced images next to the markdown (preserving subdirs)
    fig_dir = out_dir / "figures" / paper_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    src_root = src_dir.resolve()
    for img in sorted(ctx.images):
        src = (src_root / img).resolve()
        try:
            rel = src.relative_to(src_root)
        except ValueError:
            continue
        if not src.is_file() or ".." in rel.parts:
            continue
        dst = fig_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    # write extracted inline-SVG figures
    for name, markup in ctx.svg_files.items():
        (fig_dir / name).write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n' + markup,
            encoding="utf-8")
        copied += 1

    # validate: every image reference in the md must exist
    refs = set(re.findall(r"figures/([0-9A-Za-z_.-]+)/([A-Za-z0-9_./-]+)", md))
    missing = [(p, f) for p, f in refs
               if p != paper_id or not (fig_dir / f).is_file()]

    return {
        "paper": paper_id,
        "markdown": str(out_md),
        "stats": {**ctx.stats, "images_copied": copied},
        "chars": len(md),
        "missing_images": missing,
    }


def main() -> int:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="*", default=None,
                        help="only convert these arXiv ids")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out_dir

    papers = sorted(d.name for d in HTML_DIR.iterdir() if d.is_dir())
    if args.only:
        wanted = set(args.only)
        papers = [p for p in papers if p in wanted]
        unknown = wanted - set(papers)
        if unknown:
            print(f"warning: unknown paper ids: {sorted(unknown)}", file=sys.stderr)

    print(f"converting {len(papers)} papers -> {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1:
        results = [convert_paper(p, OUT_DIR) for p in papers]
    else:
        # pass out_dir explicitly: with the forkserver start method the
        # children re-import the module and would not see this global
        with Pool(args.workers) as pool:
            results = pool.starmap(convert_paper, [(p, OUT_DIR) for p in papers])

    errors = []
    total = {k: 0 for k in STATS_KEYS}
    for r in results:
        for k, v in r.get("stats", {}).items():
            total[k] = total.get(k, 0) + v
        if "error" in r or r.get("missing_images"):
            errors.append(r)

    manifest = {
        "generated_by": "html_to_markdown.py",
        "source": str(HTML_DIR),
        "papers": len(results),
        "totals": total,
        "issues": [
            {"paper": e["paper"], "error": e.get("error"),
             "missing_images": e.get("missing_images", [])}
            for e in errors
        ],
        "papers_detail": [
            {"paper": r["paper"], "stats": r["stats"], "chars": r["chars"]}
            for r in results
        ],
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"done: {len(results)} markdown files written to {OUT_DIR}")
    print(f"totals: {total}")
    if errors:
        print(f"issues in {len(errors)} papers:")
        for e in errors[:20]:
            print(f"  {e['paper']}: error={e.get('error')} missing={e.get('missing_images')}")
        return 1
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
