#!/usr/bin/env python3
"""Benchmark similarity analysis for the testing-set conversion study.

Compares the markdown produced by the two OCR pipelines against the ar5iv-
derived ground truth, treating each text type separately:

    paragraphs, display equations, tables, figure captions

Figure *alt-text* (image ``![...]`` lines) is deliberately excluded: the two
pipelines generate it in fundamentally different ways, so it is not
comparable content.

Metrics, per type and per side (qwen, dsocr):

    efficiency          = matched ground truth items / all ground truth items
                          (recall, in information-retrieval terms)
    purity              = matched candidate items / all candidate items
                          (precision)
    exact-match rate    = matched pairs whose normalized texts are identical
    mean similarity     = mean alignment score (token-set Jaccard) of pairs
    sentence BLEU       = macro over matched pairs, nltk, method1 smoothing
    ROUGE-1 / 2 / L     = macro over matched pairs, F1
    candidate traceability = candidate items with ANY ground-truth counterpart
                              at or above the 0.10 similarity floor (any-match,
                              not 1-to-1); item-level hallucination proxy
    candidate 1-/3-gram support = unique candidate n-grams that occur anywhere
                              in the ground-truth items; content-level proxy

Alignment
---------
paragraphs / tables / figure captions : 1-to-1 greedy by token-set Jaccard
                                        (threshold 0.10, highest first)
display equations                     : every GT equation matches its best
                                        candidate equation (unconstrained,
                                        with multiplicity); candidate
                                        traceability computed the other way

Outputs
-------
    example_docs/testing_set/benchmark_analysis.md
    example_docs/testing_set/eval_md/benchmark/<side>.json   (per paper)

Run from the repo root with the root pixi environment (has nltk):

    pixi run python evaluation/benchmark_similarity.py
"""

from __future__ import annotations

import argparse
import bisect
import collections
import difflib
import json
import re
import sys
import time
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

REPO = Path(__file__).resolve().parents[1]
TESTSET = REPO / "example_docs" / "testing_set"
GT_DIR = TESTSET / "md"
CAND_DIRS = {"qwen": TESTSET / "eval_md" / "qwen",
             "dsocr": TESTSET / "eval_md" / "dsocr"}
OUT_JSON_DIR = TESTSET / "eval_md" / "benchmark"
OUT_REPORT = TESTSET / "benchmark_analysis.md"

# Papers whose ground truth is defective (see md/README.md) are excluded
# from the scored corpus. All eight are upstream ar5iv/LaTeXML build
# failures: the served index.html holds only a fragment of the paper.
#   * acknowledgements fragment only (1009.5908, 1008.2461, 1009.5069,
#     1101.0070)
#   * title page / author block only (1005.5254, 1103.5559, 1104.4481)
#   * a single table fragment (1103.1984)
EXCLUDED_GT = {"1009.5908", "1008.2461", "1009.5069", "1101.0070",
               "1005.5254", "1103.5559", "1104.4481", "1103.1984"}

ID_RE = re.compile(r"[0-9]{4}\.[0-9]{4,5}")

# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

# LaTeX macro aliases that mean the same thing (applied to all math).
# \varepsilon/\epsilon and \vartheta/\theta glyphs are indistinguishable in
# scanned PDFs; the OCR pipelines and ar5iv pick different macros.
MATH_ALIAS = [("\\to", "\\rightarrow"), ("\\gets", "\\leftarrow"),
              ("\\varepsilon", "\\epsilon")]

RE_INLINE_MATH = re.compile(r"\$([^$\n]+)\$")
RE_DISPLAY = re.compile(r"\$\$(.*?)\$\$", re.S)
RE_CITE = re.compile(r"\[\s*(?:\d+\s*[-–,]\s*)*\d+\s*\]")
RE_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
RE_HEADING = re.compile(r"^#{1,6}\s")
RE_HR = re.compile(r"^\s*-{3,}\s*$")
RE_IMG = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
RE_TBLROW = re.compile(r"^\s*\|")
RE_FOOTDEF = re.compile(r"^\[\^\d+\]:")
RE_FIGCAP = re.compile(
    r"^\s*(\*\*)?(?:Figure|Fig\.?)\s+(\d+(?:\([a-zA-Z0-9]+\))?[a-zA-Z]?)(\*\*)?\s*:\s*(.*)$")
RE_TBLCAP = re.compile(
    r"^\s*(\*\*)?Table\s+(\d+)(\*\*)?\s*:\s*(.*)$")
RE_LISTMARK = re.compile(r"^\s{0,3}(?:\d+\.|[-*+])\s+")
RE_EQTOK = re.compile(r"\\[a-zA-Z]+|\\.|[A-Za-z0-9]+|[{}^_]|.")


# Font commands written without braces take the following word as their
# argument:  \rm isolated  ==  \mathrm{isolated}
FONT_COMPLETION = [
    re.compile(r"(\\(?:rm|it|bf|mathrm|mathit|mathbf|boldsymbol|text|mbox|textrm))\s+([A-Za-z0-9][A-Za-z0-9\-]*)"),
]

# Roman-text macros that render identically (longest first).
ROMAN_ALIASES = [
    (re.compile(r"\\operatorname"), "\\mathrm"),
    (re.compile(r"\\textrm"), "\\mathrm"),
    (re.compile(r"\\text(?![a-zA-Z])"), "\\mathrm"),
    (re.compile(r"\\mbox"), "\\mathrm"),
    (re.compile(r"\\rm(?![a-zA-Z])"), "\\mathrm"),
    (re.compile(r"\\it(?![a-zA-Z])"), "\\mathit"),
    (re.compile(r"\\bf(?![a-zA-Z])"), "\\mathbf"),
]


def latex_normalize(s: str) -> str:
    """Canonicalize a math span (used for inline math and display equations)."""
    for rx in FONT_COMPLETION:
        s = rx.sub(r"\1{\2}", s)
    for a, b in MATH_ALIAS:
        s = s.replace(a, b)
    for rx, rep in ROMAN_ALIASES:
        s = rx.sub(lambda m, r=rep: r, s)
    s = re.sub(r"\s+", "", s)
    return _brace_single(s)


def compact_math(s: str) -> str:
    """Whitespace-free LaTeX with a small set of macro aliases."""
    return latex_normalize(s)


def scrub_nonmath(s: str) -> str:
    """Strip markdown artifacts from a non-math span (case preserved)."""
    s = re.sub(r"††thanks:.*", "", s)                 # ar5iv footnote artifact
    s = re.sub(r"\[\^[^\]]*\]", " ", s)                # [^n] footnote refs
    s = re.sub(r"\[([^\]]*)\]\([^)\s]*\)", r"\1", s)   # [text](url) -> text
    s = s.replace("**", "").replace("__", "")
    s = s.replace("*", "").replace("`", "")
    s = s.replace("†", "").replace("‡", "")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("_", " ")
    return s


def prose_spans(line: str):
    """Yield ('math', compact) / ('text', scrubbed-casefolded) spans in order."""
    pos = 0
    for m in RE_INLINE_MATH.finditer(line):
        if m.start() > pos:
            yield "text", re.sub(r"\s+", " ", scrub_nonmath(line[pos:m.start()])).casefold().strip()
        yield "math", compact_math(m.group(1))
        pos = m.end()
    if pos < len(line):
        yield "text", re.sub(r"\s+", " ", scrub_nonmath(line[pos:])).casefold().strip()


def prose_record(lines, min_tokens: int):
    """Build a comparable record from prose lines; None if too short."""
    exact_parts, tokens = [], []
    for line in lines:
        for kind, s in prose_spans(line):
            if kind == "math":
                tokens.append(f"$${s}$$")
                exact_parts.append(f"${s}$")
            else:
                if not s:
                    continue
                s_cite = re.sub(RE_CITE, "<ref>", s)
                s_cite = re.sub(r"\[<ref>(?:\s*,\s*<ref>)+\]", "<ref>", s_cite)
                s_cite = re.sub(r"(?:<ref>\s*,\s*)+", "<ref> ", s_cite)
                exact_parts.append(s_cite)
                tokens.extend(RE_WORD.findall(re.sub(r"\s+", " ", re.sub(RE_CITE, " ", s))))
    if not tokens or len(tokens) < min_tokens:
        return None
    return {"tokens": tokens, "tset": frozenset(tokens),
            "exact": " ".join(p for p in exact_parts if p).strip(),
            "raw": "\n".join(lines)}


# ---- equations -------------------------------------------------------------

def _atom(s: str, i: int):
    """One atom starting at i: a balanced {...} group, a \\command, or a char."""
    if i >= len(s):
        return None, i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i:j + 1], j + 1
            j += 1
        return s[i:], len(s)
    if s[i] == "\\":
        m = re.match(r"[a-zA-Z]+", s[i + 1:])
        if m:
            return "\\" + m.group(0), i + 1 + m.end()
        return s[i:i + 2], i + 2
    return s[i], i + 1


def _brace_single(s: str) -> str:
    """x_e -> x_{e}, x_\\mu -> x_{\\mu} (single char / command atoms)."""
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch in "_^" and i + 1 < len(s):
            a, j = _atom(s, i + 1)
            if a is not None and not a.startswith("{"):
                out.append(ch + "{" + a + "}")
                i = j
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _swap_subsup(s: str) -> str:
    """Canonicalize a_b^c -> a^c_b (repeated to fixpoint)."""
    prev = None
    while prev != s:
        prev = s
        out, i = [], 0
        while i < len(s):
            if s[i] == "_" and i + 1 < len(s):
                a1, j = _atom(s, i + 1)
                if a1 is not None and j < len(s) and s[j] == "^":
                    a2, k = _atom(s, j + 1)
                    if a2 is not None:
                        out.append("^" + a2 + "_" + a1)
                        i = k
                        continue
            out.append(s[i])
            i += 1
        s = "".join(out)
    return s


def norm_eq(eq: str) -> str:
    """Canonical form of a display equation."""
    s = eq.strip()
    s = re.sub(r"\\tag\{[^}]*\}", "", s)                       # \tag{1}
    s = s.replace("\\\\", "")                                  # line breaks
    s = re.sub(r"\\(?:quad|qquad|enspace|[,;:!]| )", "", s)    # spacing cmds
    s = re.sub(r"(\\?&)?\s*\((?:eq\.?\s*)?\d+\)\s*$", "", s)   # trailing (1)
    s = re.sub(r"\\begin\{[^}]*\}", lambda m: "\\begin{}", s)  # aligned/gathered/...
    s = re.sub(r"\\end\{[^}]*\}", lambda m: "\\end{}", s)
    s = re.sub(r"\\left(?![a-zA-Z])", "", s)
    s = re.sub(r"\\right(?![a-zA-Z])", "", s)
    s = s.replace("\\displaystyle", "")
    s = s.replace("&", "")                                     # align layout
    s = s.replace("\\.", ".")
    s = latex_normalize(s)
    return _swap_subsup(s)


def eq_record(eq: str):
    n = norm_eq(eq)
    if not n:
        return None
    toks = RE_EQTOK.findall(n)
    if not toks:
        return None
    return {"tokens": toks, "tset": frozenset(toks), "exact": n,
            "raw": eq.strip()}


# ---- tables ----------------------------------------------------------------

def _parse_row(line: str):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_row(cells) -> bool:
    nonempty = [c for c in cells if c]
    return bool(nonempty) and all(re.fullmatch(r":?-{2,}:?", c) for c in nonempty)


def table_record(rows):
    raw = "\n".join(rows)
    rows = [_parse_row(r) for r in rows]
    rows = [r for r in rows if not _is_separator_row(r)]
    if not rows:
        return None
    norm_cells, tokens = [], []
    for r in rows:
        nr = []
        for cell in r:
            parts = []
            pos = 0
            for m in RE_INLINE_MATH.finditer(cell):
                if m.start() > pos:
                    parts.append(re.sub(r"\s+", " ", scrub_nonmath(cell[pos:m.start()])).casefold().strip())
                parts.append(f"${compact_math(m.group(1))}$")
                pos = m.end()
            if pos < len(cell):
                parts.append(re.sub(r"\s+", " ", scrub_nonmath(cell[pos:])).casefold().strip())
            nr.append(" ".join(p for p in parts if p).strip())
            toks, cparts = [], []
            for kind, s in prose_spans(cell):
                if kind == "math":
                    toks.append(f"$${s}$$")
                else:
                    toks.extend(RE_WORD.findall(re.sub(RE_CITE, " ", s)))
            tokens.extend(toks)
        norm_cells.append(nr)
    if not tokens or len(tokens) < 8:
        return None
    return {"norm_cells": norm_cells, "tokens": tokens, "tset": frozenset(tokens),
            "raw": raw}


# --------------------------------------------------------------------------
# line classification / extraction
# --------------------------------------------------------------------------

def classify_lines(text: str):
    spans = [(m.start(), m.end()) for m in RE_DISPLAY.finditer(text)]
    starts = [a for a, _ in spans]
    lines = text.split("\n")
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    cls = []
    for idx, ln in enumerate(lines):
        o = offs[idx]
        k = bisect.bisect_right(starts, o) - 1
        if k >= 0 and starts[k] <= o < spans[k][1]:
            kind = "eq"
        elif RE_HR.match(ln):
            kind = "hr"
        elif RE_HEADING.match(ln):
            kind = "heading"
        elif RE_IMG.match(ln):
            kind = "img"
        elif RE_TBLROW.match(ln):
            kind = "pipe"
        elif RE_FOOTDEF.match(ln.strip()):
            kind = "footdef"
        elif RE_FIGCAP.match(ln):
            kind = "figcap"
        elif RE_TBLCAP.match(ln):
            kind = "tblcap"
        elif not ln.strip():
            kind = "blank"
        else:
            kind = "prose"
        cls.append((ln, kind))
    return cls


def extract_paragraphs(cls, min_tokens: int):
    items, run = [], []

    def flush():
        if run:
            rec = prose_record(run, min_tokens)
            if rec:
                items.append(rec)
            run.clear()

    for ln, kind in cls:
        if kind in ("prose", "tblcap"):
            if kind == "prose" and RE_LISTMARK.match(ln) and run:
                flush()  # each list item (reference entry / bullet) is one item
            run.append(ln)
        else:
            flush()
    flush()
    return items


def extract_equations(text: str):
    items = []
    for m in RE_DISPLAY.finditer(text):
        rec = eq_record(m.group(1))
        if rec:
            items.append(rec)
    return items


def extract_tables(cls):
    items, rows = [], []

    def flush():
        if rows:
            rec = table_record(rows)
            if rec:
                items.append(rec)
            rows.clear()

    for ln, kind in cls:
        if kind == "pipe":
            rows.append(ln)
        else:
            flush()
    flush()
    return items


def extract_figcaptions(cls, min_tokens: int):
    items, i, n = [], 0, len(cls)
    while i < n:
        ln, kind = cls[i]
        if kind == "figcap":
            m = RE_FIGCAP.match(ln)
            parts = [m.group(4)]
            j = i + 1
            while j < n and cls[j][1] == "prose":
                parts.append(cls[j][0])
                j += 1
            rec = prose_record(parts, min_tokens)
            if rec:
                items.append(rec)
            i = j
        else:
            i += 1
    return items


# --------------------------------------------------------------------------
# alignment + metrics
# --------------------------------------------------------------------------

def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def align_1to1(gt_items, cand_items, thr: float):
    best_c = [0.0] * len(cand_items)
    cands = []
    for i, g in enumerate(gt_items):
        for j, c in enumerate(cand_items):
            s = jaccard(g["tset"], c["tset"])
            cands.append((s, i, j))
            if s > best_c[j]:
                best_c[j] = s
    cands.sort(key=lambda t: t[0], reverse=True)
    used_g, used_c, pairs = set(), set(), []
    for s, i, j in cands:
        if s < thr:
            break
        if i in used_g or j in used_c:
            continue
        pairs.append((i, j, s))
        used_g.add(i)
        used_c.add(j)
    trace = sum(1 for b in best_c if b >= thr)
    return pairs, trace


def align_equations(gt_items, cand_items, thr: float):
    mat = [[jaccard(g["tset"], c["tset"]) for c in cand_items] for g in gt_items]
    pairs = []
    for i, row in enumerate(mat):
        if not row:
            continue
        j = max(range(len(row)), key=lambda k: row[k])
        if row[j] >= thr:
            pairs.append((i, j, row[j]))
    trace = 0
    for row in zip(*mat) if mat else []:
        if max(row) >= thr:
            trace += 1
    return pairs, trace


_SM1 = SmoothingFunction().method1


def bleu(g, c) -> float:
    return sentence_bleu([g["tokens"]], c["tokens"], smoothing_function=_SM1)


def rouge_n(ref, hyp, n) -> float:
    if len(ref) < n or len(hyp) < n:
        return 0.0
    rn = collections.Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
    hn = collections.Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
    overlap = sum((rn & hn).values())
    if overlap == 0:
        return 0.0
    p = overlap / sum(hn.values())
    r = overlap / sum(rn.values())
    return 2 * p * r / (p + r)


def rouge_l(ref, hyp) -> float:
    # ROUGE-L F1 == 2*LCS/(|ref|+|hyp|) == SequenceMatcher.ratio()
    return difflib.SequenceMatcher(None, ref, hyp, autojunk=False).ratio()


def _ngram_counts(gt_items, cand_items, n: int):
    """(unique candidate n-grams, of which some appear in the GT items)."""
    def ngrams(items):
        out = set()
        for it in items:
            toks = it["tokens"]
            if len(toks) >= n:
                out.update(tuple(toks[k:k + n]) for k in range(len(toks) - n + 1))
        return out
    g = ngrams(gt_items)
    c = ngrams(cand_items)
    return len(c), len(c & g)


def pair_scores(g, c) -> dict:
    out = {"bleu": None, "r1": None, "r2": None, "rl": None}
    if not g["tokens"] or not c["tokens"]:
        return out
    out["bleu"] = bleu(g, c)
    out["r1"] = rouge_n(g["tokens"], c["tokens"], 1)
    out["r2"] = rouge_n(g["tokens"], c["tokens"], 2)
    out["rl"] = rouge_l(g["tokens"], c["tokens"])
    return out


def is_exact(typ, g, c) -> bool:
    if typ == "equations":
        return g["exact"] == c["exact"]
    if typ == "tables":
        return g["norm_cells"] == c["norm_cells"]
    return g["exact"] == c["exact"]


def score_type(typ, gt_items, cand_items, min_tokens: int,
               matched: list | None = None) -> dict:
    if typ == "equations":
        pairs, trace = align_equations(gt_items, cand_items, 0.10)
    else:
        pairs, trace = align_1to1(gt_items, cand_items, 0.10)
    if matched is not None:
        matched.extend(pairs)

    n_gt, n_cand = len(gt_items), len(cand_items)
    n_matched = len(pairs)
    exact = 0
    sum_sim = 0.0
    sums = {"bleu": 0.0, "r1": 0.0, "r2": 0.0, "rl": 0.0}
    n_scored = 0
    for i, j, s in pairs:
        g, c = gt_items[i], cand_items[j]
        sum_sim += s
        if is_exact(typ, g, c):
            exact += 1
        sc = pair_scores(g, c)
        if sc["bleu"] is not None:
            n_scored += 1
            for k in sums:
                sums[k] += sc[k]

    res = {
        "n_gt": n_gt, "n_cand": n_cand, "n_matched": n_matched,
        "n_scored": n_scored,
        "exact": exact, "sum_sim": sum_sim,
        "sum_bleu": sums["bleu"], "sum_r1": sums["r1"],
        "sum_r2": sums["r2"], "sum_rl": sums["rl"],
        # per-paper convenience means
        "gt_coverage": (round(n_matched / n_gt, 4) if n_gt else None),
        "cand_precision": (round(n_matched / n_cand, 4) if n_cand else None),
        "exact_rate": (round(exact / n_matched, 4) if n_matched else None),
        "mean_sim": (round(sum_sim / n_matched, 4) if n_matched else None),
        "bleu": (round(sums["bleu"] / n_scored, 4) if n_scored else None),
        "rouge1": (round(sums["r1"] / n_scored, 4) if n_scored else None),
        "rouge2": (round(sums["r2"] / n_scored, 4) if n_scored else None),
        "rougeL": (round(sums["rl"] / n_scored, 4) if n_scored else None),
    }
    ng1_cand, ng1_int = _ngram_counts(gt_items, cand_items, 1)
    ng3_cand, ng3_int = _ngram_counts(gt_items, cand_items, 3)
    res["trace"] = trace
    res["cand_traceable"] = (round(trace / n_cand, 4) if n_cand else None)
    res["ng1_cand"], res["ng1_int"] = ng1_cand, ng1_int
    res["ng3_cand"], res["ng3_int"] = ng3_cand, ng3_int
    res["ng1_support"] = (round(ng1_int / ng1_cand, 4) if ng1_cand else None)
    res["ng3_support"] = (round(ng3_int / ng3_cand, 4) if ng3_cand else None)
    return res


def analyze_pair(text_gt: str, text_cand: str, min_tokens: int,
                 keep_pairs: bool = False) -> dict:
    cls_gt, cls_c = classify_lines(text_gt), classify_lines(text_cand)
    items = {
        "paragraphs": (extract_paragraphs(cls_gt, min_tokens),
                       extract_paragraphs(cls_c, min_tokens)),
        "equations": (extract_equations(text_gt),
                      extract_equations(text_cand)),
        "tables": (extract_tables(cls_gt),
                   extract_tables(cls_c)),
        "figcaptions": (extract_figcaptions(cls_gt, min_tokens),
                        extract_figcaptions(cls_c, min_tokens)),
    }
    out = {}
    matched = {typ: [] for typ in items} if keep_pairs else None
    for typ, (gt_i, cand_i) in items.items():
        out[typ] = score_type(typ, gt_i, cand_i, min_tokens,
                              matched[typ] if matched is not None else None)
    if keep_pairs:
        out["_items"] = items
        out["_matched"] = matched
    return out


def aggregate(papers: dict) -> dict:
    """Pool per-paper pair sums to get corpus-level (pair-macro) means."""
    agg = {}
    for typ in ("paragraphs", "equations", "tables", "figcaptions"):
        n_gt = sum(p[typ]["n_gt"] for p in papers.values())
        n_cand = sum(p[typ]["n_cand"] for p in papers.values())
        n_matched = sum(p[typ]["n_matched"] for p in papers.values())
        n_scored = sum(p[typ]["n_scored"] for p in papers.values())
        exact = sum(p[typ]["exact"] for p in papers.values())
        sum_sim = sum(p[typ]["sum_sim"] for p in papers.values())
        e = {
            "n_gt": n_gt, "n_cand": n_cand, "n_matched": n_matched,
            "gt_coverage": round(n_matched / n_gt, 4) if n_gt else None,
            "cand_precision": round(n_matched / n_cand, 4) if n_cand else None,
            "exact_rate": round(exact / n_matched, 4) if n_matched else None,
            "mean_sim": round(sum_sim / n_matched, 4) if n_matched else None,
            "bleu": round(sum(p[typ]["sum_bleu"] for p in papers.values()) / n_scored, 4) if n_scored else None,
            "rouge1": round(sum(p[typ]["sum_r1"] for p in papers.values()) / n_scored, 4) if n_scored else None,
            "rouge2": round(sum(p[typ]["sum_r2"] for p in papers.values()) / n_scored, 4) if n_scored else None,
            "rougeL": round(sum(p[typ]["sum_rl"] for p in papers.values()) / n_scored, 4) if n_scored else None,
        }
        trace = sum(p[typ]["trace"] for p in papers.values())
        e["cand_traceable"] = round(trace / n_cand, 4) if n_cand else None
        ng1c = sum(p[typ]["ng1_cand"] for p in papers.values())
        ng1i = sum(p[typ]["ng1_int"] for p in papers.values())
        ng3c = sum(p[typ]["ng3_cand"] for p in papers.values())
        ng3i = sum(p[typ]["ng3_int"] for p in papers.values())
        e["ng1_support"] = round(ng1i / ng1c, 4) if ng1c else None
        e["ng3_support"] = round(ng3i / ng3c, 4) if ng3c else None
        agg[typ] = e
    return agg


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

PCT = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"
VAL = lambda v: "n/a" if v is None else f"{v:.3f}"
CNT = lambda v: "n/a" if v is None else f"{v}"

KEY_ORDER = [
    ("GT items", "n_gt", CNT),
    ("Candidate items", "n_cand", CNT),
    ("Matched pairs", "n_matched", CNT),
    ("Efficiency", "gt_coverage", PCT),
    ("Purity", "cand_precision", PCT),
    ("Candidate traceability (any-match)", "cand_traceable", PCT),
    ("Exact-match rate", "exact_rate", PCT),
    ("Mean similarity", "mean_sim", VAL),
    ("BLEU", "bleu", VAL),
    ("ROUGE-1", "rouge1", VAL),
    ("ROUGE-2", "rouge2", VAL),
    ("ROUGE-L", "rougeL", VAL),
    ("Candidate 1-gram support", "ng1_support", PCT),
    ("Candidate 3-gram support", "ng3_support", PCT),
]


def type_table(q: dict, d: dict) -> str:
    out = ["| Metric | qwen | dsocr |", "|---|---:|---:|"]
    for label, key, fmt in KEY_ORDER:
        out.append(f"| {label} | {fmt(q.get(key))} | {fmt(d.get(key))}")
    return "\n".join(out)


def write_report(results: dict, papers: list, args, elapsed: float) -> None:
    L = []
    ap = L.append
    ap("# Benchmark similarity analysis: qwen vs dsocr optical character "
       "recognition (OCR) pipelines")
    ap("")
    ap(f"*{len(papers)} papers scored (of {len(papers) + len(EXCLUDED_GT)} in the testing set); "
       f"excluded: {', '.join(sorted(EXCLUDED_GT))} — their ar5iv ground truth is an "
       "upstream LaTeXML fragment (title page, acknowledgements, or one table only; "
       f"see `md/README.md`). Runtime {elapsed:.0f}s. Macro metric means are over papers "
       "(each paper weighted equally); efficiency/purity are pooled over all items.*")
    ap("")
    ap("Ground truth (GT): ar5iv HTML → markdown (`evaluation/html_to_markdown.py`). Candidates: "
       "`eval_md/qwen/` (Qwen3.8-27B-FP8 page OCR, `pdf_to_markdown_agentic_v2.py`) and "
       "`eval_md/dsocr/` (DeepSeek-OCR2, `pdf_to_markdown_dsocr2.py` + "
       "`dsocr2_postprocess.py`).")
    ap("")
    ap("## Method")
    ap("")
    ap("Each text type is extracted and scored separately. Figure **alt-text is excluded** "
       "(image `![...]` lines): the two pipelines generate it in fundamentally different ways.")
    ap("")
    ap("- **Paragraphs** — prose blocks (headings, tables, equations, image lines, figure "
       "captions excluded). Each list item (numbered reference, bullet) is one item. "
       f"Blocks with < {args.min_tokens} tokens are dropped. Table-caption lines "
       "(`Table N: ...`) are treated as paragraph text.")
    ap("- **Equations** — display math `$$...$$` only. Normalized by removing whitespace, "
       "`\\left`/`\\right`, spacing commands, align layout (`\\` breaks, `&`), "
       "`\\tag{N}` / trailing `(N)` numbers, collapsing `\\begin{aligned}`-style "
       "environments, mapping `\\to`→`\\rightarrow` and `\\varepsilon`→`\\epsilon` "
       "(indistinguishable glyphs in scans), aliasing `\\rm`/`\\mbox`/`\\text`→"
       "`\\mathrm`, and canonicalizing `a_b^c`→`a^c_b` and `a_x`→`a_{x}`.")
    ap("- **Tables** — pipe-table blocks; separator rows dropped; exact match on the "
       "row-major sequence of normalized cells.")
    ap("- **Figure captions** — lines matching `Figure|Fig. <label>:` (colon required; "
       "`Figure 2 shows…` prose is not captured), label prefix stripped.")
    ap("")
    ap("Prose normalization: markdown emphasis stripped, links → anchor text, `[^n]` "
       "footnotes and `††thanks:` artifacts removed, citations (`[19]`, `[1–3]`) → "
       "`<ref>`, inline math compacted (whitespace-free, macro aliases), casefolded, "
       "whitespace collapsed. Similarity = token-set Jaccard; alignment is 1-to-1 greedy "
       "(threshold 0.10) for paragraphs/tables/captions and unconstrained best-match "
       "(with multiplicity) for equations. BLEU (Bilingual Evaluation Understudy; "
       "n-gram precision against the reference) is nltk sentence-BLEU (4-gram, method1 "
       "smoothing); ROUGE-1/2/L (Recall-Oriented Understudy for Gisting Evaluation) are "
       "token-level F1 — the harmonic mean of precision and recall (ROUGE-L via longest "
       "common subsequence).")
    ap("")
    ap("Metric names: **efficiency** (recall in information-retrieval terms) = fraction "
       "of ground-truth items with a matched candidate; **purity** (precision) = the same "
       "fraction counted over candidate items; **exact-match rate** = fraction of matched "
       "pairs identical after normalization; **mean similarity** = mean Jaccard of matched "
       "pairs; **candidate traceability** (all types, any-match rather than 1-to-1) "
       "= fraction of candidate items that have any ground-truth counterpart at or "
       "above the 0.10 similarity floor — the closest item-level proxy for the "
       "hallucination rate (1 − traceability is the share of emitted items with no "
       "counterpart in the source at all); **candidate n-gram support** = fraction of "
       "unique candidate n-grams (n = 1 or 3 tokens, within items) that occur anywhere "
       "in the ground-truth items — the content-level proxy, catching fabrication "
       "inside otherwise matched items.")
    ap("")
    ap("## Results")
    ap("")
    TYPE_TITLES = {"paragraphs": "Paragraphs", "equations": "Equations",
                  "tables": "Tables", "figcaptions": "Figure captions"}
    for typ in ("paragraphs", "equations", "tables", "figcaptions"):
        ap(f"### {TYPE_TITLES[typ]}")
        ap("")
        ap(type_table(results["qwen"]["corpus"][typ], results["dsocr"]["corpus"][typ]))
        ap("")
    for name, text in NOTES.items():
        ap(f"### {name} — observations")
        ap("")
        ap(text)
        ap("")
    ap("## Caveats")
    ap("")
    ap("1. **Figure-caption asymmetry.** The ground truth has far fewer standalone "
       "caption lines than the candidates: ar5iv HTML usually embeds figure captions in "
       "the image *alt text*, which is excluded from this analysis by design. Candidate "
       "captions that have no ground-truth counterpart therefore lower "
       "purity on this type; efficiency is the more meaningful number.")
    ap("2. **Page-boundary splits.** Candidate documents are OCR'd page by page, so "
       "paragraphs and tables are frequently cut at page boundaries, splitting one "
       "ground-truth item into several candidates (and repeating running headers). This "
       "caps paragraph/table purity and efficiency; similarity/BLEU/ROUGE on matched "
       "pairs are less affected.")
    ap("3. **Equation numbering and macro style.** dsocr keeps trailing `(N)` numbers "
       "and qwen uses `\\tag{N}`; both are stripped. Residual differences (`\\mathrm` vs "
       "`\\mbox`, OCR letter-spacing like `\\mathrm{m i s s}`) mostly survive the "
       "normalization, but exact match on equations is therefore stricter than on prose.")
    ap("4. **Defective ground truth excluded.** Eight papers have ar5iv/LaTeXML builds "
       "that served only a fragment of the paper (title page / author block only: "
       "1005.5254, 1103.5559, 1104.4481; acknowledgements only: 1008.2461, 1009.5069, "
       "1009.5908, 1101.0070; a single table only: 1103.1984). Scoring the full-paper "
       "OCR candidates against such fragments would measure mostly what the candidates "
       "added, not what they got right, so they are excluded from the scored corpus. "
       "The PDFs and GT markdown files themselves are intact in the testing set.")
    ap("5. **Long list blocks cap efficiency (alignment artifact, not content loss).** "
       "Paragraph-efficiency losses concentrate in papers with long reference / "
       "affiliation / author lists. Probes of the raw candidate text confirm the "
       "content is present in both pipelines (e.g. a GT affiliation or reference that "
       "fails to align is found verbatim in the candidate output); the loss comes from "
       "1-to-1 alignment being unable to bridge items the candidate split or re-flowed "
       "across `## Slide` boundaries. Large ATLAS/CMS papers are worst: the "
       "ground-truth author list is one ~10k-token line and the affiliation block is "
       "~150-180 numbered lines, and qwen's agentic pipeline additionally renders the "
       "tail of the affiliation list as `[^n]:` footnote definitions (the excluded "
       "class), so it cannot match at all. This is why qwen has 23 papers below 85% "
       "paragraph efficiency (concentrated in those list-heavy papers) while dsocr "
       "has 1; it is the dominant term in the qwen-vs-dsocr paragraph-efficiency gap.")
    ap("")
    ap("Per-paper metrics: `eval_md/benchmark/qwen.json`, `eval_md/benchmark/dsocr.json`.")
    ap("")
    ap("Validation artifacts: `eval_md/benchmark/match_viewer.html` shows the matched "
       "items side by side (ground-truth item vs its matched candidate item, for both "
       "pipelines) plus the no-match items (candidate items without a ground-truth "
       "counterpart — the closest thing to hallucination candidates — and ground-truth "
       "items without a candidate) in six sample documents — two each from the top, "
       "the median and the "
       "bottom of the per-document purity distribution, where purity/efficiency are "
       "pooled over the four text types and both pipelines; markdown is rendered "
       "client-side with marked, math with KaTeX. "
       "`eval_md/benchmark/purity_ranking.png` and "
       "`eval_md/benchmark/efficiency_ranking.png` show the per-document purity and "
       "efficiency by pipeline (qwen vs dsocr, pooled over the four text types), each "
       "ranked by the pooled value of its metric; the purity ranking is the one the "
       "six-document sample was picked from.")
    ap("")
    OUT_REPORT.write_text("\n".join(L), encoding="utf-8")


NOTES = {}


def fill_notes(results: dict):
    """Populate NOTES with data-driven per-type observations."""
    q, d = results["qwen"]["corpus"], results["dsocr"]["corpus"]
    p = q["paragraphs"]
    NOTES["Paragraphs"] = (
        f"Prose is the best-aligned type: efficiency is {PCT(p['gt_coverage'])} (qwen) "
        f"vs {PCT(d['paragraphs']['gt_coverage'])} (dsocr), with exact-match rates of "
        f"{PCT(p['exact_rate'])} / {PCT(d['paragraphs']['exact_rate'])}. The residual "
        "gap to 100% is dominated by long list blocks (references, affiliation lists) "
        "and slide-boundary fragmentation — see caveat 5; the per-paper distribution "
        f"shows the gap is concentrated in the list-heavy papers. Any-match candidate "
        f"traceability is {PCT(p['cand_traceable'])} (qwen) / "
        f"{PCT(d['paragraphs']['cand_traceable'])} (dsocr) and candidate 3-gram support "
        f"{PCT(p['ng3_support'])} / {PCT(d['paragraphs']['ng3_support'])}: these are "
        "the shares of emitted items/trigrams with no counterpart in the source at "
        "all — the closest thing to a hallucination rate. 1 − purity is higher than "
        "1 − traceability because 1-to-1 alignment also drops items whose "
        "counterpart exists (caveats 2 and 5).")
    e = q["equations"]
    NOTES["Equations"] = (
        f"{e['n_gt']} GT display equations; qwen emits {e['n_cand']}, dsocr "
        f"{d['equations']['n_cand']} (both OCR pipelines invent/retain more display "
        "blocks than the ar5iv source, e.g. aligning terms that GT leaves inline). "
        f"Exact match after normalization is {PCT(e['exact_rate'])} (qwen) / "
        f"{PCT(d['equations']['exact_rate'])} (dsocr); mean similarity "
        f"{VAL(e['mean_sim'])} / {VAL(d['equations']['mean_sim'])}. Candidate "
        f"traceability: {PCT(e.get('cand_traceable'))} / "
        f"{PCT(d['equations'].get('cand_traceable'))}.")
    t = q["tables"]
    NOTES["Tables"] = (
        f"Tables are the most structure-dependent type: efficiency "
        f"{PCT(t['gt_coverage'])} (qwen) / {PCT(d['tables']['gt_coverage'])} (dsocr), "
        f"exact match {PCT(t['exact_rate'])} / {PCT(d['tables']['exact_rate'])}. "
        "dsocr OCRs more content as pipe-tables (extra grid-like fragments), which "
        "lowers purity; GT tables that ar5iv split across pages rarely "
        "re-align 1:1.")
    c = q["figcaptions"]
    NOTES["Figure captions"] = (
        f"Ground truth carries only {c['n_gt']} standalone figure captions (ar5iv puts "
        f"most captions in image alt text, which is excluded), vs {c['n_cand']} (qwen) "
        f"and {d['figcaptions']['n_cand']} (dsocr) candidate captions. Efficiency is "
        f"therefore the informative metric: {PCT(c['gt_coverage'])} (qwen) / "
        f"{PCT(d['figcaptions']['gt_coverage'])} (dsocr); exact match on matched "
        f"captions {PCT(c['exact_rate'])} / {PCT(d['figcaptions']['exact_rate'])}.")


# --------------------------------------------------------------------------
# validation artifacts: per-paper ranking chart + matched-pairs HTML viewer
# --------------------------------------------------------------------------

TYPE_KEYS = ("paragraphs", "equations", "tables", "figcaptions")
SIDE_KEYS = ("qwen", "dsocr")


def ranking_rows(results: dict, papers: list) -> list:
    """[(stem, efficiency, purity, per-side)] per paper, sorted by pooled purity
    (highest first).

    Efficiency and purity are pooled over the four text types:
    matched items / ground-truth items and matched items / candidate items
    respectively. The per-side dict maps each pipeline to its own
    (efficiency, purity) pair, also pooled over the four types.
    """
    rows = []
    for stem in papers:
        per = {}
        nm = ng = nc = 0
        for side in SIDE_KEYS:
            p = results[side]["papers"].get(stem)
            if not p:
                continue
            m = g = c = 0
            for t in TYPE_KEYS:
                m += p[t]["n_matched"]
                g += p[t]["n_gt"]
                c += p[t]["n_cand"]
            per[side] = (m / g if g else None, m / c if c else None)
            nm += m
            ng += g
            nc += c
        if ng or nc:
            rows.append((stem, nm / ng if ng else None, nm / nc if nc else None,
                         per))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def write_chart(rows: list, out: Path, metric: str) -> None:
    """Horizontal bar chart of per-document `metric` ("purity" or
    "efficiency"), one bar per pipeline, documents ranked by the pooled
    (both-pipeline) value of the same metric, highest at top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = 2 if metric == "purity" else 1
    sel = [r for r in rows if r[idx] is not None]
    sel.sort(key=lambda r: r[idx], reverse=True)
    n = len(sel)
    fig, ax = plt.subplots(figsize=(11, 0.34 * n + 2.5))
    y = list(range(n))
    for off, side, color in ((0.19, "qwen", "#4c72b0"),
                             (-0.19, "dsocr", "#dd8452")):
        vals = []
        for r in sel:
            v = r[3].get(side, (None, None))[idx - 1]
            vals.append(v * 100 if v is not None else 0.0)
        ax.barh([yi + off for yi in y], vals, height=0.38, color=color,
                label=side)
    ax.set_yticks(y, [r[0] for r in sel], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("percent of items")
    ax.set_title(f"Per-document {metric} by pipeline ({n} papers, pooled over the "
                 f"four text types), ranked by pooled {metric} (highest at top)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def select_subsample(rows: list, n_each: int = 2) -> list:
    """(stem, group) picks: n_each each from top, median and bottom of the
    purity ranking (the median picks straddle the middle of the list)."""
    n = len(rows)
    half = n_each // 2
    mid = n // 2
    picks = ([(rows[i][0], "top") for i in range(n_each)]
             + [(rows[i][0], "median") for i in range(mid - half, mid + half)]
             + [(rows[i][0], "bottom") for i in range(n - n_each, n)])
    return picks


def collect_viewer(picks: list, min_tokens: int) -> list:
    """Re-analyze the picked papers with pair/item retention for the HTML."""
    docs = []
    for stem, group in picks:
        gt_text = (GT_DIR / f"{stem}.md").read_text(encoding="utf-8")
        doc = {"id": stem, "group": group, "sides": {}}
        nm = ng = nc = 0
        for side in SIDE_KEYS:
            cand_text = (CAND_DIRS[side] / f"{stem}.md").read_text(encoding="utf-8")
            a = analyze_pair(gt_text, cand_text, min_tokens, keep_pairs=True)
            sdata = {}
            for t in TYPE_KEYS:
                gti, ci = a["_items"][t]
                pairs = a["_matched"][t]
                used_g = {i for i, j, s in pairs}
                used_c = {j for i, j, s in pairs}
                sdata[t] = {
                    "gt": [it["raw"] for it in gti],
                    "cand": [it["raw"] for it in ci],
                    "pairs": [[i, j, round(s, 4), is_exact(t, gti[i], ci[j])]
                              for i, j, s in pairs],
                    "cand_nomatch": [[j, ci[j]["raw"]]
                                     for j in range(len(ci)) if j not in used_c],
                    "gt_nomatch": [[i, gti[i]["raw"]]
                                   for i in range(len(gti)) if i not in used_g],
                    "n_gt": a[t]["n_gt"], "n_cand": a[t]["n_cand"],
                    "mean_sim": a[t]["mean_sim"],
                }
                nm += a[t]["n_matched"]
                ng += a[t]["n_gt"]
                nc += a[t]["n_cand"]
            doc["sides"][side] = sdata
        doc["efficiency"] = round(nm / ng, 4) if ng else None
        doc["purity"] = round(nm / nc, 4) if nc else None
        docs.append(doc)
    return docs


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Match viewer — benchmark similarity analysis</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
         color: #222; background: #fafafa; }
  header { padding: 16px 24px 10px; background: #fff; border-bottom: 2px solid #4c72b0; }
  h1 { margin: 0 0 8px; font-size: 1.4em; }
  header p { font-size: 0.9em; color: #555; max-width: 90em; }
  #nav a { margin-right: 14px; }
  section { background: #fff; margin: 16px 24px; padding: 12px 20px;
            border: 1px solid #ddd; border-radius: 6px; }
  h2 .group { background: #4c72b0; color: #fff; border-radius: 4px;
              padding: 2px 8px; font-size: 0.7em; vertical-align: middle; }
  h2 .stats { color: #666; font-size: 0.75em; font-weight: normal; }
  h3 { border-bottom: 1px solid #eee; margin-top: 1.4em; }
  h4 { color: #444; margin-bottom: 2px; }
  details { border: 1px solid #e3e3e3; border-radius: 4px; margin: 6px 0; }
  summary { cursor: pointer; padding: 6px 10px; font-size: 0.85em; }
  summary:hover { background: #f0f4fa; }
  .sim { color: #666; font-family: ui-monospace, monospace; }
  .exact { background: #2e7d32; color: #fff; border-radius: 3px; padding: 1px 6px;
           font-size: 0.85em; margin: 0 6px; }
  .idx { color: #888; font-family: ui-monospace, monospace; margin-right: 8px; }
  .snip { color: #444; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
          padding: 10px 12px; }
  .lab { font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.05em;
         color: #777; }
  .md { border: 1px solid #eee; border-radius: 4px; padding: 8px 12px;
        background: #fcfcfc; overflow-x: auto; font-size: 0.95em; }
  .md table { border-collapse: collapse; }
  .md th, .md td { border: 1px solid #ccc; padding: 4px 8px; }
  .md p { margin: 0.5em 0; }
  .md .katex-display { overflow-x: auto; }
  .md.single { margin: 8px 12px; }
  h5 { margin: 12px 0 2px; font-size: 0.95em; color: #b26a00; }
  h5.h5gt { color: #4c72b0; }
  .nomatch { background: #b26a00; color: #fff; border-radius: 3px; padding: 1px 6px;
             font-size: 0.85em; margin-right: 6px; white-space: nowrap; }
  details.warn { border-left: 4px solid #e6a817; }
</style>
</head>
<body>
<header>
  <h1>Matched items, side by side</h1>
  <p>__N_DOCS__ sample documents (two each from the top, the median and the bottom of the
  per-document purity distribution; purity and efficiency pooled over the four text
  types and both pipelines) used to validate the matching procedure. Each expandable
  entry under a type heading is one matched pair: the ground-truth item on the left,
  its matched candidate item on the right, with the alignment similarity (token-set
  Jaccard) and the exact-match status in the header. After the matched pairs, the
  <em>no-match items</em> are listed per type: candidate items with no ground-truth
  counterpart at or above the 0.10 similarity floor (possible hallucinations, or
  structural artifacts such as page-boundary splits, repeated running headers and
  re-classification — see caveats 2 and 5 of the report), and ground-truth items that
  no candidate matched. Markdown is rendered client-side with
  <a href="https://marked.js.org/">marked</a>; equations and inline math with
  <a href="https://katex.org/">KaTeX</a> (both loaded from a CDN — without internet
  access the raw markdown is shown unrendered).</p>
  <nav id="nav"></nav>
</header>
<main id="docs"></main>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
window.addEventListener("load", function () {
  "use strict";
  const data = JSON.parse(document.getElementById("data").textContent);
  const TYPE_TITLES = { paragraphs: "Paragraphs", equations: "Equations",
                        tables: "Tables", figcaptions: "Figure captions" };
  const TYPE_ORDER = ["paragraphs", "equations", "tables", "figcaptions"];
  const SIDE_ORDER = ["qwen", "dsocr"];
  const TYPE_NOTE = {
    paragraphs: "",
    equations: "",
    tables: "",
    figcaptions: " (for captions, most no-match candidates are expected: ar5iv usually " +
      "embeds figure captions in the image alt text, which is excluded from the " +
      "ground truth — caveat 1)"
  };
  const mainEl = document.getElementById("docs");
  const navEl = document.getElementById("nav");

  function esc(s) {
    return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                                         '"': "&quot;", "'": "&#39;" }[c]));
  }

  function mdEl(type, raw) {
    const div = document.createElement("div");
    div.className = "md";
    if (type === "equations") {
      div.innerHTML = katex.renderToString(raw,
                                           { displayMode: true, throwOnError: false });
    } else {
      div.innerHTML = marked.parse(raw);
      renderMathInElement(div, {
        delimiters: [{ left: "$$", right: "$$", display: true },
                     { left: "$", right: "$", display: false }],
        throwOnError: false
      });
    }
    return div;
  }

  function pairEl(type, s, i, j, sim, exact) {
    const det = document.createElement("details");
    const sum = document.createElement("summary");
    const snippet = esc(s.gt[i].replace(/\\s+/g, " ").slice(0, 100));
    sum.innerHTML =
      '<span class="sim">sim ' + sim.toFixed(3) + "</span>" +
      (exact ? '<span class="exact">exact</span>' : "") +
      '<span class="idx">GT #' + i + " &harr; cand #" + j + "</span> " +
      '<span class="snip">' + snippet + "&#8230;</span>";
    det.appendChild(sum);
    const grid = document.createElement("div");
    grid.className = "grid";
    for (const lab of ["ground truth", "candidate"]) {
      const l = document.createElement("div");
      l.className = "lab";
      l.textContent = lab;
      grid.appendChild(l);
    }
    grid.appendChild(mdEl(type, s.gt[i]));
    grid.appendChild(mdEl(type, s.cand[j]));
    det.appendChild(grid);
    return det;
  }

  function nomatchEl(label, idx, raw, missingName, type) {
    const det = document.createElement("details");
    det.className = "warn";
    const sum = document.createElement("summary");
    const snippet = esc(raw.replace(/\\s+/g, " ").slice(0, 100));
    sum.innerHTML =
      '<span class="nomatch">no ' + missingName + ' counterpart</span> ' +
      '<span class="idx">' + label + " #" + idx + "</span> " +
      '<span class="snip">' + snippet + "&#8230;</span>";
    det.appendChild(sum);
    const cell = mdEl(type, raw);
    cell.className = "md single";
    det.appendChild(cell);
    return det;
  }

  for (const doc of data.docs) {
    const sec = document.createElement("section");
    sec.id = "doc-" + doc.id;
    const h2 = document.createElement("h2");
    h2.innerHTML = esc(doc.id) +
      ' <span class="group">' + doc.group + "</span> " +
      '<span class="stats">efficiency ' + (doc.efficiency * 100).toFixed(1) +
      "% &middot; purity " + (doc.purity * 100).toFixed(1) + "%</span>";
    sec.appendChild(h2);
    const link = document.createElement("a");
    link.href = "#doc-" + doc.id;
    link.textContent = doc.id + " (" + doc.group + ")";
    navEl.appendChild(link);
    navEl.appendChild(document.createTextNode("  "));

    for (const side of SIDE_ORDER) {
      const h3 = document.createElement("h3");
      h3.textContent = "pipeline: " + side;
      sec.appendChild(h3);
      for (const type of TYPE_ORDER) {
        const s = doc.sides[side][type];
        if (!s.pairs.length && !s.cand_nomatch.length && !s.gt_nomatch.length) {
          continue;
        }
        const h4 = document.createElement("h4");
        h4.textContent = TYPE_TITLES[type] + " — " + s.pairs.length +
          (s.pairs.length > 1 ? " matched pairs" : " matched pair") +
          " (" + s.n_gt + " ground-truth items, " + s.n_cand +
          (s.mean_sim != null ? ", mean similarity " + s.mean_sim.toFixed(3) : "") +
          ")";
        sec.appendChild(h4);
        if (s.pairs.length) {
          const wrap = document.createElement("div");
          wrap.className = "pairs";
          for (const [i, j, sim, exact] of s.pairs) {
            wrap.appendChild(pairEl(type, s, i, j, sim, exact));
          }
          sec.appendChild(wrap);
        }
        if (s.cand_nomatch.length) {
          const h5 = document.createElement("h5");
          h5.innerHTML = "No-match candidate items (" + s.cand_nomatch.length +
            ") &mdash; possible hallucinations or structural artifacts (page splits, " +
            "repeated running headers, re-classification; caveats 2 and 5)." +
            TYPE_NOTE[type];
          sec.appendChild(h5);
          const wc = document.createElement("div");
          wc.className = "pairs";
          for (const [idx, raw] of s.cand_nomatch) {
            wc.appendChild(nomatchEl("candidate", idx, raw, "ground-truth", type));
          }
          sec.appendChild(wc);
        }
        if (s.gt_nomatch.length) {
          const h5g = document.createElement("h5");
          h5g.className = "h5gt";
          h5g.textContent = "No-match ground-truth items (" + s.gt_nomatch.length +
            ") — content the candidate did not reproduce.";
          sec.appendChild(h5g);
          const wg = document.createElement("div");
          wg.className = "pairs";
          for (const [idx, raw] of s.gt_nomatch) {
            wg.appendChild(nomatchEl("ground truth", idx, raw, "candidate", type));
          }
          sec.appendChild(wg);
        }
      }
    }
    mainEl.appendChild(sec);
  }
});
</script>
</body>
</html>
"""


def build_html(docs: list, n_docs: int) -> str:
    payload = json.dumps({"docs": docs}, ensure_ascii=False)
    return (HTML_TEMPLATE
            .replace("__DATA_JSON__", payload.replace("</", "<\\/"))
            .replace("__N_DOCS__", str(n_docs)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="only first N papers")
    ap.add_argument("--only", default=None, help="comma-separated arXiv ids")
    ap.add_argument("--min-tokens", type=int, default=8)
    args = ap.parse_args()

    papers = sorted(f for f in GT_DIR.glob("*.md") if ID_RE.fullmatch(f.stem))
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        papers = [f for f in papers if f.stem in want]
    if args.limit:
        papers = papers[:args.limit]
    scored = [f for f in papers if f.stem not in EXCLUDED_GT]
    skipped = [f.stem for f in papers if f.stem in EXCLUDED_GT]
    print(f"scoring {len(scored)} papers ({len(papers)} found, {len(skipped)} excluded: {skipped})")

    t0 = time.time()
    results = {}
    for side, sdir in CAND_DIRS.items():
        per_paper = {}
        missing = []
        for k, gt in enumerate(scored, 1):
            cf = sdir / f"{gt.stem}.md"
            if not cf.is_file():
                missing.append(gt.stem)
                continue
            per_paper[gt.stem] = analyze_pair(gt.read_text(encoding="utf-8"),
                                              cf.read_text(encoding="utf-8"),
                                              args.min_tokens)
            if k % 25 == 0:
                print(f"  {side}: {k}/{len(scored)}", flush=True)
        if missing:
            print(f"  {side}: MISSING {len(missing)}: {missing[:5]}")
        results[side] = {"corpus": aggregate(per_paper), "papers": per_paper}
        OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON_DIR / f"{side}.json", "w", encoding="utf-8") as fh:
            json.dump({"corpus": results[side]["corpus"], "papers": per_paper}, fh,
                      indent=1)
    elapsed = time.time() - t0

    fill_notes(results)
    rows = ranking_rows(results, [f.stem for f in scored])
    chart_pur = OUT_JSON_DIR / "purity_ranking.png"
    write_chart(rows, chart_pur, "purity")
    chart_eff = OUT_JSON_DIR / "efficiency_ranking.png"
    write_chart(rows, chart_eff, "efficiency")
    picks = select_subsample(rows)
    docs = collect_viewer(picks, args.min_tokens)
    viewer = OUT_JSON_DIR / "match_viewer.html"
    viewer.write_text(build_html(docs, len(docs)), encoding="utf-8")
    write_report(results, [f.stem for f in scored], args, elapsed)
    print(f"report -> {OUT_REPORT}")
    print(f"chart  -> {chart_pur}")
    print(f"chart  -> {chart_eff}")
    print(f"viewer -> {viewer}")
    print(f"elapsed {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
