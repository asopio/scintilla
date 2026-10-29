#!/usr/bin/env python3
"""Structural QA checker for the ground-truth markdown in md/.

Checks (per paper):
  1.  first line is ``# <id>.pdf``
  2.  ``$`` balance outside code fences, ignoring escaped ``\\$``
  3.  no tight ``$$`` adjacency (display math must be spaced)
  4.  footnote refs all defined, no duplicates, no unused defs
  5.  no converter artifacts (kern/penalty/ltx_/entities/bib: keys/zwsp)
  6.  every ``figures/<id>/...`` image ref exists and is well-formed
  7.  every pipe table is rectangular (unescaped pipe count, all rows)
  8.  no leading ``\\displaystyle`` inside a ``$$ ... $$`` block

Run from anywhere:
    python qa_check.py [md_dir]
"""

import glob
import os
import re
import sys
from collections import Counter


def cell_count(line: str) -> int:
    s = line.strip()
    s = re.sub(r"^\|", "", s)
    s = re.sub(r"\|$", "", s)
    return len(s.replace("\\|", "").split("|"))


def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def main() -> int:
    md_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "md")
    all_files = sorted(glob.glob(os.path.join(md_dir, "*.md")))
    files = [f for f in all_files
             if re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}",
                             os.path.splitext(os.path.basename(f))[0])]
    skipped = len(all_files) - len(files)
    print(f"checking {len(files)} markdown files in {md_dir}"
          + (f" ({skipped} non-paper file(s) skipped)" if skipped else ""))
    if not files:
        return 1

    issues: list[str] = []
    for f in files:
        id_ = os.path.splitext(os.path.basename(f))[0]
        text = open(f, encoding="utf-8").read()
        lines = text.splitlines()

        # 1. header
        if not lines or lines[0] != f"# {id_}.pdf":
            issues.append(f"{id_}: bad first line {lines[0] if lines else ''!r}")

        tc = strip_code(text).replace("\\$", "")

        # 2. dollar balance
        if tc.count("$") % 2:
            issues.append(f"{id_}: odd $ count {tc.count('$')}")

        # 3. tight $$ adjacency
        for m in re.finditer(r"\$\$(?!\s)", tc):
            issues.append(f"{id_}: tight $$ open: {tc[m.start():m.start()+30]!r}")
        for m in re.finditer(r"(?<!\s)\$\$", tc):
            ctx = tc[max(0, m.start() - 30):m.start() + 2]
            if not ctx.rstrip().endswith("$$"):
                issues.append(f"{id_}: tight $$ close: {ctx!r}")

        # 4. footnotes (math spans dropped so [x] inside math is not a ref;
        #    refs counted in body text, defs in the trailing def block, so a
        #    def's own "[^n]:" prefix is never mistaken for an inline ref)
        tm = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", "", tc, flags=re.S)
        defline = re.compile(r"^\[\^([^\]:\s]+)\]:", re.M)
        defs = defline.findall(tm)
        m_def = defline.search(tm)
        body = tm[: m_def.start()] if m_def else tm
        refs = re.findall(r"\[\^([^\]\s]+)\]", body)
        rc, dc = Counter(refs), Counter(defs)
        missing = {k: v for k, v in rc.items() if not dc.get(k)}
        unused = {k: v for k, v in dc.items() if not rc.get(k)}
        dupes = {k: v for k, v in dc.items() if v > 1}
        if missing:
            issues.append(f"{id_}: undefined footnote refs {missing}")
        if unused:
            issues.append(f"{id_}: unused footnote defs {unused}")
        if dupes:
            issues.append(f"{id_}: duplicate footnote defs {dupes}")

        # 5. artifacts
        for pat, name in [(r"\\kern\b", "kern"), (r"\\penalty\b", "penalty"),
                          (r"\bltx_[a-z_]+\b", "ltx"), (r"&[a-z]+;", "entity"),
                          (r"bib:[A-Z]", "bibkey"), (r"\u200b", "zwsp")]:
            for m in re.finditer(pat, tc):
                issues.append(f"{id_}: artifact {name}: "
                              f"{tc[max(0, m.start()-40):m.start()+40]!r}")

        # 6. figure refs
        for m in re.finditer(r"!\[[^\]]*\]\((figures/[^)]+)\)", text):
            p = m.group(1)
            if not re.fullmatch(r"figures/([0-9A-Za-z_.-]+)/[A-Za-z0-9_./-]+", p):
                issues.append(f"{id_}: bad figure path {p!r}")
            elif not os.path.exists(os.path.join(md_dir, p)):
                issues.append(f"{id_}: missing figure file {p!r}")

        # 7. rectangular tables
        i = 0
        while i < len(lines):
            if lines[i].lstrip().startswith("|"):
                j, counts = i, []
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    counts.append(cell_count(lines[j]))
                    j += 1
                if len(set(counts)) > 1:
                    issues.append(f"{id_}: ragged table at {i+1}-{j}, {counts[:10]}")
                i = j
            else:
                i += 1

        # 8. no leading \displaystyle in display blocks
        for m in re.finditer(r"\$\$\s*\\displaystyle", tc):
            issues.append(f"{id_}: leading displaystyle: "
                          f"{tc[max(0, m.start()-20):m.start()+40]!r}")

    if issues:
        print(f"FAIL: {len(issues)} issues")
        for x in issues[:60]:
            print("  " + x)
        return 1
    print("PASS: all checks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
