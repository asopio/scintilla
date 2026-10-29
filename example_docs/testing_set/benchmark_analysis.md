# Benchmark similarity analysis: qwen vs dsocr optical character recognition (OCR) pipelines

*142 papers scored (of 150 in the testing set); excluded: 1005.5254, 1008.2461, 1009.5069, 1009.5908, 1101.0070, 1103.1984, 1103.5559, 1104.4481 — their ar5iv ground truth is an upstream LaTeXML fragment (title page, acknowledgements, or one table only; see `md/README.md`). Runtime 112s. Macro metric means are over papers (each paper weighted equally); efficiency/purity are pooled over all items.*

Ground truth (GT): ar5iv HTML → markdown (`evaluation/html_to_markdown.py`). Candidates: `eval_md/qwen/` (Qwen3.8-27B-FP8 page OCR, `pdf_to_markdown_agentic_v2.py`) and `eval_md/dsocr/` (DeepSeek-OCR2, `pdf_to_markdown_dsocr2.py` + `dsocr2_postprocess.py`).

## Method

Each text type is extracted and scored separately. Figure **alt-text is excluded** (image `![...]` lines): the two pipelines generate it in fundamentally different ways.

- **Paragraphs** — prose blocks (headings, tables, equations, image lines, figure captions excluded). Each list item (numbered reference, bullet) is one item. Blocks with < 8 tokens are dropped. Table-caption lines (`Table N: ...`) are treated as paragraph text.
- **Equations** — display math `$$...$$` only. Normalized by removing whitespace, `\left`/`\right`, spacing commands, align layout (`\` breaks, `&`), `\tag{N}` / trailing `(N)` numbers, collapsing `\begin{aligned}`-style environments, mapping `\to`→`\rightarrow` and `\varepsilon`→`\epsilon` (indistinguishable glyphs in scans), aliasing `\rm`/`\mbox`/`\text`→`\mathrm`, and canonicalizing `a_b^c`→`a^c_b` and `a_x`→`a_{x}`.
- **Tables** — pipe-table blocks; separator rows dropped; exact match on the row-major sequence of normalized cells.
- **Figure captions** — lines matching `Figure|Fig. <label>:` (colon required; `Figure 2 shows…` prose is not captured), label prefix stripped.

Prose normalization: markdown emphasis stripped, links → anchor text, `[^n]` footnotes and `††thanks:` artifacts removed, citations (`[19]`, `[1–3]`) → `<ref>`, inline math compacted (whitespace-free, macro aliases), casefolded, whitespace collapsed. Similarity = token-set Jaccard; alignment is 1-to-1 greedy (threshold 0.10) for paragraphs/tables/captions and unconstrained best-match (with multiplicity) for equations. BLEU (Bilingual Evaluation Understudy; n-gram precision against the reference) is nltk sentence-BLEU (4-gram, method1 smoothing); ROUGE-1/2/L (Recall-Oriented Understudy for Gisting Evaluation) are token-level F1 — the harmonic mean of precision and recall (ROUGE-L via longest common subsequence).

Metric names: **efficiency** (recall in information-retrieval terms) = fraction of ground-truth items with a matched candidate; **purity** (precision) = the same fraction counted over candidate items; **exact-match rate** = fraction of matched pairs identical after normalization; **mean similarity** = mean Jaccard of matched pairs; **candidate traceability** (all types, any-match rather than 1-to-1) = fraction of candidate items that have any ground-truth counterpart at or above the 0.10 similarity floor — the closest item-level proxy for the hallucination rate (1 − traceability is the share of emitted items with no counterpart in the source at all); **candidate n-gram support** = fraction of unique candidate n-grams (n = 1 or 3 tokens, within items) that occur anywhere in the ground-truth items — the content-level proxy, catching fabrication inside otherwise matched items.

## Results

### Paragraphs

| Metric | qwen | dsocr |
|---|---:|---:|
| GT items | 40972 | 40972
| Candidate items | 43904 | 53928
| Matched pairs | 38218 | 40331
| Efficiency | 93.3% | 98.4%
| Purity | 87.1% | 74.8%
| Candidate traceability (any-match) | 95.3% | 88.7%
| Exact-match rate | 15.3% | 12.2%
| Mean similarity | 0.826 | 0.784
| BLEU | 0.811 | 0.775
| ROUGE-1 | 0.888 | 0.866
| ROUGE-2 | 0.853 | 0.825
| ROUGE-L | 0.887 | 0.864
| Candidate 1-gram support | 81.5% | 72.9%
| Candidate 3-gram support | 56.6% | 53.8%

### Equations

| Metric | qwen | dsocr |
|---|---:|---:|
| GT items | 1368 | 1368
| Candidate items | 1609 | 1534
| Matched pairs | 1368 | 1368
| Efficiency | 100.0% | 100.0%
| Purity | 85.0% | 89.2%
| Candidate traceability (any-match) | 98.7% | 99.9%
| Exact-match rate | 19.4% | 15.7%
| Mean similarity | 0.854 | 0.853
| BLEU | 0.700 | 0.744
| ROUGE-1 | 0.856 | 0.880
| ROUGE-2 | 0.800 | 0.832
| ROUGE-L | 0.826 | 0.862
| Candidate 1-gram support | 79.6% | 89.2%
| Candidate 3-gram support | 74.4% | 78.9%

### Tables

| Metric | qwen | dsocr |
|---|---:|---:|
| GT items | 500 | 500
| Candidate items | 547 | 508
| Matched pairs | 478 | 468
| Efficiency | 95.6% | 93.6%
| Purity | 87.4% | 92.1%
| Candidate traceability (any-match) | 95.2% | 94.9%
| Exact-match rate | 8.2% | 5.6%
| Mean similarity | 0.629 | 0.593
| BLEU | 0.514 | 0.477
| ROUGE-1 | 0.709 | 0.675
| ROUGE-2 | 0.593 | 0.557
| ROUGE-L | 0.689 | 0.660
| Candidate 1-gram support | 67.4% | 67.7%
| Candidate 3-gram support | 50.2% | 39.4%

### Figure captions

| Metric | qwen | dsocr |
|---|---:|---:|
| GT items | 74 | 74
| Candidate items | 953 | 1007
| Matched pairs | 54 | 62
| Efficiency | 73.0% | 83.8%
| Purity | 5.7% | 6.2%
| Candidate traceability (any-match) | 13.1% | 14.3%
| Exact-match rate | 31.5% | 12.9%
| Mean similarity | 0.861 | 0.853
| BLEU | 0.834 | 0.809
| ROUGE-1 | 0.902 | 0.920
| ROUGE-2 | 0.852 | 0.853
| ROUGE-L | 0.889 | 0.919
| Candidate 1-gram support | 5.4% | 5.6%
| Candidate 3-gram support | 4.2% | 4.4%

### Paragraphs — observations

Prose is the best-aligned type: efficiency is 93.3% (qwen) vs 98.4% (dsocr), with exact-match rates of 15.3% / 12.2%. The residual gap to 100% is dominated by long list blocks (references, affiliation lists) and slide-boundary fragmentation — see caveat 5; the per-paper distribution shows the gap is concentrated in the list-heavy papers. Any-match candidate traceability is 95.3% (qwen) / 88.7% (dsocr) and candidate 3-gram support 56.6% / 53.8%: these are the shares of emitted items/trigrams with no counterpart in the source at all — the closest thing to a hallucination rate. 1 − purity is higher than 1 − traceability because 1-to-1 alignment also drops items whose counterpart exists (caveats 2 and 5).

### Equations — observations

1368 GT display equations; qwen emits 1609, dsocr 1534 (both OCR pipelines invent/retain more display blocks than the ar5iv source, e.g. aligning terms that GT leaves inline). Exact match after normalization is 19.4% (qwen) / 15.7% (dsocr); mean similarity 0.854 / 0.853. Candidate traceability: 98.7% / 99.9%.

### Tables — observations

Tables are the most structure-dependent type: efficiency 95.6% (qwen) / 93.6% (dsocr), exact match 8.2% / 5.6%. dsocr OCRs more content as pipe-tables (extra grid-like fragments), which lowers purity; GT tables that ar5iv split across pages rarely re-align 1:1.

### Figure captions — observations

Ground truth carries only 74 standalone figure captions (ar5iv puts most captions in image alt text, which is excluded), vs 953 (qwen) and 1007 (dsocr) candidate captions. Efficiency is therefore the informative metric: 73.0% (qwen) / 83.8% (dsocr); exact match on matched captions 31.5% / 12.9%.

## Caveats

1. **Figure-caption asymmetry.** The ground truth has far fewer standalone caption lines than the candidates: ar5iv HTML usually embeds figure captions in the image *alt text*, which is excluded from this analysis by design. Candidate captions that have no ground-truth counterpart therefore lower purity on this type; efficiency is the more meaningful number.
2. **Page-boundary splits.** Candidate documents are OCR'd page by page, so paragraphs and tables are frequently cut at page boundaries, splitting one ground-truth item into several candidates (and repeating running headers). This caps paragraph/table purity and efficiency; similarity/BLEU/ROUGE on matched pairs are less affected.
3. **Equation numbering and macro style.** dsocr keeps trailing `(N)` numbers and qwen uses `\tag{N}`; both are stripped. Residual differences (`\mathrm` vs `\mbox`, OCR letter-spacing like `\mathrm{m i s s}`) mostly survive the normalization, but exact match on equations is therefore stricter than on prose.
4. **Defective ground truth excluded.** Eight papers have ar5iv/LaTeXML builds that served only a fragment of the paper (title page / author block only: 1005.5254, 1103.5559, 1104.4481; acknowledgements only: 1008.2461, 1009.5069, 1009.5908, 1101.0070; a single table only: 1103.1984). Scoring the full-paper OCR candidates against such fragments would measure mostly what the candidates added, not what they got right, so they are excluded from the scored corpus. The PDFs and GT markdown files themselves are intact in the testing set.
5. **Long list blocks cap efficiency (alignment artifact, not content loss).** Paragraph-efficiency losses concentrate in papers with long reference / affiliation / author lists. Probes of the raw candidate text confirm the content is present in both pipelines (e.g. a GT affiliation or reference that fails to align is found verbatim in the candidate output); the loss comes from 1-to-1 alignment being unable to bridge items the candidate split or re-flowed across `## Slide` boundaries. Large ATLAS/CMS papers are worst: the ground-truth author list is one ~10k-token line and the affiliation block is ~150-180 numbered lines, and qwen's agentic pipeline additionally renders the tail of the affiliation list as `[^n]:` footnote definitions (the excluded class), so it cannot match at all. This is why qwen has 23 papers below 85% paragraph efficiency (concentrated in those list-heavy papers) while dsocr has 1; it is the dominant term in the qwen-vs-dsocr paragraph-efficiency gap.

Per-paper metrics: `eval_md/benchmark/qwen.json`, `eval_md/benchmark/dsocr.json`.

Validation artifacts: `eval_md/benchmark/match_viewer.html` shows the matched items side by side (ground-truth item vs its matched candidate item, for both pipelines) plus the no-match items (candidate items without a ground-truth counterpart — the closest thing to hallucination candidates — and ground-truth items without a candidate) in six sample documents — two each from the top, the median and the bottom of the per-document purity distribution, where purity/efficiency are pooled over the four text types and both pipelines; markdown is rendered client-side with marked, math with KaTeX. `eval_md/benchmark/purity_ranking.png` and `eval_md/benchmark/efficiency_ranking.png` show the per-document purity and efficiency by pipeline (qwen vs dsocr, pooled over the four text types), each ranked by the pooled value of its metric; the purity ranking is the one the six-document sample was picked from.
