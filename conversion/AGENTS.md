# AGENTS.md — LLM (large language model) Doc Converter

## Project Overview

This repo converts PDF documents into Markdown using a vision-language model pipeline, optional figure extraction, and an agentic Markdown review loop. It also includes a second-stage Markdown → HTML/EPUB book conversion pipeline, plus a few utility scripts for figure inspection, arXiv downloads, and diagnostics.

Use the current scripts and CLI flags in the repo rather than older assumptions: the main entrypoint is `pdf_to_markdown_agentic.py`, the pixi task wrapper is `pixi run convert`, and the current default OpenAI-compatible endpoint is `http://127.0.0.1:8008/v1`.

## Environment

This project uses **[pixi](https://pixi.sh/)** for environment management. Dependencies live in `pixi.toml`.

- **Python**: `==3.14`
- **Package manager**: pixi (conda-forge + PyPI)

Run project commands with `pixi run ...` or `pixi exec -- ...`.

### Common setup

```bash
cd /nfs/user/asopio-eidf107/llm-doc-converter
pixi install
```

### Key installed tools/libraries

| Tool | Source | Purpose |
|------|--------|---------|
| `pdftoppm`, `pdfinfo` | `poppler` | Render PDFs to PNG and inspect page counts |
| `pandoc` | conda-forge | Convert cleaned Markdown to HTML/EPUB |
| `qpdf` | conda-forge | PDF object inspection utilities |
| `langchain` | PyPI | LLM orchestration |
| `langchain-openai` | PyPI | OpenAI-compatible chat client |
| `deepagents` | PyPI | Agentic Markdown repair loop |
| `markdown` | PyPI | Markdown validation |
| `pillow` | PyPI | Image cropping and figure handling |
| `docling` | PyPI | Alternative PDF parsing experiments |
| `transformers` | PyPI | Model-related utilities |

## Main PDF → Markdown pipeline

### Primary entrypoint

```bash
pixi exec -- python pdf_to_markdown_agentic.py input.pdf -o output/
```

Equivalent pixi task:

```bash
pixi run convert input.pdf -o output/
```

### What the pipeline does

1. Uses `pdfinfo` to count pages.
2. Renders requested PDF pages to PNG with `pdftoppm`, in parallel up to `--npages-parallel` workers.
3. Sends each page image to a vision model for Markdown generation.
4. Optionally parses `<!-- Figure: ... -->` markers, finds figure coordinates from the page image, validates crops, retries once when the reviewer suggests a better bbox, and replaces markers with Markdown image tags.
5. Writes incremental Markdown output so interrupted runs can resume from existing page blocks.
6. Optionally runs an agentic review loop that validates and fixes Markdown syntax until it passes or the iteration limit is hit.

Figure detection now has two backends selected by `--figure-extraction-mode`:
- `vlm` — current default, using the vision model to predict figure coordinates from markdown-derived descriptions
- `heuristic` — qpdf/XObject-based detection from `list_pdf_figures.py`, paired to markdown figure markers in reading order

### Output structure

The generated Markdown file looks like:

```markdown
# filename.pdf

<page 1 markdown>

---

<page 2 markdown>

---
```

When figure extraction succeeds, figure markers are replaced with image syntax like:

```markdown
![Figure description](figures/fig_12_0.png)
```

Related outputs may include:

- `figures/` — extracted figure PNGs
- `figure_debug/` — per-page debug logs for marker parsing, coordinate extraction, validation, and replacement
- intermediate page PNGs, unless `--no-cache-dir` is used

### Important CLI flags

```bash
pixi run convert input.pdf \
  -o output/ \
  --dpi 300 \
  --llm-base-url http://127.0.0.1:8008/v1 \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --start-page 0 \
  --end-page 10 \
  --npages-parallel 8
```

Supported options in `pdf_to_markdown_agentic.py`:

- `-o, --output` — output directory
- `--dpi` — page render DPI, default `300`
- `--model` — OpenAI-compatible model name
- `--api-key` — API key, default `EMPTY`
- `--llm-base-url` — default `http://127.0.0.1:8008/v1`
- `--start-page` — inclusive 0-indexed start page
- `--end-page` — exclusive 0-indexed end page in CLI semantics; internally converted to the requested page range
- `--continue-on-error` — continue on page/LLM errors
- `--no-cache-dir` — remove rendered page PNGs after completion
- `--max-review-iterations` — review loop cap, default `10`
- `--skip-review` — skip agentic Markdown repair
- `--skip-figure-extraction` — skip figure detection/cropping/replacement
- `--figure-extraction-mode` — `vlm` or `heuristic`, default `vlm`
- `--npages-parallel` — parallelism for page rendering and page processing, default `8`

### Review loop details

The review loop lives inside `pdf_to_markdown_agentic.py` and uses a DeepAgents agent with the `validate_markdown` tool plus file tools. Its intended workflow is:

1. validate current Markdown
2. read file when validation fails
3. edit targeted syntax issues
4. re-validate

The standalone helper `run_review.py` is currently a hard-coded convenience script for one test output, not a general CLI.

## DeepSeek-OCR2 (optical character recognition) PDF → Markdown pipeline

Second converter, for scanned/slide documents. It sends one rendered page image per request to DeepSeek-OCR-2 served by vLLM (pod spec: `vllm_deepseek_ocr2.yaml` at the repo root), keeps the model's grounding/layout tokens in a raw dump, crops the figure det boxes, and finishes with `dsocr2_postprocess.py`.

### Primary entrypoint

```bash
pixi run python pdf_to_markdown_dsocr2.py input.pdf
pixi run python pdf_to_markdown_dsocr2.py input.pdf -o out/ --dpi 150 --start-page 0 --end-page 10
```

### What the pipeline does

1. Uses `pdfinfo` to count pages.
2. Renders requested pages to `pages/page-NN.png` with `pdftoppm` (parallel, cached; `--force` re-renders).
3. One chat completion per page image: prompt `<|grounding|>Convert the document to markdown.`, temperature 0, max_tokens 8180, `skip_special_tokens=false` (keeps the layout tokens).
4. Writes the raw grounding dump `<stem>_dsocr2.md` with a preamble, `Page notes:` (blank scans) and `<!-- ==================== PAGE N ==================== -->` markers — same format as `Evian1992LHC_sample_dsocr2.md`.
5. Crops each `<|ref|>image<|/ref|><|det|>` box (normalized 0–999 of the page image) from the rendered page into `figures/page-NN-fKK.png` and records `figures_manifest.json` (schema of `example_docs/evian/figures_manifest.json`, `rotated_input` always false).
6. Calls `dsocr2_postprocess.postprocess()` to write the standard markdown to `md/<stem>_dsocr2.md` (`## Slide N`, `---` separators, pipe tables, `$…$`/`$$…$$` math, `![Figure k](../figures/…)` refs).

Raw per-page model outputs are cached in `raw_pages/page-NN.txt`, so interrupted runs resume without re-OCRed pages. Pages are sent in their native orientation — rotated pages are NOT handled.

### Supported options

- `-o, --output` — output directory (default: the PDF's directory, as in `example_docs/evian/`)
- `--dpi` — page render DPI, default `150`
- `--llm-base-url` — default `http://127.0.0.1:8000/v1` (the dsocr2 pod — not the Qwen `:8008` endpoint)
- `--model` — default `deepseek-ocr2`
- `--api-key` — default `EMPTY`
- `--temperature` (default `0`), `--max-tokens` (default `8180`), `--request-timeout` (default `600`)
- `--start-page` / `--end-page` — same 0-indexed semantics as the main pipeline
- `--npages-parallel` — rendering + OCR parallelism, default `8`
- `--force` — re-render/re-OCR cached pages
- `--continue-on-error` — skip pages whose request fails
- `--skip-figures` — no crops, empty manifest
- `--no-postprocess` — stop after raw dump + figures
- `--no-cache-dir` — remove rendered page PNGs after completion

`dsocr2_postprocess.py` also works standalone (`python3 dsocr2_postprocess.py [input.md] [-o OUT] [--figures-dir DIR] [--manifest FILE]`; no-args defaults re-process the Evian sample) and is stdlib-only.

## Markdown → HTML / EPUB pipeline

The `markdown_to_book/` package turns generated Markdown into standalone HTML or EPUB via preprocessing plus pandoc.

### Main commands

```bash
# Full pipeline to HTML
pixi run python -m markdown_to_book test_outputs/file.md -o output_html/

# Full pipeline to EPUB
pixi run python -m markdown_to_book test_outputs/file.md -o output_epub/ --format epub

# Pixi wrapper for the HTML default
pixi run convert-to-html -- test_outputs/file.md -o output_html/
```

### Stages

1. `markdown_to_book.preprocess`
   - removes the synthetic top `# filename.pdf` header
   - converts `<!-- Figure: ... -->` markers into italic figure captions
   - removes `<!-- Table: start -->` / `<!-- Table: end -->`
   - replaces PDF page separators with `<div class="pagebreak"></div>`
   - strips orphan page numbers
2. `markdown_to_book.convert`
   - runs pixi-managed `pandoc`
   - emits `index.html` for HTML or `book.epub` for EPUB

### Useful book-conversion flags

- `--format html|epub`
- `--clean-output <path>`
- `--title <title>`
- `--author <author>`
- `--cover <image>`

## Other repo utilities

### Figure/PDF utilities

- `list_pdf_figures.py` — inspects PDF XObjects with `qpdf` and reports figure-like placements; also powers the `heuristic` figure-detection backend in `pdf_to_markdown_agentic.py`
- `extract_figure.py` — crops a figure from a rendered page using a supplied bbox

### Download utility

- `download_recent_hepex_arxiv.py` — downloads recent arXiv PDFs plus metadata JSON, with optional category exclusion

### Diagnostics

- `diagnostics/` contains experimental and debugging scripts, API checks, and saved debug artifacts
- Treat these as support material, not stable user-facing interfaces

## Repo structure

```text
pdf_to_markdown_agentic.py   Main PDF → Markdown pipeline
pdf_to_markdown_dsocr2.py    DeepSeek-OCR2 (vLLM) grounding pipeline
dsocr2_postprocess.py        Raw-dump post-processor (slides, tables, math, figures)
pdf_to_markdown.py           Older simpler converter without agentic review/figure workflow
markdown_to_book/            Markdown preprocessing + pandoc conversion package
run_review.py                Hard-coded review helper for a sample output
list_pdf_figures.py          qpdf-based PDF figure inspection utility
extract_figure.py            Simple bbox-based figure crop utility
download_recent_hepex_arxiv.py  arXiv download helper
diagnostics/                 Debugging and experiment scripts
activation/                  Pixi activation scripts
test_input/                  Sample PDFs
test_outputs/                Generated outputs
pixi.toml                    Environment and task definitions
```

## Working conventions for agents

- Prefer `pixi run` / `pixi exec --` for commands.
- Assume `pdf_to_markdown_agentic.py` is the current implementation unless the user explicitly wants an older script.
- Do not document `run_review.py` as a general-purpose command without checking whether it has been generalized.
- If changing docs, keep defaults aligned with code: Python 3.14 in `pixi.toml`, and `http://127.0.0.1:8008/v1` as the current default base URL.
- `diagnostics/`, `archive/`, and similar folders may contain stale experiments; verify before describing them as supported workflows.
