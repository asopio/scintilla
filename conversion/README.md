# LLM Doc Converter

Convert PDF documents into Markdown using a vision-language model, optional figure extraction, and an agentic Markdown repair loop. The repo also includes a Markdown → HTML/EPUB pipeline for turning converted documents into readable standalone outputs.

## Quick start

```bash
cd /nfs/user/asopio-eidf107/llm-doc-converter
pixi install

# Basic conversion
pixi run convert input.pdf -o output/

# Limit pages and reduce DPI
pixi run convert input.pdf --start-page 0 --end-page 10 --dpi 150 -o output/

# Skip the agentic review step
pixi run convert input.pdf --skip-review

# Skip figure extraction
pixi run convert input.pdf --skip-figure-extraction

# Use the heuristic qpdf-based figure detector instead of the VLM detector
pixi run convert input.pdf --figure-extraction-mode heuristic

# Process pages in parallel batches
pixi run convert input.pdf --npages-parallel 8 -o output/
```

## What it does

1. Counts PDF pages with `pdfinfo`
2. Renders each requested page to PNG with `pdftoppm`
3. Sends each page image to a vision-language model for Markdown extraction
4. Optionally extracts figures from `<!-- Figure: ... -->` markers using a description-guided crop-and-review loop
5. Writes incremental Markdown output and resumes from existing page blocks when re-run
6. Optionally runs an agentic Markdown review/fix loop

## Main pipeline

### Entry points

```bash
# Direct script
pixi exec -- python pdf_to_markdown_agentic.py input.pdf -o output/

# Pixi task wrapper
pixi run convert input.pdf -o output/
```

### Current defaults

| Setting | Default |
|---------|---------|
| Model | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| API endpoint | `http://127.0.0.1:8008/v1` |
| API key | `EMPTY` |
| DPI | `300` |
| Parallel pages | `8` |
| Max review iterations | `10` |
| Figure extraction mode | `vlm` |

### CLI reference

```text
pixi run convert PDF [OPTIONS]

Positionals:
  PDF                          Path to the input PDF file

Options:
  -o, --output DIRECTORY       Output directory (default: same dir as PDF)
  --dpi INTEGER                DPI for page images (default: 300)
  --model TEXT                 OpenAI-compatible model name
  --api-key TEXT               OpenAI API key (default: EMPTY)
  --llm-base-url URL           OpenAI API base URL (default: http://127.0.0.1:8008/v1)
  --start-page INTEGER         0-indexed page to start from (default: 0)
  --end-page INTEGER           0-indexed page to stop before (default: all pages)
  --continue-on-error          Continue on API errors (default: True)
  --no-cache-dir               Delete rendered page PNGs after completion
  --max-review-iterations N    Max review→fix cycles (default: 10)
  --skip-review                Skip agentic markdown review step
  --skip-figure-extraction     Skip figure detection and extraction
  --figure-extraction-mode     Figure detection backend: vlm or heuristic (default: vlm)
  --npages-parallel N          Number of pages to process in parallel per batch (default: 8)
```

## Figure extraction

Figure extraction is markdown-first and description-guided.

### Workflow

1. The page is first converted to Markdown.
2. The model inserts figure placeholders like `<!-- Figure: description -->`.
3. Those descriptions are sent back with the full page image to request figure coordinates.
4. Candidate crops are reviewed against the full page and description.
5. If the review suggests a better bbox, the crop is retried once.
6. Validated figure markers are replaced with Markdown image tags.

### Outputs

Successful figure extraction creates:

- `figures/` — cropped figure PNGs
- `figure_debug/` — per-page text logs for marker parsing, coordinate extraction, validation, and replacement

Example replacement:

```markdown
![Detector geometry overview](figures/fig_42_0.png)
```

## Resume and incremental output

The converter writes the Markdown file after each completed batch. If the output Markdown file already exists, it attempts to recover existing page blocks and continue from there.

This is useful for long runs or interrupted jobs.

## Agentic Markdown review

After page assembly, the pipeline can run a DeepAgents review loop that:

1. validates Markdown with Python's `markdown` library
2. reads the current file when validation fails
3. applies targeted edits
4. re-validates until clean or the iteration cap is reached

Disable it with:

```bash
pixi run convert input.pdf --skip-review
```

## Markdown → HTML / EPUB conversion

The `markdown_to_book/` package converts the generated Markdown into standalone HTML or EPUB through a preprocess-then-pandoc flow.

### Quick start

```bash
# Full pipeline to HTML
pixi run python -m markdown_to_book test_outputs/output.md -o output_html/

# Full pipeline to EPUB
pixi run python -m markdown_to_book test_outputs/output.md -o output_epub/ --format epub

# Pixi wrapper for HTML
pixi run convert-to-html -- test_outputs/output.md -o output_html/

# Show help
pixi run python -m markdown_to_book --help
```

### Preprocessing step

`markdown_to_book.preprocess` cleans up PDF-conversion artefacts before pandoc:

- removes the synthetic `# filename.pdf` header
- converts `<!-- Figure: ... -->` to italic figure captions
- removes `<!-- Table: start -->` / `<!-- Table: end -->`
- converts page separators to `<div class="pagebreak"></div>`
- strips orphan page numbers

### Conversion step

`markdown_to_book.convert` runs pandoc and writes:

- `index.html` for HTML output
- `book.epub` for EPUB output

### Book-conversion examples

```bash
# Preprocess only
pixi run python -m markdown_to_book.preprocess test_outputs/output.md

# Convert an already-clean markdown file to HTML
pixi run python -m markdown_to_book.convert test_outputs/output_clean.md -o output_html/

# Convert to EPUB with explicit metadata
pixi run python -m markdown_to_book test_outputs/output.md -o output_epub/ \
  --format epub \
  --title "Document Title" \
  --author "Author Name" \
  --cover figures/cover.png
```

## Requirements

- [pixi](https://pixi.sh/)
- OpenAI-compatible LLM endpoint
- GPU recommended for larger documents and higher `--npages-parallel` settings

Installed via `pixi.toml`:

- `poppler` for `pdftoppm` and `pdfinfo`
- `pandoc`
- `qpdf`
- Python `==3.14`
- `langchain`, `langchain-openai`, `deepagents`, `markdown`, `pillow`, `docling`, `transformers`

## Other utilities

- `list_pdf_figures.py` — inspect figure-like PDF XObjects with `qpdf`
- `extract_figure.py` — crop a figure from a page image using a bbox
- `download_recent_hepex_arxiv.py` — download recent arXiv PDFs and metadata
- `diagnostics/` — debugging and experiment scripts

## Repo layout

```text
pdf_to_markdown_agentic.py      Main PDF → Markdown pipeline
pdf_to_markdown.py              Older simpler converter
markdown_to_book/               Markdown preprocessing and pandoc conversion
run_review.py                   Hard-coded sample review helper
list_pdf_figures.py             qpdf-based figure inspection utility
extract_figure.py               Simple bbox-based crop helper
download_recent_hepex_arxiv.py  arXiv downloader
diagnostics/                    Debugging and experimental scripts
test_input/                     Sample PDFs
test_outputs/                   Generated outputs
```

## License

MIT
## Repo layout

```text
pdf_to_markdown_agentic.py      Main PDF → Markdown pipeline
pdf_to_markdown.py              Older simpler converter
markdown_to_book/               Markdown preprocessing and pandoc conversion
run_review.py                   Hard-coded sample review helper
list_pdf_figures.py             qpdf-based figure inspection utility
extract_figure.py               Simple bbox-based crop helper
download_recent_hepex_arxiv.py  arXiv downloader
diagnostics/                    Debugging and experimental scripts
test_input/                     Sample PDFs
test_outputs/                   Generated outputs
```

## License

MIT
