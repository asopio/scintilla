# SCINTILLA

## Scientific Corpus Intelligence and Navigation Toolkit using Large-Language-model Agents

SCINTILLA is an agentic framework for processing, organizing, and exploring
scientific document corpora. It combines workflows for document
acquisition, multimodal conversion, retrieval-oriented database construction,
and downstream evaluation.

## Package overview

SCINTILLA is organized into a small number of top-level components that reflect
the end-to-end document-processing workflow.

### `conversion/`

The `conversion/` module contains the main PDF-processing and document-conversion
pipeline. It includes:

- `pdf_to_markdown_agentic.py` for multimodal PDF-to-Markdown conversion
- `list_pdf_figures.py` for PDF figure inspection
- `extract_figure.py` for figure cropping
- `markdown_to_book/` for Markdown-to-HTML/EPUB conversion

This layer is responsible for turning scientific PDFs into structured Markdown,
with optional figure extraction and an agentic Markdown repair loop. The
`training/` subdirectory also contains utilities for building figure-detection
datasets and training, evaluating, merging, or quantizing Qwen-based figure
extraction models.

### `download/`

The `download/` module contains corpus-ingestion and batch-operations tools.
It includes:

- `export_category_pdfs.py` for retrieving presentation PDFs from Indico
- `download_recent_hepex_arxiv.py` for building arXiv-based corpora
- `batch_pdf_md_convert.py` for batch conversion workflows
- `check_jobs_running.py` for monitoring long-running conversion jobs

This layer supports both continuous ingestion of meeting material and one-off
collection of larger document sets.

### `database/`

The `database/` module contains the retrieval-storage layer. It includes:

- `build_database.py` for constructing structured slide/document datasets
- `merge_vortex_shards.py` for merging sharded Vortex outputs

This stage converts converted corpora and their metadata into retrieval-ready
Parquet or Vortex datasets with preserved provenance and optional embedded page
assets.

### `evaluation/`

The `evaluation/` module contains scripts for benchmarking and downstream
analysis. It includes:

- `build_vectorstore.py` for constructing persisted Chroma vector stores
- `category_statistics.py` for corpus statistics over Indico categories
- `evaluate_generation.py` for evaluation of downstream generation workflows

This layer is intended to support comparison of retrieval and answer-generation
strategies over the processed corpus.

### `legacy/`

The `legacy/` module contains older or superseded workflows retained for
reference. At present this includes `extract_pdf_markdown_vlm.py`, which is
useful for documenting earlier design iterations but is not intended to be the
primary conversion path.

## Intended use

SCINTILLA is intended to unify four closely related capabilities:

1. acquisition of scientific PDFs from sources such as Indico and arXiv,
2. multimodal conversion of PDFs into structured Markdown,
3. construction of retrieval-oriented databases and vector stores,
4. evaluation of retrieval and generation pipelines driven by LLM agents.

In this sense, SCINTILLA is not just a converter and not just a retrieval tool.
It is a systems repository for building scientific knowledge pipelines from raw
PDF corpora to grounded LLM-assisted exploration.

## License 

MIT