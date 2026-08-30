# Docling Architecture Analysis

Docling is a modular, multi-stage pipeline that converts arbitrary documents (primarily PDFs) into structured `DoclingDocument` objects, which can then be exported to Markdown, HTML, JSON, or other formats.

This document details how each sub-task — document parsing, OCR, layout segmentation, table structure extraction, reading-order determination, and export — is implemented, with references to the upstream source code.

---

## 1. Document Parsing (Backend Layer)

**Source**: [`docling/backend/`](https://github.com/docling-project/docling/tree/main/docling/backend)

Docling supports **20+ document backends** (PDF, DOCX, XLSX, HTML, Markdown, LaTeX, Email, etc.). Each format has its own backend class (e.g., `PdfDocumentBackend`, `MsWordBackend`, `HtmlBackend`) that implements the `AbstractDocumentBackend` interface.

For PDFs, the primary backend (`docling/backend/pdf_backend.py`) uses **pdfium** (via pypdfium2) to:

- Extract **programmatic text** (native PDF text, not OCR) as bounding-boxed characters and words
- Extract **bitmap regions** (images, scanned content) as rectangles
- Extract **hyperlink annotations**, page size, and document bookmarks/ToC

The backend produces a `ParsedPage` with:
- `char_cells`: char-level bounding boxes with raw text
- `word_cells`: word-level bounding boxes with raw text
- `bitmap_rects`: rectangles covering image/scanned areas
- `hyperlinks`: URL annotation rectangles

> **Key files**:
> - `docling/backend/pdf_backend.py` — base PDF backend
> - `docling/backend/pypdfium2_backend.py` — pypdfium2-based implementation
> - `docling/backend/abstract_backend.py` — interface definition

---

## 2. Page Preprocessing

**Source**: [`docling/models/stages/page_preprocessing/page_preprocessing_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/page_preprocessing/page_preprocessing_model.py)

This is the first pipeline stage. It:

- Renders each PDF page to a **rasterized PIL image** (at a configurable scale, default 1.0x or 2.0x)
- Extracts **text cells** from the backend's programmatic parse
- **Rates text quality** using heuristic rules:
  - Detects replacement characters (`�`)
  - Detects glyph-only text (`GLYPX<...>` patterns)
  - Detects fragmented words (`/G1/G2...` patterns)
  - Detects slash-separated garbage (`/token * /token *...`)
  - Applies penalties for repeated fragmented word patterns
  - Returns a parse confidence score per cell (0.0–1.0)

The quality score is stored in `conv_res.confidence.pages[page_no].parse_score` and later used to decide whether OCR is needed.

> **Key code**: `rate_text_quality()` method

---

## 3. OCR (Optical Character Recognition)

**Source**: [`docling/models/base_ocr_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/base_ocr_model.py)

Docling uses an **abstract factory** pattern for OCR engines. The default implementation is **EasyOCR** (with other engines pluggable).

### 3a. OCR Region Selection

The `get_ocr_rects()` method decides **where** to OCR on each page:

1. The backend provides a list of `bitmap_rects` (areas that contain images/scanned content)
2. These rectangles are drawn onto a binary image and **dilated** (10px kernel via cv2 or scipy) to merge nearby bitmaps
3. **Connected-component analysis** finds bounding boxes of bitmap clusters
4. Computes **bitmap area fraction** on the page:
   - If coverage > 75% or exceeds `bitmap_area_threshold` → **full-page OCR**
   - If coverage > threshold but < 75% → OCR only the bitmap regions
   - If below threshold → **skip OCR entirely** (page has usable programmatic text)

### 3b. OCR Cell Filtering

The `_filter_ocr_cells()` method prevents **duplicate text**:

- Builds an **R-tree spatial index** over existing programmatic cells
- For each OCR-extracted cell, checks if it **overlaps** with any programmatic cell
- **Drops overlapping OCR cells**, keeping only cells in regions where the PDF parser failed

### 3c. Cell Combination

The `_combine_cells()` method:

- Merges programmatic cells + filtered OCR cells into a single unified `TextCell` list
- Re-indexes cells sequentially
- When `force_full_page_ocr` is set, **replaces all non-OCR cells** (for PDFs where the parser output is unreliable)

### 3d. OCR Options

Configurable via `OcrOptions`:

- `force_full_page_ocr`: always OCR the entire page
- `bitmap_area_threshold`: minimum bitmap coverage to trigger OCR
- `ocr_mode`: strategy for OCR region selection (e.g., automatic, full-page)

> **Key code**: `get_ocr_rects()`, `_filter_ocr_cells()`, `_combine_cells()`

---

## 4. Layout / Document Segmentation

**Source**: [`docling/models/stages/layout/layout_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/layout/layout_model.py)

The layout model uses a **YOLO-based object detection model** from the `docling_IBM_models` package.

### 4a. Model Loading

- The model is loaded from a pre-trained `LayoutPredictor` (YOLO variant)
- Supports multiple model specs via `LayoutModelConfig` (default is DOCLING_LAYOUT_V2)
- Devices are auto-detected via `decide_device()` (CUDA, CPU)
- Models are downloaded automatically from HuggingFace via `download_hf_model()`

### 4b. Batch Prediction

- Takes rasterized page images and runs **batch inference** on the YOLO network
- Predicts bounding boxes + classification labels for each detected region (cluster)
- Label schema (~20 classes, defined as `DocItemLabel`):

| Label | Description |
|---|---|
| `Text` | Body text paragraph |
| `Title` / `SectionHeader` | Section headings |
| `Caption` | Figure/table caption text |
| `ListItem` | List item (bullet or numbered) |
| `Code` | Code block |
| `Formula` | Equation / math formula |
| `Table` | Table region |
| `Picture` | Image / figure |
| `CheckboxSelected` / `CheckboxUnselected` | Form checkboxes |
| `Form` | Form region |
| `KeyValueRegion` | Key-value pair region |
| `PageHeader` / `PageFooter` | Header/footer text |
| `DocumentIndex` | Table of contents |
| `Footnote` | Footnote text |
| `Bibliography` | References section |

### 4c. Layout Postprocessing

Raw YOLO output is refined by `LayoutPostprocessor` (`docling/utils/layout_postprocessor.py`):

- **Cluster merging**: nearby clusters of the same type are merged (e.g., split lines of the same paragraph)
- **Nested clusters**: child clusters are linked to parent clusters (e.g., list items under a list, table cells under a table, captions under figures)
- **Cell assignment**: each text/OCR cell is assigned to the bounding box cluster that contains it (via spatial overlap)

The output is a `LayoutPrediction` containing a list of `Cluster` objects, each with: id, label, confidence, bbox, and assigned cells.

### 4d. Quality Scoring

After layout prediction:

- `conv_res.confidence.pages[page_no].layout_score` = mean confidence of all detected clusters
- `conv_res.confidence.pages[page_no].ocr_score` = mean confidence of OCR-extracted cells

> **Key code**: `predict_layout()`, `LayoutPostprocessor`

---

## 5. Table Structure Extraction

**Source**: [`docling/models/base_table_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/base_table_model.py)

For clusters labeled as `Table` or `DocumentIndex`, a **TableFormer** model (Transformer-based, from `docling_IBM_models`) predicts the internal table structure:

- Determines the **grid layout** (N rows × M columns)
- For each cell, predicts: **row span**, **column span**, **row index**, **column index**
- Detects **header rows** and **column headers**
- Produces a `TableStructurePrediction` with a `table_map` mapping cluster IDs to structured `Table` objects

The table structure prediction is attached to the page's `predictions.tablestructure` and later used during assembly to construct the table's internal cell structure.

This stage is **optional** — controlled by `pipeline_options.do_table_structure` (default: `True`).

> **Key files**:
> - `docling/models/base_table_model.py` — abstract interface
> - `docling/models/factories/table_factory.py` — factory
> - TableFormer implementation in `docling_IBM_models` package

---

## 6. Figure / Picture Classification

**Source**: [`docling/models/picture_description_base_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/picture_description_base_model.py)

Pictures (clusters labeled `Picture`) can be further enriched via optional stages:

- **Chart detection**: identifies charts, enables data extraction
- **Picture description**: uses a VLM (Vision Language Model) to generate alt-text
- Controlled by `pipeline_options.do_picture_classification` and `do_picture_description`

These are enrichment stages, not required for basic conversion.

---

## 7. Page Assembly

**Source**: [`docling/models/stages/page_assemble/page_assemble_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/page_assemble/page_assemble_model.py)

This stage converts raw layout clusters + assigned cells into structured `PageElement` objects:

### 7a. Text Elements

For clusters in `TEXT_ELEM_LABELS`:

- Joins all cells' text within a cluster into a single string
- Applies **text normalization**:
  - **Hyphenation-aware line joining**: if a line ends with `-` and the next line starts with an alphanumeric word, joins them without a space (removing the hyphen)
  - **Unicode normalization**: converts `⁄` → `/`, `''` → `'`, `` ` `` → `'`, smart quotes to ASCII
  - **Ligature expansion** via regex matching Unicode Alphabetic Presentation Forms (U+FB00–U+FB06): `ﬀ` → `ff`, `ﬁ` → `fi`, `ﬂ` → `fl`, `ﬃ` → `ffi`, `ﬄ` → `ffl`, `ﬅ`/`ﬆ` → `st`
  - Dutch `Ĳ`/`ij` → `IJ`/`ij`; discards private-use glyph `U+F0A0`
- Matches **hyperlinks** by computing spatial overlap between cluster bbox and PDF annotation rectangles (requires >50% coverage)

### 7b. Table Elements

For `TABLE_LABELS` clusters:

- Attaches the `TableStructurePrediction` if available (provides grid structure)
- Falls back to a simple `Table` without structure if no prediction exists

### 7c. Figure Elements

For `PICTURE` clusters:

- Attaches classification prediction if available (chart type, etc.)
- Falls back to a bare `FigureElement`

### 7d. Container Elements

For `FORM` and `KEY_VALUE_REGION` clusters:

- Groups them as `ContainerElement` with appropriate label

Each element receives: id, label, text content, hyperlink, page number, spatial provenance, and cluster reference.

### 7e. Output

The assembled elements are grouped into:
- `elements`: all page elements in document order
- `headers`: page header/footer elements
- `body`: body elements (excludes headers)

Stored in `page.assembled = AssembledUnit(...)`.

> **Key code**: `__call__()`, `sanitize_text()`, `_match_hyperlink()`

---

## 8. Reading Order Determination

**Source**: [`docling/models/stages/reading_order/readingorder_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/reading_order/readingorder_model.py)

The assembled elements across all pages are reordered using a **reading-order predictor** (`ReadingOrderPredictor` from `docling_IBM_models`):

### 8a. Input Preparation

- Converts all page elements to `ReadingOrderPageElement` objects with: cid, ref, text, page_no, page_size, label, bbox
- All elements from all pages are flattened into a single list

### 8b. Prediction

The predictor outputs:

1. **`sorted_elements`**: elements sorted in predicted reading order
2. **`el_to_captions_mapping`**: maps element ID → list of caption element IDs
3. **`el_to_footnotes_mapping`**: maps element ID → list of footnote element IDs
4. **`el_merges_mapping`**: maps element ID → list of element IDs that should be concatenated (e.g., text flowing across columns or pages)

### 8c. Document Construction

The sorted elements are converted into a `DoclingDocument` tree:

- **Text elements**:
  - `SectionHeader` → heading node
  - `ListItem` → list item inside a list group
  - `Formula` → formula node (with original text preserved in `orig` field)
  - Others → text node with appropriate label
- **Tables**: `add_table()` with `TableData` (grid structure)
- **Pictures**: `add_picture()` with captions and footnotes attached
- **Containers**: `add_group()` for FORM and KEY_VALUE_REGION areas
- **Merges**: concatenates text with space separator (or without space for soft hyphens)
- **Captions/Footnotes**: linked to parent elements via reference items

The `ListItemMarkerProcessor` additionally infers list markers (bullets, numbers, letters).

> **Key code**: `__call__()`, `_readingorder_elements_to_docling_doc()`

---

## 9. Heading Hierarchy

**Source**: [`docling/models/stages/heading_hierarchy/heading_hierarchy_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/heading_hierarchy/heading_hierarchy_model.py)

- Builds a **heading tree** from the sequence of section headers
- Optionally uses **PDF bookmarks/ToC** (extracted from the backend's `get_document_outline()`) as additional signals
- Infers hierarchy levels (h1 → h2 → h3 → ...) based on spatial relationships and text patterns
- Produces a nested group structure in the DoclingDocument

This stage is **optional** — controlled by `heading_hierarchy_options.enabled` and `use_bookmarks`.

---

## 10. Pipeline Architecture

**Source**: [`docling/pipeline/standard_pdf_pipeline.py`](https://github.com/docling-project/docling/blob/main/docling/pipeline/standard_pdf_pipeline.py)

The modern pipeline (`StandardPdfPipeline`) is **multi-threaded**:

### 10a. Stage Wiring

Each pipeline stage runs in its own worker thread, connected by bounded queues:

```
PageProducer → Preprocess → OCR → Layout → Table → Assemble → Output
```

- Each stage batches pages for inference (configurable batch size)
- Back-pressure: producers block on full queues; queues have explicit close semantics
- Per-run isolation: each conversion uses its own queue and worker threads

### 10b. Build Phase (`_build_document`)

1. Assigns a monotonic `run_id` for tracking
2. Creates a producer thread that loads each PDF page's backend
3. Feeds `ThreadedItem(payload=page, ...)` into the preprocess queue
4. Workers process batches through each stage
5. Results are collected and integrated

### 10c. Assemble Phase (`_assemble_document`)

After all pages pass through the thread pipeline:

1. Collects assembled elements from all pages
2. Calls the **reading order model** on the full set of elements
3. Calls the **heading hierarchy model**
4. Generates optional page/element images
5. Aggregates confidence scores (layout, parse, table, OCR)
6. Adds failed/skipped pages to preserve page numbering

### 10d. Legacy Pipeline

The `LegacyStandardPdfPipeline` (`docling/pipeline/legacy_standard_pdf_pipeline.py`) is a simpler, **paginated** (non-threaded) version:

- Processes pages one at a time through the build pipe:
  ```python
  self.build_pipe = [
      PagePreprocessingModel(...),
      ocr_model,
      layout_model,
      table_model,
      PageAssembleModel(...),
  ]
  ```
- Applies enrichment pipe after assembly
- Uses `PaginatedPipeline` base class (iterates pages sequentially)

---

## 11. Model Factories

**Source**: [`docling/models/factories/__init__.py`](https://github.com/docling-project/docling/blob/main/docling/models/factories/__init__.py)

Four factories with plugin support:

| Factory | Purpose |
|---|---|
| `get_ocr_factory()` | Creates OCR engine instances (EasyOCR default) |
| `get_layout_factory()` | Creates layout detection model instances (YOLO default) |
| `get_table_structure_factory()` | Creates table structure model instances (TableFormer default) |
| `get_picture_description_factory()` | Creates picture description model instances (VLM) |

Each factory supports external plugins via `allow_external_plugins`. Models are loaded from artifacts (local directory or downloaded from HuggingFace).

---

## 12. Document Model (DoclingDocument)

**Source**: `docling-core` package (external dependency)

The output of the pipeline is a `DoclingDocument` — a tree-structured representation defined in the `docling-core` package. Key types:

- `DoclingDocument`: root document with name, origin, pages dict
- `PageItem`: page with page number and size
- `DocItem` (abstract base):
  - `TextItem` — text, heading, code, formula
  - `TableItem` — table with `TableData` (grid, cells)
  - `PictureItem` — image with captions, footnotes
  - `CodeItem` — code block
  - `ListItem` — list item with marker
- `NodeItem`: items with parent/child relationships (heading, table, picture, group)
- `GroupItem` — container (list, form area, key-value area)
- `ProvenanceItem`: spatial provenance (page_no, charspan, bbox)
- `RefItem`: cross-reference to another item (`#/{page_no}/{element_id}`)

---

## 13. Export to Markdown / HTML

**Source**: `docling-core` export converters

The `DoclingDocument` is exported via converters in `docling-core`:

- **MarkdownConverter**: iterates the document tree, emits:
  - Headings as `# ` prefixed lines
  - Lists as `- ` or numbered items
  - Tables as pipe tables or HTML `<table>`
  - Code blocks as fenced ``` blocks
  - Figures with alt-text
  - Page break markers

- **HTMLConverter**: emits semantic HTML with proper tags (`<h1>`–`<h6>`, `<table>`, `<figure>`, `<pre>`, `<ul>`, etc.)

---

## Complete Processing Flow

```
PDF Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Backend (pdfium / pypdfium2)                        │
│   • Programmatic text → char_cells, word_cells      │
│   • Bitmap regions → bitmap_rects                   │
│   • Hyperlinks, bookmarks, page size                │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Page Preprocessing                                  │
│   • Rasterize page to image                         │
│   • Extract text cells                              │
│   • Rate text quality (heuristic score)             │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ OCR (EasyOCR, selective regions)                    │
│   • Analyze bitmap coverage                         │
│   • OCR only bitmap regions or full page            │
│   • Filter overlapping OCR cells                    │
│   • Merge with programmatic cells                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Layout (YOLOv8 object detection)                    │
│   • Batch predict ~20 label classes                 │
│   • Postprocess: merge, nest, assign cells          │
│   • Output: LayoutPrediction (clusters + cells)     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Table Structure (TableFormer)                       │
│   • Predict grid layout (N rows × M cols)           │
│   • Predict row/col spans, headers                  │
│   • Output: TableStructurePrediction                │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Page Assembly                                       │
│   • Cluster → PageElement (Text, Table, Fig, ...)   │
│   • Text normalization: hyphens, ligatures, unicode │
│   • Hyperlink matching                              │
│   • Attach table structure, figure classification   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Reading Order (sequence model)                      │
│   • Predict total ordering across all pages         │
│   • Predict caption/footnote/merge mappings         │
│   • Build DoclingDocument tree                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Heading Hierarchy                                   │
│   • Build heading tree (optionally use PDF ToC)     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ DoclingDocument (tree-structured)                   │
│   • Pages with size metadata                        │
│   • DocItems with provenance                        │
│   • Captions, footnotes, hyperlinks                 │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Export                                              │
│   • MarkdownConverter → Markdown                    │
│   • HTMLConverter → HTML                            │
│   • JSON → DoclingDocument JSON                     │
└─────────────────────────────────────────────────────┘
```

---

## Key Dependencies (External)

| Package | Role |
|---|---|
| `docling-core` | Document model, export converters, base types |
| `docling-IBM-models` | Layout predictor (YOLO), reading order, list item processor, TableFormer |
| `easyocr` | OCR engine (default) |
| `pypdfium2` | PDF rendering and text extraction (pdfium bindings) |
| `Pillow` | Image manipulation |
| `opencv-python` (cv2) | Image dilation for bitmap merging |
| `scipy` | Connected-component analysis (fallback for bitmap detection) |
| `rtree` | Spatial index for OCR cell filtering |

---

## References to Upstream Source

All source code references point to the `main` branch of [`docling-project/docling`](https://github.com/docling-project/docling):

| Component | Source File |
|---|---|
| Document converter entry point | [`docling/document_converter.py`](https://github.com/docling-project/docling/blob/main/docling/document_converter.py) |
| Standard PDF pipeline (threaded) | [`docling/pipeline/standard_pdf_pipeline.py`](https://github.com/docling-project/docling/blob/main/docling/pipeline/standard_pdf_pipeline.py) |
| Legacy PDF pipeline | [`docling/pipeline/legacy_standard_pdf_pipeline.py`](https://github.com/docling-project/docling/blob/main/docling/pipeline/legacy_standard_pdf_pipeline.py) |
| Base pipeline | [`docling/pipeline/base_pipeline.py`](https://github.com/docling-project/docling/blob/main/docling/pipeline/base_pipeline.py) |
| PDF backend | [`docling/backend/pdf_backend.py`](https://github.com/docling-project/docling/blob/main/docling/backend/pdf_backend.py) |
| PyPdfium2 backend | [`docling/backend/pypdfium2_backend.py`](https://github.com/docling-project/docling/blob/main/docling/backend/pypdfium2_backend.py) |
| Page preprocessing | [`docling/models/stages/page_preprocessing/page_preprocessing_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/page_preprocessing/page_preprocessing_model.py) |
| OCR model (base) | [`docling/models/base_ocr_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/base_ocr_model.py) |
| Layout model | [`docling/models/stages/layout/layout_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/layout/layout_model.py) |
| Table model (base) | [`docling/models/base_table_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/base_table_model.py) |
| Page assembly | [`docling/models/stages/page_assemble/page_assemble_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/page_assemble/page_assemble_model.py) |
| Reading order | [`docling/models/stages/reading_order/readingorder_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/reading_order/readingorder_model.py) |
| Heading hierarchy | [`docling/models/stages/heading_hierarchy/heading_hierarchy_model.py`](https://github.com/docling-project/docling/blob/main/docling/models/stages/heading_hierarchy/heading_hierarchy_model.py) |
| Model factories | [`docling/models/factories/__init__.py`](https://github.com/docling-project/docling/blob/main/docling/models/factories/__init__.py) |
| Layout postprocessor | [`docling/utils/layout_postprocessor.py`](https://github.com/docling-project/docling/blob/main/docling/utils/layout_postprocessor.py) |
| Pipeline options | [`docling/datamodel/pipeline_options.py`](https://github.com/docling-project/docling/blob/main/docling/datamodel/pipeline_options.py) |
| Document model | [`docling/datamodel/document.py`](https://github.com/docling-project/docling/blob/main/docling/datamodel/document.py) |
| Base models | [`docling/datamodel/base_models.py`](https://github.com/docling-project/docling/blob/main/docling/datamodel/base_models.py) |

---

## Design Principles

1. **Modular staged pipeline**: each sub-task is an independent stage that can be enabled/disabled/swapped
2. **Programmatic-first, OCR-second**: prefers native PDF text, uses OCR only where the parser fails or bitmap content exists
3. **Factory pattern for models**: OCR, layout, table, and picture models are pluggable via factories
4. **Spatial reasoning**: all element relationships (parent-child, reading order, merges) are inferred from bbox geometry
5. **Multi-threaded throughput**: the threaded pipeline uses bounded queues and worker threads for parallel processing
6. **Rich provenance**: every element carries spatial provenance (page, bbox, charspan) enabling downstream tasks
7. **Quality scoring**: heuristic and model-based scores at each stage for diagnostics and fallback decisions