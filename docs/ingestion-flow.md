# Ingestion flow

The first implementation milestone is deliberately limited to document ingestion contracts.

```text
source registry
  -> download and checksum
  -> PDF classification
  -> page extraction
  -> selective OCR
  -> layout reconstruction
  -> normalization
  -> structure-aware chunking
  -> provenance validation
  -> BM25/vector/raw-page indexing
```

## Current scaffold

`backend/src/copilot/ingestion/models.py` defines the source, page, chunk, evidence, and positioned-text schemas. `stages.py` defines dependency-injected stage protocols and the pipeline coordinator. `pdf_inspector.py` is the unexecuted adapter for Firecrawl's `pdf-inspector` Python bindings: it performs detection, extracts per-page Markdown, preserves positioned text, and marks pages needing OCR.

The active ingestion path deliberately does not perform OCR. Scanned or mixed pages are parsed with whatever native text is available and retain the `requires_ocr` flag for future review. OCR remains optional and is not required for the corpus to be ingested.

Text-only parsed artifacts mirror the manual source tree:

```text
data/manuals/computers/<manual>.pdf  -> data/raw/computers/<manual>.json
data/manuals/routers/<manual>.pdf    -> data/raw/routers/<manual>.json
data/manuals/printers/<manual>.pdf   -> data/raw/printers/<manual>.json
```

The reproducible runner is `scripts/parse_text_pdfs.py`; it writes the corpus report to `data/raw/parse_report.json`.

`scripts/parse_native_pdfs.py` is the active corpus runner. It processes all 24 manuals without OCR and writes per-document `ocr_required_pages` metadata. The PP-OCR benchmark adapter remains available as an optional experiment, but is not part of the default ingestion flow.

Install the native binding with the backend extra before using it:

```bash
cd backend
pip install -e '.[dev]'
```

## Non-negotiable checks

- Original files remain immutable.
- OCR is page-selective and records confidence.
- Procedures retain prerequisites, warnings, and order.
- Every chunk has document, model, version, section, page, parser, confidence, and evidence metadata.
- A chunk without page and section evidence cannot enter retrieval indexes.
- `pdf-inspector` page positions are converted from its 0-based page values to the application's 1-based citation pages.
