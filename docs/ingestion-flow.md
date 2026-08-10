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

The reproducible runner is `PYTHONPATH=backend/src python -m copilot.ingestion.parsing.text_only` from the repository root; it writes the corpus report to `data/raw/parse_report.json`.

`PYTHONPATH=backend/src python -m copilot.ingestion.parsing.native` is the active corpus runner. It processes all 21 retained manuals without OCR and writes per-document `ocr_required_pages` metadata. The PP-OCR benchmark runner remains available as an optional experiment, but is not part of the default ingestion flow.

The next deterministic stage is `PYTHONPATH=backend/src python -m copilot.ingestion.cleaning.runner`, which writes a mirrored `data/cleaned/{computers,routers,printers}` tree. The original text and positioned spans remain immutable in `data/raw/`. Cleaned pages retain their page identity, a raw-source pointer, `span_count`, normalization counts, and `removed_fragments`; coordinates are resolved from raw by source file and page rather than duplicated.

Cleanup removes repeated running titles after preserving their first occurrence, isolated page numbers, copyright/navigation boilerplate, non-semantic HTML formatting, malformed Markdown markers, and dot leaders. Contents, empty, and exact-duplicate pages remain represented but are marked `excluded_from_chunking` and carry an exclusion reason. Warnings, procedures, tables, codes, URLs, and source page numbering are preserved.

The chunk schema and provenance rules are defined in [`docs/chunk-contract.md`](chunk-contract.md). The contract is validated before a future chunk can enter BM25 or vector indexes.

The first structure pass is documented in [`docs/structure-detection.md`](structure-detection.md). It assigns deterministic section paths but does not yet generate chunks.

The next pass, documented in [`docs/procedure-extraction.md`](procedure-extraction.md), identifies ordered procedure candidates while preserving prerequisites, warnings, step order, and page evidence.

Table rows are handled by the deterministic pass documented in [`docs/table-extraction.md`](table-extraction.md). It requires explicit Markdown table structure before emitting row candidates.

Exact technical identifiers are handled by [`docs/exact-match-extraction.md`](exact-match-extraction.md) using context-gated patterns for error codes, blink patterns, part numbers, and model numbers.

Install the native binding with the backend extra before using it:

```bash
cd backend
pip install -e '.[dev]'
```

## Non-negotiable checks

- Original files remain immutable.
- OCR is page-selective and records confidence.
- Procedures retain prerequisites, warnings, and order.
- Every chunk has document, model, version, section, page, parser, and evidence metadata.
- A chunk without page and section evidence cannot enter retrieval indexes.
- `pdf-inspector` page positions are converted from its 0-based page values to the application's 1-based citation pages.
