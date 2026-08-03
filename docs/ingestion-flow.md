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

`backend/src/copilot/ingestion/models.py` defines the source, page, chunk, and evidence schemas. `stages.py` defines dependency-injected stage protocols and the pipeline coordinator. Real PDF/OCR/index adapters are intentionally not implemented yet.

## Non-negotiable checks

- Original files remain immutable.
- OCR is page-selective and records confidence.
- Procedures retain prerequisites, warnings, and order.
- Every chunk has document, model, version, section, page, parser, confidence, and evidence metadata.
- A chunk without page and section evidence cannot enter retrieval indexes.
