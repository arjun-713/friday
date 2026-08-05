# Chunk contract

This contract defines the artifact produced by the future structure-aware chunker. It does not perform chunking or indexing.

## Required chunk metadata

Each `DocumentChunk` must contain:

- `chunk_id`: stable identifier for the generated chunk.
- `document`: resolved document identity, including model and version.
- `page`: the first one-based citation page.
- `pages`: sorted, unique one-based pages covered by the chunk.
- `section`: non-empty section path.
- `content`: non-empty retrieval text.
- `kind`: content type: `section`, `parent`, `child`, `procedure`, `table_row`, or `exact_match`.
- `parser`: parser identifier and version when available.
- `evidence`: at least one source-backed evidence record.

`kind` is the schema's content-type field. It is intentionally deterministic and does not require an LLM classification call.

## Evidence requirements

Every evidence record contains:

- `source_file`: path or registry key for the immutable file in `data/raw`.
- `page`: one-based page number.
- `section`: section label at the evidence location.
- `content`: the exact or normalized source excerpt supporting the chunk.
- `coordinates`: optional `(x, y, width, height)` coordinates resolved from the raw page spans.

Evidence pages must be included in `DocumentChunk.pages`, all evidence records must reference the same raw source file, and the first chunk page must equal `DocumentChunk.page`.

Chunks with unresolved model/version metadata, missing evidence, empty content, invalid page numbers, or inconsistent provenance must fail validation and must not enter a retrieval index. Page-level parser/OCR confidence remains on `PageRecord` and `OcrPage`; no undefined chunk-level confidence is inferred.

The cleaned corpus is retrieval text. Coordinates are resolved from the immutable raw JSON using `source_file` and page; they are not duplicated into cleaned pages.
