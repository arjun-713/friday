# Chunk generation

Chunk generation produces several deterministic retrieval views from each cleaned manual. It does not ask an LLM to decide boundaries and it does not rewrite source evidence.

## Strategies

| Strategy | Chunk kind | Boundary rule | Retrieval use |
| --- | --- | --- | --- |
| `section` | `section` | Section path, paragraph boundaries, soft character cap | Normal semantic/narrative questions |
| `parent_child` | `parent`, `child` | Bounded section parents; smaller child windows with bounded overlap | Child retrieval with parent context expansion |
| `procedure` | `procedure` | Complete ordered procedure, including detected prerequisites and warnings | Step-by-step troubleshooting |
| `table_row` | `table_row` | One verified Markdown table row plus its headers | Error-code and troubleshooting lookup |
| `exact_match` | `exact_match` | One context-gated code, blink pattern, part number, or model occurrence | Exact lexical lookup |

Section and parent/child chunks use paragraph-aware packing. A paragraph is not split unless one individual line exceeds the limit; an oversized line then falls back to word-boundary splitting. Child windows overlap only at complete paragraph units and record `overlap_before` and `overlap_after`.

Procedure, table, and exact-match chunks are atomic. This prevents a warning or table header from being separated from the evidence it qualifies. These specialized chunks intentionally overlap with narrative chunks: they are alternate retrieval representations, not duplicate source claims.

Every emitted chunk has a strategy, kind, section, ordered pages, parser marker, source evidence, source document metadata, and deterministic ID. Child chunks point to `parent_chunk_id`. Strategy and metadata fields allow later BM25/vector indexes to filter or boost representations without changing the stored evidence.

## Retrieval projections

Chunk generation does not create embeddings or connect to a vector database. It records the intended retrieval profile on every chunk so indexing can be added as a separate, measurable phase:

| Profile | Chunks | Purpose |
| --- | --- | --- |
| `exact` | Exact identifiers and table rows | Codes, model numbers, part numbers, blink patterns |
| `bm25` | Section, child, procedure-step, table-row, and exact chunks | Exact terminology, symptoms, error codes, and procedures |
| `vector` | Section, child, procedure-step, and table-row chunks | Later semantic retrieval |
| `context_store` | Parents and canonical complete procedures | Expand context after a child or step-group hit |

This keeps parent context available without forcing every large parent into the first-stage search results. Device model, manufacturer, document version, source page, section, strategy, and profile remain available for metadata filtering and rank fusion in the later retrieval phase.

Run the generator from the repository root after parsing, cleaning, and source metadata resolution:

```bash
PYTHONPATH=backend/src python -m copilot.ingestion.chunking.runner
```

For the normal workflow after adding or replacing manuals, use the root Makefile:

```bash
make ingest
```

This runs parsing, metadata resolution, cleaning, and chunk generation in order. The individual `make prepare` and `make chunk` targets are available when only one part of the RAG ingestion flow needs to be repeated.

It writes one JSONL file per manual under `data/chunks/` and a strategy-count report at `data/chunks/chunk_report.json`. Generated artifacts are ignored by Git.
