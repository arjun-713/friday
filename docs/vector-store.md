# Local Qdrant vector store

Qdrant is the local persistent dense-vector store for the retrieval phase. It runs through [`docker-compose.qdrant.yml`](../docker-compose.qdrant.yml) with storage mounted in the `qdrant_storage` Docker volume.

Start it with:

```bash
make qdrant-up
```

Stop the service with:

```bash
make qdrant-down
```

The application uses the async `QdrantVectorIndex` adapter. Retrieval code depends on the provider-neutral [`VectorIndex`](../backend/src/copilot/retrieval/contracts.py) protocol rather than Qdrant types.

## Collection contract

There is one collection: `troubleshooting_chunks`. Its dimension is created from `EmbeddingProvider.dimension`, and its distance is cosine. Reusing an existing collection with a different embedding dimension fails closed.

Only chunks whose `retrieval_profiles` contain `vector` are embedded and upserted. Context-only parents remain in the local generated JSONL context store and are fetched in one batch after child hits; they are not assigned dummy vectors.

Qdrant payload contains the complete serialized chunk plus flattened filter fields. Payload indexes are created only for fields used by the current filter contract:

```text
manufacturer
model
document_id
document_version
parent_chunk_id
kind
strategy
normalized_value
```

Qdrant point IDs are deterministic UUID5 values derived from the stable chunk IDs because Qdrant point IDs must be integers or UUIDs. The original stable `chunk_id` remains in payload and is returned by the adapter.

## Search behavior

Normal search uses HNSW. Callers can set `exact=True` for small or heavily filtered candidate sets, or leave the decision to the adapter with `exact=None` and `candidate_count`.

BM25 and exact identifier retrieval are deliberately outside Qdrant. The hybrid boundary runs dense and lexical search concurrently, fuses results with reciprocal rank fusion, applies abstention thresholds, and batch-fetches parent context through `ParentChunkStore`.

No embedding provider is bundled yet. The provider must implement `EmbeddingProvider`, expose its dimension dynamically, and support batch document embedding plus single-query embedding.

## Latency measurements

`retrieval.metrics` provides trace marks and P50/P70/P99/max summaries. Benchmark retrieval with the same query set and embedding provider across:

- filtered HNSW search;
- exact search on small filtered subsets;
- parent batch-fetch latency;
- Recall@5 and MRR;
- P50, P70, P99, and maximum retrieval latency.

Embedding, Qdrant search, lexical search, fusion, parent expansion, context readiness, first LLM token, and final output must be measured separately. Chunking and embedding ingestion are offline and must not be included in query latency.
