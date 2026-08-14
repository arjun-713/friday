"""Hybrid dense/lexical retrieval orchestration without a BM25 dependency."""

import asyncio
from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, Field

from ..ingestion.models import DocumentChunk
from .contracts import EmbeddingProvider, MetadataFilter, VectorHit, VectorIndex


class LexicalRetriever(Protocol):
    async def search(
        self,
        query: str,
        metadata_filter: MetadataFilter | None = None,
        limit: int = 10,
    ) -> list[VectorHit]: ...


class ParentChunkStore(Protocol):
    async def fetch(self, ids: Sequence[str]) -> list[DocumentChunk]: ...


class RetrievalResult(BaseModel):
    hits: list[VectorHit] = Field(default_factory=list)
    parents: list[DocumentChunk] = Field(default_factory=list)
    abstained: bool = False
    reason: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


async def retrieve(
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndex,
    lexical_retriever: LexicalRetriever | None = None,
    parent_store: ParentChunkStore | None = None,
    metadata_filter: MetadataFilter | None = None,
    limit: int = 8,
    exact: bool | None = False,
    candidate_count: int | None = None,
    score_threshold: float | None = None,
) -> RetrievalResult:
    """Run dense and lexical retrieval concurrently, then expand parents in one batch."""

    total_started = perf_counter()
    parallel_started = perf_counter()
    (query_vector, embedding_ms), (lexical_hits, lexical_ms) = await asyncio.gather(
        _timed_embedding(embedding_provider, query),
        _timed_lexical(lexical_retriever, query, metadata_filter, limit * 2),
    )
    parallel_ms = _elapsed_ms(parallel_started)
    dense_started = perf_counter()
    vector_hits = await vector_index.search(
        query_vector,
        metadata_filter=metadata_filter,
        limit=limit * 2,
        exact=exact,
        candidate_count=candidate_count,
        score_threshold=score_threshold,
    )
    dense_ms = _elapsed_ms(dense_started)
    fusion_started = perf_counter()
    hits = _rrf_fuse(vector_hits, lexical_hits, limit)
    fusion_ms = _elapsed_ms(fusion_started)
    if not hits:
        return RetrievalResult(
            abstained=True,
            reason="no_retrieval_hits",
            timings_ms={
                "parallel_embed_lexical_ms": parallel_ms,
                "embedding_ms": embedding_ms,
                "lexical_ms": lexical_ms,
                "dense_search_ms": dense_ms,
                "fusion_ms": fusion_ms,
                "parent_fetch_ms": 0.0,
                "total_ms": _elapsed_ms(total_started),
            },
        )
    parent_ids = [str(hit.payload["parent_chunk_id"]) for hit in hits if hit.payload.get("parent_chunk_id")]
    parent_started = perf_counter()
    parents = await parent_store.fetch(parent_ids) if parent_store is not None else []
    return RetrievalResult(
        hits=hits,
        parents=parents,
        timings_ms={
            "parallel_embed_lexical_ms": parallel_ms,
            "embedding_ms": embedding_ms,
            "lexical_ms": lexical_ms,
            "dense_search_ms": dense_ms,
            "fusion_ms": fusion_ms,
            "parent_fetch_ms": _elapsed_ms(parent_started),
            "total_ms": _elapsed_ms(total_started),
        },
    )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


async def _empty_hits() -> list[VectorHit]:
    return []


async def _timed_embedding(provider: EmbeddingProvider, query: str) -> tuple[list[float], float]:
    started = perf_counter()
    vector = await provider.embed_query(query)
    return vector, _elapsed_ms(started)


async def _timed_lexical(
    retriever: LexicalRetriever | None,
    query: str,
    metadata_filter: MetadataFilter | None,
    limit: int,
) -> tuple[list[VectorHit], float]:
    started = perf_counter()
    hits = await retriever.search(query, metadata_filter, limit) if retriever is not None else await _empty_hits()
    return hits, _elapsed_ms(started)


def _rrf_fuse(vector_hits: Sequence[VectorHit], lexical_hits: Sequence[VectorHit], limit: int) -> list[VectorHit]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, object]] = {}
    for results in (vector_hits, lexical_hits):
        for rank, hit in enumerate(results, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1 / (60 + rank)
            payloads.setdefault(hit.id, hit.payload)
    ranked = sorted(scores, key=lambda item: scores[item], reverse=True)[:limit]
    return [VectorHit(id=chunk_id, score=scores[chunk_id], payload=payloads[chunk_id]) for chunk_id in ranked]
