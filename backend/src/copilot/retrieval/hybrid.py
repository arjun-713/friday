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
    diagnostics: dict[str, object] = Field(default_factory=dict)


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
    candidate_limit: int | None = None,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    rrf_k: int = 60,
    diversify: bool = False,
    include_diagnostics: bool = False,
    abstention_dense_threshold: float | None = None,
) -> RetrievalResult:
    """Run dense and lexical retrieval concurrently, then expand parents in one batch."""

    if limit <= 0:
        raise ValueError("retrieval limit must be positive")
    if candidate_limit is None:
        candidate_limit = max(limit * 4, 32)
    if candidate_limit < limit:
        raise ValueError("candidate limit must be greater than or equal to final limit")
    if dense_weight < 0 or lexical_weight < 0 or dense_weight + lexical_weight == 0:
        raise ValueError("retrieval source weights must be non-negative and not both zero")
    if rrf_k <= 0:
        raise ValueError("RRF rank constant must be positive")
    if abstention_dense_threshold is not None and not 0 <= abstention_dense_threshold <= 1:
        raise ValueError("abstention dense threshold must be between zero and one")
    total_started = perf_counter()
    parallel_started = perf_counter()
    (query_vector, embedding_ms), (lexical_hits, lexical_ms) = await asyncio.gather(
        _timed_embedding(embedding_provider, query),
        _timed_lexical(lexical_retriever, query, metadata_filter, candidate_limit),
    )
    parallel_ms = _elapsed_ms(parallel_started)
    dense_started = perf_counter()
    vector_hits = await vector_index.search(
        query_vector,
        metadata_filter=metadata_filter,
        limit=candidate_limit,
        exact=exact,
        candidate_count=candidate_count,
        score_threshold=score_threshold,
    )
    dense_ms = _elapsed_ms(dense_started)
    fusion_started = perf_counter()
    hits = _rrf_fuse(
        vector_hits,
        lexical_hits,
        limit,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
        rrf_k=rrf_k,
        diversify=diversify,
    )
    fusion_ms = _elapsed_ms(fusion_started)
    top_dense_score = vector_hits[0].score if vector_hits else 0.0
    if abstention_dense_threshold is not None and top_dense_score < abstention_dense_threshold:
        return RetrievalResult(
            abstained=True,
            reason="low_dense_confidence",
            timings_ms={
                "parallel_embed_lexical_ms": parallel_ms,
                "embedding_ms": embedding_ms,
                "lexical_ms": lexical_ms,
                "dense_search_ms": dense_ms,
                "fusion_ms": fusion_ms,
                "parent_fetch_ms": 0.0,
                "total_ms": _elapsed_ms(total_started),
            },
            diagnostics=_diagnostics(vector_hits, lexical_hits, candidate_limit) if include_diagnostics else {},
        )
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
            diagnostics=_diagnostics(vector_hits, lexical_hits, candidate_limit) if include_diagnostics else {},
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
        diagnostics=_diagnostics(vector_hits, lexical_hits, candidate_limit) if include_diagnostics else {},
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


def _rrf_fuse(
    vector_hits: Sequence[VectorHit],
    lexical_hits: Sequence[VectorHit],
    limit: int,
    *,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    rrf_k: int = 60,
    diversify: bool = False,
) -> list[VectorHit]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, object]] = {}
    for weight, results in ((dense_weight, vector_hits), (lexical_weight, lexical_hits)):
        for rank, hit in enumerate(results, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (rrf_k + rank)
            payloads.setdefault(hit.id, hit.payload)
    ranked_candidates = sorted(scores, key=lambda item: scores[item], reverse=True)
    ranked = _diversified_ids(ranked_candidates, payloads, limit) if diversify else ranked_candidates[:limit]
    return [VectorHit(id=chunk_id, score=scores[chunk_id], payload=payloads[chunk_id]) for chunk_id in ranked]


def _diversified_ids(ranked_candidates: Sequence[str], payloads: dict[str, dict[str, object]], limit: int) -> list[str]:
    selected: list[str] = []
    seen_groups: set[str] = set()
    deferred: list[str] = []
    for chunk_id in ranked_candidates:
        group = _evidence_group(payloads[chunk_id])
        if group in seen_groups:
            deferred.append(chunk_id)
            continue
        selected.append(chunk_id)
        seen_groups.add(group)
        if len(selected) == limit:
            return selected
    for chunk_id in deferred:
        if len(selected) == limit:
            break
        selected.append(chunk_id)
    return selected


def _evidence_group(payload: dict[str, object]) -> str:
    parent = payload.get("parent_chunk_id")
    if isinstance(parent, str) and parent:
        return f"parent:{parent}"
    return ":".join(str(payload.get(field, "")) for field in ("document_id", "page", "section"))


def _diagnostics(
    vector_hits: Sequence[VectorHit], lexical_hits: Sequence[VectorHit], candidate_limit: int
) -> dict[str, object]:
    return {
        "candidate_limit": candidate_limit,
        "dense_candidate_count": len(vector_hits),
        "lexical_candidate_count": len(lexical_hits),
        "dense_ranks": {hit.id: rank for rank, hit in enumerate(vector_hits, start=1)},
        "lexical_ranks": {hit.id: rank for rank, hit in enumerate(lexical_hits, start=1)},
        "dense_scores": {hit.id: hit.score for hit in vector_hits},
        "lexical_scores": {hit.id: hit.score for hit in lexical_hits},
        "source_overlap_count": len({hit.id for hit in vector_hits} & {hit.id for hit in lexical_hits}),
    }
