"""Hybrid dense/lexical retrieval orchestration without a BM25 dependency."""

import asyncio
from collections.abc import Sequence
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

    query_vector, lexical_hits = await asyncio.gather(
        embedding_provider.embed_query(query),
        lexical_retriever.search(query, metadata_filter, limit * 2) if lexical_retriever is not None else _empty_hits(),
    )
    vector_hits = await vector_index.search(
        query_vector,
        metadata_filter=metadata_filter,
        limit=limit * 2,
        exact=exact,
        candidate_count=candidate_count,
        score_threshold=score_threshold,
    )
    hits = _rrf_fuse(vector_hits, lexical_hits, limit)
    if not hits:
        return RetrievalResult(abstained=True, reason="no_retrieval_hits")
    parent_ids = [str(hit.payload["parent_chunk_id"]) for hit in hits if hit.payload.get("parent_chunk_id")]
    parents = await parent_store.fetch(parent_ids) if parent_store is not None else []
    return RetrievalResult(hits=hits, parents=parents)


async def _empty_hits() -> list[VectorHit]:
    return []


def _rrf_fuse(vector_hits: Sequence[VectorHit], lexical_hits: Sequence[VectorHit], limit: int) -> list[VectorHit]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, object]] = {}
    for results in (vector_hits, lexical_hits):
        for rank, hit in enumerate(results, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1 / (60 + rank)
            payloads.setdefault(hit.id, hit.payload)
    ranked = sorted(scores, key=lambda item: scores[item], reverse=True)[:limit]
    return [VectorHit(id=chunk_id, score=scores[chunk_id], payload=payloads[chunk_id]) for chunk_id in ranked]
