"""Batch embedding and vector upsert orchestration."""

from collections.abc import Sequence

from ..ingestion.models import DocumentChunk, RetrievalProfile
from .contracts import EmbeddingProvider, VectorIndex, VectorRecord
from .qdrant import chunk_payload


def vector_chunks(chunks: Sequence[DocumentChunk]) -> list[DocumentChunk]:
    return [chunk for chunk in chunks if RetrievalProfile.VECTOR in chunk.retrieval_profiles]


async def index_chunks(
    chunks: Sequence[DocumentChunk],
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndex,
    batch_size: int = 128,
) -> int:
    selected = vector_chunks(chunks)
    await vector_index.ensure_collection(embedding_provider.dimension)
    indexed = 0
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        vectors = await embedding_provider.embed_documents([chunk.content for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError("embedding provider returned a different number of vectors than inputs")
        records = [
            VectorRecord(id=chunk.chunk_id, vector=vector, payload=chunk_payload(chunk), chunk=chunk)
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        await vector_index.upsert(records)
        indexed += len(records)
    return indexed
