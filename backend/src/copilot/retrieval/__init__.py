"""Retrieval contracts, benchmark helpers, and the Qdrant-backed dense index."""

from .benchmark import BenchmarkQuery, BenchmarkResult, run_benchmark
from .contracts import (
    EmbeddingProvider,
    MetadataFilter,
    VectorHit,
    VectorIndex,
    VectorRecord,
)
from .qdrant import QdrantSettings, QdrantVectorIndex

__all__ = [
    "BenchmarkQuery",
    "BenchmarkResult",
    "EmbeddingProvider",
    "MetadataFilter",
    "QdrantSettings",
    "QdrantVectorIndex",
    "VectorHit",
    "VectorIndex",
    "VectorRecord",
    "run_benchmark",
]
