"""Retrieval contracts, benchmark helpers, and the Qdrant-backed dense index."""

from .benchmark import BenchmarkQuery, BenchmarkResult, run_benchmark
from .bm25 import CombinedLexicalRetriever, InMemoryBM25Retriever, InMemoryExactIdentifierRetriever
from .contracts import (
    EmbeddingProvider,
    MetadataFilter,
    VectorHit,
    VectorIndex,
    VectorRecord,
)
from .granite import GraniteEmbeddingProvider, GraniteEmbeddingSettings
from .qdrant import QdrantSettings, QdrantVectorIndex

__all__ = [
    "BenchmarkQuery",
    "BenchmarkResult",
    "CombinedLexicalRetriever",
    "EmbeddingProvider",
    "GraniteEmbeddingProvider",
    "GraniteEmbeddingSettings",
    "InMemoryBM25Retriever",
    "InMemoryExactIdentifierRetriever",
    "MetadataFilter",
    "QdrantSettings",
    "QdrantVectorIndex",
    "VectorHit",
    "VectorIndex",
    "VectorRecord",
    "run_benchmark",
]
