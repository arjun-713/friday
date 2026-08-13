"""Provider-neutral contracts for embedding and dense retrieval."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from ..ingestion.models import DocumentChunk


class EmbeddingProvider(Protocol):
    """An embedding implementation supplied independently of Qdrant."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class MetadataFilter(BaseModel):
    manufacturer: str | None = None
    model: str | None = None
    document_id: str | None = None
    document_version: str | None = None
    parent_chunk_id: str | None = None
    kind: str | None = None
    strategy: str | None = None
    normalized_value: str | None = None


class VectorRecord(BaseModel):
    id: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    payload: dict[str, Any]
    chunk: DocumentChunk


class VectorHit(BaseModel):
    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorIndex(Protocol):
    async def ensure_collection(self, dimension: int) -> None: ...

    async def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    async def search(
        self,
        vector: Sequence[float],
        metadata_filter: MetadataFilter | None = None,
        limit: int = 10,
        exact: bool | None = False,
        candidate_count: int | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorHit]: ...

    async def fetch(self, ids: Sequence[str]) -> list[VectorHit]: ...


class Timing(TypedDict):
    name: str
    milliseconds: float


Clock = Callable[[], float]
AwaitableFactory = Callable[[], Awaitable[Any]]
