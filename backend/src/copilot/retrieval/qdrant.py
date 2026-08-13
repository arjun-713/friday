"""Async Qdrant implementation of the provider-neutral vector index."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import AsyncQdrantClient, models

from ..ingestion.models import DocumentChunk
from .contracts import MetadataFilter, VectorHit, VectorIndex, VectorRecord

COLLECTION_NAME = "troubleshooting_chunks"
PAYLOAD_INDEX_FIELDS = (
    "manufacturer",
    "model",
    "document_id",
    "document_version",
    "parent_chunk_id",
    "kind",
    "strategy",
    "normalized_value",
)


@dataclass(frozen=True)
class QdrantSettings:
    url: str = "http://localhost:6333"
    collection_name: str = COLLECTION_NAME
    batch_size: int = 128
    exact_candidate_threshold: int = 128


def chunk_payload(chunk: DocumentChunk) -> dict[str, Any]:
    """Flatten stable retrieval metadata into Qdrant payload fields."""

    payload = chunk.model_dump(mode="json")
    document = payload.pop("document")
    payload.update(
        {
            "text": chunk.content,
            "document_id": document["document_id"],
            "manufacturer": document["manufacturer"],
            "model": document["model"],
            "document_version": document["version"],
            "category": _category_from_source(chunk.evidence[0].source_file),
            "document_title": document["title"],
            "source_url": document["source_url"],
        }
    )
    for field in ("identifier_kind", "normalized_value"):
        if field in chunk.metadata:
            payload[field] = chunk.metadata[field]
    return payload


class QdrantVectorIndex(VectorIndex):
    """Small async adapter; no retrieval code imports Qdrant models directly."""

    def __init__(self, settings: QdrantSettings | None = None, client: AsyncQdrantClient | None = None) -> None:
        self.settings = settings or QdrantSettings()
        self.client = client or AsyncQdrantClient(url=self.settings.url)
        self._dimension: int | None = None

    async def close(self) -> None:
        await self.client.close()

    async def ensure_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        exists = await self.client.collection_exists(self.settings.collection_name)
        if exists:
            info = await self.client.get_collection(self.settings.collection_name)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict) or vectors is None:
                raise ValueError("troubleshooting collection must use a single unnamed vector")
            configured = vectors.size
            if configured != dimension:
                raise ValueError(f"collection dimension is {configured}, embedding provider dimension is {dimension}")
        else:
            await self.client.create_collection(
                collection_name=self.settings.collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        self._dimension = dimension
        for field in PAYLOAD_INDEX_FIELDS:
            await self.client.create_payload_index(
                collection_name=self.settings.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        if self._dimension is None:
            raise RuntimeError("call ensure_collection before upsert")
        for start in range(0, len(records), self.settings.batch_size):
            batch = records[start : start + self.settings.batch_size]
            await self.client.upsert(
                collection_name=self.settings.collection_name,
                points=[
                    models.PointStruct(id=_point_id(item.id), vector=item.vector, payload=item.payload)
                    for item in batch
                ],
                wait=True,
            )

    async def search(
        self,
        vector: Sequence[float],
        metadata_filter: MetadataFilter | None = None,
        limit: int = 10,
        exact: bool | None = False,
        candidate_count: int | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorHit]:
        use_exact = (
            exact
            if exact is not None
            else (candidate_count is not None and candidate_count <= self.settings.exact_candidate_threshold)
        )
        response = await self.client.query_points(
            collection_name=self.settings.collection_name,
            query=list(vector),
            query_filter=_qdrant_filter(metadata_filter),
            search_params=models.SearchParams(exact=use_exact),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [_hit(point) for point in response.points]

    async def fetch(self, ids: Sequence[str]) -> list[VectorHit]:
        if not ids:
            return []
        points = await self.client.retrieve(
            collection_name=self.settings.collection_name,
            ids=[_point_id(chunk_id) for chunk_id in ids],
            with_payload=True,
            with_vectors=False,
        )
        return [_hit(point) for point in points]


def _qdrant_filter(metadata_filter: MetadataFilter | None) -> models.Filter | None:
    if metadata_filter is None:
        return None
    must: list[models.Condition] = []
    for field, value in metadata_filter.model_dump(exclude_none=True).items():
        must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    return models.Filter(must=cast(list[models.Condition], must)) if must else None


def _category_from_source(source_file: str) -> str:
    parts = source_file.split("/")
    return parts[2] if len(parts) > 2 else "unknown"


def _hit(point: Any) -> VectorHit:
    payload = dict(point.payload or {})
    return VectorHit(
        id=str(payload.get("chunk_id", point.id)),
        score=float(getattr(point, "score", 0.0) or 0.0),
        payload=payload,
    )


def _point_id(chunk_id: str) -> str:
    """Map arbitrary stable chunk IDs to Qdrant's UUID point-ID format."""

    try:
        UUID(chunk_id)
        return chunk_id
    except ValueError:
        return str(uuid5(NAMESPACE_URL, f"friday:{chunk_id}"))
