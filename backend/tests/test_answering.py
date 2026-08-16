import asyncio
from collections.abc import Sequence

from fastapi.testclient import TestClient

from copilot.answering.models import TroubleshootingRequest
from copilot.answering.service import TroubleshootingService
from copilot.ingestion.models import ChunkKind, DocumentChunk, Evidence, RetrievalProfile, SourceDocument
from copilot.main import app, get_troubleshooting_service
from copilot.retrieval.contracts import MetadataFilter, VectorHit


class FakeEmbeddingProvider:
    dimension = 3
    model_name = "fixture"

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeVectorIndex:
    def __init__(self, hits: list[VectorHit]) -> None:
        self.hits = hits

    async def ensure_collection(self, dimension: int) -> None:
        del dimension

    async def upsert(self, records) -> None:
        del records

    async def search(
        self, vector, metadata_filter=None, limit=10, exact=False, candidate_count=None, score_threshold=None
    ):
        del vector, metadata_filter, exact, candidate_count, score_threshold
        return self.hits[:limit]

    async def fetch(self, ids):
        del ids
        return []


class FakeLexicalRetriever:
    def __init__(self, hits: list[VectorHit]) -> None:
        self.hits = hits

    async def search(self, query: str, metadata_filter: MetadataFilter | None = None, limit: int = 10):
        del query, metadata_filter
        return self.hits[:limit]


class FakeParentStore:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks

    async def fetch(self, ids: Sequence[str]) -> list[DocumentChunk]:
        return [chunk for chunk in self.chunks if chunk.chunk_id in ids]


def _chunk() -> DocumentChunk:
    source = SourceDocument(
        document_id="manual",
        title="Example Manual",
        manufacturer="Example",
        model="Example 1",
        version="v1",
        source_url="https://example.test/manual.pdf",
        retrieved_at="2026-08-15T00:00:00Z",
    )
    return DocumentChunk(
        chunk_id="parent-1",
        document=source,
        page=4,
        pages=[4],
        section="Troubleshooting > Connection",
        content="Verify that the network cable is connected.",
        kind=ChunkKind.PARENT,
        parser="fixture",
        evidence=[
            Evidence(
                source_file="manual.pdf",
                page=4,
                section="Troubleshooting > Connection",
                content="Verify that the network cable is connected.",
            )
        ],
        retrieval_profiles=[RetrievalProfile.CONTEXT_STORE],
    )


def _hit() -> VectorHit:
    return VectorHit(
        id="child-1",
        score=0.95,
        payload={
            "chunk_id": "child-1",
            "parent_chunk_id": "parent-1",
            "document_id": "manual",
            "document_title": "Example Manual",
            "manufacturer": "Example",
            "model": "Example 1",
            "document_version": "v1",
            "page": 4,
            "section": "Troubleshooting > Connection",
            "source_url": "https://example.test/manual.pdf",
        },
    )


def _service(hits: list[VectorHit]) -> TroubleshootingService:
    return TroubleshootingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex(hits),
        lexical_retriever=FakeLexicalRetriever(hits),
        parent_store=FakeParentStore([_chunk()]),
    )


def test_text_layer_returns_cited_evidence() -> None:
    response = asyncio.run(
        _service([_hit()]).answer(
            TroubleshootingRequest(query="The router cannot connect", manufacturer="Example", model="Example 1")
        )
    )

    assert response.status == "ready"
    assert response.answer == "Verify that the network cable is connected."
    assert response.citations[0].page == 4
    assert response.citations[0].section == "Troubleshooting > Connection"


def test_text_layer_abstains_and_requests_device_observations() -> None:
    response = asyncio.run(_service([]).answer(TroubleshootingRequest(query="It does not work")))

    assert response.status == "abstained"
    assert response.answer is None
    assert response.missing_observations == ["manufacturer", "model"]


def test_text_endpoint_returns_typed_response_without_qdrant() -> None:
    service = _service([_hit()])
    app.dependency_overrides[get_troubleshooting_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/v1/troubleshoot",
            json={"query": "The router cannot connect", "manufacturer": "Example", "model": "Example 1"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["citations"][0]["page"] == 4
