import asyncio
from collections.abc import Sequence

from copilot.ingestion.models import ChunkKind, DocumentChunk, Evidence, RetrievalProfile, SourceDocument
from copilot.retrieval.benchmark import BenchmarkQuery, run_benchmark
from copilot.retrieval.contracts import MetadataFilter, VectorHit, VectorRecord
from copilot.retrieval.hybrid import retrieve
from copilot.retrieval.ingest import index_chunks, vector_chunks
from copilot.retrieval.metrics import latency_summary


class FakeEmbedder:
    dimension = 3
    model_name = "fixture"

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


class FakeIndex:
    def __init__(self) -> None:
        self.records: list[VectorRecord] = []
        self.search_calls: list[dict[str, object]] = []

    async def ensure_collection(self, dimension: int) -> None:
        assert dimension == 3

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        self.records.extend(records)

    async def search(
        self,
        vector,
        metadata_filter=None,
        limit=10,
        exact=False,
        candidate_count=None,
        score_threshold=None,
    ):
        self.search_calls.append(
            {"filter": metadata_filter, "exact": exact, "candidate_count": candidate_count, "limit": limit}
        )
        return [VectorHit(id=self.records[0].id, score=0.9, payload=self.records[0].payload)]

    async def fetch(self, ids):
        return []


def _chunk(chunk_id: str, profile: RetrievalProfile) -> DocumentChunk:
    source = SourceDocument(
        document_id="manual",
        title="Manual",
        manufacturer="Example",
        model="Example 1",
        version="sha256:test",
        source_url="https://example.test/manual.pdf",
        retrieved_at="2026-08-10T00:00:00Z",
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        document=source,
        page=1,
        pages=[1],
        section="Setup",
        content="Connect the cable.",
        kind=ChunkKind.CHILD,
        parser="fixture",
        evidence=[Evidence(source_file="manual.pdf", page=1, section="Setup", content="Connect the cable.")],
        retrieval_profiles=[profile],
    )


def test_vector_ingestion_selects_only_vector_profile_and_batches() -> None:
    chunks = [_chunk("vector", RetrievalProfile.VECTOR), _chunk("parent", RetrievalProfile.CONTEXT_STORE)]
    index = FakeIndex()

    indexed = asyncio.run(index_chunks(chunks, FakeEmbedder(), index, batch_size=1))

    assert indexed == 1
    assert vector_chunks(chunks) == [chunks[0]]
    assert len(index.records) == 1


def test_hybrid_retrieval_applies_filter_and_abstains_without_hits() -> None:
    index = FakeIndex()
    chunk = _chunk("vector", RetrievalProfile.VECTOR)
    index.records.append(VectorRecord(id=chunk.chunk_id, vector=[1, 0, 0], payload={"model": "Example 1"}, chunk=chunk))

    result = asyncio.run(
        retrieve(
            "connect",
            FakeEmbedder(),
            index,
            metadata_filter=MetadataFilter(model="Example 1"),
            exact=True,
        )
    )

    assert not result.abstained
    assert index.search_calls[0]["exact"] is True
    assert index.search_calls[0]["filter"] == MetadataFilter(model="Example 1")


def test_latency_summary_reports_requested_percentiles() -> None:
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert set(summary) == {"p50_ms", "p70_ms", "p99_ms", "max_ms"}
    assert summary["max_ms"] == 5.0


def test_benchmark_reports_latency_and_relevance_metrics() -> None:
    async def retrieve_ids(query: str) -> list[str]:
        return [query, "other"]

    result = asyncio.run(
        run_benchmark(
            [BenchmarkQuery(query="answer", expected_chunk_ids=frozenset({"answer"}))],
            retrieve_ids,
        )
    )

    assert result.latency["max_ms"] >= 0
    assert result.recall_at_5 == 1.0
    assert result.mrr == 1.0
