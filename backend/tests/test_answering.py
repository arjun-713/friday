import asyncio
import json
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from copilot.answering.litellm import (
    AnswerProviderUnavailable,
    InvalidAnswerError,
    LiteLLMAnswerGenerator,
    LiteLLMSettings,
    _expand_step_citations,
)
from copilot.answering.models import DiagnosticSessionState, DiagnosticStep, TroubleshootingRequest
from copilot.answering.service import TroubleshootingService, _assemble_evidence, _relevant_evidence
from copilot.ingestion.models import ChunkKind, DocumentChunk, Evidence, RetrievalProfile, SourceDocument
from copilot.main import _runtime_path, app, get_troubleshooting_service
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

    async def delete(self, metadata_filter) -> None:
        del metadata_filter

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


class SequentialStepGenerator:
    async def generate(self, query: str, evidence) -> str:
        del query, evidence
        return "legacy"

    async def generate_step(self, query: str, evidence, state: DiagnosticSessionState) -> DiagnosticStep:
        del query, evidence
        if state.completed_steps:
            return DiagnosticStep(
                step_id="step-2",
                title="Check the connection",
                instruction="Check the upstream connection.",
                question="Is the connection seated firmly?",
                options=[],
                source_ids=["child-1"],
            )
        return DiagnosticStep(
            step_id="step-1",
            title="Check the WAN light",
            instruction="Check the WAN light.",
            question="Is it off or on?",
            options=[],
            source_ids=["child-1"],
        )


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


def _child_chunk() -> DocumentChunk:
    chunk = _chunk().model_copy(deep=True)
    chunk.chunk_id = "child-1"
    chunk.kind = ChunkKind.CHILD
    chunk.content = "Check the Internet status before changing router settings."
    chunk.evidence[0].content = chunk.content
    return chunk


def _service(hits: list[VectorHit]) -> TroubleshootingService:
    return TroubleshootingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex(hits),
        lexical_retriever=FakeLexicalRetriever(hits),
        parent_store=FakeParentStore([_chunk()]),
    )


def test_runtime_paths_are_anchored_to_the_project_root(monkeypatch) -> None:
    monkeypatch.delenv("CHUNKS_ROOT", raising=False)

    path = _runtime_path("CHUNKS_ROOT", "data/chunks")

    assert path.name == "chunks"
    assert path.parent.name == "data"
    assert path.is_absolute()


def test_supported_devices_are_derived_from_registered_manuals(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "source_registry.json"
    registry.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "source_file": "data/manuals/routers/example-router.pdf",
                        "manufacturer": "Example",
                        "model": "Router 1",
                    },
                    {
                        "source_file": "data/manuals/printers/example-printer.pdf",
                        "manufacturer": "Example",
                        "model": "Printer 1",
                    },
                    # A second manual for the same model must not create a
                    # duplicate selection in the product UI.
                    {
                        "source_file": "data/manuals/printers/example-printer-safety.pdf",
                        "manufacturer": "Example",
                        "model": "Printer 1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SOURCE_REGISTRY", str(registry))

    response = TestClient(app).get("/v1/devices")

    assert response.status_code == 200
    assert response.json() == {
        "devices": [
            {"category": "printer", "manufacturer": "Example", "model": "Printer 1"},
            {"category": "router", "manufacturer": "Example", "model": "Router 1"},
        ]
    }


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


def test_litellm_refuses_to_send_a_request_without_the_configured_key(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_PROVIDER_KEY", raising=False)

    async def completion(**request):
        raise AssertionError(f"provider call must not happen without a key: {request}")

    generator = LiteLLMAnswerGenerator(
        LiteLLMSettings(enabled=True, model="openai/example", api_key_env="MISSING_PROVIDER_KEY"),
        completion=completion,
    )

    with pytest.raises(AnswerProviderUnavailable, match="no API key"):
        asyncio.run(
            generator.generate_step(
                "The router cannot connect",
                _assemble_evidence([_hit()], [_chunk()]),
                DiagnosticSessionState(session_id="key-check"),
            )
        )


def test_health_reports_configuration_state_without_contacting_providers() -> None:
    payload = TestClient(app).get("/health").json()

    assert payload["phase"] == "conversational-troubleshooting"
    assert payload["components"]["retrieval"] == "configured"
    assert payload["components"]["llm"] in {"configured", "not_configured"}
    assert payload["components"]["voice"] in {"configured", "not_configured"}


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


def test_litellm_generator_sends_evidence_and_accepts_known_citation() -> None:
    captured: dict[str, object] = {}

    async def completion(**request):
        captured.update(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Check the cable. [source:child-1]"))]
        )

    evidence = _service([_hit()])
    response = asyncio.run(
        evidence.answer(
            TroubleshootingRequest(query="The router cannot connect", manufacturer="Example", model="Example 1")
        )
    )
    generator = LiteLLMAnswerGenerator(
        LiteLLMSettings(enabled=True, model="deepseek/deepseek-chat"), completion=completion
    )
    answer = asyncio.run(generator.generate("The router cannot connect", response.evidence))

    assert answer.endswith("[Example Manual · p. 4 · Troubleshooting > Connection]")
    assert captured["model"] == "deepseek/deepseek-chat"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "Example Manual" in messages[1]["content"]
    assert "[source:child-1]" in messages[1]["content"]
    assert "exactly one diagnostic step" in messages[0]["content"]
    assert "Do not use general world knowledge" in messages[0]["content"]
    assert "troubleshooting-v3" in messages[0]["content"]


def test_litellm_generator_uses_strict_schema_for_diagnostic_steps() -> None:
    captured: dict[str, object] = {}

    async def completion(**request):
        captured.update(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"title":"Check the cable","instruction":"Check the cable.",'
                            '"question":"Did you find a problem?","options":[],"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    generator = LiteLLMAnswerGenerator(
        LiteLLMSettings(enabled=True, model="openai/sarvam-105b-conversations", response_format="json_schema"),
        completion=completion,
    )
    asyncio.run(
        generator.generate_step(
            "The router cannot connect",
            _assemble_evidence([_hit()], [_chunk()]),
            DiagnosticSessionState(session_id="structured-output-test"),
        )
    )

    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["source_ids"]["minItems"] == 1


def test_litellm_generator_rejects_unknown_citation() -> None:
    async def completion(**request):
        del request
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Reset it. [source:not-retrieved]"))]
        )

    service = _service([_hit()])
    response = asyncio.run(
        service.answer(
            TroubleshootingRequest(query="The router cannot connect", manufacturer="Example", model="Example 1")
        )
    )
    generator = LiteLLMAnswerGenerator(completion=completion)

    try:
        asyncio.run(generator.generate("The router cannot connect", response.evidence))
    except InvalidAnswerError as error:
        assert "outside the retrieved context" in str(error)
    else:
        raise AssertionError("unknown citation should be rejected")


def test_text_endpoint_runs_litellm_answer_layer() -> None:
    async def completion(**request):
        del request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"title":"Check the cable","instruction":"Check the cable.",'
                            '"question":"Did you find a problem?","options":[],"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    base_service = _service([_hit()])
    service = TroubleshootingService(
        embedding_provider=base_service.embedding_provider,
        vector_index=base_service.vector_index,
        lexical_retriever=base_service.lexical_retriever,
        parent_store=base_service.parent_store,
        answer_generator=LiteLLMAnswerGenerator(completion=completion),
        image_manifest={
            "assets": {
                "asset-1": {
                    "path": "assets/images/asset-1.png",
                    "mime_type": "image/png",
                    "features": {"classification": "valid", "quality_score": 4},
                    "occurrences": [
                        {
                            "document_title": "Example Manual",
                            "source_file": "data/manuals/example.pdf",
                            "page": 4,
                            "chunk_ids": ["child-1"],
                        }
                    ],
                }
            }
        },
    )
    app.dependency_overrides[get_troubleshooting_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/v1/troubleshoot",
            json={"query": "The router cannot connect", "manufacturer": "Example", "model": "Example 1"},
        )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["answer"] == "Check the cable. [Example Manual · p. 4 · Troubleshooting > Connection]"
    assert body["images"][0]["asset_id"] == "asset-1"
    assert body["images"][0]["url"] == "/v1/assets/images/asset-1"


def test_litellm_stream_validates_structured_step_after_tokens() -> None:
    async def completion(**request):
        assert request["stream"] is True
        payload = (
            '{"title":"Check the cable","instruction":"Check the cable.",'
            '"question":"Did you find a problem?","options":[],"source_ids":["child-1"]}'
        )

        class Stream:
            async def __aiter__(self):
                for piece in (payload[:32], payload[32:]):
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

        return Stream()

    async def collect() -> list[object]:
        generator = LiteLLMAnswerGenerator(completion=completion)
        return [
            item
            async for item in generator.stream_generate_step(
                "The router cannot connect",
                _assemble_evidence([_hit()], [_chunk()]),
                DiagnosticSessionState(session_id="stream-test"),
            )
        ]

    events = asyncio.run(collect())
    assert len(events[:-1]) == 2
    assert "".join(str(event) for event in events[:-1]).startswith('{"title":"Check the cable"')
    assert isinstance(events[-1], DiagnosticStep)
    assert events[-1].instruction.startswith("Check the cable.")


def test_litellm_normalizes_sarvam_option_and_source_id_formatting() -> None:
    async def completion(**request):
        del request
        payload = (
            '{"title":"Check the cable","instruction":"Check the cable.",'
            '"question":"What do you see?","options":["Connected","Loose"],'
            '"source_ids":["source:child-1"]}'
        )

        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=payload))])

        return Stream()

    async def collect() -> list[object]:
        generator = LiteLLMAnswerGenerator(completion=completion)
        return [
            item
            async for item in generator.stream_generate_step(
                "The router cannot connect",
                _assemble_evidence([_hit()], [_chunk()]),
                DiagnosticSessionState(session_id="sarvam-format"),
            )
        ]

    events = asyncio.run(collect())
    step = events[-1]
    assert isinstance(step, DiagnosticStep)
    assert step.source_ids == ["child-1"]
    assert [option.label for option in step.options] == ["Connected", "Loose"]
    assert len({option.id for option in step.options}) == 2


def test_service_keeps_supported_lexical_result_when_dense_score_is_low() -> None:
    low_dense_hit = _hit().model_copy(update={"score": 0.10})
    service = TroubleshootingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex([low_dense_hit]),
        lexical_retriever=FakeLexicalRetriever([_hit()]),
        parent_store=FakeParentStore([_chunk()]),
    )

    response = asyncio.run(
        service.answer(
            TroubleshootingRequest(query="The router cannot connect", manufacturer="Example", model="Example 1")
        )
    )

    assert response.status == "ready"
    assert response.citations[0].chunk_id == "child-1"


def test_sarvam_litellm_request_uses_compatible_endpoint_and_structured_output(monkeypatch) -> None:
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    captured: dict[str, object] = {}

    async def completion(**request):
        captured.update(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"title":"Check the cable","instruction":"Check the cable.",'
                            '"question":"Did you find a problem?","options":[],"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    async def run() -> None:
        generator = LiteLLMAnswerGenerator(
            settings=LiteLLMSettings(
                enabled=True,
                model="openai/sarvam-105b-conversations",
                api_base="https://api.sarvam.ai/v1",
            ),
            completion=completion,
        )
        await generator.generate_step(
            "The router cannot connect",
            _assemble_evidence([_hit()], [_chunk()]),
            DiagnosticSessionState(session_id="test"),
        )

    asyncio.run(run())
    assert captured["model"] == "openai/sarvam-105b-conversations"
    assert captured["api_base"] == "https://api.sarvam.ai/v1"
    assert captured["api_key"] == "test-key"
    assert captured["extra_headers"] == {"api-subscription-key": "test-key"}
    assert captured["response_format"] == {"type": "json_object"}


def test_stream_endpoint_emits_tokens_and_final_response() -> None:
    async def completion(**request):
        del request
        payload = (
            '{"title":"Check the cable","instruction":"Check the cable.",'
            '"question":"Did you find a problem?","options":[],"source_ids":["child-1"]}'
        )

        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=payload))])

        return Stream()

    base_service = _service([_hit()])
    service = TroubleshootingService(
        embedding_provider=base_service.embedding_provider,
        vector_index=base_service.vector_index,
        lexical_retriever=base_service.lexical_retriever,
        parent_store=base_service.parent_store,
        answer_generator=LiteLLMAnswerGenerator(completion=completion),
    )
    app.dependency_overrides[get_troubleshooting_service] = lambda: service
    try:
        with TestClient(app).stream(
            "POST",
            "/v1/troubleshoot/stream",
            json={"query": "The router cannot connect", "manufacturer": "Example", "model": "Example 1"},
        ) as response:
            body = "".join(response.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert '"type": "token"' in body
    assert '"type": "complete"' in body


def test_session_advances_to_the_next_step_after_observation() -> None:
    base_service = _service([_hit()])
    service = TroubleshootingService(
        embedding_provider=base_service.embedding_provider,
        vector_index=base_service.vector_index,
        lexical_retriever=base_service.lexical_retriever,
        parent_store=base_service.parent_store,
        answer_generator=SequentialStepGenerator(),
    )

    first = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Wi-Fi is visible but there is no internet",
                manufacturer="Example",
                model="Example 1",
                session_id="session-1",
            )
        )
    )
    second = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Wi-Fi is visible but there is no internet",
                manufacturer="Example",
                model="Example 1",
                session_id="session-1",
                selected_option="on",
            )
        )
    )

    assert first.step is not None
    assert first.step.step_id == "step-1"
    assert second.step is not None
    assert second.step.step_id == "step-2"
    assert service.session_store.get("session-1").observations == {"step-1": "on"}
    assert second.observations == ["on"]


def test_session_does_not_complete_step_for_acknowledgement_only() -> None:
    base_service = _service([_hit()])
    service = TroubleshootingService(
        embedding_provider=base_service.embedding_provider,
        vector_index=base_service.vector_index,
        lexical_retriever=base_service.lexical_retriever,
        parent_store=base_service.parent_store,
        answer_generator=SequentialStepGenerator(),
    )

    first = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Wi-Fi is visible but there is no internet",
                manufacturer="Example",
                model="Example 1",
                session_id="session-ack",
            )
        )
    )
    second = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Yes, I got it. What's next?",
                manufacturer="Example",
                model="Example 1",
                session_id="session-ack",
                observation="Yes, I got it. What's next?",
            )
        )
    )

    state = service.session_store.get("session-ack")
    assert state.completed_steps == []
    assert state.observations == {}
    assert first.step is not None
    assert second.step == first.step
    assert second.retrieval.reason == "awaiting_current_observation"


def test_regeneration_does_not_record_the_previous_user_message_as_an_observation() -> None:
    base_service = _service([_hit()])
    service = TroubleshootingService(
        embedding_provider=base_service.embedding_provider,
        vector_index=base_service.vector_index,
        lexical_retriever=base_service.lexical_retriever,
        parent_store=base_service.parent_store,
        answer_generator=SequentialStepGenerator(),
    )
    first = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Wi-Fi is visible but there is no internet",
                manufacturer="Example",
                model="Example 1",
                session_id="session-regenerate",
            )
        )
    )
    regenerated = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Wi-Fi is visible but there is no internet",
                manufacturer="Example",
                model="Example 1",
                session_id="session-regenerate",
                regenerate=True,
            )
        )
    )

    state = service.session_store.get("session-regenerate")
    assert state.observations == {}
    assert state.completed_steps == []
    assert first.step is not None
    assert regenerated.step is not None
    assert regenerated.step.step_id == first.step.step_id


def test_session_treats_unpunctuated_what_next_as_acknowledgement() -> None:
    base_service = _service([_hit()])
    service = TroubleshootingService(
        embedding_provider=base_service.embedding_provider,
        vector_index=base_service.vector_index,
        lexical_retriever=base_service.lexical_retriever,
        parent_store=base_service.parent_store,
        answer_generator=SequentialStepGenerator(),
    )

    first = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Wi-Fi is visible but there is no internet",
                manufacturer="Example",
                model="Example 1",
                session_id="session-what-next",
            )
        )
    )
    second = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Yes, I got it. What next?",
                observation="Yes, I got it. What next?",
                manufacturer="Example",
                model="Example 1",
                session_id="session-what-next",
            )
        )
    )

    assert first.step is not None
    assert second.step == first.step
    assert second.retrieval.reason == "awaiting_current_observation"


def test_session_uses_query_as_acknowledgement_when_observation_is_omitted() -> None:
    base_service = _service([_hit()])
    service = TroubleshootingService(
        embedding_provider=base_service.embedding_provider,
        vector_index=base_service.vector_index,
        lexical_retriever=base_service.lexical_retriever,
        parent_store=base_service.parent_store,
        answer_generator=SequentialStepGenerator(),
    )
    first = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="The router cannot connect",
                manufacturer="Example",
                model="Example 1",
                session_id="query-only-ack",
            )
        )
    )
    second = asyncio.run(
        service.answer(
            TroubleshootingRequest(
                query="Yes, I got it. What next?",
                manufacturer="Example",
                model="Example 1",
                session_id="query-only-ack",
            )
        )
    )

    assert first.step is not None
    assert second.step == first.step
    assert second.retrieval.reason == "awaiting_current_observation"


def test_broad_print_failure_excludes_unreported_conditional_manual_branches() -> None:
    evidence = _assemble_evidence([_hit()], [_chunk()])
    conditional = evidence[0].model_copy(update={"section": "The printer does not print after wireless configuration"})
    generic = evidence[0].model_copy(update={"section": "Solve problems"})

    filtered = _relevant_evidence(
        [conditional, generic],
        TroubleshootingRequest(query="The printer will not print", manufacturer="Example", model="Example 1"),
    )

    assert filtered == []


def test_evidence_uses_exact_retrieved_child_not_broad_parent_context() -> None:
    parent = _chunk()
    parent.content = "Unrelated procedure. Do not use this parent as the active instruction."
    child = _child_chunk()

    evidence = _assemble_evidence([_hit()], [parent, child])

    assert evidence[0].content == child.content
    assert evidence[0].pages == [4]


def test_step_citation_display_deduplicates_identical_manual_locations() -> None:
    evidence = _assemble_evidence([_hit()], [_chunk()])
    duplicate = evidence[0].model_copy(update={"chunk_id": "child-duplicate"})
    step = DiagnosticStep(
        step_id="step",
        title="Check",
        instruction="Check the connection.",
        question="What do you see?",
        options=[],
        source_ids=["child-1", "child-duplicate"],
    )

    expanded = _expand_step_citations(step, [*evidence, duplicate])

    assert expanded.instruction.count("[Example Manual · p. 4 · Troubleshooting > Connection]") == 1
