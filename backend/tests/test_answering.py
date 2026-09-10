import asyncio
import json
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from friday.answering.litellm import (
    AnswerProviderUnavailable,
    InvalidAnswerError,
    LiteLLMAnswerGenerator,
    LiteLLMSettings,
    UnsupportedAnswerError,
    _expand_step_citations,
    _mark_cacheable_prefix,
    _prompt_cache_mode,
    _validate_turn,
)
from friday.answering.models import (
    DiagnosticFact,
    DiagnosticSessionState,
    DiagnosticStep,
    DiagnosticTurn,
    TroubleshootingRequest,
)
from friday.answering.service import (
    TroubleshootingService,
    _assemble_evidence,
    _relevant_evidence,
    _retrieval_query,
)
from friday.answering.tools import AgentToolResult
from friday.ingestion.models import ChunkKind, DocumentChunk, Evidence, RetrievalProfile, SourceDocument
from friday.main import _runtime_path, app, get_troubleshooting_service
from friday.retrieval.contracts import MetadataFilter, VectorHit


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

    def scoped(self, metadata_filter: MetadataFilter) -> "FakeLexicalRetriever":
        del metadata_filter
        return self


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
    assert response.answer == "Which manufacturer and model are you working with?"
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


def test_prompt_cache_controls_are_scoped_to_documented_providers() -> None:
    assert _prompt_cache_mode(LiteLLMSettings(model="openai/gpt-4o")) == "openai"
    assert _prompt_cache_mode(LiteLLMSettings(model="deepseek/deepseek-chat")) == "deepseek"
    assert _prompt_cache_mode(LiteLLMSettings(model="groq/openai/gpt-oss-120b")) is None
    assert (
        _prompt_cache_mode(
            LiteLLMSettings(model="openai/sarvam-105b-conversations", api_base="https://api.sarvam.ai/v1")
        )
        is None
    )


def test_prompt_cache_content_marker_only_changes_a_copy_of_the_system_message() -> None:
    messages = [{"role": "system", "content": "Stable policy"}, {"role": "user", "content": "Question"}]

    cached = _mark_cacheable_prefix(messages)

    assert messages[0]["content"] == "Stable policy"
    assert cached[0]["content"] == [{"type": "text", "text": "Stable policy", "cache_control": {"type": "ephemeral"}}]


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
    assert "natural conversation with the device owner" in messages[0]["content"]
    assert "Do not use general world knowledge" in messages[0]["content"]
    assert "troubleshooting-v" in messages[0]["content"]


def test_litellm_generator_explicitly_disables_sarvam_reasoning() -> None:
    captured: dict[str, object] = {}

    async def completion(**request):
        captured.update(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Check the cable. [source:child-1]"))]
        )

    generator = LiteLLMAnswerGenerator(
        LiteLLMSettings(
            enabled=True,
            model="openai/sarvam-105b-conversations",
            disable_reasoning=True,
        ),
        completion=completion,
    )
    asyncio.run(generator.generate("The router cannot connect", _assemble_evidence([_hit()], [_chunk()])))

    assert "reasoning_effort" in captured
    assert captured["reasoning_effort"] is None


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


def test_litellm_turn_can_resolve_without_forcing_another_question() -> None:
    async def completion(**request):
        del request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"mode":"solve","response":"The adapter is detected and the battery is near full charge.",'
                            '"interpretation":"This matches the documented high-charge indicator state.",'
                            '"next_action":null,"observation_request":null,"facts_learned":[],'
                            '"candidate_causes":[],"ruled_out_causes":["loose adapter connection"],'
                            '"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    turn = asyncio.run(
        LiteLLMAnswerGenerator(completion=completion).generate_turn(
            "The battery is at 97% and Windows says plugged in.",
            _assemble_evidence([_hit()], [_chunk()]),
            DiagnosticSessionState(session_id="solve-without-question"),
        )
    )

    assert turn.mode == "solve"
    assert turn.observation_request is None
    assert turn.next_action is None


def test_litellm_turn_rejects_a_question_for_a_known_fact() -> None:
    async def completion(**request):
        del request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"mode":"clarify","response":"I need the light state.","interpretation":null,'
                            '"next_action":null,"observation_request":{"request_id":"check-led-again",'
                            '"fact_key":"battery_led_state","question":"What color is the battery light?",'
                            '"options":[],"recheck_after_action":false},"decision_basis":'
                            '{"why_not_solved":"The light state is needed before a documented path can be selected.",'
                            '"discriminates_between":["adapter detection","battery state"],'
                            '"expected_discrimination":"The reported light state selects the applicable documented check."},"facts_learned":[], '
                            '"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    state = DiagnosticSessionState(
        session_id="repeat-guard",
        facts={
            "battery_led_state": {
                "key": "battery_led_state",
                "value": "white",
                "label": "Battery light",
                "raw": "The light is white",
            }
        },
    )
    with pytest.raises(InvalidAnswerError, match="already known"):
        asyncio.run(
            LiteLLMAnswerGenerator(completion=completion).generate_turn(
                "The light is white.", _assemble_evidence([_hit()], [_chunk()]), state
            )
        )


def test_litellm_turn_rejects_an_advance_without_diagnostic_discrimination() -> None:
    async def completion(**request):
        del request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"mode":"advance","response":"Check the adapter connection.","interpretation":null,'
                            '"next_action":{"instruction":"Reconnect the adapter.","why":"This checks the connection."},'
                            '"observation_request":null,"decision_basis":null,"facts_learned":[],'
                            '"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    with pytest.raises(InvalidAnswerError, match="what the action distinguishes"):
        asyncio.run(
            LiteLLMAnswerGenerator(completion=completion).generate_turn(
                "The battery is not charging.",
                _assemble_evidence([_hit()], [_chunk()]),
                DiagnosticSessionState(session_id="advance-without-discrimination"),
            )
        )


def test_litellm_turn_rejects_a_recheck_without_an_action() -> None:
    async def completion(**request):
        del request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"mode":"clarify","response":"Please check the light again.","interpretation":null,'
                            '"next_action":null,"observation_request":{"request_id":"recheck-led",'
                            '"fact_key":"battery_led_state","question":"What color is it now?",'
                            '"options":[],"recheck_after_action":true},"decision_basis":'
                            '{"why_not_solved":"The light state needs confirmation after the connection change.",'
                            '"discriminates_between":["adapter detection","battery state"],'
                            '"expected_discrimination":"A changed light would show whether the connection changed."},'
                            '"facts_learned":[],"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    state = DiagnosticSessionState(
        session_id="recheck-without-action",
        facts={
            "battery_led_state": {
                "key": "battery_led_state",
                "value": "amber",
                "label": "Battery light",
                "raw": "The light is amber.",
            }
        },
    )
    with pytest.raises(InvalidAnswerError, match="recheck must follow an action"):
        asyncio.run(
            LiteLLMAnswerGenerator(completion=completion).generate_turn(
                "It is still amber.", _assemble_evidence([_hit()], [_chunk()]), state
            )
        )


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
    assert body["answer"] == "Check the cable."
    assert body["citations"] == [
        {
            "chunk_id": "child-1",
            "document_id": "manual",
            "document_title": "Example Manual",
            "manufacturer": "Example",
            "model": "Example 1",
            "document_version": "v1",
            "page": 4,
            "section": "Troubleshooting > Connection",
            "source_url": "https://example.test/manual.pdf",
        }
    ]
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


def test_agentic_stream_executes_bounded_manual_tool_then_streams_final_answer() -> None:
    calls: list[dict[str, object]] = []

    async def completion(**request):
        calls.append(request)
        if len(calls) == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="search_manual",
                                        arguments='{"query":"WAN light status"}',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )

        class Stream:
            async def __aiter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Check the WAN light."))])

        return Stream()

    async def run() -> list[object]:
        generator = LiteLLMAnswerGenerator(completion=completion)
        evidence = _assemble_evidence([_hit()], [_chunk()])

        async def execute(name: str, arguments: dict[str, object]) -> AgentToolResult:
            assert name == "search_manual"
            assert arguments["query"] == "WAN light status"
            return AgentToolResult("WAN evidence", evidence)

        return [
            item
            async for item in generator.stream_agentic_conversation(
                "The router has no internet",
                evidence,
                DiagnosticSessionState(session_id="agentic-test"),
                execute,
            )
        ]

    events = asyncio.run(run())
    assert len(calls) == 2
    assert calls[0]["tools"]
    assert calls[1]["stream"] is True
    assert any(getattr(event, "tool_names", []) == ["search_manual"] for event in events)
    assert "Check the WAN light." in events[-1]


def test_single_call_planner_uses_one_call_without_tools() -> None:
    """The common no-tool turn must cost exactly one provider call."""

    calls: list[dict[str, object]] = []

    async def completion(**request):
        calls.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"mode":"solve","response":"The cable is connected.",'
                            '"interpretation":null,"next_action":null,"observation_request":null,'
                            '"decision_basis":null,"facts_learned":[],'
                            '"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    async def run():
        generator = LiteLLMAnswerGenerator(completion=completion)
        evidence = _assemble_evidence([_hit()], [_chunk()])

        async def execute(name: str, arguments: dict[str, object]) -> AgentToolResult:
            raise AssertionError("no tool should be executed")

        return await generator.generate_agent_turn(
            "The router cannot connect",
            evidence,
            DiagnosticSessionState(session_id="single-call"),
            execute,
        )

    run_result = asyncio.run(run())

    assert len(calls) == 1
    assert run_result.turn.mode == "solve"
    assert run_result.tool_names == []


def test_single_call_planner_handles_null_content_with_tool_calls() -> None:
    """Tool-call responses with null content must enrich evidence, not crash."""

    calls: list[dict[str, object]] = []

    async def completion(**request):
        calls.append(request)
        if len(calls) == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="search_manual",
                                        arguments='{"query":"WAN light status"}',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"mode":"solve","response":"The WAN light is off.",'
                            '"interpretation":null,"next_action":null,"observation_request":null,'
                            '"decision_basis":null,"facts_learned":[],'
                            '"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"]}'
                        )
                    )
                )
            ]
        )

    async def run():
        generator = LiteLLMAnswerGenerator(completion=completion)
        evidence = _assemble_evidence([_hit()], [_chunk()])

        async def execute(name: str, arguments: dict[str, object]) -> AgentToolResult:
            assert name == "search_manual"
            return AgentToolResult("WAN evidence", evidence)

        return await generator.generate_agent_turn(
            "The router has no internet",
            evidence,
            DiagnosticSessionState(session_id="tool-content-none"),
            execute,
        )

    run_result = asyncio.run(run())

    assert len(calls) == 2
    assert run_result.tool_names == ["search_manual"]
    assert run_result.turn.mode == "solve"


def test_planner_retries_once_on_correctable_validation_failure() -> None:
    """An ungrounded option must trigger one correction retry, not an abstain."""

    calls: list[dict[str, object]] = []

    def turn_payload(options: str) -> str:
        return (
            '{"mode":"clarify","response":"What is the light doing?",'
            '"interpretation":null,"next_action":null,'
            '"observation_request":{"request_id":"light-1","fact_key":"router_light",'
            '"question":"What is the router light doing?",'
            f'"options":{options},'
            '"recheck_after_action":false},'
            '"decision_basis":{"why_not_solved":"The light state selects the branch.",'
            '"discriminates_between":["power issue","connection issue"],'
            '"expected_discrimination":"Light state selects the branch."},'
            '"facts_learned":[],"candidate_causes":[],"ruled_out_causes":[],'
            '"source_ids":["child-1"]}'
        )

    async def completion(**request):
        calls.append(request)
        if len(calls) == 1:
            content = turn_payload('[{"id":"e501","label":"Error E501","value":"E501"}]')
        else:
            content = turn_payload('[{"id":"connected","label":"Connected","value":"connected"}]')
            assert "rejected and not shown" in json.dumps(request.get("messages", []))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    async def run():
        generator = LiteLLMAnswerGenerator(completion=completion)
        evidence = _assemble_evidence([_hit()], [_chunk()])

        async def execute(name: str, arguments: dict[str, object]) -> AgentToolResult:
            raise AssertionError("retry path must not use tools")

        return await generator.generate_agent_turn(
            "The router cannot connect",
            evidence,
            DiagnosticSessionState(session_id="retry-validation"),
            execute,
        )

    run_result = asyncio.run(run())

    assert len(calls) == 2
    assert run_result.turn.mode == "clarify"
    assert run_result.turn.observation_request is not None
    assert [option.label for option in run_result.turn.observation_request.options] == ["Connected"]


def test_litellm_normalizes_sarvam_option_and_source_id_formatting() -> None:
    async def completion(**request):
        del request
        payload = (
            '{"title":"Check the cable","instruction":"Check the cable.",'
            '"question":"What do you see?","options":["Connected","Cable loose"],'
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
    assert [option.label for option in step.options] == ["Connected", "Cable loose"]
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


def _delta_chunk(text: str) -> SimpleNamespace:
    """Model a provider stream delta carrying assistant content."""

    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))])


async def _stream_texts(texts: Sequence[str]):
    for piece in texts:
        yield _delta_chunk(piece)


def test_stream_endpoint_emits_tokens_and_final_response() -> None:
    payload = (
        '{"mode":"solve","response":"Check the cable. It is connected.",'
        '"interpretation":null,"next_action":null,"observation_request":null,'
        '"decision_basis":null,"facts_learned":[],'
        '"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"]}'
    )

    async def completion(**request):
        # Single-call planner: the structured request carries tools; with no
        # tool calls needed the turn parses directly from this response.
        assert "tools" in request
        if request.get("stream"):
            # Split inside the response value so the gate decodes incrementally.
            cut = payload.index("It is connected.")
            return _stream_texts([payload[:cut], payload[cut:]])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])

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
    assert "Check the cable" in body
    # Planner contract is preserved in streaming so buttons and memory work.
    assert '"turn"' in body


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
    # Acknowledgements remain unresolved, but Friday delegates the natural
    # follow-up to the planner instead of replaying a hardcoded form prompt.
    assert second.retrieval.reason != "awaiting_current_observation"


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
    assert second.retrieval.reason != "awaiting_current_observation"


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
    assert second.retrieval.reason != "awaiting_current_observation"


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


def test_evidence_deduplicates_identical_parent_and_child_content() -> None:
    parent = _chunk()
    child = parent.model_copy(deep=True)
    child.chunk_id = "child-duplicate"
    child.kind = ChunkKind.CHILD
    duplicate_hit = _hit().model_copy(update={"id": "child-duplicate"})

    evidence = _assemble_evidence([_hit(), duplicate_hit], [parent, child])

    assert len(evidence) == 1


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

    assert expanded.instruction == "Check the connection."
    assert expanded.source_ids == ["child-1", "child-duplicate"]


def test_retrieval_query_carries_state_constraints_into_next_branch() -> None:
    state = DiagnosticSessionState(session_id="router-query")
    state.facts["wan_light"] = DiagnosticFact(
        key="wan_light",
        value="off",
        label="WAN light",
        raw="The WAN light is off.",
    )
    state.completed_actions = ["POWER_CYCLE_MODEM_ROUTER", "CHECK_WAN_CABLE"]
    state.ruled_out_causes = ["modem connectivity"]

    query = _retrieval_query(
        TroubleshootingRequest(
            query="What should I try next?",
            manufacturer="TP-Link",
            model="Archer C6",
        ),
        state,
    )

    assert "WAN light: off" in query
    assert "POWER_CYCLE_MODEM_ROUTER" in query
    assert "CHECK_WAN_CABLE" in query
    assert "modem connectivity" in query


def test_lean_retrieval_query_drops_planner_only_instructions() -> None:
    from friday.answering.service import _retrieval_query_lean

    state = DiagnosticSessionState(session_id="lean-query")
    state.facts["wan_light"] = DiagnosticFact(
        key="wan_light",
        value="off",
        label="WAN light",
        raw="The WAN light is off.",
    )
    state.completed_actions = ["POWER_CYCLE_MODEM_ROUTER", "CHECK_WAN_CABLE"]
    state.ruled_out_causes = ["modem connectivity"]
    state.user_reports = ["the modem works directly", "renew did not help"]

    query = _retrieval_query_lean(
        TroubleshootingRequest(
            query="Wi-Fi connects but there is no internet",
            manufacturer="TP-Link",
            model="Archer C6",
            observation="The WAN light is off",
        ),
        state,
    )

    assert "Wi-Fi connects but there is no internet" in query
    assert "The WAN light is off" in query
    assert "WAN light: off" in query
    assert "renew did not help" in query
    assert "POWER_CYCLE_MODEM_ROUTER" not in query
    assert "modem connectivity" not in query
    assert len(query) <= 600


def test_scoped_retriever_returns_subset_matches() -> None:
    from friday.retrieval.bm25 import CombinedLexicalRetriever, InMemoryBM25Retriever

    own_chunk = _chunk().model_copy(deep=True)
    own_chunk.chunk_id = "own-1"
    own_chunk.content = "Check the WAN light state on the router before changing settings."
    own_chunk.retrieval_profiles = [RetrievalProfile.BM25]
    other_chunk = _chunk().model_copy(deep=True)
    other_chunk.chunk_id = "other-1"
    other_chunk.content = "Check the WAN light state on the other device before changing settings."
    other_chunk.document = other_chunk.document.model_copy(update={"manufacturer": "Other", "model": "Nope"})
    other_chunk.retrieval_profiles = [RetrievalProfile.BM25]
    base = CombinedLexicalRetriever(
        InMemoryBM25Retriever([own_chunk, other_chunk]),
        FakeLexicalRetriever([]),
    )
    service = TroubleshootingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex([]),
        lexical_retriever=base,
        parent_store=FakeParentStore([]),
    )
    scoped = service._scoped_retriever(MetadataFilter(manufacturer="Example", model="Example 1"))

    async def search() -> list:
        return await scoped.search("WAN light state", MetadataFilter(manufacturer="Example", model="Example 1"), 10)

    hits = asyncio.run(search())

    assert [hit.id for hit in hits] == ["own-1"]
    # No-scope requests reuse the shared base retriever without building subsets.
    assert service._scoped_retriever(MetadataFilter()) is service.lexical_retriever


def _collect_stream(
    service: TroubleshootingService, request: TroubleshootingRequest
) -> tuple[list[dict[str, object]], dict[str, object]]:
    events: list[dict[str, object]] = []

    async def collect() -> dict[str, object]:
        async for event in service.stream_answer(request):
            events.append(event)
        complete = next(event for event in events if event.get("type") == "complete")
        return complete["response"]  # type: ignore[return-value]

    response = asyncio.run(collect())
    assert isinstance(response, dict)
    return events, response


def test_streaming_persists_planner_state_across_turns() -> None:
    """Follow-up reports must advance the diagnosis instead of repeating it."""

    base_service = _service([_hit()])
    service = TroubleshootingService(
        embedding_provider=base_service.embedding_provider,
        vector_index=base_service.vector_index,
        lexical_retriever=base_service.lexical_retriever,
        parent_store=base_service.parent_store,
        answer_generator=SequentialStepGenerator(),
    )

    _, first = _collect_stream(
        service,
        TroubleshootingRequest(
            query="Wi-Fi is visible but there is no internet",
            manufacturer="Example",
            model="Example 1",
            session_id="session-stream-advance",
        ),
    )
    _, second = _collect_stream(
        service,
        TroubleshootingRequest(
            query="Wi-Fi is visible but there is no internet",
            manufacturer="Example",
            model="Example 1",
            session_id="session-stream-advance",
            observation="The WAN light is off",
        ),
    )

    assert first["turn"]["observation_request"]["request_id"] == "step-1"
    assert second["turn"]["observation_request"]["request_id"] == "step-2"
    assert second["turn"]["response"] != first["turn"]["response"]
    state = service.session_store.get("session-stream-advance")
    assert state.observations
    assert state.current_request is not None


def test_streaming_complete_includes_turn_buttons_and_citations() -> None:
    payload = (
        '{"mode":"advance","response":"The network is visible so we check upstream next.",'
        '"interpretation":"Visible Wi-Fi isolates the fault to upstream.",'
        '"next_action":{"instruction":"Check the WAN light state on the router.",'
        '"why":"The WAN light shows whether upstream is detected."},'
        '"observation_request":{"request_id":"wan-light-1","fact_key":"wan_light",'
        '"question":"What is the WAN light doing?",'
        '"options":[{"id":"off","label":"Off","value":"off"},'
        '{"id":"solid","label":"Solid","value":"solid"},'
        '{"id":"blinking","label":"Blinking","value":"blinking"}],'
        '"recheck_after_action":false},'
        '"decision_basis":{"why_not_solved":"Visible Wi-Fi does not show upstream state.",'
        '"discriminates_between":["upstream failure","local wireless issue"],'
        '"expected_discrimination":"WAN light selects the branch."},'
        '"facts_learned":[],"candidate_causes":["upstream failure"],'
        '"ruled_out_causes":[],"source_ids":["child-1"]}'
    )

    async def completion(**request):
        assert "tools" in request
        if request.get("stream"):
            cut = payload.index("so we check upstream")
            return _stream_texts([payload[:cut], payload[cut:]])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])

    wan_chunk = _chunk()
    wan_chunk = wan_chunk.model_copy(
        update={
            "content": (
                "Check the router light labelled Internet or WAN and note "
                "whether it is off, solid, or blinking before changing settings."
            ),
            "section": "Troubleshooting > WAN light",
        }
    )
    wan_chunk.evidence[0] = wan_chunk.evidence[0].model_copy(
        update={
            "content": wan_chunk.content,
            "section": "Troubleshooting > WAN light",
        }
    )
    service = TroubleshootingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex([_hit()]),
        lexical_retriever=FakeLexicalRetriever([_hit()]),
        parent_store=FakeParentStore([wan_chunk]),
        answer_generator=LiteLLMAnswerGenerator(completion=completion),
    )
    events, response = _collect_stream(
        service,
        TroubleshootingRequest(
            query="Wi-Fi connects but there is no internet",
            manufacturer="Example",
            model="Example 1",
            session_id="session-stream-buttons",
        ),
    )

    assert any(event.get("type") == "token" for event in events)
    turn = response["turn"]
    assert turn["mode"] == "advance"
    labels = [option["label"] for option in turn["observation_request"]["options"]]
    assert labels == ["Off", "Solid", "Blinking"]
    assert response["citations"][0]["page"] == 4
    assert response["awaiting_observation"] is True


def test_planner_rejects_invented_option_states() -> None:
    evidence = _assemble_evidence([_hit()], [_chunk()])
    turn = DiagnosticTurn.model_validate(
        {
            "turn_id": "turn-invented",
            "mode": "clarify",
            "response": "What is the light doing?",
            "interpretation": None,
            "next_action": None,
            "observation_request": {
                "request_id": "light-1",
                "fact_key": "router_light",
                "question": "What is the router light doing?",
                "options": [{"id": "e501", "label": "Error E501", "value": "E501"}],
                "recheck_after_action": False,
            },
            "decision_basis": {
                "why_not_solved": "The light state selects the branch.",
                "discriminates_between": ["power issue", "connection issue"],
                "expected_discrimination": "Light state selects the branch.",
            },
            "facts_learned": [],
            "candidate_causes": [],
            "ruled_out_causes": [],
            "source_ids": ["child-1"],
        }
    )

    with pytest.raises(InvalidAnswerError, match="not grounded"):
        _validate_turn(turn, evidence, DiagnosticSessionState(session_id="grounding-check"))


def test_planner_allows_paraphrased_observation_states() -> None:
    """Benign paraphrases must not fail validation and drive abstains."""

    evidence = _assemble_evidence([_hit()], [_chunk()])
    turn = DiagnosticTurn.model_validate(
        {
            "turn_id": "turn-paraphrase",
            "mode": "clarify",
            "response": "What is the light doing?",
            "interpretation": None,
            "next_action": None,
            "observation_request": {
                "request_id": "light-1",
                "fact_key": "router_light",
                "question": "What is the router light doing?",
                "options": [{"id": "valid", "label": "Valid IP address", "value": "valid ip address"}],
                "recheck_after_action": False,
            },
            "decision_basis": {
                "why_not_solved": "The light state selects the branch.",
                "discriminates_between": ["power issue", "connection issue"],
                "expected_discrimination": "Light state selects the branch.",
            },
            "facts_learned": [],
            "candidate_causes": [],
            "ruled_out_causes": [],
            "source_ids": ["child-1"],
        }
    )

    _validate_turn(turn, evidence, DiagnosticSessionState(session_id="paraphrase-check"))


def test_planner_rejects_repeated_completed_action() -> None:
    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="repeat-check")
    state.completed_actions = ["check the wan light state on the router."]
    turn = DiagnosticTurn.model_validate(
        {
            "turn_id": "turn-repeat",
            "mode": "advance",
            "response": "Check the WAN light again.",
            "interpretation": None,
            "next_action": {
                "instruction": "Check the WAN light state on the router.",
                "why": "The WAN light shows upstream state.",
            },
            "observation_request": None,
            "decision_basis": {
                "why_not_solved": "Upstream state unknown.",
                "discriminates_between": ["upstream failure", "local issue"],
                "expected_discrimination": "Light selects branch.",
            },
            "facts_learned": [],
            "candidate_causes": [],
            "ruled_out_causes": [],
            "source_ids": ["child-1"],
        }
    )

    with pytest.raises(InvalidAnswerError, match="repeated a completed action"):
        _validate_turn(turn, evidence, state)


def test_chunked_planner_response_roundtrips_exactly() -> None:
    from friday.answering.service import _chunk_planned_response

    response = "The Wi-Fi is visible, so the local link is up. Check the WAN light next."
    pieces = _chunk_planned_response(response)

    assert len(pieces) > 1
    assert "".join(pieces) == response


def _repair_fixture(**overrides: object) -> DiagnosticTurn:
    payload: dict[str, object] = {
        "turn_id": "turn-repair",
        "mode": "advance",
        "response": "Check the cable state next.",
        "interpretation": None,
        "next_action": None,
        "observation_request": {
            "request_id": "cable-1",
            "fact_key": "cable_state",
            "question": "What is the cable state?",
            "options": [],
            "recheck_after_action": False,
        },
        "decision_basis": {
            "why_not_solved": "The cable state selects the branch.",
            "discriminates_between": ["loose cable", "faulty port"],
            "expected_discrimination": "Cable state selects the branch.",
        },
        "facts_learned": [],
        "candidate_causes": [],
        "ruled_out_causes": [],
        "source_ids": ["child-1"],
    }
    payload.update(overrides)
    return DiagnosticTurn.model_validate(payload)


def test_repair_demotes_actionless_advance_to_clarify() -> None:
    from friday.answering.litellm import _attempt_repair

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="repair-advance")
    repaired = _attempt_repair(_repair_fixture(), _assemble_evidence([_hit()], [_chunk()]))

    assert repaired is not None
    assert repaired.mode == "clarify"
    _validate_turn(repaired, evidence, state)


def test_repair_strips_extras_from_solve_turn() -> None:
    from friday.answering.litellm import _attempt_repair

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="repair-solve")
    turn = _repair_fixture(
        mode="solve",
        next_action={"instruction": "Check again.", "why": "Just in case."},
        decision_basis=None,
    )
    repaired = _attempt_repair(turn, _assemble_evidence([_hit()], [_chunk()]))

    assert repaired is not None
    assert repaired.next_action is None
    assert repaired.observation_request is None
    assert repaired.decision_basis is None
    _validate_turn(repaired, evidence, state)


def test_repair_drops_action_from_clarify_turn() -> None:
    from friday.answering.litellm import _attempt_repair

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="repair-clarify")
    turn = _repair_fixture(
        mode="clarify", next_action={"instruction": "Check the cable.", "why": "It selects the branch."}
    )
    repaired = _attempt_repair(turn, _assemble_evidence([_hit()], [_chunk()]))

    assert repaired is not None
    assert repaired.next_action is None
    _validate_turn(repaired, evidence, state)


def test_repair_declines_repeated_action_without_invention() -> None:
    from friday.answering.litellm import _attempt_repair

    state = DiagnosticSessionState(session_id="repair-repeat")
    state.completed_actions = ["check the cable state on the router."]
    turn = _repair_fixture(
        next_action={"instruction": "Check the cable state on the router.", "why": "It selects the branch."},
        observation_request=None,
    )

    assert _attempt_repair(turn, _assemble_evidence([_hit()], [_chunk()])) is None


def test_repeated_actions_ignores_mentions_and_flags_recommendations() -> None:
    from friday.answering.session import repeated_actions_in_response

    completed = ["RENEW_DHCP", "POWER_CYCLE_ROUTER"]
    mention = (
        "Since renewing the connection did not help, check the router's WAN IP status next. "
        "You already restarted the router, so no need to power it off again."
    )
    assert repeated_actions_in_response(mention, completed) == []

    recommendation = "Please renew the connection now, then power off the router for one minute."
    flagged = repeated_actions_in_response(recommendation, completed)
    assert set(flagged) == {"RENEW_DHCP", "POWER_CYCLE_ROUTER"}


def test_scanner_tolerates_truncation_and_nesting() -> None:
    from friday.answering.litellm import _decode_partial_json_string, _scan_top_level_fields

    raw = '{"mode": "advance", "next_action": {"instruction": "Check it", "why": "x"}, "tags": ["a", {"b": 1}], "half": "abc'
    fields = _scan_top_level_fields(raw)
    assert set(fields) == {"mode", "next_action", "tags", "half"}
    assert all(end is not None for key, (_, end) in fields.items() if key != "half")
    assert fields["half"][1] is None
    assert _decode_partial_json_string(raw, fields["half"][0]) == "abc"
    assert _decode_partial_json_string('{"a": 1}', 5) is None


def test_partial_decoder_matches_stdlib_on_tricky_strings() -> None:
    import json as _json

    from friday.answering.litellm import _decode_partial_json_string

    tricky = 'He said "hi" \\ backslash \n newline \t tab caf\u00e9 \U0001f50c end'
    encoded = _json.dumps(tricky)
    assert _decode_partial_json_string(encoded, 0) == tricky
    # Every truncation point must decode to a prefix of the full string.
    for cut in range(1, len(encoded)):
        decoded = _decode_partial_json_string(encoded[:cut], 0)
        assert decoded is not None
        assert tricky.startswith(decoded), cut


def test_gate_buffers_response_until_fields_validate() -> None:
    from friday.answering.litellm import _ResponseGate

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="gate-buffer")
    gate = _ResponseGate(evidence, state)
    head = '{"mode":"solve","interpretation":null,"next_action":null,"observation_request":null,'
    # Response arrives before the remaining fields: nothing may leak yet.
    assert gate.feed(head + '"response":"Check th') == []
    tail = 'e cable.","decision_basis":null,"facts_learned":[],"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"]}'
    assert gate.feed(tail) == ["Check the cable."]
    assert gate.complete_turn().mode == "solve"


def test_gate_withholds_repeat_action_prose() -> None:
    from friday.answering.litellm import _ResponseGate

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="gate-repeat")
    state.completed_actions = ["check the cable state on the router."]
    gate = _ResponseGate(evidence, state)
    turn_json = (
        '{"mode":"advance","interpretation":null,'
        '"next_action":{"instruction":"Check the cable state on the router.","why":"It selects the branch."},'
        '"observation_request":null,'
        '"decision_basis":{"why_not_solved":"The cable state is unknown.","discriminates_between":["loose cable","faulty port"],"expected_discrimination":"The reported state selects the branch."},'
        '"facts_learned":[],"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"],'
        '"response":"Check the cable state on the router now."}'
    )
    # The repeat is only detectable once next_action arrives; until then the
    # gate must stay shut even though response text is already buffered.
    assert gate.feed(turn_json[: turn_json.index('"response"')]) == []
    assert gate.feed(turn_json[turn_json.index('"response"') :]) == []
    with pytest.raises(InvalidAnswerError, match="repeated a completed action"):
        gate.complete_turn()


def test_tool_accumulator_reassembles_split_arguments() -> None:
    from friday.answering.litellm import _ToolCallAccumulator

    acc = _ToolCallAccumulator()
    acc.add([(0, "call-1", "search_manual", '{"query": "WAN')])
    acc.add([(0, "", "", ' light"}'), (1, "call-2", "open_manual_page", '{"document_id": "d", "page": 3}')])
    acc.add([(9, "", "", "orphan")])
    assert acc.calls() == [
        ("search_manual", {"query": "WAN light"}, "call-1"),
        ("open_manual_page", {"document_id": "d", "page": 3}, "call-2"),
    ]


def test_stream_agent_turn_speaks_before_completion() -> None:
    payload = (
        '{"mode":"solve","interpretation":null,"next_action":null,"observation_request":null,'
        '"decision_basis":null,"facts_learned":[],"candidate_causes":[],"ruled_out_causes":[],'
        '"source_ids":["child-1"],"response":"Check the cable. It is connected."}'
    )

    async def completion(**request):
        assert request.get("stream") is True
        assert "tools" in request
        cut = payload.index("It is connected.")
        return _stream_texts([payload[:cut], payload[cut:]])

    async def run():
        generator = LiteLLMAnswerGenerator(completion=completion)
        evidence = _assemble_evidence([_hit()], [_chunk()])

        async def execute(name: str, arguments: dict[str, object]) -> AgentToolResult:
            raise AssertionError("no tool should be executed")

        items = [
            item
            async for item in generator.stream_agent_turn(
                "The router cannot connect",
                evidence,
                DiagnosticSessionState(session_id="gate-speech"),
                execute,
            )
        ]
        return items

    items = asyncio.run(run())
    tokens = [item for item in items if isinstance(item, str)]
    runs = [item for item in items if not isinstance(item, str)]
    assert "".join(tokens) == "Check the cable. It is connected."
    assert len(runs) == 1
    assert runs[0].turn.mode == "solve"
    assert len(runs[0].llm_calls) == 1
    assert runs[0].llm_calls[0].stream is True
    assert runs[0].llm_calls[0].ttft_ms is not None


def test_stream_agent_turn_executes_streamed_tool_calls() -> None:
    followup = (
        '{"mode":"solve","interpretation":null,"next_action":null,"observation_request":null,'
        '"decision_basis":null,"facts_learned":[],"candidate_causes":[],"ruled_out_causes":[],'
        '"source_ids":["child-1"],"response":"The WAN light is off."}'
    )
    calls: list[dict[str, object]] = []

    async def completion(**request):
        calls.append(request)
        if len(calls) == 1:
            assert request.get("stream") is True

            async def tool_stream():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call-1",
                                        function=SimpleNamespace(
                                            name="search_manual", arguments='{"query":"WAN light"}'
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                )

            return tool_stream()
        return _stream_texts([followup])

    async def run():
        generator = LiteLLMAnswerGenerator(completion=completion)
        evidence = _assemble_evidence([_hit()], [_chunk()])

        async def execute(name: str, arguments: dict[str, object]) -> AgentToolResult:
            assert name == "search_manual"
            assert arguments == {"query": "WAN light"}
            return AgentToolResult("WAN evidence", evidence)

        return [
            item
            async for item in generator.stream_agent_turn(
                "The router has no internet",
                evidence,
                DiagnosticSessionState(session_id="gate-tools"),
                execute,
            )
        ]

    items = asyncio.run(run())
    kinds = [type(item).__name__ for item in items]
    assert "AgentStreamEvidence" in kinds
    assert items[-1].turn.response == "The WAN light is off."
    assert items[-1].tool_names == ["search_manual"]
    assert len(items[-1].llm_calls) == 2


def test_gate_optimistic_releases_response_before_validation() -> None:
    from friday.answering.litellm import _ResponseGate

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="gate-optimistic")
    gate = _ResponseGate(evidence, state, optimistic=True)
    # Response arrives first: strict mode withholds everything...
    strict = _ResponseGate(evidence, state)
    assert strict.feed('{"response":"Check the cable."') == []
    # ...while optimistic mode speaks it immediately.
    assert gate.feed('{"response":"Check the cable."') == ["Check the cable."]
    rest = ',"mode":"solve","interpretation":null,"next_action":null,"observation_request":null,"decision_basis":null,"facts_learned":[],"candidate_causes":[],"ruled_out_causes":[],"source_ids":["child-1"]}'
    assert gate.feed(rest) == []
    assert gate.complete_turn().mode == "solve"


def test_response_first_schema_and_prompt_ordering() -> None:
    from friday.answering.litellm import _diagnostic_turn_schema
    from friday.prompts import build_messages

    first = _diagnostic_turn_schema(response_first=True)
    assert next(iter(first["properties"])) == "response"
    last = _diagnostic_turn_schema(response_first=False)
    assert list(last["properties"])[-1] == "response"
    assert set(first["required"]) == set(last["required"])
    default_line = build_messages("q", [], None)[1]["content"].splitlines()[-1]
    assert "`mode`" in default_line and default_line.index("`mode`") < default_line.index("`response`")
    first_line = build_messages("q", [], None, response_first=True)[1]["content"].splitlines()[-1]
    assert first_line.index("`response`") < first_line.index("`mode`")


def test_repair_filters_hallucinated_source_ids() -> None:
    from friday.answering.litellm import _attempt_repair

    evidence = _assemble_evidence([_hit()], [_chunk()])
    turn = _repair_fixture(source_ids=["child-1", "ghost-chunk"])
    repaired = _attempt_repair(turn, evidence)

    assert repaired is not None
    assert repaired.source_ids == ["child-1"]
    _validate_turn(repaired, evidence, DiagnosticSessionState(session_id="repair-ids"))


def test_repair_declines_fully_hallucinated_citations() -> None:
    from friday.answering.litellm import _attempt_repair

    evidence = _assemble_evidence([_hit()], [_chunk()])
    turn = _repair_fixture(source_ids=["ghost-chunk"])

    assert _attempt_repair(turn, evidence) is None


def test_gate_logs_abandoned_optimistic_speech(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from friday.answering.litellm import _ResponseGate

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="gate-abandoned")
    gate = _ResponseGate(evidence, state, optimistic=True)
    assert gate.feed('{"response":"Check the cable now."') != []
    with caplog.at_level(logging.INFO, logger="friday.answering.litellm"), pytest.raises(InvalidAnswerError):
        gate.complete_turn()
    assert "optimistic_speech_abandoned" in caplog.text


def test_bare_sentinel_abstain_turn_maps_to_unsupported() -> None:
    """A JSON abstain turn carrying bare UNSUPPORTED prose must not reach users."""

    from friday.answering.litellm import _parse_turn_text

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="sentinel-check")
    payload = json.dumps(
        {
            "mode": "abstain",
            "response": "UNSUPPORTED",
            "interpretation": None,
            "next_action": None,
            "observation_request": None,
            "decision_basis": None,
            "facts_learned": [],
            "candidate_causes": [],
            "ruled_out_causes": [],
            "source_ids": ["child-1"],
        }
    )
    with pytest.raises(UnsupportedAnswerError):
        _parse_turn_text(payload, evidence, state)


def test_mentioning_unsupported_in_prose_stays_valid() -> None:
    """Exact-match only: prose merely containing the word must still validate."""

    from friday.answering.litellm import _parse_turn_text

    evidence = _assemble_evidence([_hit()], [_chunk()])
    state = DiagnosticSessionState(session_id="sentinel-prose-check")
    payload = json.dumps(
        {
            "mode": "solve",
            "response": "The old method is unsupported, so check the cable instead. It is connected.",
            "interpretation": None,
            "next_action": None,
            "observation_request": None,
            "decision_basis": None,
            "facts_learned": [],
            "candidate_causes": [],
            "ruled_out_causes": [],
            "source_ids": ["child-1"],
        }
    )
    assert _parse_turn_text(payload, evidence, state).mode == "solve"
