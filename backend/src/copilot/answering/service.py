"""Safe text-only answer orchestration over the hybrid retriever."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from ..ingestion.assets.images import images_for_chunks
from ..ingestion.models import DocumentChunk
from ..retrieval.cache import RetrievalSessionCache
from ..retrieval.contracts import EmbeddingProvider, MetadataFilter, VectorHit, VectorIndex
from ..retrieval.hybrid import LexicalRetriever
from .litellm import AgentRun, InvalidAnswerError, UnsupportedAnswerError
from .models import (
    Citation,
    DiagnosticAction,
    DiagnosticSessionState,
    DiagnosticStep,
    DiagnosticTurn,
    EvidenceContext,
    ManualImage,
    ObservationRequest,
    RetrievalSummary,
    TroubleshootingRequest,
    TroubleshootingResponse,
)
from .session import DiagnosticSessionStore
from .tools import AgentToolExecutor, AgentToolResult


class AnswerGenerator(Protocol):
    async def generate(self, query: str, evidence: Sequence[EvidenceContext]) -> str: ...

    async def generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> DiagnosticStep: ...

    async def generate_turn(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> DiagnosticTurn: ...

    def stream_generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> AsyncIterator[str | DiagnosticStep]: ...

    def stream_generate_turn(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> AsyncIterator[str | DiagnosticTurn]: ...


class ParentChunkStore(Protocol):
    async def fetch(self, ids: Sequence[str]) -> list[DocumentChunk]: ...


class EvidenceOnlyAnswerGenerator:
    """Return source text verbatim until a separately evaluated LLM is added."""

    async def generate(self, query: str, evidence: Sequence[EvidenceContext]) -> str:
        del query
        if not evidence:
            raise ValueError("cannot generate an answer without evidence")
        return evidence[0].content

    async def generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> DiagnosticStep:
        del query
        if not evidence:
            raise InvalidAnswerError("cannot generate a step without evidence")
        first = evidence[0]
        return DiagnosticStep(
            step_id=f"evidence-{first.chunk_id}",
            title="Next diagnostic step",
            instruction=first.content,
            question="Did you complete this step, and what did you observe?",
            source_ids=[first.chunk_id],
        )

    async def generate_turn(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> DiagnosticTurn:
        step = await self.generate_step(query, evidence, state)
        return _turn_from_step(step)

    async def stream_generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> AsyncIterator[str | DiagnosticStep]:
        del query
        step = await self.generate_step("", evidence, state)
        yield step.instruction
        yield step

    async def stream_generate_turn(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> AsyncIterator[str | DiagnosticTurn]:
        turn = await self.generate_turn(query, evidence, state)
        yield turn.response
        yield turn


class TroubleshootingService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        lexical_retriever: LexicalRetriever,
        parent_store: ParentChunkStore,
        answer_generator: AnswerGenerator | None = None,
        session_cache: RetrievalSessionCache | None = None,
        session_store: DiagnosticSessionStore | None = None,
        image_manifest: dict[str, object] | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.lexical_retriever = lexical_retriever
        self.parent_store = parent_store
        self.answer_generator = answer_generator or EvidenceOnlyAnswerGenerator()
        self.session_cache = session_cache or RetrievalSessionCache()
        self.session_store = session_store or DiagnosticSessionStore()
        self.image_manifest = image_manifest or {"assets": {}}

    def _tool_executor(
        self,
        request: TroubleshootingRequest,
        state: DiagnosticSessionState,
        initial_evidence: Sequence[EvidenceContext],
    ) -> AgentToolExecutor:
        async def execute(name: str, arguments: dict[str, object]) -> AgentToolResult:
            if name in {"search_manual", "find_error_code"}:
                value = arguments.get("query") if name == "search_manual" else arguments.get("code")
                query = str(value or "").strip()
                if not query:
                    return AgentToolResult("The requested manual search needs a non-empty query.")
                prefix = "error code " if name == "find_error_code" else ""
                result = await self.session_cache.retrieve(
                    prefix + query,
                    self.embedding_provider,
                    self.vector_index,
                    lexical_retriever=self.lexical_retriever,
                    parent_store=self.parent_store,
                    metadata_filter=MetadataFilter(manufacturer=request.manufacturer, model=request.model),
                    limit=5,
                    candidate_limit=32,
                    dense_weight=1.0,
                    lexical_weight=1.5,
                    rrf_k=30,
                    diversify=True,
                    include_diagnostics=False,
                    abstention_dense_threshold=None,
                )
                evidence = _assemble_evidence(result.hits, result.parents) if result.hits else []
                return AgentToolResult(_tool_evidence_text(evidence), evidence)
            if name == "open_manual_page":
                document_id = str(arguments.get("document_id", ""))
                try:
                    page = int(str(arguments.get("page", 0)))
                except (TypeError, ValueError):
                    page = 0
                matches = [
                    item
                    for item in initial_evidence
                    if item.citation.document_id == document_id and item.citation.page == page
                ]
                return AgentToolResult(_tool_evidence_text(matches), matches)
            if name == "get_diagnostic_state":
                return AgentToolResult(state.model_dump_json())
            return AgentToolResult("This tool is unavailable.")

        return execute

    def delete_session(self, session_id: str) -> None:
        self.session_store.delete(session_id)

    async def answer(self, request: TroubleshootingRequest) -> TroubleshootingResponse:
        state = (
            self.session_store.get(request.session_id)
            if request.regenerate
            else self.session_store.record_turn(request)
        )
        metadata_filter = MetadataFilter(
            manufacturer=request.manufacturer,
            model=request.model,
        )
        result = await self.session_cache.retrieve(
            _retrieval_query(request),
            self.embedding_provider,
            self.vector_index,
            lexical_retriever=self.lexical_retriever,
            parent_store=self.parent_store,
            metadata_filter=metadata_filter,
            limit=5,
            candidate_limit=32,
            dense_weight=1.0,
            lexical_weight=1.5,
            rrf_k=30,
            diversify=True,
            include_diagnostics=True,
            abstention_dense_threshold=None,
        )
        retrieval = RetrievalSummary(
            abstained=result.abstained,
            reason=result.reason,
            timings_ms=result.timings_ms,
        )
        if result.abstained or not result.hits:
            missing = _missing_observations(request)
            return TroubleshootingResponse(
                status="abstained",
                session_id=request.session_id,
                answer=_observation_request(missing),
                observations=_confirmed_observations(state),
                missing_observations=missing,
                retrieval=retrieval,
            )

        evidence = _assemble_evidence(result.hits, result.parents)
        evidence = _relevant_evidence(evidence, request)
        if not evidence:
            missing = _missing_observations(request, require_symptom_detail=True)
            return TroubleshootingResponse(
                status="abstained",
                session_id=request.session_id,
                answer=_observation_request(missing),
                observations=_confirmed_observations(state),
                missing_observations=missing,
                retrieval=RetrievalSummary(
                    abstained=True,
                    reason="retrieved_evidence_not_specific_to_symptom",
                    timings_ms=result.timings_ms,
                ),
            )
        try:
            run = await _generate_turn(
                self.answer_generator,
                request.query,
                evidence,
                state,
                self._tool_executor(request, state, evidence),
            )
        except UnsupportedAnswerError:
            missing = _missing_observations(request)
            return TroubleshootingResponse(
                status="abstained",
                session_id=request.session_id,
                answer=_observation_request(missing),
                observations=_confirmed_observations(state),
                missing_observations=missing,
                retrieval=RetrievalSummary(
                    abstained=True,
                    reason="answer_not_supported_by_retrieved_evidence",
                    timings_ms=result.timings_ms,
                ),
            )
        except InvalidAnswerError:
            missing = _missing_observations(request)
            return TroubleshootingResponse(
                status="abstained",
                session_id=request.session_id,
                answer=_observation_request(missing),
                observations=_confirmed_observations(state),
                missing_observations=missing,
                retrieval=RetrievalSummary(
                    abstained=True,
                    reason="answer_failed_evidence_validation",
                    timings_ms=result.timings_ms,
                ),
            )
        evidence = run.evidence
        response = _turn_response(
            request.session_id,
            run.turn,
            state,
            evidence,
            _images_for_evidence(self.image_manifest, evidence),
            retrieval,
        )
        self.session_store.apply_turn(state, run.turn)
        response.observations = _confirmed_observations(state)
        self.session_store.save(state)
        return response

    async def stream_answer(self, request: TroubleshootingRequest) -> AsyncIterator[dict[str, object]]:
        """Stream provider tokens while keeping the final response contract strict."""

        state = (
            self.session_store.get(request.session_id)
            if request.regenerate
            else self.session_store.record_turn(request)
        )
        result = await self.session_cache.retrieve(
            _retrieval_query(request),
            self.embedding_provider,
            self.vector_index,
            lexical_retriever=self.lexical_retriever,
            parent_store=self.parent_store,
            metadata_filter=MetadataFilter(manufacturer=request.manufacturer, model=request.model),
            limit=5,
            candidate_limit=32,
            dense_weight=1.0,
            lexical_weight=1.5,
            rrf_k=30,
            diversify=True,
            include_diagnostics=True,
            abstention_dense_threshold=None,
        )
        retrieval = RetrievalSummary(abstained=result.abstained, reason=result.reason, timings_ms=result.timings_ms)
        yield {"type": "retrieval", "retrieval": retrieval.model_dump()}
        if result.abstained or not result.hits:
            missing = _missing_observations(request)
            yield {
                "type": "complete",
                "response": TroubleshootingResponse(
                    status="abstained",
                    session_id=request.session_id,
                    answer=_observation_request(missing),
                    observations=_confirmed_observations(state),
                    missing_observations=missing,
                    retrieval=retrieval,
                ).model_dump(),
            }
            return
        evidence = _assemble_evidence(result.hits, result.parents)
        evidence = _relevant_evidence(evidence, request)
        if not evidence:
            missing = _missing_observations(request, require_symptom_detail=True)
            yield {
                "type": "complete",
                "response": TroubleshootingResponse(
                    status="abstained",
                    session_id=request.session_id,
                    answer=_observation_request(missing),
                    observations=_confirmed_observations(state),
                    missing_observations=missing,
                    retrieval=RetrievalSummary(
                        abstained=True,
                        reason="retrieved_evidence_not_specific_to_symptom",
                        timings_ms=result.timings_ms,
                    ),
                ).model_dump(),
            }
            return
        try:
            run = await _generate_turn(
                self.answer_generator,
                request.query,
                evidence,
                state,
                self._tool_executor(request, state, evidence),
            )
        except (UnsupportedAnswerError, InvalidAnswerError) as error:
            missing = _missing_observations(request)
            yield {
                "type": "complete",
                "response": TroubleshootingResponse(
                    status="abstained",
                    session_id=request.session_id,
                    answer=_observation_request(missing),
                    observations=_confirmed_observations(state),
                    missing_observations=missing,
                    retrieval=RetrievalSummary(
                        abstained=True,
                        reason="answer_not_supported_by_retrieved_evidence"
                        if isinstance(error, UnsupportedAnswerError)
                        else "answer_failed_evidence_validation",
                        timings_ms=result.timings_ms,
                    ),
                ).model_dump(),
            }
            return
        evidence = run.evidence
        response = _turn_response(
            request.session_id,
            run.turn,
            state,
            evidence,
            _images_for_evidence(self.image_manifest, evidence),
            retrieval,
        )
        self.session_store.apply_turn(state, run.turn)
        response.observations = _confirmed_observations(state)
        self.session_store.save(state)
        # The final natural-language turn is emitted only after source and
        # schema validation. This preserves the streaming event contract while
        # avoiding partial JSON leaking into the chat or voice layer.
        yield {"type": "token", "text": run.turn.response}
        yield {"type": "complete", "response": response.model_dump()}


async def _generate_turn(
    generator: AnswerGenerator,
    query: str,
    evidence: Sequence[EvidenceContext],
    state: DiagnosticSessionState,
    execute_tool: AgentToolExecutor,
) -> AgentRun:
    """Use the new planner contract while preserving local test generators."""

    generate_agent_turn = getattr(generator, "generate_agent_turn", None)
    if callable(generate_agent_turn):
        return await generate_agent_turn(query, evidence, state, execute_tool)
    generate_turn = getattr(generator, "generate_turn", None)
    if callable(generate_turn):
        return AgentRun(turn=await generate_turn(query, evidence, state), evidence=list(evidence))
    return AgentRun(turn=_turn_from_step(await generator.generate_step(query, evidence, state)), evidence=list(evidence))


def _turn_from_step(step: DiagnosticStep) -> DiagnosticTurn:
    """Adapt a legacy generator without changing its diagnostic policy."""

    return DiagnosticTurn(
        turn_id=f"legacy-{step.step_id}",
        mode="clarify",
        response=step.instruction,
        next_action=DiagnosticAction(instruction=step.instruction),
        observation_request=ObservationRequest(
            request_id=step.step_id,
            fact_key=f"observation_{step.step_id.removeprefix('step-').replace('-', '_')[:48]}",
            question=step.question,
            options=step.options,
        ),
        source_ids=step.source_ids,
    )


def _turn_step(turn: DiagnosticTurn) -> DiagnosticStep | None:
    """Expose an action/request compatibility view for older consumers."""

    if turn.next_action is None or turn.observation_request is None:
        return None
    request = turn.observation_request
    return DiagnosticStep(
        step_id=request.request_id,
        title="Next check",
        instruction=turn.next_action.instruction,
        question=request.question,
        options=request.options,
        source_ids=turn.source_ids,
    )


def _turn_response(
    session_id: str,
    turn: DiagnosticTurn,
    state: DiagnosticSessionState,
    evidence: Sequence[EvidenceContext],
    images: list[ManualImage],
    retrieval: RetrievalSummary,
) -> TroubleshootingResponse:
    step = _turn_step(turn)
    citations = _source_citations(turn.source_ids, evidence)
    return TroubleshootingResponse(
        session_id=session_id,
        status="abstained" if turn.mode == "abstain" else "ready",
        answer=turn.response,
        turn=turn,
        step=step,
        awaiting_observation=turn.observation_request is not None,
        images=images,
        evidence=list(evidence),
        citations=citations,
        observations=_confirmed_observations(state),
        retrieval=retrieval,
    )


def _step_citations(step: DiagnosticStep, evidence: Sequence[EvidenceContext]) -> list[Citation]:
    """Return only the manual locations the verified step explicitly uses."""

    return _source_citations(step.source_ids, evidence)


def _source_citations(source_ids: Sequence[str], evidence: Sequence[EvidenceContext]) -> list[Citation]:
    """Return compact, deduplicated evidence for a validated planner turn."""

    by_id = {item.chunk_id: item.citation for item in evidence}
    citations: list[Citation] = []
    seen: set[tuple[str, int, str]] = set()
    for source_id in source_ids:
        citation = by_id[source_id]
        key = (citation.document_id, citation.page, citation.section)
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return citations


def _awaiting_current_observation(state: DiagnosticSessionState, session_id: str) -> TroubleshootingResponse:
    """Repeat the unresolved check without retrieval or a provider call."""

    step = state.current_step
    assert step is not None
    return TroubleshootingResponse(
        session_id=session_id,
        status="ready",
        answer=f"Before I choose the next step, please report: {step.question}",
        step=step,
        awaiting_observation=True,
        images=state.current_images,
        evidence=state.current_evidence,
        citations=state.current_citations,
        observations=_confirmed_observations(state),
        retrieval=RetrievalSummary(
            abstained=False,
            reason="awaiting_current_observation",
        ),
    )


def _confirmed_observations(state: DiagnosticSessionState) -> list[str]:
    """Expose completed-step results in their original diagnostic order."""

    return list(state.observations.values())


def _tool_evidence_text(evidence: Sequence[EvidenceContext]) -> str:
    """Return compact, citation-addressable manual evidence to an agent tool call."""

    if not evidence:
        return "No matching manufacturer evidence was found."
    return "\n\n".join(
        "\n".join(
            (
                f"[source:{item.chunk_id}]",
                f"Document: {item.citation.document_title}",
                f"Page: {item.citation.page}",
                f"Section: {item.citation.section}",
                f"Content: {item.content}",
            )
        )
        for item in evidence
    )


def _assemble_evidence(hits: Sequence[VectorHit], parents: Sequence[DocumentChunk]) -> list[EvidenceContext]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in parents}
    evidence: list[EvidenceContext] = []
    seen: set[str] = set()
    for hit in hits:
        exact_chunk = chunks_by_id.get(hit.id)
        parent_id = hit.payload.get("parent_chunk_id")
        parent = chunks_by_id.get(str(parent_id)) if parent_id else None
        # Retrieval chose the exact child for a reason. Never replace it with
        # a broad parent section, which may contain neighbouring procedures
        # and a different heading. Payload text is used for BM25-only tests;
        # the local exact chunk is the dense-search path.
        content = (
            str(hit.payload.get("text", "")).strip()
            or (exact_chunk.content if exact_chunk is not None else "")
            or (parent.content if parent is not None else "")
        )
        if not content or hit.id in seen:
            continue
        citation = _citation(hit, exact_chunk or parent)
        if citation is None:
            continue
        seen.add(hit.id)
        evidence.append(
            EvidenceContext(
                chunk_id=hit.id,
                content=content,
                section=citation.section,
                pages=(
                    exact_chunk.pages
                    if exact_chunk is not None
                    else parent.pages
                    if parent is not None
                    else [citation.page]
                ),
                citation=citation,
            )
        )
    return evidence


def _citation(hit: VectorHit, parent: DocumentChunk | None) -> Citation | None:
    payload = hit.payload
    required = (
        "document_id",
        "document_title",
        "manufacturer",
        "model",
        "document_version",
        "page",
        "section",
        "source_url",
    )
    if any(field not in payload or payload[field] in (None, "") for field in required):
        if parent is None:
            return None
        document = parent.document
        evidence = parent.evidence[0]
        return Citation(
            chunk_id=hit.id,
            document_id=document.document_id,
            document_title=document.title,
            manufacturer=document.manufacturer,
            model=document.model or "",
            document_version=document.version or "",
            page=evidence.page,
            section=evidence.section,
            source_url=document.source_url,
        )
    return Citation(
        chunk_id=hit.id,
        document_id=str(payload["document_id"]),
        document_title=str(payload["document_title"]),
        manufacturer=str(payload["manufacturer"]),
        model=str(payload["model"]),
        document_version=str(payload["document_version"]),
        page=int(payload["page"]),
        section=str(payload["section"]),
        source_url=str(payload["source_url"]),
    )


def _missing_observations(request: TroubleshootingRequest, *, require_symptom_detail: bool = False) -> list[str]:
    missing: list[str] = []
    if not request.manufacturer:
        missing.append("manufacturer")
    if not request.model:
        missing.append("model")
    if require_symptom_detail and _is_broad_print_failure(request.query):
        missing.append("any message or error code shown on the printer control panel")
    return missing


def _observation_request(missing: Sequence[str]) -> str:
    """Ask only for the fact that the deterministic safety gate needs."""

    if missing == ["any message or error code shown on the printer control panel"]:
        return "What exactly appears on the printer control panel—an error code, a message, or its normal status?"
    if missing == ["manufacturer", "model"]:
        return "Which manufacturer and model are you working with?"
    if missing == ["manufacturer"]:
        return "Which manufacturer made the device?"
    if missing == ["model"]:
        return "What is the device model?"
    return "I could not verify a safe next step from the available manual evidence."


def _retrieval_query(request: TroubleshootingRequest) -> str:
    parts = [request.query]
    if request.observation:
        parts.append(f"Observed: {request.observation}")
    if request.selected_option:
        parts.append(f"Selected result: {request.selected_option}")
    return " ".join(parts)


def _relevant_evidence(evidence: Sequence[EvidenceContext], request: TroubleshootingRequest) -> list[EvidenceContext]:
    """Keep conditional manual branches out of an unqualified symptom report.

    A heading such as "does not print after wireless configuration" is useful
    only after the user has said the printer is on Wi-Fi. Selecting it for a
    generic "won't print" report produces a plausible but misleading repair
    path. This intentionally small deterministic guard asks for the missing
    observation instead.
    """

    if not _is_broad_print_failure(request.query):
        return list(evidence)
    query = request.query.casefold()
    condition_groups = (
        (("wireless", "wi-fi", "wifi", "wlan"), ("wireless", "wi-fi", "wifi", "wlan")),
        (("firewall",), ("firewall",)),
        (
            ("multiple sheets", "misfeed", "pick up paper", "paper feed"),
            ("multiple sheets", "misfeed", "pick up paper", "paper feed"),
        ),
        (
            ("print quality", "image defect", "toner", "streak", "blur", "blank page"),
            ("print quality", "image defect", "toner", "streak", "blur", "blank page"),
        ),
        (
            ("job storage", "stored job", "private print", "delayed print"),
            ("job storage", "stored job", "private print", "delayed print"),
        ),
    )
    relevant: list[EvidenceContext] = []
    for item in evidence:
        haystack = f"{item.section}\n{item.content}".casefold()
        section = item.section.casefold()
        if "does not print" not in section and "cannot print" not in section:
            continue
        if any(
            any(term in haystack for term in evidence_terms) and not any(term in query for term in query_terms)
            for evidence_terms, query_terms in condition_groups
        ):
            continue
        # A generic failure does not establish a manual branch whose heading
        # itself depends on a circumstance the user did not report.
        if "after " in section:
            continue
        relevant.append(item)
    return relevant


def _is_broad_print_failure(query: str) -> bool:
    normalized = query.casefold()
    return "print" in normalized and any(
        term in normalized for term in ("not", "cannot", "unable", "won't", "wont", "fails", "offline")
    )


def _images_for_evidence(manifest: dict[str, object], evidence: Sequence[EvidenceContext]) -> list[ManualImage]:
    references = images_for_chunks(manifest, {item.chunk_id for item in evidence})
    return [
        ManualImage(
            asset_id=str(reference["asset_id"]),
            url=f"/v1/assets/images/{reference['asset_id']}",
            mime_type=str(reference["mime_type"]),
            document_title=str(reference["document_title"]),
            page=int(reference["page"]),
        )
        for reference in references
    ]
