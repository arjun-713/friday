"""Safe text-only answer orchestration over the hybrid retriever."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from ..ingestion.assets.images import images_for_chunks
from ..ingestion.models import DocumentChunk
from ..retrieval.cache import RetrievalSessionCache
from ..retrieval.contracts import EmbeddingProvider, MetadataFilter, VectorHit, VectorIndex
from ..retrieval.hybrid import LexicalRetriever
from .litellm import InvalidAnswerError, UnsupportedAnswerError
from .models import (
    Citation,
    DiagnosticSessionState,
    DiagnosticStep,
    EvidenceContext,
    ManualImage,
    RetrievalSummary,
    TroubleshootingRequest,
    TroubleshootingResponse,
)
from .session import DiagnosticSessionStore


class AnswerGenerator(Protocol):
    async def generate(self, query: str, evidence: Sequence[EvidenceContext]) -> str: ...

    async def generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> DiagnosticStep: ...

    def stream_generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> AsyncIterator[str | DiagnosticStep]: ...


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

    async def stream_generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> AsyncIterator[str | DiagnosticStep]:
        del query
        step = await self.generate_step("", evidence, state)
        yield step.instruction
        yield step


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

    async def answer(self, request: TroubleshootingRequest) -> TroubleshootingResponse:
        state = self.session_store.record_turn(request)
        if state.last_turn_was_acknowledgement and state.current_step is not None:
            return _awaiting_current_observation(state, request.session_id)
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
            include_diagnostics=True,
            abstention_dense_threshold=None,
        )
        retrieval = RetrievalSummary(
            abstained=result.abstained,
            reason=result.reason,
            timings_ms=result.timings_ms,
        )
        if result.abstained or not result.hits:
            return TroubleshootingResponse(
                status="abstained",
                session_id=request.session_id,
                missing_observations=_missing_observations(request),
                retrieval=retrieval,
            )

        evidence = _assemble_evidence(result.hits, result.parents)
        if not evidence:
            return TroubleshootingResponse(
                status="abstained",
                session_id=request.session_id,
                missing_observations=_missing_observations(request),
                retrieval=RetrievalSummary(
                    abstained=True,
                    reason="retrieved_chunks_have_no_citation_evidence",
                    timings_ms=result.timings_ms,
                ),
            )
        try:
            step = await self.answer_generator.generate_step(request.query, evidence, state)
        except UnsupportedAnswerError:
            return TroubleshootingResponse(
                status="abstained",
                session_id=request.session_id,
                missing_observations=_missing_observations(request),
                retrieval=RetrievalSummary(
                    abstained=True,
                    reason="answer_not_supported_by_retrieved_evidence",
                    timings_ms=result.timings_ms,
                ),
            )
        except InvalidAnswerError:
            return TroubleshootingResponse(
                status="abstained",
                missing_observations=_missing_observations(request),
                retrieval=RetrievalSummary(
                    abstained=True,
                    reason="answer_failed_evidence_validation",
                    timings_ms=result.timings_ms,
                ),
            )
        images = _images_for_evidence(self.image_manifest, evidence)
        response = TroubleshootingResponse(
            session_id=request.session_id,
            status="ready",
            answer=step.instruction,
            step=step,
            awaiting_observation=True,
            images=images,
            evidence=evidence,
            citations=[item.citation for item in evidence],
            retrieval=retrieval,
        )
        _remember_current_step(state, response)
        return response

    async def stream_answer(self, request: TroubleshootingRequest) -> AsyncIterator[dict[str, object]]:
        """Stream provider tokens while keeping the final response contract strict."""

        state = self.session_store.record_turn(request)
        if state.last_turn_was_acknowledgement and state.current_step is not None:
            yield {"type": "complete", "response": _awaiting_current_observation(state, request.session_id).model_dump()}
            return
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
            include_diagnostics=True,
            abstention_dense_threshold=None,
        )
        retrieval = RetrievalSummary(abstained=result.abstained, reason=result.reason, timings_ms=result.timings_ms)
        yield {"type": "retrieval", "retrieval": retrieval.model_dump()}
        if result.abstained or not result.hits:
            yield {
                "type": "complete",
                "response": TroubleshootingResponse(
                    status="abstained",
                    session_id=request.session_id,
                    missing_observations=_missing_observations(request),
                    retrieval=retrieval,
                ).model_dump(),
            }
            return
        evidence = _assemble_evidence(result.hits, result.parents)
        if not evidence:
            yield {
                "type": "complete",
                "response": TroubleshootingResponse(
                    status="abstained",
                    session_id=request.session_id,
                    missing_observations=_missing_observations(request),
                    retrieval=RetrievalSummary(
                        abstained=True,
                        reason="retrieved_chunks_have_no_citation_evidence",
                        timings_ms=result.timings_ms,
                    ),
                ).model_dump(),
            }
            return
        try:
            step: DiagnosticStep | None = None
            async for item in self.answer_generator.stream_generate_step(request.query, evidence, state):
                if isinstance(item, str):
                    yield {"type": "token", "text": item}
                else:
                    step = item
            if step is None:
                raise InvalidAnswerError("LLM stream ended without a diagnostic step")
        except (UnsupportedAnswerError, InvalidAnswerError) as error:
            yield {
                "type": "complete",
                "response": TroubleshootingResponse(
                    status="abstained",
                    session_id=request.session_id,
                    missing_observations=_missing_observations(request),
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
        response = TroubleshootingResponse(
            session_id=request.session_id,
            status="ready",
            answer=step.instruction,
            step=step,
            awaiting_observation=True,
            images=_images_for_evidence(self.image_manifest, evidence),
            evidence=evidence,
            citations=[item.citation for item in evidence],
            retrieval=retrieval,
        )
        _remember_current_step(state, response)
        yield {"type": "complete", "response": response.model_dump()}


def _remember_current_step(state: DiagnosticSessionState, response: TroubleshootingResponse) -> None:
    """Retain the active evidence-backed check for acknowledgement-only turns."""

    if response.step is None:
        return
    state.current_step_id = response.step.step_id
    state.current_step = response.step
    state.current_evidence = response.evidence
    state.current_images = response.images
    state.current_citations = response.citations
    state.last_turn_was_acknowledgement = False


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
        retrieval=RetrievalSummary(
            abstained=False,
            reason="awaiting_current_observation",
        ),
    )


def _assemble_evidence(hits: Sequence[VectorHit], parents: Sequence[DocumentChunk]) -> list[EvidenceContext]:
    parents_by_id = {chunk.chunk_id: chunk for chunk in parents}
    evidence: list[EvidenceContext] = []
    seen: set[str] = set()
    for hit in hits:
        parent_id = hit.payload.get("parent_chunk_id")
        parent = parents_by_id.get(str(parent_id)) if parent_id else None
        content = parent.content if parent is not None else str(hit.payload.get("text", "")).strip()
        if not content or hit.id in seen:
            continue
        citation = _citation(hit, parent)
        if citation is None:
            continue
        seen.add(hit.id)
        evidence.append(
            EvidenceContext(
                chunk_id=hit.id,
                content=content,
                section=citation.section,
                pages=parent.pages if parent is not None else [citation.page],
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


def _missing_observations(request: TroubleshootingRequest) -> list[str]:
    missing: list[str] = []
    if not request.manufacturer:
        missing.append("manufacturer")
    if not request.model:
        missing.append("model")
    return missing


def _retrieval_query(request: TroubleshootingRequest) -> str:
    parts = [request.query]
    if request.observation:
        parts.append(f"Observed: {request.observation}")
    if request.selected_option:
        parts.append(f"Selected result: {request.selected_option}")
    return " ".join(parts)


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
