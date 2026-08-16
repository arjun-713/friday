"""Safe text-only answer orchestration over the hybrid retriever."""

from collections.abc import Sequence
from typing import Protocol

from ..ingestion.models import DocumentChunk
from ..retrieval.cache import RetrievalSessionCache
from ..retrieval.contracts import EmbeddingProvider, MetadataFilter, VectorHit, VectorIndex
from ..retrieval.hybrid import LexicalRetriever
from .models import Citation, EvidenceContext, RetrievalSummary, TroubleshootingRequest, TroubleshootingResponse


class AnswerGenerator(Protocol):
    async def generate(self, query: str, evidence: Sequence[EvidenceContext]) -> str: ...


class ParentChunkStore(Protocol):
    async def fetch(self, ids: Sequence[str]) -> list[DocumentChunk]: ...


class EvidenceOnlyAnswerGenerator:
    """Return source text verbatim until a separately evaluated LLM is added."""

    async def generate(self, query: str, evidence: Sequence[EvidenceContext]) -> str:
        del query
        if not evidence:
            raise ValueError("cannot generate an answer without evidence")
        return evidence[0].content


class TroubleshootingService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        lexical_retriever: LexicalRetriever,
        parent_store: ParentChunkStore,
        answer_generator: AnswerGenerator | None = None,
        session_cache: RetrievalSessionCache | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.lexical_retriever = lexical_retriever
        self.parent_store = parent_store
        self.answer_generator = answer_generator or EvidenceOnlyAnswerGenerator()
        self.session_cache = session_cache or RetrievalSessionCache()

    async def answer(self, request: TroubleshootingRequest) -> TroubleshootingResponse:
        metadata_filter = MetadataFilter(
            manufacturer=request.manufacturer,
            model=request.model,
        )
        result = await self.session_cache.retrieve(
            request.query,
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
            abstention_dense_threshold=0.84,
        )
        retrieval = RetrievalSummary(
            abstained=result.abstained,
            reason=result.reason,
            timings_ms=result.timings_ms,
        )
        if result.abstained or not result.hits:
            return TroubleshootingResponse(
                status="abstained",
                missing_observations=_missing_observations(request),
                retrieval=retrieval,
            )

        evidence = _assemble_evidence(result.hits, result.parents)
        if not evidence:
            return TroubleshootingResponse(
                status="abstained",
                missing_observations=_missing_observations(request),
                retrieval=RetrievalSummary(
                    abstained=True,
                    reason="retrieved_chunks_have_no_citation_evidence",
                    timings_ms=result.timings_ms,
                ),
            )
        answer = await self.answer_generator.generate(request.query, evidence)
        return TroubleshootingResponse(
            status="ready",
            answer=answer,
            evidence=evidence,
            citations=[item.citation for item in evidence],
            retrieval=retrieval,
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
