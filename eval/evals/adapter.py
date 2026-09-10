"""In-process Friday adapter for DeepEval suites.

Runs the REAL production pipeline (same retriever, planner, tools, validators
as the deployed backend) with an isolated in-memory session store, so eval
turns never pollute ``data/index/diagnostic_sessions.sqlite3``.

Two entry points:

- :func:`run_turn` — one troubleshooting turn, returning the full
  ``TroubleshootingResponse`` plus flattened fields metrics need.
- :class:`FridaySession` — a stateful multi-turn conversation handle that keeps
  one ``session_id`` across turns, mirroring the frontend session contract.
"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from friday.answering.models import TroubleshootingRequest, TroubleshootingResponse
from friday.answering.service import TroubleshootingService
from friday.answering.session import DiagnosticSessionStore
from friday.main import _answer_generator, _lexical_retriever, _load_image_manifest
from friday.paths import chunks_dir
from friday.retrieval.cache import RetrievalSessionCache
from friday.retrieval.context_store import JsonlParentChunkStore
from friday.retrieval.contracts import MetadataFilter
from friday.retrieval.granite import GraniteEmbeddingProvider
from friday.retrieval.hybrid import LexicalRetriever
from friday.retrieval.indexer import load_all_chunks
from friday.retrieval.qdrant import QdrantSettings, QdrantVectorIndex

from .tracing import instrument, run_traced_turn


@dataclass
class TurnResult:
    """Everything an eval needs from one Friday turn, without re-plumbing."""

    response: TroubleshootingResponse
    answer: str
    status: str
    progress: str | None
    abstained: bool
    retrieval_context: list[str]
    chunk_ids: list[str]
    citations: list[dict[str, Any]]
    tools_called: list[str]
    tool_calls: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    options: list[dict[str, Any]]
    images: list[dict[str, Any]]
    retrieval_ms: float | None
    state_summary: dict[str, Any] = field(default_factory=dict)


class FridayAdapter:
    """Owns one isolated service instance shared across an eval session.

    All app coroutines run on ONE dedicated event loop owned by the adapter.
    This is load-bearing: the Qdrant gRPC client binds to its creating loop,
    so per-test ``asyncio.run()`` calls break reuse with "Event loop is
    closed". :meth:`run_turn_sync` works from sync pytest bodies and from
    foreign running loops (e.g. the ConversationSimulator callback).
    """

    def __init__(self) -> None:
        all_chunks = load_all_chunks(chunks_dir())
        # Same constructor production uses: vector-scoped exact index (Round 2).
        lexical: LexicalRetriever = _lexical_retriever(all_chunks)
        self.service = TroubleshootingService(
            embedding_provider=GraniteEmbeddingProvider(),
            vector_index=QdrantVectorIndex(QdrantSettings()),
            lexical_retriever=lexical,
            parent_store=JsonlParentChunkStore.from_chunks(all_chunks),
            answer_generator=_answer_generator(),
            session_cache=RetrievalSessionCache(),
            session_store=DiagnosticSessionStore(),
            image_manifest=_load_image_manifest(),
        )
        instrument(self.service)
        # The loop runs forever on its own thread; every turn is submitted
        # via run_coroutine_threadsafe. This is load-bearing twice: the Qdrant
        # gRPC client binds to its creating loop (so per-test asyncio.run()
        # breaks reuse), and concurrent callers (simulator threads) must never
        # share run_until_complete on one loop.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="friday-eval-loop", daemon=True
        )
        self._loop_thread.start()
        # Warm the embedding model once (mirrors main.py startup warmup) so
        # measured retrieval latencies never include one-time model load.
        self._run(self.service.embedding_provider.embed_query("Friday eval warmup"))
        atexit.register(self.close)

    def close(self) -> None:
        try:
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=10)
                self._loop.close()
        except RuntimeError:
            pass

    def _run(self, coro):
        # Propagate the submitter's context (deepeval trace + eval capture)
        # into the loop thread: bare threadsafe submit would run under the
        # loop thread's empty context and silently detach spans and capture.
        context = contextvars.copy_context()

        async def _with_context():
            for variable, value in context.items():
                variable.set(value)
            return await coro

        future = asyncio.run_coroutine_threadsafe(_with_context(), self._loop)
        return future.result(timeout=300)

    def chunk_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """Fetch normalized chunk texts (sync). Used for text-aware evidence
        matching: the app dedups textually identical parent/child chunks, so
        an expected chunk ID may be represented by its duplicate's text."""

        async def _fetch() -> dict[str, str]:
            chunks = await self.service.parent_store.fetch(chunk_ids)
            return {chunk.chunk_id: " ".join(chunk.content.split()) for chunk in chunks}

        return self._run(_fetch())

    def run_turn_sync(
        self,
        query: str,
        *,
        session_id: str,
        manufacturer: str | None = None,
        model: str | None = None,
        observation: str | None = None,
    ) -> TurnResult:
        return self._run(
            self.run_turn(
                query,
                session_id=session_id,
                manufacturer=manufacturer,
                model=model,
                observation=observation,
            )
        )

    async def run_turn(
        self,
        query: str,
        *,
        session_id: str,
        manufacturer: str | None = None,
        model: str | None = None,
        observation: str | None = None,
    ) -> TurnResult:
        request = TroubleshootingRequest(
            query=query,
            session_id=session_id,
            manufacturer=manufacturer,
            model=model,
            observation=observation,
        )
        response, capture = await run_traced_turn(self.service, request)
        return self._summarize(response, capture)

    def _summarize(
        self, response: TroubleshootingResponse, capture: list[dict[str, Any]]
    ) -> TurnResult:
        turn = response.turn
        evidence = response.evidence or []
        citations = [citation.model_dump() for citation in (response.citations or [])]
        tools = [
            entry["tool"] for entry in capture if entry.get("span") == "friday_tool"
        ]
        tool_calls = [
            {"name": entry["tool"], "arguments": entry.get("arguments", {})}
            for entry in capture
            if entry.get("span") == "friday_tool"
        ]
        llm_calls = [
            entry for entry in capture if entry.get("span") == "friday_plan_llm"
        ]
        validations = [
            entry for entry in capture if entry.get("span") == "friday_validate"
        ]
        options: list[dict[str, Any]] = []
        if turn is not None and turn.observation_request is not None:
            options = [
                option.model_dump() for option in turn.observation_request.options
            ]
        # Manual figure references ride along for future multimodal cases.
        # Friday currently has no image-input path, so no image metrics are
        # attached; the field keeps multimodal test cases addable without
        # re-plumbing the adapter.
        images = [
            {"asset_id": image.asset_id, "url": image.url}
            for image in (response.images or [])
        ]
        timings = response.retrieval.timings_ms or {}
        total_ms = timings.get("total_ms")
        return TurnResult(
            response=response,
            answer=response.answer or "",
            status=response.status,
            progress=response.diagnostic_progress,
            abstained=response.status == "abstained",
            retrieval_context=[item.content for item in evidence],
            chunk_ids=[item.chunk_id for item in evidence],
            citations=citations,
            tools_called=tools,
            tool_calls=tool_calls,
            llm_calls=llm_calls,
            options=options,
            images=images,
            retrieval_ms=float(total_ms) if total_ms is not None else None,
            state_summary={
                "validation_failures": sum(
                    1 for entry in validations if not entry.get("valid", True)
                ),
            },
        )


class FridaySession:
    """A stateful conversation handle: one session_id, sequential turns."""

    def __init__(self, adapter: FridayAdapter, **device: Any) -> None:
        self.adapter = adapter
        self.session_id = f"deepeval-{uuid4().hex[:12]}"
        self.device = device
        self.turns: list[TurnResult] = []

    def ask(self, query: str, **kwargs: Any) -> TurnResult:
        params = {**self.device, **kwargs}
        result = self.adapter.run_turn_sync(query, session_id=self.session_id, **params)
        self.turns.append(result)
        return result

    def metadata_filter(self) -> MetadataFilter:
        return MetadataFilter(
            manufacturer=self.device.get("manufacturer"),
            model=self.device.get("model"),
        )
