"""Session-scoped retrieval caching for repeated troubleshooting turns."""

import asyncio
import json
from collections import OrderedDict
from time import monotonic, perf_counter

from .contracts import EmbeddingProvider, MetadataFilter, VectorIndex
from .hybrid import LexicalRetriever, ParentChunkStore, RetrievalResult, retrieve


class RetrievalSessionCache:
    """Cache completed retrieval turns inside one device troubleshooting session.

    The caller supplies the confirmed device scope. The cache never guesses a
    manufacturer or model from ambiguous user text, preserving the abstention
    and missing-observation rules of the assistant.
    """

    def __init__(self, max_entries: int = 32, ttl_seconds: float = 900.0) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._scope: MetadataFilter | None = None
        self._entries: OrderedDict[str, tuple[float, RetrievalResult]] = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def scope(self) -> MetadataFilter | None:
        return self._scope

    def set_scope(self, metadata_filter: MetadataFilter) -> None:
        """Set the confirmed device/document scope and clear stale results."""

        if self._scope != metadata_filter:
            self._scope = metadata_filter
            self._entries.clear()

    def clear(self) -> None:
        self._entries.clear()

    async def retrieve(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        lexical_retriever: LexicalRetriever | None = None,
        parent_store: ParentChunkStore | None = None,
        metadata_filter: MetadataFilter | None = None,
        limit: int = 8,
        exact: bool | None = False,
        candidate_count: int | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        effective_filter = self._effective_filter(metadata_filter)
        key = self._key(query, effective_filter, limit, exact, candidate_count, score_threshold)
        now = monotonic()
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and now - entry[0] <= self.ttl_seconds:
                self._entries.move_to_end(key)
                cached = entry[1].model_copy(deep=True)
                cached.timings_ms = {"cache_hit": 1.0, "total_ms": 0.0}
                return cached
            if entry is not None:
                del self._entries[key]

        started = perf_counter()
        result = await retrieve(
            query,
            embedding_provider,
            vector_index,
            lexical_retriever=lexical_retriever,
            parent_store=parent_store,
            metadata_filter=effective_filter,
            limit=limit,
            exact=exact,
            candidate_count=candidate_count,
            score_threshold=score_threshold,
        )
        result.timings_ms["cache_hit"] = 0.0
        result.timings_ms["cache_total_ms"] = (perf_counter() - started) * 1000
        async with self._lock:
            self._entries[key] = (monotonic(), result.model_copy(deep=True))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return result

    def _effective_filter(self, metadata_filter: MetadataFilter | None) -> MetadataFilter | None:
        if self._scope is None:
            return metadata_filter
        if metadata_filter is None:
            return self._scope
        scope_values = self._scope.model_dump(exclude_none=True)
        query_values = metadata_filter.model_dump(exclude_none=True)
        for field, value in query_values.items():
            if field in scope_values and scope_values[field] != value:
                raise ValueError(f"query filter conflicts with session scope for {field}")
            scope_values[field] = value
        return MetadataFilter(**scope_values)

    @staticmethod
    def _key(
        query: str,
        metadata_filter: MetadataFilter | None,
        limit: int,
        exact: bool | None,
        candidate_count: int | None,
        score_threshold: float | None,
    ) -> str:
        return json.dumps(
            {
                "query": " ".join(query.lower().split()),
                "filter": metadata_filter.model_dump(exclude_none=True) if metadata_filter else None,
                "limit": limit,
                "exact": exact,
                "candidate_count": candidate_count,
                "score_threshold": score_threshold,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
