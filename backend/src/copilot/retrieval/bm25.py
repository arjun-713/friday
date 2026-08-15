"""Deterministic BM25 and exact-identifier retrieval over chunk JSONL."""

import math
import re
from asyncio import gather
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from ..ingestion.models import DocumentChunk, RetrievalProfile
from .contracts import MetadataFilter, VectorHit
from .qdrant import chunk_payload

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)+|[a-z]+\d+|\d+[a-z]+|\b\d{2,}\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class InMemoryBM25Retriever:
    """Small lexical index designed for the current local corpus."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        k1: float = 1.2,
        b: float = 0.75,
        scope_filter: MetadataFilter | None = None,
    ) -> None:
        self._chunks = [chunk for chunk in chunks if RetrievalProfile.BM25 in chunk.retrieval_profiles]
        self._k1 = k1
        self._b = b
        self._scope_filter = scope_filter
        self._tokens = [tokenize(chunk.content) for chunk in self._chunks]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        for index, tokens in enumerate(self._tokens):
            for token, frequency in Counter(tokens).items():
                self._postings[token][index] = frequency

    @classmethod
    def from_directory(cls, root: Path) -> "InMemoryBM25Retriever":
        chunks: list[DocumentChunk] = []
        for path in sorted(root.glob("*/*.jsonl")):
            chunks.extend(
                DocumentChunk.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
        return cls(chunks)

    async def search(
        self,
        query: str,
        metadata_filter: MetadataFilter | None = None,
        limit: int = 10,
    ) -> list[VectorHit]:
        query_tokens = _lexical_query_tokens(query, metadata_filter)
        if not query_tokens or not self._chunks:
            return []
        scores: dict[int, float] = defaultdict(float)
        document_count = len(self._chunks)
        for token in query_tokens:
            posting = self._postings.get(token)
            if not posting:
                continue
            inverse_document_frequency = math.log(1 + (document_count - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, frequency in posting.items():
                chunk = self._chunks[index]
                if self._scope_filter == metadata_filter or _matches(chunk, metadata_filter):
                    denominator = frequency + self._k1 * (
                        1 - self._b + self._b * self._lengths[index] / max(self._average_length, 1)
                    )
                    scores[index] += inverse_document_frequency * frequency * (self._k1 + 1) / denominator
        ranked = sorted(scores, key=lambda index: scores[index], reverse=True)[:limit]
        return [
            VectorHit(id=self._chunks[index].chunk_id, score=scores[index], payload=chunk_payload(self._chunks[index]))
            for index in ranked
        ]

    def scoped(self, metadata_filter: MetadataFilter) -> "InMemoryBM25Retriever":
        """Build a smaller lexical index for a confirmed device scope."""

        chunks = [chunk for chunk in self._chunks if _matches(chunk, metadata_filter)]
        return InMemoryBM25Retriever(chunks, self._k1, self._b, scope_filter=metadata_filter)


class InMemoryExactIdentifierRetriever:
    """Exact lookup for normalized error codes, model numbers, and identifiers."""

    def __init__(self, chunks: Sequence[DocumentChunk], scope_filter: MetadataFilter | None = None) -> None:
        self._chunks = list(chunks)
        self._scope_filter = scope_filter
        self._by_value: dict[str, list[DocumentChunk]] = defaultdict(list)
        for chunk in self._chunks:
            if RetrievalProfile.EXACT in chunk.retrieval_profiles:
                value = chunk.metadata.get("normalized_value")
                if isinstance(value, str):
                    self._by_value[_normalize_identifier(value)].append(chunk)

    async def search(
        self,
        query: str,
        metadata_filter: MetadataFilter | None = None,
        limit: int = 10,
    ) -> list[VectorHit]:
        matches: list[DocumentChunk] = []
        for raw_value in IDENTIFIER_PATTERN.findall(query):
            matches.extend(
                chunk
                for chunk in self._by_value.get(_normalize_identifier(raw_value), [])
                if self._scope_filter == metadata_filter or _matches(chunk, metadata_filter)
            )
        return [VectorHit(id=chunk.chunk_id, score=1.0, payload=chunk_payload(chunk)) for chunk in matches[:limit]]

    def scoped(self, metadata_filter: MetadataFilter) -> "InMemoryExactIdentifierRetriever":
        """Build a smaller exact-identifier index for a confirmed device scope."""

        chunks = [chunk for chunk in self._chunks if _matches(chunk, metadata_filter)]
        return InMemoryExactIdentifierRetriever(chunks, scope_filter=metadata_filter)


class CombinedLexicalRetriever:
    """Run exact identifier and BM25 retrieval together outside Qdrant."""

    def __init__(self, bm25: InMemoryBM25Retriever, exact: InMemoryExactIdentifierRetriever) -> None:
        self.bm25 = bm25
        self.exact = exact

    async def search(
        self,
        query: str,
        metadata_filter: MetadataFilter | None = None,
        limit: int = 10,
    ) -> list[VectorHit]:
        exact_hits, bm25_hits = await gather(
            self.exact.search(query, metadata_filter, limit),
            self.bm25.search(query, metadata_filter, limit),
        )
        merged: list[VectorHit] = []
        seen: set[str] = set()
        for hit in [*exact_hits, *bm25_hits]:
            if hit.id not in seen:
                seen.add(hit.id)
                merged.append(hit)
        return merged[:limit]

    def scoped(self, metadata_filter: MetadataFilter) -> "CombinedLexicalRetriever":
        return CombinedLexicalRetriever(self.bm25.scoped(metadata_filter), self.exact.scoped(metadata_filter))


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _lexical_query_tokens(query: str, metadata_filter: MetadataFilter | None) -> list[str]:
    """Exclude confirmed device identity terms from scope-local BM25 scoring.

    Model and manufacturer names are useful for routing, but once the caller
    has supplied a confirmed scope they occur in many chunks and add no
    evidence about the user's symptom or requested procedure.
    """

    tokens = tokenize(query)
    if metadata_filter is None:
        return tokens
    identity_tokens = set(
        tokenize(" ".join(value for value in (metadata_filter.manufacturer, metadata_filter.model) if value))
    )
    return [token for token in tokens if token not in identity_tokens]


def _matches(chunk: DocumentChunk, metadata_filter: MetadataFilter | None) -> bool:
    if metadata_filter is None:
        return True
    values = metadata_filter.model_dump(exclude_none=True)
    fields = {
        "manufacturer": chunk.document.manufacturer,
        "model": chunk.document.model,
        "document_id": chunk.document.document_id,
        "document_version": chunk.document.version,
        "parent_chunk_id": chunk.parent_chunk_id,
        "kind": chunk.kind.value,
        "strategy": chunk.strategy.value,
        "normalized_value": chunk.metadata.get("normalized_value"),
    }
    return all(fields.get(field) == value for field, value in values.items())
