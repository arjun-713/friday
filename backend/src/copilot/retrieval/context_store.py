"""Local parent-context store backed by generated JSONL chunks."""

import json
from collections.abc import Sequence
from pathlib import Path

from ..ingestion.models import DocumentChunk


class JsonlParentChunkStore:
    def __init__(self, chunks: dict[str, DocumentChunk]) -> None:
        self._chunks = chunks

    @classmethod
    def from_directory(cls, root: Path) -> "JsonlParentChunkStore":
        chunks: dict[str, DocumentChunk] = {}
        for path in sorted(root.glob("*/*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                chunk = DocumentChunk.model_validate_json(line)
                if chunk.metadata.get("role") in {"parent", None}:
                    chunks[chunk.chunk_id] = chunk
        return cls(chunks)

    async def fetch(self, ids: Sequence[str]) -> list[DocumentChunk]:
        return [self._chunks[chunk_id] for chunk_id in ids if chunk_id in self._chunks]

    def save_manifest(self, path: Path) -> None:
        path.write_text(
            json.dumps({chunk_id: chunk.model_dump(mode="json") for chunk_id, chunk in self._chunks.items()}),
            encoding="utf-8",
        )
