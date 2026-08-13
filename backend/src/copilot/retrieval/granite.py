"""Local Granite embedding provider backed by Sentence Transformers."""

import asyncio
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import EmbeddingProvider

DEFAULT_MODEL = "ibm-granite/granite-embedding-small-english-r2"
DEFAULT_DIMENSION = 384
DEFAULT_MAX_TOKENS = 8192


@dataclass(frozen=True)
class GraniteEmbeddingSettings:
    model_name: str = DEFAULT_MODEL
    dimension: int = DEFAULT_DIMENSION
    normalize_embeddings: bool = True
    batch_size: int = 8
    max_tokens: int = DEFAULT_MAX_TOKENS
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "GraniteEmbeddingSettings":
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_DIMENSION))),
            normalize_embeddings=_env_bool("EMBEDDING_NORMALIZE", True),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
            max_tokens=int(os.getenv("EMBEDDING_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        )


class GraniteEmbeddingProvider(EmbeddingProvider):
    """Async, CPU-first provider with explicit length and vector validation."""

    def __init__(self, settings: GraniteEmbeddingSettings | None = None, model: Any | None = None) -> None:
        self.settings = settings or GraniteEmbeddingSettings.from_env()
        if self.settings.dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if self.settings.batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        if self.settings.max_tokens <= 0:
            raise ValueError("embedding max tokens must be positive")
        self._model = model

    @property
    def dimension(self) -> int:
        return self.settings.dimension

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        return await self._encode(values)

    async def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query text must not be empty")
        vectors = await self._encode([text])
        return vectors[0]

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        lengths = await asyncio.to_thread(self._token_lengths, model, texts)
        oversized = [length for length in lengths if length > self.settings.max_tokens]
        if oversized:
            raise ValueError(
                f"embedding input has {max(oversized)} tokens; re-chunk before embedding "
                f"(limit={self.settings.max_tokens})"
            )
        encoded = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalize_embeddings,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        vectors = [list(vector) for vector in encoded]
        if len(vectors) != len(texts):
            raise ValueError("embedding model returned a different number of vectors than inputs")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(f"embedding dimension mismatch: expected {self.dimension}, got {len(vector)}")
            if not all(math.isfinite(float(value)) for value in vector):
                raise ValueError("embedding model returned a non-finite vector value")
        return [[float(value) for value in vector] for vector in vectors]

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Sentence Transformers is required for Granite embeddings; install the embeddings extra"
                ) from error
            self._model = SentenceTransformer(self.settings.model_name, device=self.settings.device)
        actual_dimension = self._model.get_sentence_embedding_dimension()
        if actual_dimension != self.dimension:
            raise ValueError(
                f"embedding model dimension is {actual_dimension}, configured dimension is {self.dimension}"
            )
        return self._model

    def _token_lengths(self, model: Any, texts: Sequence[str]) -> list[int]:
        tokenized = model.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        return [len(input_ids) for input_ids in tokenized["input_ids"]]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"{name} must be a boolean value")
    return value.lower() in {"1", "true", "yes"}
