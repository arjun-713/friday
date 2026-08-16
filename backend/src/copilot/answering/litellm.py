"""LiteLLM-backed answer generation with deterministic evidence validation."""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..prompts import build_messages
from .models import EvidenceContext


class AnswerGenerationError(RuntimeError):
    """Base error for failures after retrieval has produced evidence."""


class AnswerProviderUnavailable(ConnectionError, AnswerGenerationError):
    """The configured LLM provider could not produce a response."""


class UnsupportedAnswerError(AnswerGenerationError):
    """The model explicitly declined because the evidence was insufficient."""


class InvalidAnswerError(AnswerGenerationError):
    """The model response did not satisfy the evidence contract."""


@dataclass(frozen=True)
class LiteLLMSettings:
    """Runtime configuration for LiteLLM without storing provider secrets."""

    enabled: bool = False
    model: str = "deepseek/deepseek-chat"
    api_base: str | None = None
    temperature: float = 0.0
    max_tokens: int = 400
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> LiteLLMSettings:
        return cls(
            enabled=_env_bool("LLM_ENABLED", default=False),
            model=os.getenv("LLM_MODEL", "deepseek/deepseek-chat"),
            api_base=os.getenv("LLM_API_BASE") or None,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "400")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        )


CompletionFunction = Callable[..., Awaitable[Any]]
_SOURCE_MARKER = re.compile(r"\[source:([^\]]+)\]")
_UNSUPPORTED = "UNSUPPORTED"


class LiteLLMAnswerGenerator:
    """Generate one cited troubleshooting answer through LiteLLM's async SDK."""

    def __init__(
        self,
        settings: LiteLLMSettings | None = None,
        completion: CompletionFunction | None = None,
    ) -> None:
        self.settings = settings or LiteLLMSettings.from_env()
        self._completion = completion

    async def generate(self, query: str, evidence: Sequence[EvidenceContext]) -> str:
        if not evidence:
            raise InvalidAnswerError("cannot generate an answer without evidence")

        response = await self._complete(query, evidence)
        answer = _response_text(response).strip()
        if answer.upper() == _UNSUPPORTED:
            raise UnsupportedAnswerError("the model could not answer from the supplied evidence")
        _validate_answer(answer, evidence)
        return _expand_citations(answer, evidence)

    async def _complete(self, query: str, evidence: Sequence[EvidenceContext]) -> Any:
        completion = self._completion
        if completion is None:
            try:
                from litellm import acompletion
            except ImportError as error:  # pragma: no cover - dependency is installed in supported environments
                raise AnswerProviderUnavailable("LiteLLM is not installed") from error
            completion = acompletion

        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": build_messages(query, evidence),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "timeout": self.settings.timeout_seconds,
        }
        if self.settings.api_base:
            request["api_base"] = self.settings.api_base
        try:
            return await completion(**request)
        except Exception as error:  # LiteLLM maps provider failures to its own exception hierarchy.
            raise AnswerProviderUnavailable("configured LLM provider is unavailable") from error


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise InvalidAnswerError("LLM response did not contain assistant content") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(str(part) for part in parts)
    raise InvalidAnswerError("LLM response content was not text")


def _validate_answer(answer: str, evidence: Sequence[EvidenceContext]) -> None:
    if not answer:
        raise InvalidAnswerError("LLM response was empty")
    known_ids = {item.chunk_id for item in evidence}
    markers = _SOURCE_MARKER.findall(answer)
    if not markers:
        raise InvalidAnswerError("LLM response omitted evidence citations")
    if any(marker not in known_ids for marker in markers):
        raise InvalidAnswerError("LLM response cited evidence outside the retrieved context")


def _expand_citations(answer: str, evidence: Sequence[EvidenceContext]) -> str:
    evidence_by_id = {item.chunk_id: item for item in evidence}

    def replace(match: re.Match[str]) -> str:
        item = evidence_by_id[match.group(1)]
        citation = item.citation
        return f"[{citation.document_title} · p. {citation.page} · {citation.section}]"

    return _SOURCE_MARKER.sub(replace, answer)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
