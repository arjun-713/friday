"""LiteLLM-backed answer generation with deterministic evidence validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import config_section, load_runtime_config
from ..prompts import build_messages
from .models import DiagnosticSessionState, DiagnosticStep, EvidenceContext


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
    api_key_env: str | None = None
    response_format: str | None = "json_object"
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls) -> LiteLLMSettings:
        values = config_section(load_runtime_config(), "llm")
        return cls(
            enabled=bool(values.get("enabled", False)),
            model=str(values.get("model", "openai/sarvam-105b-conversations")),
            api_base=str(values["api_base"]) if values.get("api_base") else None,
            temperature=float(values.get("temperature", 0.0)),
            max_tokens=int(values.get("max_tokens", 400)),
            timeout_seconds=float(values.get("timeout_seconds", 20.0)),
            api_key_env=str(values["api_key_env"]) if values.get("api_key_env") else None,
            response_format=str(values["response_format"]) if values.get("response_format") else None,
            reasoning_effort=str(values["reasoning_effort"]) if values.get("reasoning_effort") else None,
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

    async def generate_step(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
    ) -> DiagnosticStep:
        response = await self._complete(query, evidence, state, structured=True)
        answer = _response_text(response).strip()
        if answer.upper() == _UNSUPPORTED:
            raise UnsupportedAnswerError("the model could not answer from the supplied evidence")
        try:
            payload = json.loads(_strip_json_fence(answer))
            if not isinstance(payload, dict):
                raise TypeError("step response must be a JSON object")
            step = DiagnosticStep(
                step_id=_step_id(payload),
                title=payload["title"],
                instruction=payload["instruction"],
                question=payload["question"],
                options=payload.get("options", []),
                source_ids=payload["source_ids"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise InvalidAnswerError("LLM response was not a valid diagnostic step") from error
        _validate_step(step, evidence)
        return _expand_step_citations(step, evidence)

    async def stream_generate_step(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
    ) -> AsyncIterator[str | DiagnosticStep]:
        """Stream provider text, then yield the validated structured step."""

        response = await self._complete(query, evidence, state, stream=True, structured=True)
        pieces: list[str] = []
        async for chunk in response:
            text = _stream_text(chunk)
            if text:
                pieces.append(text)
                yield text
        answer = "".join(pieces).strip()
        if answer.upper() == _UNSUPPORTED:
            raise UnsupportedAnswerError("the model could not answer from the supplied evidence")
        try:
            payload = json.loads(_strip_json_fence(answer))
            if not isinstance(payload, dict):
                raise TypeError("step response must be a JSON object")
            step = DiagnosticStep(
                step_id=_step_id(payload),
                title=payload["title"],
                instruction=payload["instruction"],
                question=payload["question"],
                options=payload.get("options", []),
                source_ids=payload["source_ids"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise InvalidAnswerError("LLM response was not a valid diagnostic step") from error
        _validate_step(step, evidence)
        yield _expand_step_citations(step, evidence)

    async def _complete(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState | None = None,
        stream: bool = False,
        structured: bool = False,
    ) -> Any:
        completion = self._completion
        if completion is None:
            try:
                from litellm import acompletion
            except ImportError as error:  # pragma: no cover - dependency is installed in supported environments
                raise AnswerProviderUnavailable("LiteLLM is not installed") from error
            completion = acompletion

        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": build_messages(query, evidence, state),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "timeout": self.settings.timeout_seconds,
            "stream": stream,
        }
        if self.settings.api_base:
            request["api_base"] = self.settings.api_base
        api_key_env = self.settings.api_key_env
        if api_key_env is None and "sarvam" in self.settings.model.casefold():
            api_key_env = "SARVAM_API_KEY"
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                request["api_key"] = api_key
                request["extra_headers"] = {"api-subscription-key": api_key}
        if structured and self.settings.response_format:
            request["response_format"] = {"type": self.settings.response_format}
        if self.settings.reasoning_effort:
            request["reasoning_effort"] = self.settings.reasoning_effort
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


def _stream_text(chunk: Any) -> str:
    try:
        content = chunk.choices[0].delta.content
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise InvalidAnswerError("LLM stream chunk did not contain assistant content") from error
    return content if isinstance(content, str) else ""


def _strip_json_fence(answer: str) -> str:
    if answer.startswith("```") and answer.endswith("```"):
        lines = answer.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return answer


def _step_id(payload: dict[str, Any]) -> str:
    stable = f"{payload.get('title', '')}|{payload.get('instruction', '')}|{payload.get('source_ids', '')}"
    digest = hashlib.sha256(stable.encode()).hexdigest()[:16]
    return f"step-{digest}"


def _validate_step(step: DiagnosticStep, evidence: Sequence[EvidenceContext]) -> None:
    known_ids = {item.chunk_id for item in evidence}
    if any(source_id not in known_ids for source_id in step.source_ids):
        raise InvalidAnswerError("diagnostic step cited evidence outside the retrieved context")
    if len({option.id for option in step.options}) != len(step.options):
        raise InvalidAnswerError("diagnostic step contains duplicate options")
    evidence_text = " ".join(item.content.casefold() for item in evidence)
    for option in step.options:
        if option.label.casefold() not in evidence_text and option.label.casefold() not in {"not sure", "unknown"}:
            raise InvalidAnswerError("diagnostic option was not supported by retrieved evidence")


def _expand_step_citations(step: DiagnosticStep, evidence: Sequence[EvidenceContext]) -> DiagnosticStep:
    by_id = {item.chunk_id: item for item in evidence}
    citation_text = " ".join(
        f"[{by_id[source_id].citation.document_title} · p. {by_id[source_id].citation.page} · "
        f"{by_id[source_id].citation.section}]"
        for source_id in step.source_ids
    )
    return step.model_copy(update={"instruction": f"{step.instruction} {citation_text}"})


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
