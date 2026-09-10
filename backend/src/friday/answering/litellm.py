"""LiteLLM-backed answer generation with deterministic evidence validation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

from ..config import config_section, load_runtime_config
from ..observability import trace_event
from ..prompts import AGENT_TOOL_FOLLOWUP_PROMPT, build_conversation_messages, build_messages
from .models import (
    DecisionBasis,
    DiagnosticAction,
    DiagnosticFact,
    DiagnosticOption,
    DiagnosticSessionState,
    DiagnosticStep,
    DiagnosticTurn,
    EvidenceContext,
    ObservationRequest,
)
from .tools import AGENT_TOOLS, AgentToolExecutor, AgentToolResult


class AnswerGenerationError(RuntimeError):
    """Base error for failures after retrieval has produced evidence."""

    llm_calls: tuple[LLMCallRecord, ...] = ()


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
    max_retries: int = 0
    api_key_env: str | None = None
    response_format: str | None = "json_object"
    reasoning_effort: str | None = None
    disable_reasoning: bool = False
    prompt_cache_enabled: bool = True
    prompt_cache_key: str = "friday-grounded-conversation"
    prompt_cache_retention: str | None = None
    prompt_cache_ttl: str | None = None
    verbosity: str | None = None
    service_tier: str | None = None
    api_mode: str = "chat_completions"
    agentic_enabled: bool = True
    max_tool_calls: int = 2
    planner_response_position: str = "last"
    streaming_gate: str = "strict"

    @classmethod
    def from_env(cls) -> LiteLLMSettings:
        values = config_section(load_runtime_config(), "llm")
        return cls(
            enabled=bool(values.get("enabled", False)),
            model=_env_or_config("FRIDAY_LLM_MODEL", values, "model", "openai/sarvam-105b-conversations")
            or "openai/sarvam-105b-conversations",
            api_base=str(values["api_base"]) if values.get("api_base") else None,
            temperature=float(values.get("temperature", 0.0)),
            max_tokens=int(values.get("max_tokens", 400)),
            timeout_seconds=float(values.get("timeout_seconds", 20.0)),
            max_retries=int(values.get("max_retries", 0)),
            api_key_env=_env_or_config("FRIDAY_LLM_API_KEY_ENV", values, "api_key_env"),
            response_format=_env_or_config("FRIDAY_LLM_RESPONSE_FORMAT", values, "response_format"),
            reasoning_effort=_env_or_config("FRIDAY_LLM_REASONING_EFFORT", values, "reasoning_effort"),
            disable_reasoning=bool(values.get("disable_reasoning", False)),
            prompt_cache_enabled=bool(values.get("prompt_cache_enabled", True)),
            prompt_cache_key=str(values.get("prompt_cache_key", "friday-grounded-conversation")),
            prompt_cache_retention=(
                str(values["prompt_cache_retention"]) if values.get("prompt_cache_retention") else None
            ),
            prompt_cache_ttl=str(values["prompt_cache_ttl"]) if values.get("prompt_cache_ttl") else None,
            verbosity=str(values["verbosity"]) if values.get("verbosity") else None,
            service_tier=str(values["service_tier"]) if values.get("service_tier") else None,
            api_mode=str(values.get("api_mode", "chat_completions")),
            agentic_enabled=bool(values.get("agentic_enabled", True)),
            max_tool_calls=max(0, min(int(values.get("max_tool_calls", 2)), 2)),
            planner_response_position=str(
                os.getenv("FRIDAY_PLANNER_RESPONSE_POSITION") or values.get("planner_response_position", "last")
            ).lower(),
            streaming_gate=str(os.getenv("FRIDAY_STREAMING_GATE") or values.get("streaming_gate", "strict")).lower(),
        )


CompletionFunction = Callable[..., Awaitable[Any]]


def _env_or_config(name: str, values: dict[str, Any], key: str, default: object = None) -> str | None:
    """Prefer an explicit environment override; an empty value unsets the option.

    Used for A/B experiments (model, key env, response format, reasoning
    effort) without editing the baked-in config file.
    """

    if name in os.environ:
        return os.environ[name] or None
    value = values.get(key, default)
    return str(value) if value is not None else None


_SOURCE_MARKER = re.compile(r"\[source:([^\]]+)\]")
_UNSUPPORTED = "UNSUPPORTED"
# One optional retrieval/tool refinement is enough after the initial hybrid
# retrieval. More rounds turn a single diagnostic turn into a slow questionnaire
# and delay the first TTS audio without improving the response contract.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRun:
    """The final planner decision and all evidence it used."""

    turn: DiagnosticTurn
    evidence: list[EvidenceContext]
    tool_names: list[str] | None = None
    llm_calls: tuple[LLMCallRecord, ...] = ()


@dataclass(frozen=True)
class LLMCallRecord:
    """Per-provider-call timing and token accounting for one turn.

    TTFT (time to first token) isolates provider queue/scheduling delay from
    generation speed: ``tokens_per_sec`` is computed over post-TTFT time only.
    """

    index: int
    model: str
    stream: bool
    structured: bool
    tools_attached: bool
    prompt_chars: int
    max_tokens: int | None
    ttft_ms: float | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    tokens_per_sec: float | None
    tool_names: tuple[str, ...] = ()
    note: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Render a JSON-safe record for SSE timings events and benchmarks."""

        return {
            "index": self.index,
            "model": self.model,
            "stream": self.stream,
            "structured": self.structured,
            "tools_attached": self.tools_attached,
            "prompt_chars": self.prompt_chars,
            "max_tokens": self.max_tokens,
            "ttft_ms": self.ttft_ms,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "tokens_per_sec": self.tokens_per_sec,
            "tool_names": list(self.tool_names),
            "note": self.note,
        }


@dataclass(frozen=True)
class AgentStreamEvidence:
    """Evidence discovered during a tool-assisted streaming turn."""

    evidence: list[EvidenceContext]
    tool_names: list[str]


_DIAGNOSTIC_TURN_PROPERTIES: dict[str, object] = {
    "mode": {"type": "string", "enum": ["solve", "advance", "clarify", "abstain"]},
    "interpretation": {"type": ["string", "null"]},
    "next_action": {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {"instruction": {"type": "string"}, "why": {"type": ["string", "null"]}},
        "required": ["instruction", "why"],
    },
    "observation_request": {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {
            "request_id": {"type": "string"},
            "fact_key": {"type": "string"},
            "question": {"type": "string"},
            "options": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["id", "label", "value"],
                },
            },
            "recheck_after_action": {"type": "boolean"},
        },
        "required": ["request_id", "fact_key", "question", "options", "recheck_after_action"],
    },
    "decision_basis": {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {
            "why_not_solved": {"type": "string"},
            "discriminates_between": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"type": "string"},
            },
            "expected_discrimination": {"type": "string"},
        },
        "required": ["why_not_solved", "discriminates_between", "expected_discrimination"],
    },
    "facts_learned": {
        "type": "array",
        "maxItems": 12,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
                "label": {"type": "string"},
                "raw": {"type": "string"},
            },
            "required": ["key", "value", "label", "raw"],
        },
    },
    "candidate_causes": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
    "ruled_out_causes": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
    "source_ids": {
        "type": "array",
        "minItems": 1,
        "maxItems": 8,
        "items": {"type": "string"},
    },
    # Speakable prose defaults last: the strict streaming gate validates every
    # other field before the first response character reaches text or voice, so
    # nothing unvalidated is ever spoken. The "first" order (response up front
    # with optimistic streaming) is an explicit experiment behind settings.
    "response": {"type": "string"},
}


_RESPONSE_PROPERTY: dict[str, object] = {"type": "string"}


def _diagnostic_turn_schema(*, response_first: bool = False) -> dict[str, object]:
    properties = dict(_DIAGNOSTIC_TURN_PROPERTIES)
    response = properties.pop("response", None) or dict(_RESPONSE_PROPERTY)
    ordered = {"response": response, **properties} if response_first else {**properties, "response": response}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": ordered,
        "required": list(ordered),
    }


_DIAGNOSTIC_TURN_SCHEMA = _diagnostic_turn_schema()


class LiteLLMAnswerGenerator:
    """Generate one cited troubleshooting answer through LiteLLM's async SDK."""

    def __init__(
        self,
        settings: LiteLLMSettings | None = None,
        completion: CompletionFunction | None = None,
    ) -> None:
        self.settings = settings or LiteLLMSettings.from_env()
        self._completion = completion
        self.agentic_enabled = self.settings.agentic_enabled

    async def generate(self, query: str, evidence: Sequence[EvidenceContext]) -> str:
        if not evidence:
            raise InvalidAnswerError("cannot generate an answer without evidence")

        response = await self._complete(query, evidence)
        answer = _response_text(response).strip()
        if answer.upper() == _UNSUPPORTED:
            raise UnsupportedAnswerError("the model could not answer from the supplied evidence")
        _validate_answer(answer, evidence)
        return _expand_citations(answer, evidence)

    async def generate_turn(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
    ) -> DiagnosticTurn:
        response = await self._complete(query, evidence, state, structured=True)
        answer = _response_text(response).strip()
        if answer.upper() == _UNSUPPORTED:
            raise UnsupportedAnswerError("the model could not answer from the supplied evidence")
        try:
            payload = json.loads(_strip_json_fence(answer))
            if not isinstance(payload, dict):
                raise TypeError("turn response must be a JSON object")
            turn = _diagnostic_turn(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise InvalidAnswerError("LLM response was not a valid diagnostic turn") from error
        _validate_turn(turn, evidence, state)
        return turn

    async def generate_agent_turn(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
        execute_tool: AgentToolExecutor,
    ) -> AgentRun:
        """Generate one tool-assisted planner turn with a single LLM call when possible.

        The structured planner request carries the read-only tools, so the common
        no-tool turn costs exactly one provider round trip. Tool calls are only
        followed by a second structured call when the model actually requested
        evidence. Responses that contain tool calls may have null content, which
        is normal and must not be treated as an error.
        """

        records: list[LLMCallRecord] = []
        try:
            return await self._generate_agent_turn_inner(query, evidence, state, execute_tool, records)
        except (UnsupportedAnswerError, InvalidAnswerError) as error:
            # Surface per-call accounting on the failure path so benchmarks
            # can count provider calls even for turns that abstain.
            error.llm_calls = tuple(records)
            raise

    async def _generate_agent_turn_inner(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
        execute_tool: AgentToolExecutor,
        records: list[LLMCallRecord],
    ) -> AgentRun:
        use_tools = self.settings.max_tool_calls > 0
        response_first = self.settings.planner_response_position == "first"
        messages = build_messages(query, evidence, state, response_first=response_first)
        response = await self._complete_messages(
            messages,
            stream=False,
            structured=True,
            tools=AGENT_TOOLS if use_tools else None,
            max_tokens=max(self.settings.max_tokens, 1200),
            record_to=records,
            call_index=len(records),
        )
        tool_calls = _tool_calls(response) if use_tools else []
        if not tool_calls:
            try:
                run = self._parse_turn_response(response, list(evidence), state, [])
                return AgentRun(
                    turn=run.turn, evidence=run.evidence, tool_names=run.tool_names, llm_calls=tuple(records)
                )
            except InvalidAnswerError as error:
                # One targeted retry with the failure reason: a correctable
                # contract violation (ungrounded option, repeated action) is
                # cheaper to fix than to abandon the turn and make the user
                # re-ask. Genuine UNSUPPORTED judgments are never retried.
                return await self._retry_turn(messages, list(evidence), state, [], error, records)
        extra_evidence: list[EvidenceContext] = []
        tool_names: list[str] = []
        tool_results: list[AgentToolResult] = []
        for call in tool_calls[: self.settings.max_tool_calls]:
            name, arguments, _call_id = _tool_call_parts(call)
            tool_names.append(name)
            trace_event(logger, "agent_tool_started", tool=name)
            tool_started = perf_counter()
            result: AgentToolResult = await execute_tool(name, arguments)
            tool_results.append(result)
            trace_event(
                logger,
                "agent_tool_completed",
                tool=name,
                elapsed_ms=round((perf_counter() - tool_started) * 1000, 2),
                evidence_count=len(result.evidence),
            )
            extra_evidence.extend(result.evidence)
        combined_evidence = [*evidence, *_dedupe_evidence(extra_evidence)]
        followup_messages = [
            *messages,
            _assistant_message(response),
            *self._tool_messages(response, tool_results),
        ]
        followup = await self._complete_messages(
            followup_messages,
            stream=False,
            structured=True,
            max_tokens=max(self.settings.max_tokens, 1200),
            record_to=records,
            call_index=len(records),
            note=f"tool-continuation: {','.join(tool_names)}" if tool_names else None,
        )
        try:
            run = self._parse_turn_response(followup, combined_evidence, state, tool_names)
            return AgentRun(turn=run.turn, evidence=run.evidence, tool_names=run.tool_names, llm_calls=tuple(records))
        except InvalidAnswerError as error:
            return await self._retry_turn(followup_messages, combined_evidence, state, tool_names, error, records)

    async def stream_agent_turn(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
        execute_tool: AgentToolExecutor,
    ) -> AsyncIterator[str | AgentStreamEvidence | AgentRun]:
        """Stream validated planner prose as early as the provider emits it.

        The structured call runs with ``stream=True``. A response gate buffers
        the JSON, validates every non-``response`` field the moment it is
        complete, and only then releases decoded ``response`` characters, so
        text and voice speak nothing unvalidated. Tool calls stream as deltas;
        when present, tools execute and the followup streams through a second
        gate. Validation failures fall back to one non-streaming correction.
        """

        records: list[LLMCallRecord] = []
        try:
            async for item in self._stream_agent_turn_inner(query, evidence, state, execute_tool, records):
                yield item
        except (UnsupportedAnswerError, InvalidAnswerError) as error:
            # Surface per-call accounting on the failure path so benchmarks
            # can count provider calls even for turns that abstain.
            error.llm_calls = tuple(records)
            raise

    async def _stream_agent_turn_inner(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
        execute_tool: AgentToolExecutor,
        records: list[LLMCallRecord],
    ) -> AsyncIterator[str | AgentStreamEvidence | AgentRun]:
        use_tools = self.settings.max_tool_calls > 0
        response_first = self.settings.planner_response_position == "first"
        optimistic = self.settings.streaming_gate == "optimistic"
        messages = build_messages(query, evidence, state, response_first=response_first)
        response_iter = await self._complete_messages(
            messages,
            stream=True,
            structured=True,
            tools=AGENT_TOOLS if use_tools else None,
            max_tokens=max(self.settings.max_tokens, 1200),
            record_to=records,
            call_index=len(records),
        )
        gate = _ResponseGate(evidence, state, optimistic=optimistic)
        tool_acc = _ToolCallAccumulator()
        async for chunk in response_iter:
            try:
                piece = _stream_text(chunk)
                fragments = _delta_tool_calls(chunk)
            except InvalidAnswerError:
                continue
            tool_acc.add(fragments)
            for token in gate.feed(piece):
                yield token
        tool_calls = tool_acc.calls() if use_tools else []
        if not tool_calls:
            try:
                turn = gate.complete_turn()
            except InvalidAnswerError as error:
                retried = await self._retry_turn(messages, list(evidence), state, [], error, records)
                if retried.turn.response:
                    yield retried.turn.response
                yield retried
                return
            except UnsupportedAnswerError:
                raise
            yield AgentRun(turn=turn, evidence=list(evidence), tool_names=[], llm_calls=tuple(records))
            return
        extra_evidence: list[EvidenceContext] = []
        tool_names: list[str] = []
        tool_results: list[AgentToolResult] = []
        for name, arguments, _call_id in tool_calls[: self.settings.max_tool_calls]:
            tool_names.append(name)
            trace_event(logger, "agent_tool_started", tool=name)
            tool_started = perf_counter()
            result: AgentToolResult = await execute_tool(name, arguments)
            tool_results.append(result)
            trace_event(
                logger,
                "agent_tool_completed",
                tool=name,
                elapsed_ms=round((perf_counter() - tool_started) * 1000, 2),
                evidence_count=len(result.evidence),
            )
            extra_evidence.extend(result.evidence)
        combined_evidence = [*evidence, *_dedupe_evidence(extra_evidence)]
        yield AgentStreamEvidence(evidence=combined_evidence, tool_names=tool_names)
        followup_messages: list[dict[str, Any]] = [
            *messages,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id or f"friday-tool-{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                    for index, (name, arguments, call_id) in enumerate(tool_calls[: self.settings.max_tool_calls])
                ],
            },
            *self._tool_messages_from_parts(tool_calls[: self.settings.max_tool_calls], tool_results),
        ]
        followup_iter = await self._complete_messages(
            followup_messages,
            stream=True,
            structured=True,
            max_tokens=max(self.settings.max_tokens, 1200),
            record_to=records,
            call_index=len(records),
            note=f"tool-continuation: {','.join(tool_names)}" if tool_names else None,
        )
        followup_gate = _ResponseGate(combined_evidence, state, optimistic=optimistic)
        async for chunk in followup_iter:
            try:
                piece = _stream_text(chunk)
            except InvalidAnswerError:
                continue
            for token in followup_gate.feed(piece):
                yield token
        try:
            turn = followup_gate.complete_turn()
        except InvalidAnswerError as error:
            retried = await self._retry_turn(followup_messages, combined_evidence, state, tool_names, error, records)
            if retried.turn.response:
                yield retried.turn.response
            yield retried
            return
        yield AgentRun(turn=turn, evidence=combined_evidence, tool_names=tool_names, llm_calls=tuple(records))

    @staticmethod
    def _tool_messages_from_parts(
        calls: Sequence[tuple[str, dict[str, Any], str]],
        results: Sequence[AgentToolResult],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for index, (_, _, call_id) in enumerate(calls):
            result = results[index] if index < len(results) else None
            content = result.content[:9000] if result is not None else "Tool returned no result."
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id or f"friday-tool-{index}",
                    "content": content,
                }
            )
        return messages

    async def _retry_turn(
        self,
        messages: Sequence[dict[str, Any]],
        evidence: list[EvidenceContext],
        state: DiagnosticSessionState,
        tool_names: list[str],
        error: InvalidAnswerError,
        records: list[LLMCallRecord],
    ) -> AgentRun:
        """Re-ask once with the validation failure spelled out, without tools."""

        reason = str(error)[:200]
        logger.info("planner retrying turn after validation failure reason=%s", reason)
        correction = {
            "role": "user",
            "content": (
                "Your previous turn was rejected and not shown to the user: "
                f"{error}. Return exactly one corrected JSON object that fixes "
                "this issue while keeping every technical detail, option label, "
                "and source ID grounded in the retrieved evidence."
            ),
        }
        response = await self._complete_messages(
            [*messages, correction],
            stream=False,
            structured=True,
            max_tokens=max(self.settings.max_tokens, 1200),
            record_to=records,
            call_index=len(records),
            note=f"validation-retry: {reason}",
        )
        run = self._parse_turn_response(response, evidence, state, tool_names)
        return AgentRun(turn=run.turn, evidence=run.evidence, tool_names=run.tool_names, llm_calls=tuple(records))

    def _parse_turn_response(
        self,
        response: Any,
        evidence: list[EvidenceContext],
        state: DiagnosticSessionState,
        tool_names: list[str],
    ) -> AgentRun:
        """Validate one structured planner response into an AgentRun."""

        turn = _parse_turn_text(_response_text(response).strip(), evidence, state)
        return AgentRun(turn=turn, evidence=evidence, tool_names=tool_names)

    async def stream_agentic_conversation(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
        execute_tool: AgentToolExecutor,
    ) -> AsyncIterator[str | AgentStreamEvidence]:
        """Stream a response, invoking at most two read-only evidence tools."""

        started = perf_counter()
        messages = build_conversation_messages(query, evidence, state)
        response, extra_evidence, tool_names, tool_results = await self._run_agent_tools(
            messages, evidence, execute_tool
        )
        combined_evidence = [*evidence, *extra_evidence]
        if extra_evidence:
            yield AgentStreamEvidence(evidence=combined_evidence, tool_names=tool_names)
            final_messages = [
                *messages,
                _assistant_message(response),
                *self._tool_messages(response, tool_results),
                {
                    "role": "system",
                    "content": AGENT_TOOL_FOLLOWUP_PROMPT,
                },
            ]
            response = await self._complete_messages(final_messages, stream=True, structured=False)
        else:
            # No tool was needed. The provider's direct content is already the
            # lowest-latency answer and should not incur a second completion.
            direct = _response_text(response).strip()
            if direct:
                yield direct
                return
            raise InvalidAnswerError("the model returned neither content nor a tool call")

        sequence = 0
        async for chunk in response:
            text = _stream_text(chunk)
            if text:
                sequence += 1
                trace_event(logger, "llm_provider_piece", started=started, sequence=sequence, chars=len(text))
                yield text

    async def _run_agent_tools(
        self,
        messages: Sequence[dict[str, Any]],
        initial_evidence: Sequence[EvidenceContext],
        execute_tool: AgentToolExecutor,
    ) -> tuple[Any, list[EvidenceContext], list[str], list[AgentToolResult]]:
        """Let the model request at most two read-only retrieval operations."""

        response = await self._complete_messages(
            messages,
            stream=False,
            structured=False,
            tools=AGENT_TOOLS,
            max_tokens=min(max(self.settings.max_tokens, 160), 320),
        )
        tool_calls = _tool_calls(response)
        if not tool_calls:
            return response, [], [], []

        evidence: list[EvidenceContext] = []
        names: list[str] = []
        results: list[AgentToolResult] = []
        for call in tool_calls[: self.settings.max_tool_calls]:
            name, arguments, _call_id = _tool_call_parts(call)
            names.append(name)
            trace_event(logger, "agent_tool_started", tool=name)
            tool_started = perf_counter()
            result: AgentToolResult = await execute_tool(name, arguments)
            results.append(result)
            trace_event(
                logger,
                "agent_tool_completed",
                tool=name,
                elapsed_ms=round((perf_counter() - tool_started) * 1000, 2),
                evidence_count=len(result.evidence),
            )
            evidence.extend(result.evidence)
        return response, _dedupe_evidence(evidence), names, results

    @staticmethod
    def _tool_messages(response: Any, results: Sequence[AgentToolResult]) -> list[dict[str, Any]]:
        calls = _tool_calls(response)
        messages: list[dict[str, Any]] = []
        for index, call in enumerate(calls[:2]):
            _, _, call_id = _tool_call_parts(call)
            result = results[index] if index < len(results) else None
            content = result.content[:9000] if result is not None else "Tool returned no result."
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id or f"friday-tool-{index}",
                    "content": content,
                }
            )
        return messages

    async def _instrumented_stream(
        self,
        chunks: Any,
        *,
        started: float,
        record_to: list[LLMCallRecord] | None,
        call_index: int,
        model: str,
        structured: bool,
        tools_attached: bool,
        prompt_chars: int,
        max_tokens: Any,
        note: str | None,
    ) -> AsyncIterator[Any]:
        """Forward provider stream chunks while capturing TTFT for voice metrics."""

        first_token_at: float | None = None
        count = 0
        async for chunk in chunks:
            count += 1
            if first_token_at is None and (_stream_text(chunk) or _delta_tool_calls(chunk)):
                first_token_at = perf_counter()
                first_ttft_ms = round((first_token_at - started) * 1000, 2)
                trace_event(logger, "llm_first_token", started=started, call_index=call_index, ttft_ms=first_ttft_ms)
            yield chunk
        latency_ms = (perf_counter() - started) * 1000
        ttft_ms = round((first_token_at - started) * 1000, 2) if first_token_at is not None else None
        if record_to is not None:
            record_to.append(
                LLMCallRecord(
                    index=call_index,
                    model=model,
                    stream=True,
                    structured=structured,
                    tools_attached=tools_attached,
                    prompt_chars=prompt_chars,
                    max_tokens=max_tokens if isinstance(max_tokens, int) else None,
                    ttft_ms=ttft_ms,
                    latency_ms=latency_ms,
                    prompt_tokens=None,
                    completion_tokens=None,
                    cached_tokens=None,
                    tokens_per_sec=None,
                    tool_names=(),
                    note=note,
                )
            )
        trace_event(
            logger,
            "llm_call_complete",
            started=started,
            call_index=call_index,
            model=model,
            ttft_ms=ttft_ms,
            latency_ms=round(latency_ms, 2),
            chunks=count,
        )

    async def stream_conversation(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
    ) -> AsyncIterator[str]:
        """Stream one grounded natural-language reply without planner JSON."""

        started = perf_counter()
        trace_event(logger, "llm_stream_started", started=started, query_chars=len(query), evidence_count=len(evidence))
        response = await self._complete_messages(
            build_conversation_messages(query, evidence, state),
            stream=True,
            structured=False,
        )
        sequence = 0
        async for chunk in response:
            text = _stream_text(chunk)
            if text:
                sequence += 1
                trace_event(
                    logger,
                    "llm_provider_piece",
                    started=started,
                    sequence=sequence,
                    chars=len(text),
                )
                yield text
        trace_event(logger, "llm_stream_complete", started=started, pieces=sequence)

    async def _generate_agent_turn(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
    ) -> AgentRun:

        messages: list[dict[str, Any]] = build_messages(query, evidence, state)
        # A diagnostic turn contains several required fields, including the
        # evidence IDs. Keep conversational streaming concise, but do not let
        # the shared short voice budget truncate the agent JSON mid-document.
        # GPT-OSS may spend part of this budget on its internal reasoning even
        # with low reasoning effort, so the structured path needs headroom.
        response = await self._complete_messages(
            messages,
            structured=True,
            max_tokens=max(self.settings.max_tokens, 1200),
        )
        answer = _response_text(response).strip()
        if answer.upper() == _UNSUPPORTED:
            raise UnsupportedAnswerError("the model could not answer from the supplied evidence")
        try:
            payload = json.loads(_strip_json_fence(answer))
            if not isinstance(payload, dict):
                raise TypeError("turn response must be a JSON object")
            turn = _diagnostic_turn(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise InvalidAnswerError("LLM response was not a valid diagnostic turn") from error
        try:
            _validate_turn(turn, evidence, state)
        except InvalidAnswerError as error:
            logger.warning("LLM diagnostic turn failed validation reason=%s", str(error)[:200])
            raise
        return AgentRun(turn=turn, evidence=list(evidence))

    async def stream_generate_turn(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState,
    ) -> AsyncIterator[str | DiagnosticTurn]:
        """Stream provider text, then yield the validated diagnostic turn."""

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
                raise TypeError("turn response must be a JSON object")
            turn = _diagnostic_turn(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise InvalidAnswerError("LLM response was not a valid diagnostic turn") from error
        _validate_turn(turn, evidence, state)
        yield turn

    # Compatibility for focused callers that still need the pre-v4 action
    # shape. The service itself consumes ``DiagnosticTurn``.
    async def generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> DiagnosticStep:
        turn = await self.generate_turn(query, evidence, state)
        return _turn_step(turn)

    async def stream_generate_step(
        self, query: str, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState
    ) -> AsyncIterator[str | DiagnosticStep]:
        async for item in self.stream_generate_turn(query, evidence, state):
            yield item if isinstance(item, str) else _turn_step(item)

    async def _complete(
        self,
        query: str,
        evidence: Sequence[EvidenceContext],
        state: DiagnosticSessionState | None = None,
        stream: bool = False,
        structured: bool = False,
    ) -> Any:
        return await self._complete_messages(
            build_messages(query, evidence, state), stream=stream, structured=structured
        )

    async def _complete_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        stream: bool = False,
        structured: bool = False,
        tools: Sequence[dict[str, object]] | None = None,
        max_tokens: int | None = None,
        record_to: list[LLMCallRecord] | None = None,
        call_index: int = 0,
        note: str | None = None,
    ) -> Any:
        completion = self._completion
        if completion is None:
            try:
                from litellm import acompletion, aresponses
            except ImportError as error:  # pragma: no cover - dependency is installed in supported environments
                raise AnswerProviderUnavailable("LiteLLM is not installed") from error
            completion = aresponses if self.settings.api_mode == "responses" else acompletion

        cache_mode = _prompt_cache_mode(self.settings)
        request_messages: Sequence[dict[str, Any]] = messages
        if cache_mode == "content":
            request_messages = _mark_cacheable_prefix(messages)

        if self.settings.api_mode == "responses" and self._completion is None:
            request: dict[str, Any] = {
                "model": self.settings.model.removeprefix("openai/"),
                "input": list(request_messages),
                "max_output_tokens": max_tokens or self.settings.max_tokens,
                "timeout": self.settings.timeout_seconds,
                "num_retries": self.settings.max_retries,
                "stream": stream,
            }
        else:
            request = {
                "model": self.settings.model,
                "messages": list(request_messages),
                "temperature": self.settings.temperature,
                "max_tokens": max_tokens or self.settings.max_tokens,
                "timeout": self.settings.timeout_seconds,
                "num_retries": self.settings.max_retries,
                "stream": stream,
            }
        if cache_mode in {"openai", "deepseek"} and self.settings.prompt_cache_key:
            request["prompt_cache_key"] = self.settings.prompt_cache_key
        if cache_mode == "openai" and self.settings.prompt_cache_retention:
            request["prompt_cache_retention"] = self.settings.prompt_cache_retention
        if cache_mode == "openai" and self.settings.prompt_cache_ttl:
            request["prompt_cache_options"] = {"ttl": self.settings.prompt_cache_ttl}
        if self.settings.verbosity and self.settings.model.casefold().startswith("openai/"):
            if self.settings.api_mode == "responses" and self._completion is None:
                request["text"] = {"verbosity": self.settings.verbosity}
            else:
                request["verbosity"] = self.settings.verbosity
        if self.settings.service_tier and self.settings.model.casefold().startswith("openai/"):
            request["service_tier"] = self.settings.service_tier
        if self.settings.api_base:
            request["api_base"] = self.settings.api_base
        api_key_env = self.settings.api_key_env
        if api_key_env is None and "sarvam" in self.settings.model.casefold():
            api_key_env = "SARVAM_API_KEY"
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise AnswerProviderUnavailable("configured LLM provider has no API key")
            request["api_key"] = api_key
            if "sarvam" in self.settings.model.casefold():
                request["extra_headers"] = {"api-subscription-key": api_key}
        if structured and self.settings.response_format:
            response_first = self.settings.planner_response_position == "first"
            if self.settings.api_mode == "responses" and self._completion is None:
                request["text"] = {
                    "format": _response_format(self.settings.response_format, response_first=response_first)
                }
            else:
                request["response_format"] = _response_format(
                    self.settings.response_format, response_first=response_first
                )
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"
        if self.settings.disable_reasoning and self.settings.api_mode == "responses" and self._completion is None:
            request["reasoning"] = {"effort": "none"}
        elif self.settings.disable_reasoning:
            # Sarvam treats an omitted value as its default reasoning mode.
            # The explicit JSON null is therefore intentional, not a missing
            # setting: it removes hidden reasoning tokens from the latency path.
            request["reasoning_effort"] = None
        elif self.settings.reasoning_effort:
            if self.settings.api_mode == "responses" and self._completion is None:
                request["reasoning"] = {"effort": self.settings.reasoning_effort}
            else:
                request["reasoning_effort"] = self.settings.reasoning_effort
        started = perf_counter()
        prompt_chars = sum(len(str(message.get("content", ""))) for message in request_messages)
        trace_event(
            logger,
            "llm_request_sent",
            started=started,
            call_index=call_index,
            model=self.settings.model,
            stream=stream,
            structured=structured,
            tools_attached=bool(tools),
            prompt_chars=prompt_chars,
            max_tokens=request.get("max_tokens", request.get("max_output_tokens")),
        )
        try:
            result = await completion(**request)
            if stream:
                result = self._instrumented_stream(
                    result,
                    started=started,
                    record_to=record_to,
                    call_index=call_index,
                    model=self.settings.model,
                    structured=structured,
                    tools_attached=bool(tools),
                    prompt_chars=prompt_chars,
                    max_tokens=request.get("max_tokens", request.get("max_output_tokens")),
                    note=note,
                )
                return result
            latency_ms = (perf_counter() - started) * 1000
            prompt_tokens, completion_tokens = _usage_tokens(result)
            cached_tokens = _cached_input_tokens(result)
            generation_ms = latency_ms  # Non-streaming responses have no first-token split.
            tokens_per_sec = (
                round(completion_tokens / (generation_ms / 1000), 2)
                if completion_tokens and generation_ms > 0
                else None
            )
            if record_to is not None:
                record_to.append(
                    LLMCallRecord(
                        index=call_index,
                        model=self.settings.model,
                        stream=False,
                        structured=structured,
                        tools_attached=bool(tools),
                        prompt_chars=prompt_chars,
                        max_tokens=request.get("max_tokens", request.get("max_output_tokens")),
                        ttft_ms=None,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cached_tokens=cached_tokens,
                        tokens_per_sec=tokens_per_sec,
                        tool_names=tuple(_tool_call_names(result)),
                        note=note,
                    )
                )
            logger.info(
                "llm_completion_complete stream=%s structured=%s max_tokens=%s prompt_cache=%s latency_ms=%.1f",
                stream,
                structured,
                request.get("max_tokens", request.get("max_output_tokens")),
                cache_mode or "unsupported",
                latency_ms,
            )
            cached_tokens_logged = cached_tokens
            if cached_tokens_logged is not None:
                logger.info("llm_prompt_cache_usage mode=%s cached_tokens=%d", cache_mode, cached_tokens_logged)
            trace_event(
                logger,
                "llm_call_complete",
                started=started,
                call_index=call_index,
                model=self.settings.model,
                ttft_ms=None,
                latency_ms=round(latency_ms, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_per_sec=tokens_per_sec,
            )
            return result
        except Exception as error:  # LiteLLM maps provider failures to its own exception hierarchy.
            api_key = os.getenv(api_key_env) if api_key_env else None
            safe_message = str(error).replace(api_key or "", "<redacted>")[:500]
            logger.warning(
                "LLM provider request failed type=%s message=%s",
                type(error).__name__,
                safe_message,
            )
            raise AnswerProviderUnavailable("configured LLM provider is unavailable") from error


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
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
    if getattr(chunk, "type", None) == "response.output_text.delta":
        delta = getattr(chunk, "delta", None)
        return delta if isinstance(delta, str) else ""
    try:
        if not chunk.choices:
            return ""
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


def _diagnostic_step(payload: dict[str, Any]) -> DiagnosticStep:
    """Normalize harmless provider formatting differences before strict validation."""

    raw_options = payload.get("options", [])
    if not isinstance(raw_options, list):
        raise TypeError("options must be an array")
    options: list[DiagnosticOption] = []
    for index, option in enumerate(raw_options, start=1):
        if isinstance(option, str):
            label = option.strip()
            option_id = _option_id(label, index)
        elif isinstance(option, dict):
            label = str(option.get("label", "")).strip()
            option_id = str(option.get("id") or _option_id(label, index)).strip()
        else:
            raise TypeError("each option must be an object or string")
        if not label or not option_id:
            raise ValueError("each option needs an id and label")
        options.append(DiagnosticOption(id=option_id, label=label))
    raw_source_ids = payload["source_ids"]
    if not isinstance(raw_source_ids, list):
        raise TypeError("source_ids must be an array")
    source_ids = [_canonical_source_id(value) for value in raw_source_ids]
    return DiagnosticStep(
        step_id=_step_id(payload),
        title=payload["title"],
        instruction=payload["instruction"],
        question=payload["question"],
        options=options,
        source_ids=source_ids,
    )


def _diagnostic_turn(payload: dict[str, Any]) -> DiagnosticTurn:
    """Parse the LLM-owned diagnostic decision without prescribing a branch."""

    # Permit stored fixtures and third-party generators to migrate from the
    # v3 step object. Production prompts and schema validation request v4.
    if "mode" not in payload and {"title", "instruction", "question", "options", "source_ids"} <= payload.keys():
        return _legacy_step_turn(_diagnostic_step(payload))

    raw_request = payload.get("observation_request")
    request: ObservationRequest | None = None
    if raw_request is not None:
        if not isinstance(raw_request, dict):
            raise TypeError("observation_request must be an object or null")
        raw_options = raw_request.get("options", [])
        if not isinstance(raw_options, list):
            raise TypeError("observation_request options must be an array")
        options: list[DiagnosticOption] = []
        for index, option in enumerate(raw_options, start=1):
            if not isinstance(option, dict):
                raise TypeError("each observation option must be an object")
            label = str(option.get("label", "")).strip()
            option_id = str(option.get("id") or _option_id(label, index)).strip()
            value = str(option.get("value") or label).strip()
            if not label or not option_id or not value:
                raise ValueError("each observation option needs id, label, and value")
            options.append(DiagnosticOption(id=option_id, label=label, value=value))
        request = ObservationRequest(
            request_id=str(raw_request["request_id"]).strip(),
            fact_key=str(raw_request["fact_key"]).strip(),
            question=str(raw_request["question"]).strip(),
            options=options,
            recheck_after_action=bool(raw_request.get("recheck_after_action", False)),
        )

    raw_action = payload.get("next_action")
    action = None
    if raw_action is not None:
        if not isinstance(raw_action, dict):
            raise TypeError("next_action must be an object or null")
        why = raw_action.get("why")
        action = DiagnosticAction(
            instruction=str(raw_action["instruction"]).strip(),
            why=str(why).strip() if why is not None and str(why).strip() else None,
        )

    raw_basis = payload.get("decision_basis")
    basis = None
    if raw_basis is not None:
        if not isinstance(raw_basis, dict):
            raise TypeError("decision_basis must be an object or null")
        raw_causes = raw_basis.get("discriminates_between", [])
        if not isinstance(raw_causes, list):
            raise TypeError("decision_basis discriminates_between must be an array")
        basis = DecisionBasis(
            why_not_solved=str(raw_basis["why_not_solved"]).strip(),
            discriminates_between=[str(value).strip() for value in raw_causes if str(value).strip()],
            expected_discrimination=str(raw_basis["expected_discrimination"]).strip(),
        )

    raw_facts = payload.get("facts_learned", [])
    if not isinstance(raw_facts, list):
        raise TypeError("facts_learned must be an array")
    facts = [DiagnosticFact.model_validate(fact) for fact in raw_facts]
    source_ids = payload.get("source_ids")
    if not isinstance(source_ids, list):
        raise TypeError("source_ids must be an array")
    mode = str(payload["mode"]).strip()
    if mode not in {"solve", "advance", "clarify", "abstain"}:
        raise ValueError("mode must be solve, advance, clarify, or abstain")
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    turn_id = f"turn-{hashlib.sha256(stable.encode()).hexdigest()[:16]}"
    return DiagnosticTurn(
        turn_id=turn_id,
        mode=cast(Literal["solve", "advance", "clarify", "abstain"], mode),
        response=str(payload["response"]).strip(),
        interpretation=(str(payload["interpretation"]).strip() if payload.get("interpretation") else None),
        next_action=action,
        observation_request=request,
        decision_basis=basis,
        facts_learned=facts,
        candidate_causes=[str(value).strip() for value in payload.get("candidate_causes", []) if str(value).strip()],
        ruled_out_causes=[str(value).strip() for value in payload.get("ruled_out_causes", []) if str(value).strip()],
        source_ids=[_canonical_source_id(value) for value in source_ids],
    )


def _legacy_step_turn(step: DiagnosticStep) -> DiagnosticTurn:
    return DiagnosticTurn(
        turn_id=f"legacy-{step.step_id}",
        mode="advance",
        response=step.instruction,
        next_action=DiagnosticAction(
            instruction=step.instruction,
            why="This legacy check needs the requested observation before the documented path can continue.",
        ),
        observation_request=ObservationRequest(
            request_id=step.step_id,
            fact_key=f"observation_{step.step_id.removeprefix('step-').replace('-', '_')[:48]}",
            question=step.question,
            options=step.options,
        ),
        decision_basis=DecisionBasis(
            why_not_solved="The legacy response did not provide a supported resolution.",
            discriminates_between=["reported symptom", "manual-supported next check"],
            expected_discrimination="The requested observation determines whether the next documented check applies.",
        ),
        source_ids=step.source_ids,
    )


def _turn_step(turn: DiagnosticTurn) -> DiagnosticStep:
    """Provide the old action surface only when the turn requests a result."""

    if turn.next_action is None or turn.observation_request is None:
        raise InvalidAnswerError("diagnostic turn has no action/request step")
    request = turn.observation_request
    return DiagnosticStep(
        step_id=request.request_id,
        title="Next check",
        instruction=turn.next_action.instruction,
        question=request.question,
        options=request.options,
        source_ids=turn.source_ids,
    )


def _option_id(label: str, index: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
    return (normalized[:56] or "option") + f"-{index}"


def _canonical_source_id(value: object) -> str:
    source_id = str(value).strip()
    return source_id.removeprefix("[source:").removeprefix("source:").removesuffix("]").strip()


def _validate_step(step: DiagnosticStep, evidence: Sequence[EvidenceContext]) -> None:
    known_ids = {item.chunk_id for item in evidence}
    if not step.source_ids:
        raise InvalidAnswerError("diagnostic step omitted source evidence")
    if any(source_id not in known_ids for source_id in step.source_ids):
        raise InvalidAnswerError("diagnostic step cited evidence outside the retrieved context")
    if len({option.id for option in step.options}) != len(step.options):
        raise InvalidAnswerError("diagnostic step contains duplicate options")


_GENERIC_OPTION_VALUES = {
    "yes",
    "no",
    "not sure",
    "unsure",
    "none",
    "other",
    "on",
    "off",
}


def _evidence_text_blob(evidence: Sequence[EvidenceContext]) -> str:
    return "\n".join(f"{item.section}\n{item.content}" for item in evidence).casefold()


def _validate_options_grounded(
    options: Sequence[DiagnosticOption],
    evidence: Sequence[EvidenceContext],
) -> None:
    """Reject fabricated specifics in clickable options without blocking paraphrase.

    Option labels capture user observations ("Valid IP address", "Available"),
    which are often legitimate paraphrases of manual checks rather than quotes,
    so plain qualitative words are governed by prompt discipline, not substring
    matching (measured: word overlap rejects benign options and drives abstains,
    while embedding similarity cannot separate good from bad options either).
    The hard rule targets what must never be invented: specific technical
    content such as codes, numbers, addresses, versions, and menu paths. Any
    such token absent from the retrieved evidence fails the turn.
    """

    if not options:
        return
    blob = _evidence_text_blob(evidence)
    for option in options:
        for text in (option.label, option.value or ""):
            normalized = " ".join(str(text).casefold().split())
            if not normalized or normalized in _GENERIC_OPTION_VALUES:
                continue
            for token in _specific_tokens(normalized):
                if token not in blob:
                    raise InvalidAnswerError(
                        f"diagnostic option {option.label!r} is not grounded in retrieved evidence"
                    )


def _specific_tokens(normalized: str) -> list[str]:
    """Extract tokens that must match the manual exactly to be trustworthy."""

    specific: list[str] = []
    for raw in re.split(r"\s+", normalized):
        token = raw.strip(".,;:!?()[]{}\"'").casefold()
        if len(token) <= 2:
            continue
        if any(character.isdigit() for character in token) or any(
            separator in token for separator in (".", "/", ">", "\\", "_", "-", ":", "@")
        ):
            specific.append(token)
    return specific


def _validate_action_not_repeated(instruction: str, state: DiagnosticSessionState) -> None:
    """Block paraphrased repeats of an already-completed diagnostic action."""

    normalized = " ".join(instruction.casefold().split())
    if not normalized:
        return
    for completed in state.completed_actions:
        done = " ".join(str(completed).casefold().split())
        if not done:
            continue
        if normalized == done or normalized in done or done in normalized:
            raise InvalidAnswerError("diagnostic turn repeated a completed action")
        # Token-overlap guard catches paraphrases such as "open Wi-Fi settings
        # and join the network" after "select the wireless network to connect".
        new_words = {word for word in re.split(r"[^a-z0-9]+", normalized) if len(word) > 3}
        old_words = {word for word in re.split(r"[^a-z0-9]+", done) if len(word) > 3}
        if len(new_words) >= 4 and len(old_words) >= 4:
            overlap = len(new_words & old_words) / min(len(new_words), len(old_words))
            if overlap >= 0.8:
                raise InvalidAnswerError("diagnostic turn repeated a completed action")


def _attempt_repair(turn: DiagnosticTurn, evidence: Sequence[EvidenceContext]) -> DiagnosticTurn | None:
    """Fix mechanical contract violations without another provider call.

    Only mismatches that need no new content are repaired, keyed off the turn
    itself rather than error-message wording: stray fields on solve turns are
    dropped, a clarify carrying an action sheds the action, an advance without
    an action but with a valid observation request demotes to clarify, a stray
    recheck flag is cleared, and hallucinated source IDs are filtered to the
    retrieved set. Fixes compose: every applicable rule runs before the caller
    re-validates once. Returns None when nothing applied, so judgment calls
    (missing reasons, repeated actions, ungrounded specifics, known facts,
    fully hallucinated citations) still retry or abstain instead of inventing.
    """

    repaired = turn
    if repaired.mode == "solve" and (
        repaired.next_action is not None
        or repaired.observation_request is not None
        or repaired.decision_basis is not None
    ):
        repaired = repaired.model_copy(
            update={"next_action": None, "observation_request": None, "decision_basis": None}
        )
    if repaired.mode == "clarify" and repaired.next_action is not None:
        repaired = repaired.model_copy(update={"next_action": None})
    if (
        repaired.mode == "advance"
        and repaired.next_action is None
        and repaired.observation_request is not None
        and repaired.decision_basis is not None
    ):
        repaired = repaired.model_copy(update={"mode": "clarify"})
    request = repaired.observation_request
    if request is not None and request.recheck_after_action and repaired.next_action is None:
        repaired = repaired.model_copy(
            update={"observation_request": request.model_copy(update={"recheck_after_action": False})}
        )
    known_ids = {item.chunk_id for item in evidence}
    if any(source_id not in known_ids for source_id in repaired.source_ids):
        kept = [source_id for source_id in repaired.source_ids if source_id in known_ids]
        if not kept:
            # Nothing citable remains; a sourceless turn can never validate.
            return None
        repaired = repaired.model_copy(update={"source_ids": kept})
    return repaired if repaired != turn else None


def _parse_turn_text(
    answer: str,
    evidence: Sequence[EvidenceContext],
    state: DiagnosticSessionState,
) -> DiagnosticTurn:
    """Validate structured planner text into a turn, repairing shapes locally."""

    if answer.upper() == _UNSUPPORTED:
        raise UnsupportedAnswerError("the model could not answer from the supplied evidence")
    try:
        payload = json.loads(_strip_json_fence(answer))
        if not isinstance(payload, dict):
            raise TypeError("turn response must be a JSON object")
        turn = _diagnostic_turn(payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise InvalidAnswerError("LLM response was not a valid diagnostic turn") from error
    if turn.response.strip().upper() == _UNSUPPORTED:
        # A JSON abstain turn carrying the bare sentinel as prose: treat it as
        # an abstention judgment (never retried) so the service renders the
        # graceful observation message instead of leaking "UNSUPPORTED" to the
        # user and voice. Exact match only; prose merely mentioning the word
        # is unaffected.
        raise UnsupportedAnswerError("the model abstained without a user-facing message")
    try:
        _validate_turn(turn, list(evidence), state)
    except InvalidAnswerError as error:
        logger.warning("planner turn failed validation reason=%s", str(error)[:300])
        repaired = _attempt_repair(turn, evidence)
        if repaired is not None:
            try:
                _validate_turn(repaired, list(evidence), state)
            except InvalidAnswerError:
                repaired = None
        if repaired is None:
            raise
        trace_event(logger, "planner_turn_repaired", reason=str(error)[:200])
        turn = repaired
    return turn


_GATE_REQUIRED_KEYS = (
    "mode",
    "interpretation",
    "next_action",
    "observation_request",
    "decision_basis",
    "facts_learned",
    "candidate_causes",
    "ruled_out_causes",
    "source_ids",
)

_JSON_WS = " \t\r\n"
_JSON_HEXDIGITS = frozenset("0123456789abcdefABCDEF")


def _read_json_string(raw: str, pos: int) -> tuple[str, int, bool]:
    """Read a JSON string starting at the opening quote at ``pos``.

    Returns ``(decoded, end, closed)`` where ``end`` is exclusive. Unterminated
    input yields ``closed=False`` with the decoded prefix; an invalid escape
    also terminates with ``closed=False`` so the final strict parse reports it.
    Lone high surrogates without a following low surrogate are passed through
    exactly as the stdlib decoder would for the completed string.
    """

    assert raw[pos] == '"'
    out: list[str] = []
    i = pos + 1
    n = len(raw)
    simple = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    while i < n:
        ch = raw[i]
        if ch == '"':
            return "".join(out), i + 1, True
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            return "".join(out), n, False
        esc = raw[i + 1]
        if esc in simple:
            out.append(simple[esc])
            i += 2
            continue
        if esc != "u":
            return "".join(out), n, False
        hexpart = raw[i + 2 : i + 6]
        if len(hexpart) < 4 or any(c not in _JSON_HEXDIGITS for c in hexpart):
            return "".join(out), n, False
        code = int(hexpart, 16)
        if 0xD800 <= code <= 0xDBFF:
            if raw[i + 6 : i + 8] == "\\u":
                lowpart = raw[i + 8 : i + 12]
                if len(lowpart) == 4 and all(c in _JSON_HEXDIGITS for c in lowpart):
                    low = int(lowpart, 16)
                    if 0xDC00 <= low <= 0xDFFF:
                        out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                        i += 12
                        continue
                if len(lowpart) < 4:
                    return "".join(out), n, False
            out.append(chr(code))
            i += 6
            continue
        out.append(chr(code))
        i += 6
    return "".join(out), n, False


def _scan_json_value_end(raw: str, pos: int) -> int | None:
    """Exclusive end index of the JSON value starting at ``pos``, if complete."""

    n = len(raw)
    while pos < n and raw[pos] in _JSON_WS:
        pos += 1
    if pos >= n:
        return None
    ch = raw[pos]
    if ch == '"':
        _, end, closed = _read_json_string(raw, pos)
        return end if closed else None
    if ch in "{[":
        depth = 0
        i = pos
        while i < n:
            c = raw[i]
            if c == '"':
                _, i, _ = _read_json_string(raw, i)
                continue
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return None
    i = pos
    while i < n and raw[i] not in ",}]" and raw[i] not in _JSON_WS:
        i += 1
    return i if i < n else None


def _scan_top_level_fields(raw: str) -> dict[str, tuple[int, int | None]]:
    """Map top-level object keys to ``(value_start, value_end_or_None)`` spans.

    Never raises: truncated or malformed input simply yields the complete
    prefix, so the gate can wait for more stream chunks.
    """

    fields: dict[str, tuple[int, int | None]] = {}
    n = len(raw)
    i = 0
    while i < n and raw[i] in _JSON_WS:
        i += 1
    if i >= n or raw[i] != "{":
        return fields
    i += 1
    while True:
        while i < n and raw[i] in _JSON_WS:
            i += 1
        if i >= n or raw[i] == "}":
            return fields
        if raw[i] == ",":
            i += 1
            continue
        if raw[i] != '"':
            return fields
        key, after, closed = _read_json_string(raw, i)
        if not closed:
            return fields
        i = after
        while i < n and raw[i] in _JSON_WS:
            i += 1
        if i >= n or raw[i] != ":":
            return fields
        i += 1
        while i < n and raw[i] in _JSON_WS:
            i += 1
        vstart = i
        vend = _scan_json_value_end(raw, i)
        fields[key] = (vstart, vend)
        if vend is None:
            return fields
        i = vend
    return fields


def _decode_partial_json_string(raw: str, start: int) -> str | None:
    """Decode the longest complete prefix of the JSON string at ``start``."""

    n = len(raw)
    i = start
    while i < n and raw[i] in _JSON_WS:
        i += 1
    if i >= n or raw[i] != '"':
        return None
    decoded, _, closed = _read_json_string(raw, i)
    if not closed and decoded and 0xD800 <= ord(decoded[-1]) <= 0xDBFF:
        # A complete high surrogate at the buffer edge may still be the first
        # half of a pair; hold it back rather than emitting a lone surrogate
        # that strict JSON consumers (and the final parse) would choke on.
        decoded = decoded[:-1]
    return decoded


class _ToolCallAccumulator:
    """Reassemble streamed ``delta.tool_calls`` fragments into whole calls."""

    def __init__(self) -> None:
        self._parts: dict[int, dict[str, str]] = {}

    def add(self, fragments: Sequence[tuple[int, str, str, str]]) -> None:
        for index, call_id, name, arguments in fragments:
            part = self._parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if call_id and not part["id"]:
                part["id"] = call_id
            part["name"] += name
            part["arguments"] += arguments

    def calls(self) -> list[tuple[str, dict[str, Any], str]]:
        assembled: list[tuple[str, dict[str, Any], str]] = []
        for index in sorted(self._parts):
            part = self._parts[index]
            name = part["name"].strip()
            if not name:
                continue
            try:
                arguments = json.loads(part["arguments"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            assembled.append((name, arguments, part["id"]))
        return assembled


class _ResponseGate:
    """Forward planner prose only after the rest of the turn validates.

    The provider streams the whole diagnostic turn as one JSON object. This
    gate buffers it, validates every non-``response`` field the moment it is
    complete, and only then releases decoded ``response`` characters to text
    and voice. Nothing unvalidated is ever spoken; if the model emits
    ``response`` first, its characters wait in the buffer.
    """

    def __init__(
        self, evidence: Sequence[EvidenceContext], state: DiagnosticSessionState, *, optimistic: bool = False
    ) -> None:
        self._evidence = list(evidence)
        self._state = state
        self._optimistic = optimistic
        self._raw = ""
        self._gate_open = False
        self._response_start: int | None = None
        self._emitted = ""
        self._parsed: dict[str, Any] = {}
        self._spoke_early_chars = 0

    def feed(self, text: str) -> list[str]:
        """Offer streamed content; returns newly releasable response text."""

        if text:
            self._raw += text
        if not self._gate_open:
            self._maybe_open()
            if not self._gate_open:
                if not self._optimistic:
                    return []
                return self._release_early()
        if self._response_start is None:
            return []
        return self._release_gated()

    def _release_early(self) -> list[str]:
        """Optimistic path: speak response bytes before validation completes.

        Only enabled behind an explicit experiment flag. Divergence from the
        final validated turn is logged at completion for correction-rate
        measurement.
        """

        fields = _scan_top_level_fields(self._raw)
        if "response" not in fields:
            return []
        decoded = _decode_partial_json_string(self._raw, fields["response"][0])
        if not decoded:
            return []
        new = decoded[len(self._emitted) :] if decoded.startswith(self._emitted) else decoded
        self._emitted = decoded
        self._spoke_early_chars += len(new)
        return [new] if new else []

    def _release_gated(self) -> list[str]:
        assert self._response_start is not None
        try:
            decoded = _decode_partial_json_string(self._raw, self._response_start)
        except Exception:
            return []
        if not decoded:
            return []
        # Resync defensively; monotonic growth makes the else unreachable.
        new = decoded[len(self._emitted) :] if decoded.startswith(self._emitted) else decoded
        self._emitted = decoded
        return [new] if new else []

    def complete_turn(self) -> DiagnosticTurn:
        """Parse and validate the finished stream buffer."""

        raw = self._raw.strip()
        if not raw:
            raise InvalidAnswerError("LLM response was empty")
        try:
            turn = _parse_turn_text(raw, self._evidence, self._state)
        except (InvalidAnswerError, UnsupportedAnswerError):
            if self._spoke_early_chars:
                # Optimistic text already reached the client, but the turn did
                # not validate: voice may have spoken it. Loud by design so
                # abandonment-rate experiments cannot silently pass.
                trace_event(
                    logger,
                    "optimistic_speech_abandoned",
                    early_chars=self._spoke_early_chars,
                    raw_chars=len(raw),
                )
            raise
        if self._spoke_early_chars and not turn.response.startswith(self._emitted):
            # Optimistic speech diverged from the validated turn: the client
            # already rendered text the planner did not confirm. Loud by design
            # so correction-rate experiments cannot silently pass.
            trace_event(
                logger,
                "optimistic_speech_diverged",
                early_chars=self._spoke_early_chars,
                final_chars=len(turn.response),
            )
        self._emitted = turn.response
        return turn

    def _maybe_open(self) -> None:
        try:
            fields = _scan_top_level_fields(self._raw)
        except Exception:
            return
        complete = {key: span for key, span in fields.items() if span[1] is not None}
        if any(key not in complete for key in _GATE_REQUIRED_KEYS) or "response" not in fields:
            return
        try:
            parsed: dict[str, Any] = {}
            for key in _GATE_REQUIRED_KEYS:
                start, end = complete[key]
                assert end is not None
                parsed[key] = json.loads(self._raw[start:end])
            decoded = _decode_partial_json_string(self._raw, fields["response"][0])
        except Exception:
            return
        if not decoded:
            return
        payload = {**parsed, "response": decoded}
        try:
            turn = _diagnostic_turn(payload)
            _validate_turn(turn, self._evidence, self._state)
        except (KeyError, TypeError, ValueError, InvalidAnswerError):
            return
        self._parsed = parsed
        self._gate_open = True
        self._response_start = fields["response"][0]


def _validate_turn(
    turn: DiagnosticTurn,
    evidence: Sequence[EvidenceContext],
    state: DiagnosticSessionState,
) -> None:
    """Validate grounding and stop semantic repetition without a fixed flow."""

    known_ids = {item.chunk_id for item in evidence}
    if not turn.source_ids:
        raise InvalidAnswerError("diagnostic turn omitted source evidence")
    if any(source_id not in known_ids for source_id in turn.source_ids):
        raise InvalidAnswerError("diagnostic turn cited evidence outside the retrieved context")
    if len({fact.key for fact in turn.facts_learned}) != len(turn.facts_learned):
        raise InvalidAnswerError("diagnostic turn contains duplicate fact updates")
    request = turn.observation_request
    if request is not None:
        if len({option.id for option in request.options}) != len(request.options):
            raise InvalidAnswerError("diagnostic turn contains duplicate options")
        if request.fact_key in state.facts and not request.recheck_after_action:
            raise InvalidAnswerError("diagnostic turn asked for a fact already known in this session")
        if request.recheck_after_action and turn.next_action is None:
            raise InvalidAnswerError("a fact recheck must follow an action that could change it")
        if len(request.options) > 4:
            raise InvalidAnswerError("diagnostic turn offered too many options; keep at most 4")
        _validate_options_grounded(request.options, evidence)
    if turn.next_action is not None:
        _validate_action_not_repeated(turn.next_action.instruction, state)
    if turn.mode == "advance":
        if turn.next_action is None:
            raise InvalidAnswerError("advance turn omitted its concrete diagnostic action")
        if not turn.next_action.why:
            raise InvalidAnswerError("advance turn omitted its diagnosis-specific reason")
        if turn.decision_basis is None or len(turn.decision_basis.discriminates_between) < 2:
            raise InvalidAnswerError("advance turn omitted what the action distinguishes")
    if turn.mode == "clarify":
        if request is None:
            raise InvalidAnswerError("clarification turn omitted its requested observation")
        if turn.next_action is not None:
            raise InvalidAnswerError("clarification turn must not contain a diagnostic action")
        if turn.decision_basis is None:
            raise InvalidAnswerError("clarification turn omitted why the observation is essential")
    if turn.mode == "solve":
        if request is not None or turn.next_action is not None:
            raise InvalidAnswerError("solution turn must not continue a questionnaire")
        if turn.decision_basis is not None:
            raise InvalidAnswerError("solution turn must not justify more diagnosis")
    if turn.mode == "abstain" and (turn.next_action is not None or request is not None):
        raise InvalidAnswerError("abstention turn must not contain a diagnostic action")
    if turn.mode == "abstain" and turn.decision_basis is not None:
        raise InvalidAnswerError("abstention turn must not justify more diagnosis")


def _expand_step_citations(step: DiagnosticStep, evidence: Sequence[EvidenceContext]) -> DiagnosticStep:
    """Keep instructions readable; the UI renders verified sources in its evidence footer.

    ``source_ids`` remains mandatory and is validated before this point.  Keeping
    citations structured avoids duplicating a long manual reference in spoken
    and displayed instruction text.
    """

    del evidence
    return step


def _response_format(mode: str, *, response_first: bool = False) -> dict[str, object]:
    """Use strict JSON Schema when the selected provider supports it.

    JSON object mode remains available for OpenAI-compatible providers that do
    not offer schema-constrained structured output.
    """

    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "diagnostic_turn",
                "description": "One evidence-backed troubleshooting decision.",
                "strict": True,
                "schema": _diagnostic_turn_schema(response_first=response_first),
            },
        }
    return {"type": mode}


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


def _prompt_cache_mode(settings: LiteLLMSettings) -> Literal["openai", "deepseek", "content"] | None:
    """Select only cache controls documented for the configured provider."""

    if not settings.prompt_cache_enabled:
        return None
    model = settings.model.casefold()
    api_base = (settings.api_base or "").casefold()
    if "sarvam.ai" in api_base:
        return None
    if model.startswith("openai/"):
        return "openai"
    if model.startswith("deepseek/"):
        return "deepseek"
    if model.startswith(("anthropic/", "bedrock/", "gemini/", "vertex_ai/", "vertex_ai_beta/")):
        return "content"
    return None


def _mark_cacheable_prefix(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the stable system prompt for Anthropic/Gemini-style caching."""

    copied = [dict(message) for message in messages]
    for message in copied:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        message["content"] = [
            {
                "type": "text",
                "text": message["content"],
                "cache_control": {"type": "ephemeral"},
            }
        ]
        break
    return copied


def _cached_input_tokens(response: Any) -> int | None:
    """Read provider cache usage without depending on a response class."""

    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None
    details = (
        usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
    )
    cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    if cached is not None:
        return int(cached)
    direct = (
        usage.get("cache_read_input_tokens")
        if isinstance(usage, dict)
        else getattr(usage, "cache_read_input_tokens", None)
    )
    return int(direct) if direct is not None else None


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a LiteLLM object or dictionary without coupling to its response class."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _tool_calls(response: Any) -> list[Any]:
    choices = _field(response, "choices", []) or []
    if not choices:
        return []
    message = _field(choices[0], "message")
    calls = _field(message, "tool_calls", []) or []
    return list(calls)


def _tool_call_names(response: Any) -> list[str]:
    """Best-effort tool names from a completed response, tolerating fakes."""

    names: list[str] = []
    try:
        for call in _tool_calls(response):
            name, _, _ = _tool_call_parts(call)
            if name:
                names.append(name)
    except Exception:
        pass
    return names


def _delta_tool_calls(chunk: Any) -> list[tuple[int, str, str, str]]:
    """Extract (index, id, name fragment, arguments fragment) from a stream delta."""

    try:
        choices = _field(chunk, "choices", []) or []
        if not choices:
            return []
        delta = _field(choices[0], "delta")
        calls = _field(delta, "tool_calls", []) or []
    except Exception:
        return []
    fragments: list[tuple[int, str, str, str]] = []
    for call in calls if isinstance(calls, list) else []:
        try:
            index = _field(call, "index", 0)
            function = _field(call, "function")
            fragments.append(
                (
                    int(index) if isinstance(index, int) else 0,
                    str(_field(call, "id", "") or ""),
                    str(_field(function, "name", "") or "") if function is not None else "",
                    str(_field(function, "arguments", "") or "") if function is not None else "",
                )
            )
        except Exception:
            continue
    return fragments


def _usage_tokens(response: Any) -> tuple[int | None, int | None]:
    """Read (prompt, completion) token counts without coupling to a response class."""

    try:
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        if usage is None:
            return None, None
        prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else getattr(usage, "prompt_tokens", None)
        completion = (
            usage.get("completion_tokens") if isinstance(usage, dict) else getattr(usage, "completion_tokens", None)
        )
        return (int(prompt) if prompt is not None else None, int(completion) if completion is not None else None)
    except Exception:
        return None, None


def _tool_call_parts(call: Any) -> tuple[str, dict[str, Any], str]:
    function = _field(call, "function")
    name = str(_field(function, "name", "") or "")
    raw_arguments = _field(function, "arguments", "{}") or "{}"
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        arguments = {}
    call_id = str(_field(call, "id", "") or "")
    return name, arguments, call_id


def _assistant_message(response: Any) -> dict[str, Any]:
    choices = _field(response, "choices", []) or []
    message = _field(choices[0], "message") if choices else None
    calls = _tool_calls(response)
    return {
        "role": "assistant",
        "content": _field(message, "content") if message is not None else None,
        "tool_calls": [
            {
                "id": str(_field(call, "id", "")),
                "type": "function",
                "function": {
                    "name": _tool_call_parts(call)[0],
                    "arguments": _field(_field(call, "function"), "arguments", "{}"),
                },
            }
            for call in calls[:2]
        ],
    }


def _dedupe_evidence(items: Sequence[EvidenceContext]) -> list[EvidenceContext]:
    seen: set[str] = set()
    result: list[EvidenceContext] = []
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        result.append(item)
    return result


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
