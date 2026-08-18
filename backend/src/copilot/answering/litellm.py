"""LiteLLM-backed answer generation with deterministic evidence validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..config import config_section, load_runtime_config
from ..prompts import build_messages
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
from .tools import AGENT_TOOLS, AgentToolExecutor


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
_MAX_AGENT_TOOL_ROUNDS = 2


@dataclass(frozen=True)
class AgentRun:
    """The final planner decision and all evidence it used."""

    turn: DiagnosticTurn
    evidence: list[EvidenceContext]

_DIAGNOSTIC_TURN_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["solve", "advance", "clarify", "abstain"]},
        "response": {"type": "string"},
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
                        "properties": {"id": {"type": "string"}, "label": {"type": "string"}, "value": {"type": "string"}},
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
                    "key": {"type": "string"}, "value": {"type": "string"}, "label": {"type": "string"}, "raw": {"type": "string"},
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
    },
    "required": [
        "mode", "response", "interpretation", "next_action", "observation_request", "decision_basis", "facts_learned",
        "candidate_causes", "ruled_out_causes", "source_ids",
    ],
}


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
        """Let one LLM choose local RAG tools before producing its final turn."""

        messages: list[dict[str, Any]] = build_messages(query, evidence, state)
        active_evidence = list(evidence)
        for _ in range(_MAX_AGENT_TOOL_ROUNDS + 1):
            response = await self._complete_messages(messages, structured=True, tools=AGENT_TOOLS)
            calls = _tool_calls(response)
            if calls:
                messages.append(_assistant_tool_message(response, calls))
                for call in calls:
                    result = await execute_tool(call["name"], call["arguments"])
                    active_evidence = _merge_evidence(active_evidence, result.evidence)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result.content,
                        }
                    )
                continue
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
            _validate_turn(turn, active_evidence, state)
            return AgentRun(turn=turn, evidence=active_evidence)
        raise InvalidAnswerError("diagnostic agent exceeded its local tool-call budget")

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
            "messages": list(messages),
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
            if not api_key:
                raise AnswerProviderUnavailable("configured LLM provider has no API key")
            request["api_key"] = api_key
            request["extra_headers"] = {"api-subscription-key": api_key}
        if structured and self.settings.response_format:
            request["response_format"] = _response_format(self.settings.response_format)
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"
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


def _tool_calls(response: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI-compatible tool calls and reject malformed requests."""

    try:
        raw_calls = response.choices[0].message.tool_calls
    except (AttributeError, IndexError, KeyError, TypeError):
        return []
    if not raw_calls:
        return []
    calls: list[dict[str, Any]] = []
    for raw in raw_calls:
        try:
            call_id = str(raw.id)
            name = str(raw.function.name)
            arguments = json.loads(str(raw.function.arguments or "{}"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise InvalidAnswerError("agent requested an invalid tool call") from error
        if name not in {"search_manual", "find_error_code", "open_manual_page", "get_diagnostic_state"}:
            raise InvalidAnswerError("agent requested an unapproved tool")
        if not isinstance(arguments, dict):
            raise InvalidAnswerError("tool arguments must be a JSON object")
        calls.append({"id": call_id, "name": name, "arguments": arguments})
    return calls


def _assistant_tool_message(response: Any, calls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Retain the provider's call IDs while making the next turn portable."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        content = None
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
            }
            for call in calls
        ],
    }


def _merge_evidence(
    current: Sequence[EvidenceContext], additions: Sequence[EvidenceContext]
) -> list[EvidenceContext]:
    seen: set[str] = set()
    merged: list[EvidenceContext] = []
    for item in [*current, *additions]:
        if item.chunk_id not in seen:
            seen.add(item.chunk_id)
            merged.append(item)
    return merged


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
        if request.fact_key in state.facts and request.recheck_after_action and turn.next_action is None:
            raise InvalidAnswerError("a fact recheck must follow an action that could change it")
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


def _response_format(mode: str) -> dict[str, object]:
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
                "schema": _DIAGNOSTIC_TURN_SCHEMA,
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


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
