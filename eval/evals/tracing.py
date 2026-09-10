"""DeepEval tracing for Friday without touching production code.

Production modules never import deepeval. Instead, :func:`instrument` wraps the
bound methods of a live :class:`TroubleshootingService` with ``@observe`` spans
at eval time only, so production latency and imports are unaffected:

- ``friday_turn`` (agent): one troubleshooting turn, the trace root.
- ``friday_retrieve`` (retriever): hybrid retrieval incl. tool searches.
- ``friday_plan_llm`` (llm): each structured planner/provider call.
- ``friday_tool`` (tool): each agent tool execution with arguments.
- ``friday_validate`` (agent): the evidence-contract validation gate.

Per-turn compact records are also collected in a context-local capture list so
eval adapters can score tool use, retries, and latencies deterministically
without parsing spans.
"""

from __future__ import annotations

from contextvars import ContextVar
from time import perf_counter
from typing import Any

_current_capture: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "friday_eval_capture", default=None
)


def _capture(record: dict[str, Any]) -> None:
    capture = _current_capture.get()
    if capture is not None:
        capture.append(record)


def _truncate(value: object, limit: int = 300) -> object:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    if isinstance(value, dict):
        return {key: _truncate(item, limit) for key, item in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return [_truncate(item, limit) for item in value[:12]]
    return value


def instrument(service: Any) -> Any:
    """Attach @observe spans to a live service instance. Idempotent."""

    if getattr(service, "_deepeval_instrumented", False):
        return service
    from deepeval.tracing import observe, update_current_span

    original_retrieve = service.session_cache.retrieve

    @observe(type="retriever")
    async def friday_retrieve(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        result = await original_retrieve(*args, **kwargs)
        hits = getattr(result, "hits", []) or []
        timings = getattr(result, "timings_ms", {}) or {}
        update_current_span(
            input=str(args[1]) if len(args) > 1 else str(kwargs.get("query", "")),
            output=[getattr(hit, "chunk_id", "?") for hit in hits[:8]],
            metadata={
                "hit_count": len(hits),
                "abstained": bool(getattr(result, "abstained", False)),
                "timings_ms": {
                    k: round(v, 2)
                    for k, v in timings.items()
                    if isinstance(v, (int, float))
                },
            },
        )
        _capture(
            {
                "span": "friday_retrieve",
                "hit_count": len(hits),
                "chunk_ids": [getattr(hit, "chunk_id", "?") for hit in hits[:8]],
                "latency_ms": round((perf_counter() - started) * 1000, 2),
            }
        )
        return result

    service.session_cache.retrieve = friday_retrieve

    generator = service.answer_generator
    original_complete = getattr(generator, "_complete_messages", None)
    if callable(original_complete):

        @observe(type="llm")
        async def friday_plan_llm(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            result = await original_complete(*args, **kwargs)
            update_current_span(
                input={
                    "model": getattr(
                        getattr(generator, "settings", None), "model", "?"
                    ),
                    "stream": kwargs.get("stream", False),
                    "structured": kwargs.get("structured", False),
                    "tools_attached": bool(kwargs.get("tools")),
                    "max_tokens": kwargs.get("max_tokens"),
                },
                output={"stream_result": kwargs.get("stream", False)},
                metadata={"latency_ms": round((perf_counter() - started) * 1000, 2)},
            )
            _capture(
                {
                    "span": "friday_plan_llm",
                    "stream": bool(kwargs.get("stream", False)),
                    "structured": bool(kwargs.get("structured", False)),
                    "tools_attached": bool(kwargs.get("tools")),
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                }
            )
            return result

        # Assigned as a plain instance attribute (no __get__ binding): attribute
        # functions do not receive self, so the wrapper signature matches the
        # already-bound original exactly.
        generator._complete_messages = friday_plan_llm  # type: ignore[method-assign]

    original_executor_factory = service._tool_executor

    def friday_tool_factory(*args: Any, **kwargs: Any) -> Any:
        execute = original_executor_factory(*args, **kwargs)

        @observe(type="tool")
        async def friday_tool(name: str, arguments: dict[str, Any]) -> Any:
            started = perf_counter()
            result = await execute(name, arguments)
            evidence = getattr(result, "evidence", []) or []
            update_current_span(
                input={"tool": name, "arguments": _truncate(arguments)},
                output={"evidence_count": len(evidence)},
                metadata={"latency_ms": round((perf_counter() - started) * 1000, 2)},
            )
            _capture(
                {
                    "span": "friday_tool",
                    "tool": name,
                    "arguments": _truncate(arguments),
                    "evidence_count": len(evidence),
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                }
            )
            return result

        return friday_tool

    service._tool_executor = friday_tool_factory

    try:
        import friday.answering.litellm as litellm_module

        original_validate = litellm_module._validate_turn

        @observe(type="agent")
        def friday_validate(*args: Any, **kwargs: Any) -> Any:
            turn = args[0] if args else kwargs.get("turn")
            try:
                outcome: Any = original_validate(*args, **kwargs)
                update_current_span(
                    input={"mode": getattr(turn, "mode", "?")},
                    output={"valid": True},
                )
                _capture(
                    {
                        "span": "friday_validate",
                        "mode": getattr(turn, "mode", "?"),
                        "valid": True,
                    }
                )
                return outcome
            except Exception as error:
                update_current_span(
                    input={"mode": getattr(turn, "mode", "?")},
                    output={"valid": False, "reason": str(error)[:300]},
                )
                _capture(
                    {
                        "span": "friday_validate",
                        "mode": getattr(turn, "mode", "?"),
                        "valid": False,
                        "reason": str(error)[:300],
                    }
                )
                raise

        litellm_module._validate_turn = friday_validate
    except ImportError:
        pass

    service._deepeval_instrumented = True
    return service


async def run_traced_turn(
    service: Any, request: Any
) -> tuple[Any, list[dict[str, Any]]]:
    """Run one turn under a ``friday_turn`` root trace; return (response, capture)."""

    from deepeval.tracing import observe, update_current_trace

    capture: list[dict[str, Any]] = []
    token = _current_capture.set(capture)

    @observe(type="agent")
    async def friday_turn() -> Any:
        response = await service.answer(request)
        turn = response.turn
        update_current_trace(
            input=request.query,
            output=response.answer or "",
            tags=["friday", "troubleshooting"],
            metadata={
                "session_id": request.session_id,
                "manufacturer": request.manufacturer,
                "model": request.model,
                "status": response.status,
                "diagnostic_progress": response.diagnostic_progress,
                "abstained": response.status == "abstained",
                "planner_mode": turn.mode if turn is not None else None,
                "citation_count": len(response.citations or []),
                "evidence_count": len(response.evidence or []),
            },
        )
        return response

    try:
        return await friday_turn(), capture
    finally:
        _current_capture.reset(token)
