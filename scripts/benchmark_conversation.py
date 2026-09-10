"""Run a ten-turn grounded troubleshooting conversation and save timings.

This exercises the same SSE endpoint used by the frontend. It deliberately
keeps one session so retrieval caching, session state, and prompt growth are
measured across a realistic conversation rather than isolated requests.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from uuid import uuid4

import httpx
from friday.paths import index_dir

QUESTIONS = [
    "My TP-Link Archer C6 is powered on, my phone connects to Wi-Fi, but there is no internet.",
    "The 2.4 GHz and 5 GHz lights are solid. What does that rule out, and what should I check next?",
    "The Internet/WAN light is off. Use the manual-search capability to look up the Archer C6 guidance for that exact state before suggesting a fix.",
    "The Ethernet cable is firmly connected between the modem and the blue WAN port. Is there a documented modem-to-router restart order?",
    "I followed that restart order: the modem Internet light is now solid, and I waited before powering the router back on.",
    "The router is back on, but its Internet/WAN light is still off. What does this change tell us?",
    "A laptop gets internet when connected directly to the modem, so the modem and service appear to work without the router.",
    "The Archer C6 admin page says the WAN connection is disconnected. Use the manual-search capability to find the relevant WAN setup or status page.",
    "The WAN type is Dynamic IP, and renewing the connection did not help. What safe router-side check is supported next?",
    "I have not changed settings or reset the router. Based on everything confirmed so far, what should I try next and when should I contact my ISP?",
]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "min_ms": round(min(values), 2),
        "p50_ms": round(_percentile(values, 50), 2),
        "p95_ms": round(_percentile(values, 95), 2),
        "p99_ms": round(_percentile(values, 99), 2),
        "max_ms": round(max(values), 2),
        "mean_ms": round(statistics.mean(values), 2),
    }


try:
    from friday.voice.bridge import _is_actionable_sentence as _bridge_is_actionable
    from friday.voice.bridge import _take_tts_sentences as _bridge_sentences
except Exception:  # noqa: BLE001 - Benchmark must run even if voice deps are unavailable.
    _bridge_is_actionable = None
    _bridge_sentences = None


def _sentence_milestones(
    token_events: list[tuple[float, str]],
) -> dict[str, float | None]:
    """Earliest client-observed moment each speakable milestone completes.

    Uses the voice bridge splitters when importable so the benchmark measures
    the same sentence boundaries the TTS path would speak; otherwise falls
    back to complete-sentence detection on the token stream.
    """

    import re as _re

    first_sentence_ms: float | None = None
    first_actionable_ms: float | None = None
    acc = ""
    seen = 0
    for received_ms, piece in token_events:
        acc += piece
        if _bridge_sentences is not None:
            sentences, _ = _bridge_sentences(acc)
        else:
            sentences = [
                candidate.strip()
                for candidate in _re.split(r"(?<=[.!?])\s+", acc)
                if candidate.strip() and _re.search(r"[.!?]$", candidate.strip())
            ]
        for sentence in sentences[seen:]:
            if first_sentence_ms is None:
                first_sentence_ms = received_ms
            if _bridge_is_actionable is not None:
                actionable = _bridge_is_actionable(sentence)
            else:
                actionable = sentence.strip().endswith("?")
            if first_actionable_ms is None and actionable:
                first_actionable_ms = received_ms
            if first_sentence_ms is not None and first_actionable_ms is not None:
                break
        seen = len(sentences)
        if first_sentence_ms is not None and first_actionable_ms is not None:
            break
    return {
        "first_sentence_ms": first_sentence_ms,
        "first_actionable_sentence_ms": first_actionable_ms,
    }


def _compact_turn(
    number: int,
    question: str,
    answer: str,
    complete: dict[str, Any],
    retrieval: dict[str, Any],
    timings: dict[str, Any],
    llm_calls: list[dict[str, Any]],
    milestones: dict[str, float | None],
) -> dict[str, Any]:
    progress = complete.get("diagnostic_progress", "RESPONSE_COMPLETED")
    retrieval_timings = retrieval.get("timings_ms", {})
    top_evidence = retrieval.get("diagnostics", {}).get("top_evidence", [])
    retrieved_ids = {item.get("chunk_id") for item in top_evidence}
    citations = complete.get("citations", [])
    tool_events = [
        item for item in timings.pop("_events", []) if item.get("type") == "tool"
    ]
    citation_ids = {item.get("chunk_id") for item in citations}
    grounding = {
        "citations_present": bool(citations),
        "citation_count": len(citations),
        "citations_match_top_evidence": bool(citation_ids)
        and citation_ids.issubset(retrieved_ids),
        "semantic_claim_support_checked": False,
    }
    compact_timing = {
        "response_ms": timings.get("complete_ms"),
        "first_token_ms": timings.get("first_token_ms"),
        "first_sentence_ms": milestones.get("first_sentence_ms"),
        "first_actionable_sentence_ms": milestones.get("first_actionable_sentence_ms"),
        "backend_first_token_ms": timings.get("backend_first_token_ms"),
        "backend_complete_ms": timings.get("backend_complete_ms"),
        "retrieval_total_ms": retrieval_timings.get("total_ms"),
        "embedding_ms": retrieval_timings.get("embedding_ms"),
        "parallel_embed_lexical_ms": retrieval_timings.get("parallel_embed_lexical_ms"),
        "lexical_search_ms": retrieval_timings.get("lexical_ms"),
        "dense_search_ms": retrieval_timings.get("dense_search_ms"),
        "fusion_ms": retrieval_timings.get("fusion_ms"),
        "parent_fetch_ms": retrieval_timings.get("parent_fetch_ms"),
    }
    return {
        "turn": number,
        "question": question,
        "response": answer,
        "transport_success": True,
        "provider_success": bool(answer),
        "response_completed": bool(complete),
        "timing": {
            key: value for key, value in compact_timing.items() if value is not None
        },
        "state_delta": {
            "facts": complete.get("facts", {}),
            "observations": complete.get("observations", []),
            "completed_actions": complete.get("completed_actions", []),
            "next_branch": complete.get("current_next_branch"),
            "user_reports": complete.get("user_reports", []),
        },
        "diagnostic_progress": progress,
        "agent_tools": [
            name for event in tool_events for name in event.get("tools", [])
        ],
        "repeated_actions": complete.get("repeated_actions", []),
        "retrieval": {
            "top_evidence": top_evidence,
        },
        "citations": citations,
        "grounding": grounding,
        "llm": {
            "calls": len(llm_calls),
            "retries": sum(
                1
                for call in llm_calls
                if str(call.get("note") or "").startswith("validation-retry")
            ),
            "tool_continuations": sum(
                1
                for call in llm_calls
                if str(call.get("note") or "").startswith("tool-continuation")
            ),
            "total_latency_ms": round(
                sum(call.get("latency_ms") or 0 for call in llm_calls), 2
            ),
            "min_ttft_ms": min(
                [
                    call["ttft_ms"]
                    for call in llm_calls
                    if call.get("ttft_ms") is not None
                ]
                or [None]
            ),
            "prompt_tokens": sum(call.get("prompt_tokens") or 0 for call in llm_calls),
            "completion_tokens": sum(
                call.get("completion_tokens") or 0 for call in llm_calls
            ),
            "tokens_per_sec": [
                call.get("tokens_per_sec")
                for call in llm_calls
                if call.get("tokens_per_sec") is not None
            ],
            "records": llm_calls,
        },
    }


def _read_sse(
    response: httpx.Response, started: float
) -> tuple[list[dict[str, Any]], dict[str, float], list[tuple[float, str]]]:
    events: list[dict[str, Any]] = []
    token_events: list[tuple[float, str]] = []
    first_token_ms: float | None = None
    backend_first_token_ms: float | None = None
    backend_complete_ms: float | None = None
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        event["received_at"] = datetime.now(UTC).isoformat()
        event["received_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        events.append(event)
        if event.get("type") == "token":
            token_events.append(
                (event["received_elapsed_ms"], str(event.get("text", "")))
            )
            if first_token_ms is None:
                first_token_ms = event["received_elapsed_ms"]
                backend_first_token_ms = event.get("backend_elapsed_ms")
        if event.get("type") == "complete":
            backend_complete_ms = event.get("backend_elapsed_ms")
    complete_ms = round((time.perf_counter() - started) * 1000, 2)
    return (
        events,
        {
            "first_token_ms": first_token_ms or complete_ms,
            "complete_ms": complete_ms,
            "backend_first_token_ms": backend_first_token_ms or complete_ms,
            "backend_complete_ms": backend_complete_ms or complete_ms,
        },
        token_events,
    )


def run(
    base_url: str,
    output: Path,
    timeout_seconds: float,
    *,
    debug_events: bool = False,
    pace_seconds: float = 0,
) -> dict[str, Any]:
    session_id = f"conversation-benchmark-{uuid4().hex}"
    turns: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
        for number, question in enumerate(QUESTIONS, start=1):
            if pace_seconds > 0 and number > 1:
                # Rate-limit pacing for tier-limited providers (e.g. Groq TPM).
                # Sleeps between turns; per-turn timings start at request time
                # so pacing never pollutes the measured latencies.
                time.sleep(pace_seconds)
            payload = {
                "query": question,
                "session_id": session_id,
                "manufacturer": "TP-Link",
                "model": "Archer C6",
            }
            started = time.perf_counter()
            try:
                with client.stream(
                    "POST", "/v1/troubleshoot/stream", json=payload
                ) as response:
                    response.raise_for_status()
                    events, client_timings, token_events = _read_sse(response, started)
                retrieval = next(
                    (
                        item.get("retrieval", {})
                        for item in events
                        if item.get("type") == "retrieval"
                    ),
                    {},
                )
                complete = next(
                    (
                        item.get("response", {})
                        for item in reversed(events)
                        if item.get("type") == "complete"
                    ),
                    {},
                )
                llm_calls = next(
                    (
                        item.get("llm_calls", {})
                        for item in reversed(events)
                        if item.get("type") == "complete"
                    ),
                    {},
                )
                milestones = _sentence_milestones(token_events)
                answer = "".join(
                    str(item.get("text", ""))
                    for item in events
                    if item.get("type") == "token"
                )
                errors = [
                    item.get("message")
                    for item in events
                    if item.get("type") == "error"
                ]
                timings = {
                    **client_timings,
                    "retrieval_total_ms": retrieval.get("timings_ms", {}).get(
                        "total_ms"
                    ),
                    "embedding_ms": retrieval.get("timings_ms", {}).get("embedding_ms"),
                    "parallel_embed_lexical_ms": retrieval.get("timings_ms", {}).get(
                        "parallel_embed_lexical_ms"
                    ),
                    "lexical_search_ms": retrieval.get("timings_ms", {}).get(
                        "lexical_ms"
                    ),
                    "dense_search_ms": retrieval.get("timings_ms", {}).get(
                        "dense_search_ms"
                    ),
                    "fusion_ms": retrieval.get("timings_ms", {}).get("fusion_ms"),
                    "parent_fetch_ms": retrieval.get("timings_ms", {}).get(
                        "parent_fetch_ms"
                    ),
                }
                turn = _compact_turn(
                    number,
                    question,
                    answer,
                    complete,
                    retrieval,
                    {**timings, "_events": events},
                    llm_calls if isinstance(llm_calls, list) else [],
                    milestones,
                )
                if errors:
                    turn["errors"] = errors
                if debug_events:
                    turn["debug_events"] = events
                turns.append(turn)
            except (
                httpx.HTTPError,
                OSError,
                ValueError,
            ) as error:  # Keep one failed turn visible in the report.
                turns.append(
                    {
                        "turn": number,
                        "question": question,
                        "response": "",
                        "timing": {
                            "response_ms": round(
                                (time.perf_counter() - started) * 1000, 2
                            )
                        },
                        "diagnostic_progress": "ERROR",
                        "errors": [f"{type(error).__name__}: {error}"],
                    }
                )

    metric_names = [
        "retrieval_total_ms",
        "embedding_ms",
        "lexical_search_ms",
        "dense_search_ms",
        "fusion_ms",
        "parent_fetch_ms",
        "first_token_ms",
        "first_sentence_ms",
        "first_actionable_sentence_ms",
        "complete_ms",
        "backend_first_token_ms",
        "backend_complete_ms",
    ]
    metrics = {
        name: _summary(
            [
                float(
                    turn["timing"]["response_ms"]
                    if name == "complete_ms"
                    else turn["timing"].get(
                        "lexical_search_ms" if name == "lexical_search_ms" else name
                    )
                )
                for turn in turns
                if (
                    turn["timing"].get("response_ms")
                    if name == "complete_ms"
                    else turn["timing"].get(
                        "lexical_search_ms" if name == "lexical_search_ms" else name
                    )
                )
                is not None
            ]
        )
        for name in metric_names
    }
    llm_call_counts = [
        turn.get("llm", {}).get("calls", 0) for turn in turns if "llm" in turn
    ]
    report = {
        "benchmark": "conversation_rag.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": urljoin(base_url, "/v1/troubleshoot/stream"),
        "model": "openai/gpt-5.6-luna",
        "device": {"manufacturer": "TP-Link", "model": "Archer C6"},
        "turn_count": len(turns),
        "successful_turns": sum(not turn.get("errors") for turn in turns),
        "failed_turns": sum(bool(turn.get("errors")) for turn in turns),
        "metrics_ms": metrics,
        "llm_summary": {
            "mean_calls_per_turn": round(sum(llm_call_counts) / len(llm_call_counts), 2)
            if llm_call_counts
            else 0,
            "max_calls_in_turn": max(llm_call_counts) if llm_call_counts else 0,
            "total_retries": sum(
                turn.get("llm", {}).get("retries", 0) for turn in turns
            ),
            "total_tool_continuations": sum(
                turn.get("llm", {}).get("tool_continuations", 0) for turn in turns
            ),
            "retry_notes": sorted(
                {
                    call.get("note")
                    for turn in turns
                    for call in turn.get("llm", {}).get("records", [])
                    if str(call.get("note") or "").startswith("validation-retry")
                }
            ),
        },
        "turns": turns,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output", type=Path, default=index_dir() / "conversation_rag_benchmark.json"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--pace-seconds", type=float, default=0.0)
    parser.add_argument(
        "--debug-events",
        action="store_true",
        help="Include raw SSE events in each turn.",
    )
    args = parser.parse_args()
    report = run(
        args.base_url,
        args.output,
        args.timeout,
        debug_events=args.debug_events,
        pace_seconds=args.pace_seconds,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "benchmark",
                    "turn_count",
                    "successful_turns",
                    "failed_turns",
                    "metrics_ms",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
