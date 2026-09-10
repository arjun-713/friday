"""Agent/tool/trajectory eval: the planner path scored from real traces.

Friday has a structured planner with read-only manual tools. These cases
probe tool routing deliberately: explicit manual-search requests must route
to ``search_manual``; direct factual questions answerable from the first
retrieval must NOT burn tool calls. ``tools_called`` comes from the trace
capture, not from the model, so routing failures are measured, not guessed.

Deterministic trajectory guards (no duplicate identical tool calls, bounded
LLM calls per turn) run alongside the judges — loops and wasted steps fail
fast without spending judge tokens.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall

from .adapter import FridayAdapter
from .config import judge_model, require_openai_key

ADVERTISED_TOOLS = frozenset({"search_manual", "find_error_code", "open_manual_page"})

TOOL_CASES = [
    {
        "case_id": "tool-routed-search",
        "input": "The Internet/WAN light is off. Use the manual-search capability to look up the Archer C6 guidance for that exact state before suggesting a fix.",
        "manufacturer": "TP-Link",
        "model": "Archer C6",
        "expected_tools": ["search_manual"],
    },
    {
        "case_id": "tool-not-needed",
        "input": "What does the Toner LED indicate on a Brother HL-L2350DW?",
        "manufacturer": "Brother",
        "model": "HL-L2350DW",
        "expected_tools": [],
    },
    {
        "case_id": "tool-routed-wan-setup",
        "input": "The Archer C6 admin page says the WAN connection is disconnected. Use the manual-search capability to find the relevant WAN setup or status page.",
        "manufacturer": "TP-Link",
        "model": "Archer C6",
        "expected_tools": ["search_manual"],
    },
]

_adapter: FridayAdapter | None = None


def get_adapter() -> FridayAdapter:
    global _adapter
    if _adapter is None:
        _adapter = FridayAdapter()
    return _adapter


def agent_metrics():
    judge = judge_model()
    # TaskCompletion needs Friday's task definition: the product contract is
    # ONE safe evidence-backed step per turn, not full resolution in one shot.
    # Without this, the judge demands a comprehensive guide and fails correct
    # incremental troubleshooting (verified: 0.4 on a correct one-step answer).
    incremental_task = (
        "Provide ONE safe evidence-backed next diagnostic step for the reported "
        "symptom, give a direct evidence-backed answer for factual questions, "
        "or abstain honestly when the manuals do not support the next branch. "
        "Do not dump multiple alternative fixes at once."
    )
    return [
        ToolCorrectnessMetric(model=judge, threshold=0.5),
        ArgumentCorrectnessMetric(model=judge, threshold=0.5),
        TaskCompletionMetric(model=judge, threshold=0.5, task=incremental_task),
    ]


@pytest.mark.parametrize("case", TOOL_CASES, ids=lambda case: case["case_id"])
def test_agent_tool_routing(case: dict) -> None:
    require_openai_key()
    result = get_adapter().run_turn_sync(
        case["input"],
        session_id=f"agent-{case['case_id']}",
        manufacturer=case["manufacturer"],
        model=case["model"],
    )
    # Deterministic trajectory guards: loops and waste fail without a judge.
    assert set(result.tools_called) <= ADVERTISED_TOOLS, (
        f"unadvertised tool called: {result.tools_called}"
    )
    from collections import Counter

    counts = Counter(result.tools_called)
    assert all(count <= 2 for count in counts.values()), f"tool loop detected: {counts}"
    assert len(result.llm_calls) <= 4, (
        f"too many LLM calls in one turn: {len(result.llm_calls)}"
    )
    for expected in case["expected_tools"]:
        assert expected in result.tools_called, (
            f"{case['case_id']}: expected tool {expected} not called (got {result.tools_called})"
        )
    if not case["expected_tools"]:
        assert result.tools_called == [], (
            f"{case['case_id']}: needless tool calls {result.tools_called}"
        )

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=result.answer,
        retrieval_context=result.retrieval_context,
        tools_called=[
            ToolCall(name=call["name"], input_parameters=call.get("arguments") or {})
            for call in result.tool_calls
        ],
        expected_tools=[
            ToolCall(name=name, input_parameters={}) for name in case["expected_tools"]
        ],
    )
    assert_test(test_case=test_case, metrics=agent_metrics())
