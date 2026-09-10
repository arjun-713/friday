"""Multi-turn conversation eval: scripted troubleshooting arcs, real backend.

Friday is evaluated as a conversation, not isolated QA turns. Each curated
arc in ``datasets/conversations_curated.json`` runs through a stateful
``FridaySession`` (one session_id, sequential turns, production session
store semantics). Turns carry their ACTUAL per-turn retrieval context and
tool calls, so turn-level metrics score what Friday really retrieved.

- fast tier: short arcs only (red-wan, abstain).
- full tier: all arcs including the 10-turn Archer C6 diagnostic flow.
"""

from __future__ import annotations

import json
import os

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    ConversationCompletenessMetric,
    TurnFaithfulnessMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn

from .adapter import FridayAdapter, FridaySession
from .config import datasets_dir, judge_model, require_openai_key
from .friday_metrics import troubleshooting_policy

DATASET = datasets_dir() / "conversations_curated.json"

_adapter: FridayAdapter | None = None


def get_adapter() -> FridayAdapter:
    global _adapter
    if _adapter is None:
        _adapter = FridayAdapter()
    return _adapter


def load_conversations() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))["conversations"]


def conversation_metrics(*, fast: bool):
    judge = judge_model()
    if fast:
        return [
            TurnRelevancyMetric(model=judge, threshold=0.5),
            troubleshooting_policy(),
        ]
    return [
        TurnRelevancyMetric(model=judge, threshold=0.5),
        TurnFaithfulnessMetric(model=judge, threshold=0.5),
        ConversationCompletenessMetric(model=judge, threshold=0.5),
        troubleshooting_policy(),
    ]


def assistant_turn(result) -> Turn:
    return Turn(
        role="assistant",
        content=result.answer,
        retrieval_context=result.retrieval_context,
        tools_called=[
            ToolCall(name=name, input_parameters={}) for name in result.tools_called
        ],
        metadata={
            "status": result.status,
            "diagnostic_progress": result.progress,
            "chunk_ids": result.chunk_ids,
            "options": result.options,
            "llm_call_count": len(result.llm_calls),
        },
    )


def run_conversation(conversation: dict, *, fast: bool) -> ConversationalTestCase:
    require_openai_key()
    device = conversation.get("device", {}) or {}
    session = FridaySession(get_adapter(), **{k: v for k, v in device.items() if v})
    turns: list[Turn] = []
    for user_text in conversation["user_turns"]:
        turns.append(Turn(role="user", content=user_text))
        turns.append(assistant_turn(session.ask(user_text)))
    return ConversationalTestCase(
        turns=turns,
        scenario=conversation["scenario"],
        expected_outcome=conversation["expected_outcome"],
        user_description=conversation.get("user_description"),
        context=[f"{device.get('manufacturer', '')} {device.get('model', '')}".strip()],
    )


ALL_CONVERSATIONS = load_conversations()


def _tier() -> str:
    return os.getenv("FRIDAY_EVAL_TIER", "fast")


@pytest.mark.parametrize(
    "conversation",
    [c for c in ALL_CONVERSATIONS if c.get("tier", "full") == "fast"],
    ids=lambda c: c["conversation_id"],
)
def test_conversation_fast(conversation: dict) -> None:
    test_case = run_conversation(conversation, fast=True)
    assert_test(test_case=test_case, metrics=conversation_metrics(fast=True))


@pytest.mark.parametrize(
    "conversation", ALL_CONVERSATIONS, ids=lambda c: c["conversation_id"]
)
def test_conversation_full(conversation: dict) -> None:
    if _tier() == "fast":
        pytest.skip("full tier only (FRIDAY_EVAL_TIER=full)")
    test_case = run_conversation(conversation, fast=False)
    assert_test(test_case=test_case, metrics=conversation_metrics(fast=False))
