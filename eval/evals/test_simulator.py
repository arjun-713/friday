"""ConversationSimulator eval: synthetic users vs the REAL Friday backend.

The simulator plays the persona from each golden in
``datasets/simulator_goldens.json``; ``friday_callback`` answers through the
production pipeline with per-thread session state. Resulting conversations
are scored with turn-level metrics plus the Friday troubleshooting-policy
judge — the strongest test of state management and repeat suppression.

Heavy and variable (simulator LLM + real Luna turns): run explicitly, never
in the PR gate. See the Makefile ``eval-deepeval-sim`` target.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from deepeval import assert_test
from deepeval.dataset import ConversationalGolden, EvaluationDataset
from deepeval.metrics import TurnRelevancyMetric
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import ToolCall, Turn

from .adapter import FridayAdapter, FridaySession
from .config import (
    SIM_PACE_SECONDS,
    SIMULATOR_GOLDENS,
    SIMULATOR_MODEL,
    datasets_dir,
    judge_model,
    require_openai_key,
)
from .friday_metrics import troubleshooting_policy

DATASET = datasets_dir() / "simulator_goldens.json"

_adapter: FridayAdapter | None = None


def get_adapter() -> FridayAdapter:
    global _adapter
    if _adapter is None:
        _adapter = FridayAdapter()
    return _adapter


def make_callback(device: dict):
    """Bind one golden's device scope: the simulator callback never sees the
    golden itself, so each golden simulates with its own session factory."""

    sessions: dict[str, FridaySession] = {}

    def friday_callback(
        input: str, turns: list[Turn] | None = None, thread_id: str = ""
    ) -> Turn:
        _ = turns
        if SIM_PACE_SECONDS > 0:
            time.sleep(SIM_PACE_SECONDS)
        session = sessions.get(thread_id)
        if session is None:
            session = FridaySession(get_adapter(), **device)
            sessions[thread_id] = session
        result = session.ask(input)
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
                "llm_call_count": len(result.llm_calls),
            },
        )

    return friday_callback


def load_goldens() -> list[ConversationalGolden]:
    raw = json.loads(DATASET.read_text(encoding="utf-8"))["goldens"]
    return [
        ConversationalGolden(
            scenario=item["scenario"],
            expected_outcome=item["expected_outcome"],
            user_description=item.get("user_description"),
            context=item.get("context"),
            additional_metadata=item.get("additional_metadata"),
            multimodal=False,
        )
        for item in raw[:SIMULATOR_GOLDENS]
    ]


def simulator_metrics():
    judge = judge_model()
    return [
        TurnRelevancyMetric(model=judge, threshold=0.5),
        troubleshooting_policy(),
    ]


def test_simulator_conversations() -> None:
    require_openai_key()
    if os.getenv("FRIDAY_EVAL_SIM", "0") != "1":
        pytest.skip("simulator tier only (FRIDAY_EVAL_SIM=1)")
    dataset = EvaluationDataset(goldens=load_goldens())
    # Sequential personas: the judge/simulator account is TPM-limited and each
    # turn also costs real Luna calls. Never raise concurrency here to "save
    # time" — it just converts time into 429s.
    goldens = [
        golden for golden in dataset.goldens if isinstance(golden, ConversationalGolden)
    ]
    # One simulation per golden so each persona runs inside its own device
    # scope; the callback API exposes no golden reference.
    test_cases = []
    for golden in goldens:
        device = dict((golden.additional_metadata or {}).get("device", {}))
        simulator = ConversationSimulator(
            model_callback=make_callback(device),  # type: ignore[arg-type] # richer sync callback supported at runtime
            simulator_model=SIMULATOR_MODEL,
            max_concurrent=1,
        )
        test_cases.extend(simulator.simulate([golden], max_user_simulations=6))
    assert test_cases, "simulator produced no conversations"
    for test_case in test_cases:
        assert_test(test_case=test_case, metrics=simulator_metrics())
