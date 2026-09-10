"""Shared DeepEval metric lists for the Friday harness.

Standard metrics live here as factory functions (fresh instances per suite, so
threshold tuning never leaks between suites). Friday product judges live in
``friday_metrics.py``. Eval test files import these lists; they never build
ad hoc metrics inline.
"""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from . import friday_metrics
from .config import judge_model


def rag_correctness_metrics():
    """Single-turn RAG metrics needing only input + actual + retrieved context."""

    judge = judge_model()
    return [
        FaithfulnessMetric(model=judge, threshold=0.5),
        AnswerRelevancyMetric(model=judge, threshold=0.5),
        ContextualRelevancyMetric(model=judge, threshold=0.5),
    ]


def rag_ranking_metrics():
    """Retrieval ranking metrics; require goldens with expected outputs."""

    judge = judge_model()
    return [
        ContextualPrecisionMetric(model=judge, threshold=0.5),
        ContextualRecallMetric(model=judge, threshold=0.5),
    ]


def friday_contract_metrics():
    """Product-contract judges for single troubleshooting turns."""

    return [
        friday_metrics.evidence_grounding(),
        friday_metrics.one_safe_step(),
        friday_metrics.abstention_quality(),
        friday_metrics.observation_button_quality(),
        friday_metrics.voice_concision(),
    ]


def fast_suite_metrics():
    """Minimal PR-gate set: grounding truth + contract essentials."""

    judge = judge_model()
    return [
        FaithfulnessMetric(model=judge, threshold=0.5),
        friday_metrics.evidence_grounding(),
        friday_metrics.one_safe_step(),
    ]


def abstention_metrics():
    """Judges for abstained turns.

    One Safe Step is deliberately excluded: the judge systematically refuses
    to credit abstentions under it (verified in both criteria and steps
    forms), and abstention quality is what Abstention Quality measures.
    Each metric judges what it was designed for.
    """

    judge = judge_model()
    return [
        FaithfulnessMetric(model=judge, threshold=0.5),
        friday_metrics.evidence_grounding(),
        friday_metrics.abstention_quality(),
    ]
