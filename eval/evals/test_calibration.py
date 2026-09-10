"""Judge calibration: prove the metrics discriminate before trusting them.

Each case in ``datasets/calibration.json`` states the EXPECTED verdict per
metric. This suite fails when a judge misjudges — the fix is to improve the
rubric or replace the metric, never to edit the calibration case or lower a
threshold to hide the failure.
"""

from __future__ import annotations

import json

import pytest
from deepeval.test_case import LLMTestCase

from . import friday_metrics
from .config import datasets_dir, require_openai_key

DATASET = datasets_dir() / "calibration.json"

METRIC_BY_NAME = {
    "Evidence Grounding": friday_metrics.evidence_grounding,
    "One Safe Step": friday_metrics.one_safe_step,
    "Abstention Quality": friday_metrics.abstention_quality,
    "Observation Button Quality": friday_metrics.observation_button_quality,
    "Voice Concision": friday_metrics.voice_concision,
}


def load_cases() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))["cases"]


def params() -> list[tuple[str, str, object]]:
    out = []
    for case in load_cases():
        for metric_name, expected in case["expect"].items():
            out.append((case["id"], metric_name, expected))
    return out


@pytest.mark.parametrize("case_id,metric_name,expected", params())
def test_judge_calibration(case_id: str, metric_name: str, expected: object) -> None:
    require_openai_key()
    case = next(item for item in load_cases() if item["id"] == case_id)
    metric = METRIC_BY_NAME[metric_name]()
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        retrieval_context=case["retrieval_context"],
    )
    metric.measure(test_case)
    verdict = metric.is_successful()
    assert verdict == expected, (
        f"{case_id}/{metric_name}: judge said {'pass' if verdict else 'fail'}, "
        f"expected {'pass' if expected else 'fail'} (score={metric.score}). "
        f"Reason: {metric.reason}"
    )
