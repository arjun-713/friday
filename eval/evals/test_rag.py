"""Single-turn RAG eval: real Friday pipeline, actual retrieved chunks.

Each golden runs one traced turn through the production pipeline
(state-aware retrieval → tool-assisted planner → validation). The deterministic
expected-chunk check (authoritative, from ``eval/retrieval_cases.jsonl``) runs
first; DeepEval LLM metrics then score the SAME actual output and the SAME
actual retrieval context Friday produced — never an idealized context.

Tiers (via FRIDAY_EVAL_MAX_CASES or make targets):
- fast: 8 curated smoke cases (PR gate).
- full: all 35 curated cases.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase

from .adapter import FridayAdapter
from .config import datasets_dir, require_openai_key
from .metrics import (
    abstention_metrics,
    fast_suite_metrics,
    friday_contract_metrics,
    rag_correctness_metrics,
)

DATASET = datasets_dir() / "rag_curated.json"

# PR-gate smoke set: every question type and device family represented.
FAST_CASE_IDS = [
    "router-007",  # error_code, ASUS
    "router-001",  # procedure, ASUS
    "printer-001",  # factual LED, Brother
    "computer-003",  # factual, Dell
    "computer-001",  # procedure, Dell
    "computer-005",  # symptom, Dell
    "unsupported-001",  # unanswerable, wrong-device
    "unanswerable-medical-device",  # unanswerable, out-of-domain
]

_adapter: FridayAdapter | None = None


def get_adapter() -> FridayAdapter:
    global _adapter
    if _adapter is None:
        _adapter = FridayAdapter()
    return _adapter


def load_goldens() -> list[Golden]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(file_path=str(DATASET))
    return [golden for golden in dataset.goldens if isinstance(golden, Golden)]


def select_goldens(all_goldens: list[Golden], fast: bool) -> list[Golden]:
    if fast:
        wanted = set(FAST_CASE_IDS)
        return [
            golden
            for golden in all_goldens
            if (golden.additional_metadata or {}).get("case_id") in wanted
        ]
    return all_goldens


def run_case(golden: Golden, *, fast: bool):
    require_openai_key()
    meta = golden.additional_metadata or {}
    adapter = get_adapter()
    result = adapter.run_turn_sync(
        golden.input,
        session_id=f"rag-{meta.get('case_id', 'unknown')}",
        manufacturer=meta.get("manufacturer"),
        model=meta.get("model"),
    )
    # Authoritative deterministic check on the ACTUAL retrieved chunks, with
    # the SAME semantics as eval/run_retrieval.py: any acceptable-evidence
    # overlap counts (recall-style), not necessarily the single expected chunk.
    if meta.get("should_abstain"):
        assert result.abstained, (
            f"{meta.get('case_id')}: expected abstention, got {result.status}"
        )
    else:
        expected = set(meta.get("expected_chunk_ids", []))
        acceptable = set(meta.get("acceptable_chunk_ids", [])) or expected
        retrieved = set(result.chunk_ids)
        overlap = acceptable & retrieved
        assert overlap, (
            f"{meta.get('case_id')}: no acceptable evidence {sorted(acceptable)} "
            f"in retrieved {sorted(retrieved)} (expected {sorted(expected)})"
        )
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.answer,
        retrieval_context=result.retrieval_context,
    )
    metrics = (
        fast_suite_metrics()
        if fast
        else rag_correctness_metrics() + friday_contract_metrics()
    )
    # Abstained turns are scored by the abstention judges only: One Safe Step
    # systematically refuses to credit abstentions (verified in both criteria
    # and steps forms), so each metric judges what it was designed for.
    if result.abstained:
        metrics = abstention_metrics()
    assert_test(test_case=test_case, metrics=metrics)
    return result


ALL_GOLDENS = load_goldens()
FAST_GOLDENS = select_goldens(ALL_GOLDENS, fast=True)


@pytest.mark.parametrize(
    "golden",
    FAST_GOLDENS,
    ids=lambda golden: (golden.additional_metadata or {}).get("case_id"),
)
def test_rag_fast(golden: Golden) -> None:
    """PR-gate smoke: 8 curated cases, grounding + contract essentials."""

    run_case(golden, fast=True)


FULL_GOLDENS = select_goldens(ALL_GOLDENS, fast=False)


@pytest.mark.parametrize(
    "golden",
    FULL_GOLDENS,
    ids=lambda golden: (golden.additional_metadata or {}).get("case_id"),
)
def test_rag_full(golden: Golden) -> None:
    """Full curated set (35 cases). Skipped in the fast tier via -k filtering."""

    import os

    if os.getenv("FRIDAY_EVAL_TIER", "fast") == "fast":
        pytest.skip("full tier only (FRIDAY_EVAL_TIER=full)")
    run_case(golden, fast=False)


def test_dataset_contract() -> None:
    """The curated dataset keeps the fields the suite depends on."""

    import json

    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    assert len(raw) >= len(FAST_CASE_IDS)
    for entry in raw:
        assert entry["input"], "golden without input"
        meta = entry["additional_metadata"]
        assert meta.get("case_id"), "golden without case_id"
        assert meta.get("provenance", "").startswith("curated:"), "unmarked provenance"
