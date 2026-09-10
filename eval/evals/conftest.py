"""Shared pytest setup for the Friday DeepEval suites.

Makes the repo root (for ``eval.evals``) and ``backend/src`` (for ``friday``)
importable under both ``deepeval test run`` and plain pytest, and exposes the
case-cap helper used to keep iteration loops cheap.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

for entry in (str(REPO_ROOT), str(REPO_ROOT / "backend" / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def case_cap(default: int | None = None) -> int | None:
    """Optional per-suite case limit via FRIDAY_EVAL_MAX_CASES ( CI speed )."""

    raw = os.getenv("FRIDAY_EVAL_MAX_CASES")
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default
