"""Central DeepEval configuration for the Friday harness.

Friday's runtime model configuration lives in ``backend/config.yml`` and stays
at temperature 0.1. Everything here configures the *judge* side only, which
uses a deterministic temperature-0 setup for reproducibility.
"""

from __future__ import annotations

import os
from pathlib import Path

# Judges are separate from Friday's runtime model on purpose. gpt-4o supports
# the log-probability scoring GEval-family metrics require.
JUDGE_MODEL = os.getenv("FRIDAY_JUDGE_MODEL", "gpt-4o")
JUDGE_TEMPERATURE = 0.0

# Friday's application temperature. Recorded here as a tripwire, never applied:
# eval code must not change the runtime sampling configuration.
APP_TEMPERATURE = 0.1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def eval_dir() -> Path:
    return repo_root() / "eval"


def evals_dir() -> Path:
    return eval_dir() / "evals"


def datasets_dir() -> Path:
    return evals_dir() / "datasets"


# Tier budgets: the fast PR suite stays small and deterministic-first.
FAST_RAG_CASES = 8
FULL_RAG_CASES = 30
# Simulator user-model: NOT the judge. Personas just need plausible users;
# scoring stays on gpt-4o. Mini has a separate, roomier TPM budget, which
# matters because Friday's own Luna turns share the gpt-4o pool.
SIMULATOR_MODEL = os.getenv("FRIDAY_SIMULATOR_MODEL", "gpt-4o-mini")
# Pacing between simulated turns (seconds). The account TPM ceiling covers
# Luna turns + simulator + judges combined; without pacing, bursts 429.
SIM_PACE_SECONDS = float(os.getenv("FRIDAY_SIM_PACE_SECONDS", "15"))
SIMULATOR_GOLDENS = int(os.getenv("FRIDAY_SIM_GOLDENS", "4"))
MAX_CONVERSATION_TURNS = 10


def require_openai_key() -> str:
    """Return the judge API key or skip the test with a clear reason."""

    import pytest

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip(
            "OPENAI_API_KEY is not set (source backend/.env, never commit keys)"
        )
    return key


def judge_model():
    """Shared temperature-0 judge instance for all DeepEval metrics."""

    from deepeval.models import GPTModel

    return GPTModel(model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE)
