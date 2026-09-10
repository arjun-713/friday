"""Voice tier: speakability of REAL Friday replies plus latency boundaries.

DeepEval 3.9.9 ships no native voice simulation (no VoiceConversationSimulator,
connectors, or voice metrics), so this tier evaluates the voice path as far as
the architecture supports without inventing a networking layer:

- ``voice_concision`` judge over live replies (speakable, action early).
- Friday's own actionability heuristic (``voice.bridge``) applied to live
  replies: a spoken turn must contain an actionable sentence, and it must not
  be buried after a long acknowledgement.
- Generous latency boundaries (retrieval < 5 s, turn < 120 s): regression
  tripwires for pathological slowdowns, not millisecond assertions — provider
  TTFT variance makes exact timing flaky by nature.
- Barge-in classification stays in ``backend/tests/test_voice_bridge.py``
  (deterministic); it is referenced, not duplicated, here.

Separate suite: heavier and more variable than the PR gate.
"""

from __future__ import annotations

from time import perf_counter

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from .adapter import FridayAdapter
from .config import require_openai_key
from .friday_metrics import voice_concision

try:
    from friday.voice.bridge import _is_actionable_sentence, _take_tts_sentences

    _BRIDGE_AVAILABLE = True
except ImportError:  # pragma: no cover - voice deps are optional for evals
    _BRIDGE_AVAILABLE = False

VOICE_TURNS = [
    {
        "query": "My ASUS RT-AX3000 has a red WAN LED. What does that mean?",
        "manufacturer": "ASUS",
        "model": "RT-AX3000",
    },
    {
        "query": "The cable between the modem and the WAN port is firmly connected.",
        "manufacturer": "ASUS",
        "model": "RT-AX3000",
    },
]

_adapter: FridayAdapter | None = None


def get_adapter() -> FridayAdapter:
    global _adapter
    if _adapter is None:
        _adapter = FridayAdapter()
    return _adapter


@pytest.mark.parametrize("turn", VOICE_TURNS, ids=lambda turn: turn["query"][:40])
def test_voice_turn_is_speakable(turn: dict) -> None:
    require_openai_key()
    started = perf_counter()
    result = get_adapter().run_turn_sync(
        turn["query"],
        session_id="voice-tier",
        manufacturer=turn["manufacturer"],
        model=turn["model"],
    )
    elapsed_ms = (perf_counter() - started) * 1000
    assert result.answer, "empty reply on the voice path"
    if result.retrieval_ms is not None:
        assert result.retrieval_ms < 5000, (
            f"retrieval regression: {result.retrieval_ms:.0f} ms"
        )
    assert elapsed_ms < 120_000, f"turn took {elapsed_ms:.0f} ms"
    if _BRIDGE_AVAILABLE:
        sentences, _ = _take_tts_sentences(result.answer, final=True)
        assert sentences, "no complete TTS sentence in the reply"
        first_actionable = next(
            (
                index
                for index, sentence in enumerate(sentences)
                if _is_actionable_sentence(sentence)
            ),
            None,
        )
        assert first_actionable is not None, (
            "spoken reply contains no actionable sentence"
        )
        assert first_actionable <= 1, (
            "actionable sentence buried beyond the second sentence"
        )
    assert_test(
        test_case=LLMTestCase(input=turn["query"], actual_output=result.answer),
        metrics=[voice_concision()],
    )
