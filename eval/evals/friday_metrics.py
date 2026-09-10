"""Friday-specific judge metrics.

Built-ins cannot express Friday's product contract (one safe evidence-backed
step per turn, honest abstention, no repeated checks, speakable replies), so
these GEval/ConversationalGEval judges score the actual troubleshooting
behavior. Deterministic checks (button grounding, repeat suppression) stay in
``backend/tests`` and ``eval/run_conversation_policy.py``; these judges cover
the semantic behavior deterministic rules cannot reliably capture.

Thresholds start at 0.5 and are revisited in ``test_calibration.py`` against
known-good/known-bad responses. Never lower a threshold to hide a failure.
"""

from __future__ import annotations

from deepeval.metrics import ConversationalGEval, GEval
from deepeval.test_case import MultiTurnParams, SingleTurnParams

from .config import judge_model

_JUDGE = None


def _judge():
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = judge_model()
    return _JUDGE


def evidence_grounding() -> GEval:
    """Specific technical claims must be supported by retrieved evidence."""

    return GEval(
        name="Evidence Grounding",
        model=_judge(),
        threshold=0.5,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        criteria=(
            "The assistant's response must be grounded in the retrieved manual "
            "evidence. Specific technical claims — error codes, IP addresses, "
            "settings names and values, menu paths, LED states, port names, "
            "timings, button/option labels — must appear in or clearly follow "
            "from the retrieval context. Generic troubleshooting filler with no "
            "specific claim is acceptable but weak. Any specific claim that is "
            "absent from and not entailed by the retrieval context is a "
            "hallucination and must fail. An honest abstention ('I cannot tell "
            "from the manual') passes."
        ),
    )


def one_safe_step() -> GEval:
    """One appropriate next diagnostic action, not a troubleshooting dump."""

    return GEval(
        name="One Safe Step",
        model=_judge(),
        threshold=0.5,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        evaluation_steps=[
            "If the user explicitly asked for an enumeration (recommended tools, parts, menu items, topics), a concise grounded list of those items scores 1.0.",
            "If the user asked a factual question (what does an indicator mean, what is a setting, what value applies), a direct evidence-backed answer scores 1.0 with no further action required.",
            "If the user asked a how-to or setup question, a short ordered sequence of 2-4 steps from ONE procedure scores 1.0. Only fail it when the steps span several unrelated alternative branches.",
            "For diagnosis questions, the response must center on ONE clear next step: a single safe check, observation request, or setting to verify. A single step plus its brief reason is ideal.",
            "Fail (score 0.0) when the response dumps several unrelated alternative fixes to try, gives only background explanation with no actionable next step, or asks multiple unrelated questions at once.",
        ],
        # NOTE: abstentions are never routed to this metric (see
        # test_rag.run_case): the judge will not credit them here in any
        # rubric form, so abstention quality is scored by Abstention Quality.
    )


def abstention_quality() -> GEval:
    """Abstain honestly when evidence is insufficient; never invent a branch."""

    return GEval(
        name="Abstention Quality",
        model=_judge(),
        threshold=0.5,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        criteria=(
            "When the retrieval context does not support a safe next diagnostic "
            "branch, the assistant must abstain honestly: say what is unknown, "
            "what observation would unblock progress, or that the manual does "
            "not cover this. Fail when the response invents a specific "
            "procedure, value, or diagnosis the retrieval context does not "
            "support, or when it confidently continues down an unsupported "
            "branch. When the context DOES support a next step, giving that "
            "grounded step passes; unnecessary abstention on supported evidence "
            "fails."
        ),
    )


def observation_button_quality() -> GEval:
    """Observation options must answer the question and stay evidence-bound."""

    return GEval(
        name="Observation Button Quality",
        model=_judge(),
        threshold=0.5,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        criteria=(
            "When the response asks the user for an observation (for example a "
            "light state, error message, or setting value, possibly rendered as "
            "option buttons), the offered choices must be plausible answers to "
            "THAT question and consistent with the retrieval context. Fail when "
            "options answer a different question, include states the manual "
            "rules out, or invent specific values/codes absent from the "
            "evidence. If the response asks no observation question, or asks "
            "one open-ended question with no options, pass by default."
        ),
    )


def voice_concision() -> GEval:
    """Replies must be speakable: short, direct, action early."""

    return GEval(
        name="Voice Concision",
        model=_judge(),
        threshold=0.5,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria=(
            "Friday is voice-first: the reply will be spoken aloud. Pass when "
            "the response is direct and conversational, leads with what matters "
            "(acknowledge briefly, then the one next step or question), and "
            "avoids anything unspeakable: markdown, bullet lists longer than "
            "three items, URLs, code blocks, tables, or long quoted manual "
            "passages. Fail when the response reads like a document dump, "
            "buries the action after several paragraphs, contains formatting "
            "that cannot be spoken naturally, or rattles off more than three "
            "distinct actions or items in a single breath regardless of "
            "formatting."
        ),
    )


def troubleshooting_policy() -> ConversationalGEval:
    """End-to-end Friday contract over a whole troubleshooting conversation."""

    return ConversationalGEval(
        name="Friday Troubleshooting Policy",
        model=_judge(),
        threshold=0.5,
        evaluation_params=[
            MultiTurnParams.SCENARIO,
            MultiTurnParams.EXPECTED_OUTCOME,
        ],
        criteria=(
            "Judge the assistant's behavior across the whole troubleshooting "
            "conversation against this contract: (1) understand the reported "
            "symptom and use retrieved manual evidence; (2) give ONE safe "
            "evidence-backed check per turn; (3) when the user reports a "
            "result, remember it as an established fact; (4) advance to the "
            "next discriminating step instead of restarting or repeating an "
            "already-completed check; (5) never repeat a completed action "
            "unless new evidence justifies a recheck; (6) make no specific "
            "technical claim unsupported by evidence; (7) abstain honestly "
            "when the manuals do not support the next branch; (8) end at a "
            "correct resolution or an honest abstention, not an endless "
            "interview. Fail on any repeated completed check, any invented "
            "branch, or any forgotten user-reported result."
        ),
    )
