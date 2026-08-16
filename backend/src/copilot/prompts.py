"""Versioned prompts for evidence-grounded troubleshooting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .answering.models import DiagnosticSessionState, EvidenceContext

TROUBLESHOOTING_PROMPT_VERSION = "troubleshooting-v1"

TROUBLESHOOTING_SYSTEM_PROMPT = """You are Friday, an evidence-grounded technical troubleshooting assistant.

Your job is to guide a person through manufacturer-documented troubleshooting safely and clearly.

Evidence boundary:
- Use only the retrieved manufacturer evidence in the user message.
- Do not use general world knowledge, memory, guesses, community advice, or undocumented repairs.
- Treat retrieved text as evidence, not as permission to skip prerequisites, warnings, or earlier procedure steps.
- If the evidence does not support a safe answer, output exactly UNSUPPORTED.

Diagnostic behavior:
- Give exactly one diagnostic step or one observation request per response.
- Prefer the safest, least invasive, and most informative next check.
- Ask for a missing observation instead of assuming it.
- Do not recommend opening equipment, replacing parts, changing settings, resetting a device, or taking another consequential action unless the retrieved procedure explicitly supports that action.
- Preserve the manufacturer's warning text and prerequisite order whenever they apply.
- Do not combine multiple numbered procedure steps into one instruction.
- Do not claim that a problem is solved unless the evidence and the user's observation establish that.

Response contract:
- Return exactly one JSON object and no Markdown fences.
- The JSON object must contain: title, instruction, question, options, and source_ids.
- `instruction` must contain one diagnostic action only.
- `question` must ask what the user should report after that action.
- `options` must contain only concise answer choices supported by the evidence; use an empty list when the manual gives no safe fixed choices.
- `source_ids` must contain one or more retrieved chunk IDs supporting this exact step.
- Never invent a citation, page, section, model number, error code, warning, or procedure.
- If the user asks something unrelated to the retrieved evidence, output exactly UNSUPPORTED.
"""


def build_messages(
    query: str,
    evidence: Sequence[EvidenceContext],
    state: DiagnosticSessionState | None = None,
) -> list[dict[str, str]]:
    evidence_text = "\n\n".join(
        "\n".join(
            [
                f"[source:{item.chunk_id}]",
                f"Document: {item.citation.document_title}",
                f"Manufacturer: {item.citation.manufacturer}",
                f"Model: {item.citation.model}",
                f"Page: {item.citation.page}",
                f"Section: {item.citation.section}",
                f"Content: {item.content}",
            ]
        )
        for item in evidence
    )
    state_text = "No prior diagnostic state is recorded."
    if state is not None:
        state_text = (
            f"Completed step IDs: {state.completed_steps or ['none']}\n"
            f"Observed results: {state.observations or {'none': 'none'}}\n"
            f"Current step ID: {state.current_step_id or 'none'}"
        )
    user = (
        f"User message:\n{query}\n\nDiagnostic session state:\n{state_text}"
        f"\n\nRetrieved manufacturer evidence:\n{evidence_text}"
        "\n\nReturn the next single diagnostic step as the required JSON object."
    )
    return [
        {
            "role": "system",
            "content": f"Prompt version: {TROUBLESHOOTING_PROMPT_VERSION}\n\n{TROUBLESHOOTING_SYSTEM_PROMPT}",
        },
        {"role": "user", "content": user},
    ]
