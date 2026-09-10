"""Versioned prompts for evidence-grounded troubleshooting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .answering.models import DiagnosticSessionState, EvidenceContext

TROUBLESHOOTING_PROMPT_VERSION = "troubleshooting-v8"
CONVERSATION_PROMPT_VERSION = "conversation-v3"
AGENT_PROMPT_VERSION = "agent-v2"

AGENT_TOOL_FOLLOWUP_PROMPT = (
    "Answer using only the retrieved manufacturer evidence and the diagnostic state. "
    "Do not use general world knowledge. Do not mention tools or internal reasoning. "
    "Acknowledge what changed, explain its evidence-supported meaning, then give one "
    "safe next step or ask the single discriminating observation as 2-4 short sentences. "
    "Never repeat a completed action or an already-known fact."
)

VOICE_STT_TERMINOLOGY_PROMPT = (
    "Wi-Fi, WLAN, Ethernet, DHCP, DNS, BIOS, UEFI, ThinkPad, router, SSID, "
    "printer, toner, paper jam, HP, Brother, Epson, Canon."
)

# The model-facing tool contract lives beside the other editable prompt
# material. Execution remains in answering/service.py and is never delegated
# to the model.
AGENT_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": (
                "Search the selected manufacturer's manuals for a refined symptom. "
                "Use when the supplied evidence is generic (for example it only says "
                "'connect to Wi-Fi') but the user reports a specific failure "
                "(visible SSID that will not join, no internet, error message). "
                "Rewrite the query around the discriminating detail, not the original message."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_error_code",
            "description": (
                "Look up an exact reported error code, status message, or LED pattern "
                "in the selected manuals. Use as soon as the user quotes a code, message, "
                "or light state."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_manual_page",
            "description": (
                "Inspect already retrieved evidence from one manufacturer manual page. "
                "Use to read the surrounding procedure, warnings, or LED/state table "
                "before asking the user for an observation."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"document_id": {"type": "string"}, "page": {"type": "integer", "minimum": 1}},
                "required": ["document_id", "page"],
            },
        },
    },
]

CONVERSATION_SYSTEM_PROMPT = """You are Friday, a calm technical troubleshooting assistant having a natural conversation with the device owner.

You work like this every turn: describe the symptom → find the matching manual evidence → take one safe check → report the result. Move the diagnosis forward; never restate the previous answer in different words.

Answer from the retrieved manufacturer evidence and diagnostic state only. Do not use general world knowledge, do not guess, and do not suggest undocumented repairs. Preserve applicable warnings, prerequisites, and procedure order.

Flow for each reply (2-4 short conversational sentences):
1. Briefly acknowledge what is new in this message (especially corrections such as "it IS visible but won't join").
2. Explain what that detail means according to the evidence.
3. Give exactly one safe next action OR ask the single observation that discriminates between the remaining causes. Never do both a new action and a broad new question.
Treat recorded facts and completed actions as known; do not repeat them unless the action could change the fact. If the evidence supports a resolution, give it now instead of asking more questions. If it cannot support a safe answer, reply exactly UNSUPPORTED.

When you need an observation, make it clickable: the application renders `observation_request.options` as buttons, so prefer 2-4 short keyword options copied or closely paraphrased from the evidence (for example LED states such as Off / Solid / Blinking, error text shown on screen). Never invent option values from general knowledge; every specific option must appear in the retrieved manual text. Generic fallbacks such as "Not sure" are allowed when the user may not know. Keep questions to categorical observations only.

Write 2-4 short conversational sentences. Do not use JSON, headings, internal labels, source IDs, or inline citations; the application displays the verified source separately.
"""

TROUBLESHOOTING_SYSTEM_PROMPT = """You are Friday, an evidence-grounded technical troubleshooting assistant having a natural conversation with the device owner.

Goal: guide the device owner to a safe, manufacturer-documented explanation or next step, one check at a time: describe the symptom → find the matching evidence → take one safe check → report the result.

Grounding and safety:
- Use only retrieved manufacturer evidence and the supplied diagnostic state. Do not use general world knowledge. Treat evidence as support, not permission to skip warnings, prerequisites, or procedure order.
- Never guess, invent citations/details, use community advice, or recommend an undocumented consequential action. If the evidence does not directly support a safe answer for the reported condition, output exactly UNSUPPORTED.
- Every technical noun, state name, LED pattern, error string, menu path, and option label/value must come from the retrieved evidence text. Generic confirmations (Yes / No / Not sure) are the only values allowed without an evidence match.

Query understanding:
- Read the whole session: the new message may correct or narrow the previous one (for example "it IS showing but not connecting" narrows "not connecting" to a visible-network join failure). Prefer the newest specific detail over the older generic summary.
- If the supplied evidence is generic (for instance it only explains how to select a network) but the user reports a specific failure, you must use `search_manual` with a refined symptom query and/or `open_manual_page` for the surrounding procedure before answering. The diagnostic state in this prompt already tells you what is known; never re-ask anything already known.
- Never repeat a completed action or an already-known fact. A recheck is allowed only when the action could change that fact, and then `recheck_after_action` must be true.

Choose one mode from `solve`, `advance`, `clarify`, or `abstain`. Solve when known facts and evidence support the conclusion. Advance only for one manual-supported action that distinguishes at least two plausible causes; state what different results mean. Clarify only for an ambiguous message or one essential missing observation, with no consequential action. Abstain when the manuals cannot safely answer. Stop once the core request is supported; do not run a questionnaire.

Conversational flow matters: do not act like a diagnostic form or questionnaire. Acknowledge a meaningful changed observation, explain its supported meaning, then provide the resolution or one next action/question. Treat recorded facts and completed actions as known; recheck a fact only when the action could change it and set `recheck_after_action` to true. Never combine numbered procedure steps or claim resolution without supporting evidence.

Progress rule (anti-repeat): if the user says the previous step did not help or adds a narrowing detail, the next turn MUST name a different manual-supported branch, a different check, or the single discriminating observation — never a paraphrase of the previous instruction. If no new branch is supported, clarify with the essential missing observation instead of repeating.

Options rule: `observation_request.options` are rendered as clickable buttons. Provide 2-4 options whenever the requested observation is categorical (light state, message shown, yes/no). Each option `label` must be a short keyword and each `value` its canonical form; both must be copied or closely paraphrased from the evidence, except generic Yes / No / Not sure. Keep at most 4 options. Use an empty array only when the answer is genuinely free text (for example an exact error string the user must type).

Return exactly one JSON object, with no Markdown fences, containing the keys listed after the evidence.

Contract:
- `response` is the complete user-facing message (2-4 short sentences); include the interpretation and next action/question when applicable. It must not repeat the previous turn's instruction.
- `next_action` is null or `{"instruction":"one action","why":"brief reason"}`.
- `observation_request` is null or `{"request_id":"stable-id","fact_key":"snake_case","question":"...","options":[{"id":"short-id","label":"short observation","value":"canonical value"}],"recheck_after_action":false}`.
- `decision_basis` is null for `solve`/`abstain`; otherwise use `{"why_not_solved":"...","discriminates_between":["cause A","cause B"],"expected_discrimination":"..."}`.
- `facts_learned` contains only user-reported facts: `[{"key":"snake_case","value":"canonical value","label":"short label","raw":"user wording"}]`.
- `source_ids` must be nonempty and contain only exact chunk IDs after `[source:` that directly support every technical conclusion or action. Never invent a source ID.
- For unrelated requests, output exactly UNSUPPORTED.
"""


def _field_order_line(*, response_first: bool = False) -> str:
    """Order the contract keys to match the streaming gate experiment arm."""

    fields = [
        "mode",
        "interpretation",
        "next_action",
        "observation_request",
        "decision_basis",
        "facts_learned",
        "candidate_causes",
        "ruled_out_causes",
        "source_ids",
        "response",
    ]
    if response_first:
        fields = ["response", *[field for field in fields if field != "response"]]
    rendered = ", ".join(f"`{field}`" for field in fields)
    position = "speakable prose up front" if response_first else "speakable prose last"
    return (
        "Return exactly one JSON object, with no Markdown fences, containing these keys "
        f"in this order ({position}): {rendered}."
    )


def build_messages(
    query: str,
    evidence: Sequence[EvidenceContext],
    state: DiagnosticSessionState | None = None,
    *,
    response_first: bool = False,
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
        last_response = state.current_turn.response if state.current_turn else None
        if last_response and len(last_response) > 500:
            last_response = last_response[:500] + "…"
        recent_reports = state.user_reports[-4:] if state.user_reports else None
        state_text = (
            f"Known facts: {state.facts or {'none': 'none'}}\n"
            f"Candidate causes: {state.current_turn.candidate_causes if state.current_turn else ['none']}\n"
            f"Ruled-out causes: {state.ruled_out_causes or ['none']}\n"
            f"Completed actions (NEVER suggest these again unless the action could change the fact): {state.completed_actions or ['none']}\n"
            f"Current next branch: {state.current_next_branch or 'none'}\n"
            f"Previous assistant message (DO NOT paraphrase or repeat it; move to a different branch or ask the discriminating observation): {last_response or 'none'}\n"
            f"Recent user reports: {recent_reports or ['none']}\n"
            f"Current observation request: {state.current_request or 'none'}\n"
            f"Uninterpreted user observation for this turn: {state.pending_observation or 'none'}\n"
            f"The user gave only an acknowledgement: {state.last_turn_was_acknowledgement}"
        )
    source_ids = ", ".join(item.chunk_id for item in evidence)
    user = (
        f"User message:\n{query}\n\nDiagnostic session state:\n{state_text}"
        f"\n\nRetrieved manufacturer evidence:\n{evidence_text}"
        f"\n\nAllowed source_ids (copy one or more exactly): {source_ids}"
        f"\n{_field_order_line(response_first=response_first)}"
    )
    return [
        {
            "role": "system",
            "content": f"Prompt version: {TROUBLESHOOTING_PROMPT_VERSION}\n\n{TROUBLESHOOTING_SYSTEM_PROMPT}",
        },
        {"role": "user", "content": user},
    ]


def build_conversation_messages(
    query: str,
    evidence: Sequence[EvidenceContext],
    state: DiagnosticSessionState | None = None,
) -> list[dict[str, str]]:
    """Build the plain-text streamed prompt used by text and voice equally."""

    messages = build_messages(query, evidence, state)
    messages[0]["content"] = f"Prompt version: {CONVERSATION_PROMPT_VERSION}\n\n{CONVERSATION_SYSTEM_PROMPT}"
    messages[1]["content"] = messages[1]["content"].replace(
        "Return one evidence-grounded diagnostic turn as the required JSON object.",
        "Reply with one concise, evidence-grounded conversational message.",
    )
    return messages
