"""Typed contracts for the text-only troubleshooting layer."""

from typing import Literal

from pydantic import BaseModel, Field


class TroubleshootingRequest(BaseModel):
    query: str = Field(min_length=1)
    manufacturer: str | None = None
    model: str | None = None
    session_id: str = Field(default="default", min_length=1, max_length=128)
    observation: str | None = Field(default=None, min_length=1)
    selected_option: str | None = Field(default=None, min_length=1, max_length=64)
    regenerate: bool = False


class DiagnosticOption(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    # The visible label stays short (for example, ``Amber``).  ``value`` is
    # the canonical fact value the planner stores after the user selects it.
    value: str | None = Field(default=None, max_length=160)


class DiagnosticStep(BaseModel):
    step_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=160)
    instruction: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[DiagnosticOption] = Field(default_factory=list, max_length=6)
    source_ids: list[str] = Field(min_length=1, max_length=8)


class DiagnosticFact(BaseModel):
    """A user-reported fact retained independently of the chat transcript."""

    key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    value: str = Field(min_length=1, max_length=320)
    label: str = Field(min_length=1, max_length=120)
    raw: str = Field(min_length=1, max_length=1000)


class ObservationRequest(BaseModel):
    """The one fact Friday is asking the user to report, if any."""

    request_id: str = Field(min_length=1, max_length=128)
    fact_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    question: str = Field(min_length=1, max_length=500)
    options: list[DiagnosticOption] = Field(default_factory=list, max_length=6)
    recheck_after_action: bool = False


class DiagnosticAction(BaseModel):
    """One evidence-supported action; procedures remain one action at a time."""

    instruction: str = Field(min_length=1, max_length=1200)
    why: str | None = Field(default=None, max_length=700)


class DiagnosticTurn(BaseModel):
    """A planner-owned turn: solve, advance, clarify, or abstain."""

    turn_id: str = Field(min_length=1, max_length=128)
    mode: Literal["solve", "advance", "clarify", "abstain"]
    response: str = Field(min_length=1, max_length=2000)
    interpretation: str | None = Field(default=None, max_length=1200)
    next_action: DiagnosticAction | None = None
    observation_request: ObservationRequest | None = None
    facts_learned: list[DiagnosticFact] = Field(default_factory=list, max_length=12)
    candidate_causes: list[str] = Field(default_factory=list, max_length=6)
    ruled_out_causes: list[str] = Field(default_factory=list, max_length=6)
    source_ids: list[str] = Field(min_length=1, max_length=8)


class ManualImage(BaseModel):
    asset_id: str
    url: str
    mime_type: str
    document_title: str
    page: int = Field(gt=0)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    manufacturer: str
    model: str
    document_version: str
    page: int = Field(gt=0)
    section: str
    source_url: str


class EvidenceContext(BaseModel):
    chunk_id: str
    content: str = Field(min_length=1)
    section: str
    pages: list[int] = Field(min_length=1)
    citation: Citation


class RetrievalSummary(BaseModel):
    abstained: bool
    reason: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class TroubleshootingResponse(BaseModel):
    session_id: str = "default"
    status: Literal["ready", "abstained"]
    answer: str | None = None
    turn: DiagnosticTurn | None = None
    # Kept temporarily for older clients and tests. New clients render
    # ``turn`` directly so a final answer is not forced to include a question.
    step: DiagnosticStep | None = None
    awaiting_observation: bool = False
    images: list[ManualImage] = Field(default_factory=list)
    evidence: list[EvidenceContext] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    # Confirmed results from completed diagnostic steps.  This is intentionally
    # separate from the free-form conversation transcript: the client can show
    # an evidence ledger without treating every sentence as a verified fact.
    observations: list[str] = Field(default_factory=list)
    missing_observations: list[str] = Field(default_factory=list)
    retrieval: RetrievalSummary


class DiagnosticSessionState(BaseModel):
    session_id: str
    facts: dict[str, DiagnosticFact] = Field(default_factory=dict)
    pending_observation: str | None = None
    pending_option_id: str | None = None
    current_request: ObservationRequest | None = None
    current_turn: DiagnosticTurn | None = None
    observations: dict[str, str] = Field(default_factory=dict)
    completed_steps: list[str] = Field(default_factory=list)
    ruled_out_causes: list[str] = Field(default_factory=list)
    current_step_id: str | None = None
    current_step: DiagnosticStep | None = None
    current_evidence: list[EvidenceContext] = Field(default_factory=list)
    current_images: list[ManualImage] = Field(default_factory=list)
    current_citations: list[Citation] = Field(default_factory=list)
    last_turn_was_acknowledgement: bool = False
