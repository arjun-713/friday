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


class DiagnosticOption(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)


class DiagnosticStep(BaseModel):
    step_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=160)
    instruction: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[DiagnosticOption] = Field(default_factory=list, max_length=6)
    source_ids: list[str] = Field(min_length=1, max_length=8)


class DiagnosticSessionState(BaseModel):
    session_id: str
    observations: dict[str, str] = Field(default_factory=dict)
    completed_steps: list[str] = Field(default_factory=list)
    ruled_out_causes: list[str] = Field(default_factory=list)
    current_step_id: str | None = None


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
    step: DiagnosticStep | None = None
    awaiting_observation: bool = False
    images: list[ManualImage] = Field(default_factory=list)
    evidence: list[EvidenceContext] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    missing_observations: list[str] = Field(default_factory=list)
    retrieval: RetrievalSummary
