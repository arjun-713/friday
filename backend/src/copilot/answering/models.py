"""Typed contracts for the text-only troubleshooting layer."""

from typing import Literal

from pydantic import BaseModel, Field


class TroubleshootingRequest(BaseModel):
    query: str = Field(min_length=1)
    manufacturer: str | None = None
    model: str | None = None


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
    status: Literal["ready", "abstained"]
    answer: str | None = None
    evidence: list[EvidenceContext] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    missing_observations: list[str] = Field(default_factory=list)
    retrieval: RetrievalSummary
