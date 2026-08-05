"""Conservative extraction of ordered procedure candidates."""

import re
from typing import Iterable

from pydantic import BaseModel, Field, model_validator

from ..models import Evidence
from .structure import StructuredDocument, StructuredLine, structure_document


ORDERED_STEP = re.compile(r"^\s*(?:\*\*)?(\d+)[.)](?:\*\*)?\s+(.+?)\s*$")
WARNING_MARKER = re.compile(r"\b(?:warning|caution|danger|notice|important)\b", re.IGNORECASE)
PREREQUISITE_MARKER = re.compile(
    r"^(?:before you begin|prerequisite(?:s)?|preparation|prepare|requirements?|required tools?|you will need)\b",
    re.IGNORECASE,
)


class ProcedureStep(BaseModel):
    number: int = Field(ge=0)
    page: int = Field(gt=0)
    section: str = Field(min_length=1)
    content: str = Field(min_length=1)
    evidence: Evidence


class ProcedureCandidate(BaseModel):
    procedure_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    section: str = Field(min_length=1)
    pages: list[int] = Field(min_length=1)
    prerequisites: list[Evidence] = Field(default_factory=list)
    warnings: list[Evidence] = Field(default_factory=list)
    steps: list[ProcedureStep] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_pages(self) -> "ProcedureCandidate":
        if self.pages != sorted(set(self.pages)):
            raise ValueError("procedure pages must be sorted and unique")
        if any(step.page not in self.pages for step in self.steps):
            raise ValueError("every procedure step page must belong to procedure pages")
        return self


def _step_match(line: StructuredLine) -> re.Match[str] | None:
    if line.is_heading:
        return None
    return ORDERED_STEP.fullmatch(line.text)


def _is_prerequisite(text: str) -> bool:
    normalized = re.sub(r"^[#*_\s]+|[#*_\s]+$", "", text).strip()
    return bool(PREREQUISITE_MARKER.match(normalized))


def _evidence(source_file: str, line: StructuredLine) -> Evidence:
    return Evidence(
        source_file=source_file,
        page=line.page,
        section=line.section,
        content=line.text,
    )


def _pages_for(
    steps: Iterable[ProcedureStep],
    prerequisites: Iterable[Evidence],
    warnings: Iterable[Evidence],
) -> list[int]:
    pages = {step.page for step in steps}
    pages.update(item.page for item in prerequisites)
    pages.update(item.page for item in warnings)
    return sorted(pages)


def _candidate(
    source_file: str,
    procedure_number: int,
    parser: str,
    section: str,
    steps: list[ProcedureStep],
    prerequisites: list[Evidence],
    warnings: list[Evidence],
) -> ProcedureCandidate | None:
    if not section or len(steps) < 2:
        return None
    return ProcedureCandidate(
        procedure_id=f"{source_file}:procedure:{procedure_number}",
        source_file=source_file,
        parser=parser,
        section=section,
        pages=_pages_for(steps, prerequisites, warnings),
        prerequisites=prerequisites,
        warnings=warnings,
        steps=steps,
    )


def extract_procedures(document: dict) -> list[ProcedureCandidate]:
    """Extract only ordered blocks with at least two numbered steps.

    The input may be a cleaned document or an already structured document.
    No prose is synthesized and no single numbered line is promoted to a
    procedure candidate.
    """

    structured = document if isinstance(document, StructuredDocument) else structure_document(document)
    flat_lines = [line for page in structured.pages if not page.excluded_from_chunking for line in page.lines]
    parser_by_page = {page.page: page.parser for page in structured.pages}
    source_file = structured.source_file
    candidates: list[ProcedureCandidate] = []
    steps: list[ProcedureStep] = []
    prerequisites: list[Evidence] = []
    warnings: list[Evidence] = []
    section = ""
    parser = ""
    procedure_number = 0
    last_line_was_blank = False

    def finish() -> None:
        nonlocal steps, prerequisites, warnings, section, parser, procedure_number
        procedure_number += 1
        candidate = _candidate(
            source_file, procedure_number, parser, section, steps, prerequisites, warnings
        )
        if candidate is not None:
            candidates.append(candidate)
        steps = []
        prerequisites = []
        warnings = []
        section = ""
        parser = ""

    for line in flat_lines:
        match = _step_match(line)
        if match:
            number = int(match.group(1))
            if steps and (line.section != section or number <= steps[-1].number):
                finish()
            if not steps:
                section = line.section
                parser = parser_by_page.get(line.page, "unknown")
            evidence = _evidence(source_file, line)
            steps.append(ProcedureStep(
                number=number,
                page=line.page,
                section=line.section,
                content=match.group(2),
                evidence=evidence,
            ))
            if WARNING_MARKER.search(line.text):
                warnings.append(_evidence(source_file, line))
            last_line_was_blank = False
            continue

        if not steps:
            if line.text.strip() and line.section and _is_prerequisite(line.text):
                prerequisites.append(_evidence(source_file, line))
            continue

        if line.section != section or line.is_heading:
            finish()
            if line.text.strip() and line.section and _is_prerequisite(line.text):
                prerequisites.append(_evidence(source_file, line))
            continue

        if not line.text.strip():
            last_line_was_blank = True
            continue

        if last_line_was_blank and not WARNING_MARKER.search(line.text):
            # Do not absorb an unrelated paragraph after a completed list.
            finish()
            last_line_was_blank = False
            continue

        steps[-1].content = f"{steps[-1].content}\n{line.text}"
        steps[-1].evidence.content = f"{steps[-1].evidence.content}\n{line.text}"
        if WARNING_MARKER.search(line.text):
            warnings.append(_evidence(source_file, line))
        last_line_was_blank = False

    if steps:
        finish()
    return candidates
