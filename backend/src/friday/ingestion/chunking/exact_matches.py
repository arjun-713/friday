"""Context-gated extraction of exact technical identifiers."""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ..models import Evidence
from .structure import StructuredDocument, StructuredLine, structure_document


class ExactMatchKind(StrEnum):
    ERROR_CODE = "error_code"
    BLINK_PATTERN = "blink_pattern"
    PART_NUMBER = "part_number"
    MODEL_NUMBER = "model_number"


ERROR_CONTEXT = re.compile(r"\b(?:error|failure|fault|diagnostic|message)\b", re.IGNORECASE)
BLINK_CONTEXT = re.compile(r"\b(?:blink(?:ing)?|flash(?:ing)?|LED|light pattern|failure code)\b", re.IGNORECASE)
PART_CONTEXT = re.compile(r"\b(?:part|spare part|replacement part|FRU|component)\s+number\b", re.IGNORECASE)
MODEL_CONTEXT = re.compile(r"\b(?:model|product)\s+(?:number|name|code)\b", re.IGNORECASE)

BRACKET_CODE = re.compile(r"\[\s*\d{1,2}(?:\s*[,/-]\s*\d{1,2})+\s*\]")
SEQUENCE_CODE = re.compile(r"\b(?:ON|OFF|FLASH|BLINK)(?:[-/](?:ON|OFF|FLASH|BLINK)){2,}\b", re.IGNORECASE)
SEPARATED_NUMBERS = re.compile(r"\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{1,2})*\b")
SPACED_NUMBERS = re.compile(r"(?<![A-Za-z0-9])\d{1,2}\s+\d{1,2}(?![A-Za-z0-9])")
PART_NUMBER = re.compile(r"\b[A-Z]\d{4,6}-(?:\d{3}|[A-Z0-9]{3})\b", re.IGNORECASE)
MODEL_NUMBER = re.compile(
    r"\b(?=[A-Z0-9-]*\d)(?:[A-Z]{1,8}(?:-[A-Z0-9]{1,8}){1,3}|[A-Z]{1,6}\d{2,6}[A-Z0-9]*)\b",
    re.IGNORECASE,
)
GENERIC_MODEL_TOKENS = {"ERROR", "FAILURE", "SYSTEM", "PRODUCT", "NUMBER", "CODE", "MODEL"}


class ExactMatchCandidate(BaseModel):
    match_id: str = Field(min_length=1)
    kind: ExactMatchKind
    value: str = Field(min_length=1)
    normalized_value: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    section: str = Field(min_length=1)
    page: int = Field(gt=0)
    context: str = Field(min_length=1)
    evidence: Evidence


def _evidence(source_file: str, line: StructuredLine) -> Evidence:
    return Evidence(
        source_file=source_file,
        page=line.page,
        section=line.section,
        content=line.text,
        line_start=line.line_index,
        line_end=line.line_index,
    )


def _normalise(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _values(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0).strip() for match in pattern.finditer(text)))


def _add_candidates(
    output: list[ExactMatchCandidate],
    seen: set[tuple[ExactMatchKind, str, int, str]],
    source_file: str,
    parser: str,
    line: StructuredLine,
    kind: ExactMatchKind,
    values: list[str],
    occurrence: int,
) -> int:
    for value in values:
        normalized = _normalise(value)
        key = (kind, normalized, line.page, line.section)
        if not normalized or key in seen:
            continue
        seen.add(key)
        output.append(
            ExactMatchCandidate(
                match_id=f"{source_file}:exact:{occurrence}",
                kind=kind,
                value=value,
                normalized_value=normalized,
                source_file=source_file,
                parser=parser,
                section=line.section,
                page=line.page,
                context=line.text,
                evidence=_evidence(source_file, line),
            )
        )
        occurrence += 1
    return occurrence


def extract_exact_matches(document: dict[str, Any] | StructuredDocument) -> list[ExactMatchCandidate]:
    """Extract technical identifiers only when their line supplies context."""

    structured = document if isinstance(document, StructuredDocument) else structure_document(document)
    parser_by_page = {page.page: page.parser for page in structured.pages}
    source_file = structured.source_file
    output: list[ExactMatchCandidate] = []
    seen: set[tuple[ExactMatchKind, str, int, str]] = set()
    occurrence = 1

    for page in structured.pages:
        if page.excluded_from_chunking:
            continue
        for line in page.lines:
            if not line.section:
                continue
            text = line.text
            parser = parser_by_page.get(line.page, "unknown")
            if ERROR_CONTEXT.search(text):
                values = _values(BRACKET_CODE, text) + _values(SEQUENCE_CODE, text)
                occurrence = _add_candidates(
                    output, seen, source_file, parser, line, ExactMatchKind.ERROR_CODE, values, occurrence
                )
            if BLINK_CONTEXT.search(text):
                values = _values(BRACKET_CODE, text) + _values(SEQUENCE_CODE, text) + _values(SEPARATED_NUMBERS, text)
                if re.search(r"\b(?:blink|flash|LED|light pattern)\b", text, re.IGNORECASE):
                    values += _values(SPACED_NUMBERS, text)
                occurrence = _add_candidates(
                    output, seen, source_file, parser, line, ExactMatchKind.BLINK_PATTERN, values, occurrence
                )
            if PART_CONTEXT.search(text):
                occurrence = _add_candidates(
                    output,
                    seen,
                    source_file,
                    parser,
                    line,
                    ExactMatchKind.PART_NUMBER,
                    _values(PART_NUMBER, text),
                    occurrence,
                )
            if MODEL_CONTEXT.search(text):
                values = [value for value in _values(MODEL_NUMBER, text) if value.upper() not in GENERIC_MODEL_TOKENS]
                occurrence = _add_candidates(
                    output,
                    seen,
                    source_file,
                    parser,
                    line,
                    ExactMatchKind.MODEL_NUMBER,
                    values,
                    occurrence,
                )

    return output
