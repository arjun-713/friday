"""Strict extraction of Markdown table rows."""

import re
from typing import Any

from pydantic import BaseModel, Field

from ..models import Evidence
from .structure import StructuredDocument, StructuredLine, structure_document


PIPE_ROW = re.compile(r"^\s*\|[^\n]*\|\s*$")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$")


class TableRowCandidate(BaseModel):
    table_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    source_file: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    section: str = Field(min_length=1)
    page: int = Field(gt=0)
    headers: list[str] = Field(min_length=1)
    cells: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)
    evidence: Evidence


def _is_pipe_row(line: StructuredLine) -> bool:
    return not line.is_heading and bool(PIPE_ROW.fullmatch(line.text))


def _is_separator(line: StructuredLine) -> bool:
    return _is_pipe_row(line) and bool(TABLE_SEPARATOR.fullmatch(line.text))


def _cells(text: str) -> list[str]:
    value = text.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return [cell.strip() for cell in value.split("|")]


def _evidence(source_file: str, line: StructuredLine) -> Evidence:
    return Evidence(
        source_file=source_file,
        page=line.page,
        section=line.section,
        content=line.text,
    )


def extract_table_rows(document: dict[str, Any] | StructuredDocument) -> list[TableRowCandidate]:
    """Extract rows only from tables with an explicit Markdown separator."""

    structured = document if isinstance(document, StructuredDocument) else structure_document(document)
    flat_lines = [line for page in structured.pages if not page.excluded_from_chunking for line in page.lines]
    parser_by_page = {page.page: page.parser for page in structured.pages}
    source_file = structured.source_file
    candidates: list[TableRowCandidate] = []
    table_number = 0
    index = 0

    while index < len(flat_lines) - 1:
        header = flat_lines[index]
        separator = flat_lines[index + 1]
        if not (_is_pipe_row(header) and _is_separator(separator)):
            index += 1
            continue

        headers = _cells(header.text)
        if not headers or not any(headers):
            index += 1
            continue

        table_number += 1
        table_id = f"{source_file}:table:{table_number}"
        row_index = 0
        index += 2
        while index < len(flat_lines):
            row = flat_lines[index]
            if not _is_pipe_row(row) or row.section != header.section:
                break
            cells = _cells(row.text)
            if not any(cells):
                index += 1
                continue
            evidence = _evidence(source_file, row)
            candidates.append(TableRowCandidate(
                table_id=table_id,
                row_index=row_index,
                source_file=source_file,
                parser=parser_by_page.get(row.page, "unknown"),
                section=row.section,
                page=row.page,
                headers=headers,
                cells=cells,
                content=" | ".join(cells),
                evidence=evidence,
            ))
            row_index += 1
            index += 1

        continue

    return candidates
