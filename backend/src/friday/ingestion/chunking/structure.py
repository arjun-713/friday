"""Deterministic heading detection and section-path assignment."""

import re
from typing import Any

from pydantic import BaseModel, Field

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# Some manuals encode FAQ headings as a full bold line rather than Markdown
# headings. Treat only numbered FAQ questions as structural headings; broad
# bold emphasis is intentionally left as narrative text.
FAQ_HEADING = re.compile(r"^\*\*(Q\d+\.\s+.+?)\*\*$", re.IGNORECASE)


class HeadingRecord(BaseModel):
    page: int = Field(gt=0)
    line_index: int = Field(ge=0)
    level: int = Field(ge=1, le=6)
    title: str = Field(min_length=1)
    section: str = Field(min_length=1)


class StructuredLine(BaseModel):
    page: int = Field(gt=0)
    line_index: int = Field(ge=0)
    text: str
    section: str = ""
    is_heading: bool = False
    heading_level: int | None = Field(default=None, ge=1, le=6)


class StructuredPage(BaseModel):
    page: int = Field(gt=0)
    parser: str = Field(min_length=1)
    excluded_from_chunking: bool = False
    exclusion_reason: str | None = None
    lines: list[StructuredLine] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    source_file: str = Field(min_length=1)
    pages: list[StructuredPage] = Field(default_factory=list)
    headings: list[HeadingRecord] = Field(default_factory=list)


def _heading_title(match: re.Match[str]) -> str:
    return re.sub(r"\s+#+\s*$", "", match.group(2)).strip()


def _heading(text: str) -> tuple[int, str] | None:
    if match := HEADING.fullmatch(text.strip()):
        return len(match.group(1)), _heading_title(match)
    if match := FAQ_HEADING.fullmatch(text.strip()):
        # FAQs in the TP-Link corpus otherwise follow a level-two Q heading.
        # Matching that level maintains the manual's chapter > FAQ hierarchy.
        return 2, match.group(1).strip()
    return None


def _section_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


def structure_document(document: dict[str, Any]) -> StructuredDocument:
    """Assign section paths without merging or rewriting source lines."""

    stack: list[tuple[int, str]] = []
    headings: list[HeadingRecord] = []
    structured_pages: list[StructuredPage] = []

    for page in document.get("pages", []):
        page_number = int(page["page_number"])
        excluded = bool(page.get("excluded_from_chunking", False))
        lines: list[StructuredLine] = []

        if not excluded:
            for line_index, text in enumerate(page.get("text", "").splitlines()):
                heading = _heading(text)
                if heading:
                    level, title = heading
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                    section = _section_path(stack)
                    headings.append(
                        HeadingRecord(
                            page=page_number,
                            line_index=line_index,
                            level=level,
                            title=title,
                            section=section,
                        )
                    )
                    lines.append(
                        StructuredLine(
                            page=page_number,
                            line_index=line_index,
                            text=text,
                            section=section,
                            is_heading=True,
                            heading_level=level,
                        )
                    )
                else:
                    lines.append(
                        StructuredLine(
                            page=page_number,
                            line_index=line_index,
                            text=text,
                            section=_section_path(stack),
                        )
                    )

        structured_pages.append(
            StructuredPage(
                page=page_number,
                parser=str(page.get("parser", "unknown")),
                excluded_from_chunking=excluded,
                exclusion_reason=page.get("exclusion_reason"),
                lines=lines,
            )
        )

    return StructuredDocument(
        source_file=str(document.get("source_file", "unknown")),
        pages=structured_pages,
        headings=headings,
    )
