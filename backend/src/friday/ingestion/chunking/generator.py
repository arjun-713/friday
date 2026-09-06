"""Materialize multiple retrieval-oriented chunk strategies.

The generator keeps source structure intact and emits overlapping windows only
for narrative sections that exceed a bounded child size. Procedures, table
rows, and exact identifiers remain atomic so retrieval cannot separate a
warning from its procedure or a table value from its header.
"""

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ..models import ChunkKind, ChunkStrategy, DocumentChunk, Evidence, RetrievalProfile, SourceDocument
from .exact_matches import ExactMatchCandidate, extract_exact_matches
from .procedures import ProcedureCandidate, extract_procedures
from .structure import StructuredLine, structure_document
from .tables import TableRowCandidate, extract_table_rows

SECTION_MAX_CHARS = 2_400
PARENT_MAX_CHARS = 6_000
CHILD_MAX_CHARS = 1_400
CHILD_OVERLAP_CHARS = 240
PROCEDURE_STEP_GROUP_MAX_CHARS = 1_800
PROCEDURE_STEP_GROUP_SIZE = 4


class ChunkingConfig(BaseModel):
    section_max_chars: int = Field(default=SECTION_MAX_CHARS, ge=200)
    parent_max_chars: int = Field(default=PARENT_MAX_CHARS, ge=500)
    child_max_chars: int = Field(default=CHILD_MAX_CHARS, ge=200)
    child_overlap_chars: int = Field(default=CHILD_OVERLAP_CHARS, ge=0)


class BlockType(StrEnum):
    NARRATIVE = "narrative"
    HEADING = "heading"
    WARNING = "warning"
    PREREQUISITE = "prerequisite"
    ORDERED_LIST = "ordered_list"
    TABLE = "table"
    TECHNICAL_IDENTIFIER = "technical_identifier"


class _Unit(BaseModel):
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    block_type: BlockType = BlockType.NARRATIVE


def _stable_id(
    document: SourceDocument,
    strategy: ChunkStrategy,
    section: str,
    pages: Sequence[int],
    content: str,
) -> str:
    identity = "|".join(
        [
            document.sha256 or document.document_id,
            strategy.value,
            section,
            ",".join(str(page) for page in pages),
            content,
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"{strategy.value}:{digest}"


def _dedupe_evidence(items: Iterable[Evidence]) -> list[Evidence]:
    result: list[Evidence] = []
    seen: set[tuple[str, int, str, str]] = set()
    for item in items:
        key = (item.source_file, item.page, item.section, item.content)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _evidence_for_units(units: Iterable[_Unit]) -> Iterator[Evidence]:
    for unit in units:
        yield from unit.evidence


def _block_type(line: StructuredLine) -> BlockType:
    text = line.text.strip()
    if line.is_heading:
        return BlockType.HEADING
    if re.search(r"\b(?:warning|caution|danger|notice|important)\b", text, re.IGNORECASE):
        return BlockType.WARNING
    if re.match(
        r"^(?:before you begin|prerequisite|preparation|prepare|requirements?|required tools?)\b", text, re.IGNORECASE
    ):
        return BlockType.PREREQUISITE
    if re.match(r"^(?:\*\*)?\d+[.)](?:\*\*)?\s+", text):
        return BlockType.ORDERED_LIST
    if text.startswith("|") and text.endswith("|"):
        return BlockType.TABLE
    if re.search(
        r"\b(?:error code|failure code|blink(?:ing)? pattern|part number|model number)\b", text, re.IGNORECASE
    ):
        return BlockType.TECHNICAL_IDENTIFIER
    return BlockType.NARRATIVE


def _units_for_section(source_file: str, lines: Sequence[StructuredLine]) -> list[_Unit]:
    units: list[_Unit] = []
    paragraph: list[str] = []
    paragraph_evidence: list[Evidence] = []
    paragraph_type = BlockType.NARRATIVE

    def flush() -> None:
        nonlocal paragraph_type
        if paragraph:
            units.append(
                _Unit(
                    text="\n".join(paragraph),
                    evidence=paragraph_evidence.copy(),
                    block_type=paragraph_type,
                )
            )
            paragraph.clear()
            paragraph_evidence.clear()
            paragraph_type = BlockType.NARRATIVE

    for line in lines:
        text = line.text.strip()
        if not text:
            flush()
            continue
        block_type = _block_type(line)
        if paragraph and block_type != paragraph_type:
            flush()
        paragraph_type = block_type
        evidence = Evidence(
            source_file=source_file,
            page=line.page,
            section=line.section,
            content=line.text,
            line_start=line.line_index,
            line_end=line.line_index,
        )
        paragraph.append(text)
        paragraph_evidence.append(evidence)
        if line.is_heading:
            flush()
    flush()
    return units


def _pack(units: Sequence[_Unit], max_chars: int) -> list[list[_Unit]]:
    """Pack paragraph units, splitting only oversized individual lines."""

    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_chars = 0
    for unit in units:
        if current and current_chars + len(unit.text) + 1 > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        if len(unit.text) <= max_chars:
            current.append(unit)
            current_chars += len(unit.text) + 1
            continue
        if current:
            groups.append(current)
            current = []
            current_chars = 0
        words = unit.text.split()
        piece: list[str] = []
        piece_chars = 0
        for word in words:
            if piece and piece_chars + len(word) + 1 > max_chars:
                groups.append(
                    [
                        _Unit(
                            text=" ".join(piece),
                            evidence=[item.model_copy(update={"content": " ".join(piece)}) for item in unit.evidence],
                        )
                    ]
                )
                piece = []
                piece_chars = 0
            piece.append(word)
            piece_chars += len(word) + 1
        if piece:
            groups.append(
                [
                    _Unit(
                        text=" ".join(piece),
                        evidence=[item.model_copy(update={"content": " ".join(piece)}) for item in unit.evidence],
                    )
                ]
            )
    if current:
        groups.append(current)
    return groups


def _overlap_groups(units: Sequence[_Unit], max_chars: int, overlap_chars: int) -> list[tuple[list[_Unit], int, int]]:
    effective_max = max_chars if overlap_chars == 0 else max(1, max_chars - overlap_chars)
    groups = _pack(units, effective_max)
    if overlap_chars == 0 or len(groups) < 2:
        return [(group, 0, 0) for group in groups]

    result: list[tuple[list[_Unit], int, int]] = []
    for index, group in enumerate(groups):
        before: list[_Unit] = []
        chars = 0
        if index:
            for unit in reversed(groups[index - 1]):
                if chars + len(unit.text) + 1 > overlap_chars:
                    break
                before.insert(0, unit)
                chars += len(unit.text) + 1
        result.append((before + group, chars, 0))
    return result


def _chunk(
    *,
    document: SourceDocument,
    source_file: str,
    parser: str,
    kind: ChunkKind,
    strategy: ChunkStrategy,
    ordinal: int,
    section: str,
    content: str,
    evidence: Iterable[Evidence],
    parent_chunk_id: str | None = None,
    overlap_before: int = 0,
    overlap_after: int = 0,
    metadata: dict[str, str | int | bool] | None = None,
    retrieval_profiles: list[RetrievalProfile] | None = None,
) -> DocumentChunk:
    evidence_list = _dedupe_evidence(evidence)
    pages = sorted({item.page for item in evidence_list})
    return DocumentChunk(
        chunk_id=_stable_id(document, strategy, section, pages, content),
        document=document,
        page=pages[0],
        pages=pages,
        section=section,
        content=content,
        kind=kind,
        parser=parser,
        evidence=evidence_list,
        source_parser=parser,
        chunker="chunking.v2",
        retrieval_profiles=retrieval_profiles or _retrieval_profiles(kind, metadata or {}),
        strategy=strategy,
        ordinal=ordinal,
        parent_chunk_id=parent_chunk_id,
        overlap_before=overlap_before,
        overlap_after=overlap_after,
        metadata=metadata or {},
    )


def _retrieval_profiles(
    kind: ChunkKind,
    metadata: dict[str, str | int | bool],
) -> list[RetrievalProfile]:
    if kind == ChunkKind.EXACT_MATCH:
        return [RetrievalProfile.EXACT, RetrievalProfile.BM25]
    if kind == ChunkKind.TABLE_ROW:
        return [RetrievalProfile.EXACT, RetrievalProfile.BM25, RetrievalProfile.VECTOR]
    if kind == ChunkKind.PROCEDURE_STEP_GROUP:
        return [RetrievalProfile.BM25, RetrievalProfile.VECTOR]
    if kind == ChunkKind.PROCEDURE:
        return [RetrievalProfile.CONTEXT_STORE]
    if metadata.get("role") == "parent":
        return [RetrievalProfile.CONTEXT_STORE]
    return [RetrievalProfile.BM25, RetrievalProfile.VECTOR]


def _narrative_chunks(
    document: SourceDocument,
    structured: Any,
    config: ChunkingConfig,
) -> list[DocumentChunk]:
    by_section: dict[str, list[StructuredLine]] = defaultdict(list)
    for page in structured.pages:
        if page.excluded_from_chunking:
            continue
        for line in page.lines:
            if line.section and line.text.strip():
                by_section[line.section].append(line)

    chunks: list[DocumentChunk] = []
    ordinal = 0
    for section, lines in by_section.items():
        units = _units_for_section(structured.source_file, lines)
        for group in _pack(units, config.section_max_chars):
            ordinal += 1
            chunks.append(
                _chunk(
                    document=document,
                    source_file=structured.source_file,
                    parser="pdf-inspector",
                    kind=ChunkKind.SECTION,
                    strategy=ChunkStrategy.SECTION,
                    ordinal=ordinal,
                    section=section,
                    content="\n\n".join(unit.text for unit in group),
                    evidence=_evidence_for_units(group),
                    metadata={
                        "split": "block_and_paragraph_boundary",
                        "max_chars": config.section_max_chars,
                        "block_types": ",".join(sorted({unit.block_type.value for unit in group})),
                    },
                )
            )
    return chunks


def _parent_child_chunks(document: SourceDocument, structured: Any, config: ChunkingConfig) -> list[DocumentChunk]:
    by_section: dict[str, list[StructuredLine]] = defaultdict(list)
    for page in structured.pages:
        if not page.excluded_from_chunking:
            for line in page.lines:
                if line.section and line.text.strip():
                    by_section[line.section].append(line)

    chunks: list[DocumentChunk] = []
    ordinal = 0
    for section, lines in by_section.items():
        units = _units_for_section(structured.source_file, lines)
        for parent_group in _pack(units, config.parent_max_chars):
            ordinal += 1
            parent = _chunk(
                document=document,
                source_file=structured.source_file,
                parser="pdf-inspector",
                kind=ChunkKind.PARENT,
                strategy=ChunkStrategy.PARENT_CHILD,
                ordinal=ordinal,
                section=section,
                content="\n\n".join(unit.text for unit in parent_group),
                evidence=_evidence_for_units(parent_group),
                metadata={"role": "parent", "max_chars": config.parent_max_chars},
            )
            chunks.append(parent)
            for child_group, before, after in _overlap_groups(
                parent_group, config.child_max_chars, config.child_overlap_chars
            ):
                ordinal += 1
                chunks.append(
                    _chunk(
                        document=document,
                        source_file=structured.source_file,
                        parser="pdf-inspector",
                        kind=ChunkKind.CHILD,
                        strategy=ChunkStrategy.PARENT_CHILD,
                        ordinal=ordinal,
                        section=section,
                        content="\n\n".join(unit.text for unit in child_group),
                        evidence=_evidence_for_units(child_group),
                        parent_chunk_id=parent.chunk_id,
                        overlap_before=before,
                        overlap_after=after,
                        metadata={
                            "role": "child",
                            "max_chars": config.child_max_chars,
                            "overlap_policy": "trailing_context",
                            "block_types": ",".join(sorted({unit.block_type.value for unit in child_group})),
                        },
                    )
                )
    return chunks


def _procedure_chunks(
    document: SourceDocument,
    candidate: ProcedureCandidate,
    ordinal: int,
) -> tuple[list[DocumentChunk], int]:
    context = [
        *[f"Prerequisite: {item.content}" for item in candidate.prerequisites],
        *[f"Warning: {item.content}" for item in candidate.warnings],
    ]
    all_evidence = [*candidate.prerequisites, *candidate.warnings, *(step.evidence for step in candidate.steps)]
    parent_content = "\n".join([*context, *[f"{step.number}. {step.content}" for step in candidate.steps]])
    parent = _chunk(
        document=document,
        source_file=candidate.source_file,
        parser=candidate.parser,
        kind=ChunkKind.PROCEDURE,
        strategy=ChunkStrategy.PROCEDURE,
        ordinal=ordinal,
        section=candidate.section,
        content=parent_content,
        evidence=all_evidence,
        metadata={"role": "parent", "step_count": len(candidate.steps), "procedure_id": candidate.procedure_id},
    )
    chunks = [parent]
    group: list[Any] = []
    group_chars = 0
    context_chars = len("\n".join(context)) + (1 if context else 0)
    step_budget = max(200, PROCEDURE_STEP_GROUP_MAX_CHARS - context_chars - 100)
    for step in candidate.steps:
        for fragment in _split_step(step, step_budget):
            if group and (len(group) >= PROCEDURE_STEP_GROUP_SIZE or group_chars + len(fragment.content) > step_budget):
                ordinal += 1
                chunks.append(_procedure_step_group(document, candidate, parent, group, ordinal, context))
                group = []
                group_chars = 0
            group.append(fragment)
            group_chars += len(fragment.content)
    if group:
        ordinal += 1
        chunks.append(_procedure_step_group(document, candidate, parent, group, ordinal, context))
    return chunks, ordinal


def _split_step(step: Any, max_chars: int) -> list[Any]:
    if len(step.content) <= max_chars:
        return [step]
    fragments: list[Any] = []
    words = step.content.split()
    current: list[str] = []
    current_chars = 0
    for word in words:
        if current and current_chars + len(word) + 1 > max_chars:
            fragments.append(step.model_copy(update={"content": " ".join(current)}))
            current = []
            current_chars = 0
        current.append(word)
        current_chars += len(word) + 1
    if current:
        fragments.append(step.model_copy(update={"content": " ".join(current)}))
    return fragments


def _procedure_step_group(
    document: SourceDocument,
    candidate: ProcedureCandidate,
    parent: DocumentChunk,
    steps: list[Any],
    ordinal: int,
    context: list[str],
) -> DocumentChunk:
    evidence = [*candidate.prerequisites, *candidate.warnings, *(step.evidence for step in steps)]
    content = "\n".join([*context, *[f"{step.number}. {step.content}" for step in steps]])
    return _chunk(
        document=document,
        source_file=candidate.source_file,
        parser=candidate.parser,
        kind=ChunkKind.PROCEDURE_STEP_GROUP,
        strategy=ChunkStrategy.PROCEDURE,
        ordinal=ordinal,
        section=candidate.section,
        content=content,
        evidence=evidence,
        parent_chunk_id=parent.chunk_id,
        metadata={
            "role": "step_group",
            "procedure_id": candidate.procedure_id,
            "step_start": steps[0].number,
            "step_end": steps[-1].number,
        },
    )


def _candidate_chunks(
    document: SourceDocument,
    candidate: ProcedureCandidate | TableRowCandidate | ExactMatchCandidate,
    ordinal: int,
) -> tuple[list[DocumentChunk], int]:
    if isinstance(candidate, ProcedureCandidate):
        return _procedure_chunks(document, candidate, ordinal)
    if isinstance(candidate, TableRowCandidate):
        return [
            _chunk(
                document=document,
                source_file=candidate.source_file,
                parser=candidate.parser,
                kind=ChunkKind.TABLE_ROW,
                strategy=ChunkStrategy.TABLE_ROW,
                ordinal=ordinal,
                section=candidate.section,
                content=" | ".join(
                    f"{header}: {cell}" for header, cell in zip(candidate.headers, candidate.cells, strict=False)
                ),
                evidence=[candidate.evidence],
                metadata={"table_id": candidate.table_id, "row_index": candidate.row_index},
            )
        ], ordinal
    return [
        _chunk(
            document=document,
            source_file=candidate.source_file,
            parser=candidate.parser,
            kind=ChunkKind.EXACT_MATCH,
            strategy=ChunkStrategy.EXACT_MATCH,
            ordinal=ordinal,
            section=candidate.section,
            content=candidate.context,
            evidence=[candidate.evidence],
            metadata={"identifier_kind": candidate.kind.value, "normalized_value": candidate.normalized_value},
        )
    ], ordinal


def generate_chunks(
    document: dict[str, Any],
    source_document: SourceDocument,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Generate all retrieval views for one cleaned document."""

    active_config = config or ChunkingConfig()
    structured = structure_document(document)
    chunks = _narrative_chunks(source_document, structured, active_config)
    chunks.extend(_parent_child_chunks(source_document, structured, active_config))
    candidates: list[ProcedureCandidate | TableRowCandidate | ExactMatchCandidate] = [
        *extract_procedures(document),
        *extract_table_rows(document),
        *extract_exact_matches(document),
    ]
    ordinal = len(chunks)
    for candidate in candidates:
        ordinal += 1
        candidate_chunks, ordinal = _candidate_chunks(source_document, candidate, ordinal)
        chunks.extend(candidate_chunks)
    return chunks
