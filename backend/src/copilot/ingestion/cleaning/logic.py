"""Deterministic, auditable cleanup for pdf-inspector Markdown pages."""

import re
from collections import Counter
from typing import Any

PAGE_NUMBER = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,8})$", re.IGNORECASE)
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?$")
STRUCTURAL = re.compile(r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|[A-Za-z][.)]\s|\||```)")
DOT_LEADER = re.compile(r"\.{3,}")
UNDERSCORE_LEADER = re.compile(r"_{3,}")
STANDALONE_LAYOUT_RULE = re.compile(r"^-{12,}$")
HTML_FORMAT_TAG = re.compile(
    r"</?(?:u|span|div|p|font|strong|em|b|i|sub|sup)(?:\s[^>]*)?>|<br\s*/?>",
    re.IGNORECASE,
)
MALFORMED_BOLD_LINK = re.compile(r"\*\*\[([^\]]+?)\*\*\]\(([^)]+?)\*\*\)")
EXCESS_ASTERISKS = re.compile(r"\*{4,}")
NAVIGATION_LABEL = re.compile(
    r"^(?:#{1,6}\s*)?(?:related (?:information|references|tasks|topics)|parent topic)\s*:?$",
    re.IGNORECASE,
)
COPYRIGHT_LINE = re.compile(r"^(?:#{1,6}\s*)?(?:©|copyright\b|all rights reserved\b)", re.IGNORECASE)
TOC_HEADING = re.compile(r"^#{1,6}\s+(?:table of )?contents\s*$", re.IGNORECASE)
TOC_ENTRY = re.compile(r"^(?!#{1,6}\s).{3,}?\s+\d{1,4}\s*\*{0,2}$")
PROTECTED_REPEATED_HEADINGS = {
    "cause",
    "what to do",
    "warning",
    "caution",
    "danger",
    "important",
    "note",
}


def _repair_markdown(line: str) -> tuple[str, int]:
    repaired, links = MALFORMED_BOLD_LINK.subn(r"**[\1](\2)**", line)
    repaired, excess_asterisks = EXCESS_ASTERISKS.subn("**", repaired)
    return repaired, links + excess_asterisks


def normalize_line(line: str) -> str:
    line = HTML_FORMAT_TAG.sub("", line)
    line, _ = _repair_markdown(line)
    line = DOT_LEADER.sub(" ", line)
    line = UNDERSCORE_LEADER.sub(" ", line)
    return re.sub(r"[ \t]+", " ", line.replace("\u00a0", " ")).strip()


def is_page_number(line: str) -> bool:
    return bool(PAGE_NUMBER.fullmatch(normalize_line(line)))


def _boundary_lines(text: str) -> tuple[list[str], list[str]]:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    return lines[:3], lines[-3:]


def _plain_markdown(line: str) -> str:
    return re.sub(r"^[#*_\s]+|[#*_\s]+$", "", line).strip()


def _safe_boundary_candidate(line: str) -> bool:
    normalized = normalize_line(line)
    plain = _plain_markdown(normalized)
    if is_page_number(normalized) or normalized.upper() == "ENWW":
        return True
    if TABLE_SEPARATOR.fullmatch(normalized) or (STRUCTURAL.match(normalized) and not normalized.startswith("#")):
        return False
    if plain.lower() in PROTECTED_REPEATED_HEADINGS:
        return False
    if len(plain) == 1:
        return plain.isalnum() or plain == "."
    if not plain or len(plain) > 140 or re.search(r"[.!?;:]$", plain):
        return False
    return len(plain.split()) <= 14


def _repeated_boundary_lines(pages: list[dict[str, Any]]) -> set[str]:
    counts: Counter[str] = Counter()
    for page in pages:
        top, bottom = _boundary_lines(page.get("text", ""))
        counts.update(set(top + bottom))
    # Cap the threshold so long manuals with chapter-local running titles are cleaned too.
    minimum = max(3, min(20, (len(pages) + 9) // 10))
    return {line for line, count in counts.items() if count >= minimum and _safe_boundary_candidate(line)}


def _join_wrapped_lines(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        if not line:
            if result and result[-1] != "":
                result.append("")
            continue
        if result and result[-1] and not STRUCTURAL.match(result[-1]) and not STRUCTURAL.match(line):
            previous = result[-1]
            if not re.search(r"[.!?:;]$", previous) and (line[0].islower() or line[0].isdigit()):
                if previous.endswith("-") and line[0].islower():
                    result[-1] = previous[:-1] + line
                else:
                    result[-1] = previous + " " + line
                continue
        result.append(line)
    while result and result[-1] == "":
        result.pop()
    return result


def _table_of_contents_signals(lines: list[str], leader_runs: int) -> tuple[bool, bool]:
    nonempty = [line for line in lines if line]
    if not nonempty:
        return False, False
    entries = sum(bool(TOC_ENTRY.fullmatch(line)) for line in nonempty)
    has_heading = any(TOC_HEADING.fullmatch(line) for line in nonempty)
    has_page_heading = any(line.startswith("#") for line in nonempty)
    starts_contents = (has_heading and entries >= 2) or (leader_runs >= 8 and has_page_heading)
    looks_like_continuation = (entries >= 8 and entries / len(nonempty) >= 0.5) or leader_runs >= 8
    return starts_contents, looks_like_continuation


def clean_document(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    pages = document["pages"]
    repeated = _repeated_boundary_lines(pages)
    cleaned_pages: list[dict[str, Any]] = []
    removed_counts: Counter[str] = Counter()
    normalization_counts: Counter[str] = Counter()
    seen_page_text: dict[str, int] = {}
    seen_repeated_boundaries: set[str] = set()
    inside_table_of_contents = False

    for page in pages:
        raw_text = page.get("text", "")
        removed: list[dict[str, str]] = []
        kept: list[str] = []
        top, bottom = _boundary_lines(raw_text)
        boundary = set(top + bottom)
        page_normalizations = {
            "dot_leaders": sum(len(DOT_LEADER.findall(line)) for line in raw_text.splitlines()),
            "underscore_leaders": sum(len(UNDERSCORE_LEADER.findall(line)) for line in raw_text.splitlines()),
            "html_format_tags": sum(len(HTML_FORMAT_TAG.findall(line)) for line in raw_text.splitlines()),
            "markdown_repairs": sum(_repair_markdown(line)[1] for line in raw_text.splitlines()),
        }

        for raw_line in raw_text.splitlines():
            html_tags = len(HTML_FORMAT_TAG.findall(raw_line))
            repaired_line, markdown_repairs = _repair_markdown(raw_line)
            dot_leaders = len(DOT_LEADER.findall(repaired_line))
            underscore_leaders = len(UNDERSCORE_LEADER.findall(repaired_line))
            normalization_counts["html_format_tags"] += html_tags
            normalization_counts["markdown_repairs"] += markdown_repairs
            normalization_counts["dot_leaders"] += dot_leaders
            normalization_counts["underscore_leaders"] += underscore_leaders
            line = normalize_line(repaired_line)

            if not line:
                kept.append("")
            elif is_page_number(line):
                removed.append({"text": line, "reason": "page_number"})
                removed_counts["page_number"] += 1
            elif COPYRIGHT_LINE.match(line):
                removed.append({"text": line, "reason": "copyright_boilerplate"})
                removed_counts["copyright_boilerplate"] += 1
            elif NAVIGATION_LABEL.fullmatch(line):
                removed.append({"text": line, "reason": "navigation_label"})
                removed_counts["navigation_label"] += 1
            elif line in {".", "®"}:
                removed.append({"text": line, "reason": "standalone_artifact"})
                removed_counts["standalone_artifact"] += 1
            elif STANDALONE_LAYOUT_RULE.fullmatch(line):
                removed.append({"text": line, "reason": "layout_rule"})
                removed_counts["layout_rule"] += 1
            elif (
                line in repeated
                and line in boundary
                and (
                    line in seen_repeated_boundaries
                    or is_page_number(line)
                    or line.upper() == "ENWW"
                    or len(_plain_markdown(line)) == 1
                )
            ):
                removed.append({"text": line, "reason": "repeated_page_boundary"})
                removed_counts["repeated_page_boundary"] += 1
            else:
                kept.append(line)
                if line in repeated and line in boundary:
                    seen_repeated_boundaries.add(line)

        cleaned_lines = _join_wrapped_lines(kept)
        cleaned_text = "\n".join(cleaned_lines)
        exclusion_reason: str | None = None

        starts_contents, continues_contents = _table_of_contents_signals(
            cleaned_lines,
            page_normalizations["dot_leaders"] + page_normalizations["underscore_leaders"],
        )
        is_contents_page = starts_contents or (inside_table_of_contents and continues_contents)
        inside_table_of_contents = is_contents_page

        if is_contents_page:
            exclusion_reason = "table_of_contents"
            removed.extend({"text": line, "reason": exclusion_reason} for line in cleaned_lines if line)
            removed_counts[exclusion_reason] += sum(bool(line) for line in cleaned_lines)
            cleaned_text = ""
        elif not cleaned_text.strip():
            exclusion_reason = "empty_page"
            removed_counts[exclusion_reason] += 1
        elif cleaned_text in seen_page_text:
            exclusion_reason = "duplicate_page"
            removed.append(
                {
                    "text": f"Duplicate of page {seen_page_text[cleaned_text]}",
                    "reason": exclusion_reason,
                }
            )
            removed_counts[exclusion_reason] += 1
            cleaned_text = ""
        else:
            seen_page_text[cleaned_text] = page.get("page_number", len(cleaned_pages) + 1)

        page_copy = dict(page)
        spans = page_copy.pop("spans", [])
        page_copy["span_count"] = len(spans)
        page_copy["text"] = cleaned_text
        page_copy["excluded_from_chunking"] = exclusion_reason is not None
        if exclusion_reason:
            page_copy["exclusion_reason"] = exclusion_reason
        page_copy["removed_fragments"] = removed
        page_copy["normalizations"] = page_normalizations
        cleaned_pages.append(page_copy)

    cleaned = dict(document)
    cleaned["schema_version"] = "cleaned.v2"
    cleaned["cleaning"] = {
        "method": "deterministic-retrieval-text-cleaning",
        "raw_source_file": document.get("source_file"),
        "coordinates_retained_in_raw": True,
        "repeated_boundary_candidates": sorted(repeated),
        "removed_fragment_count": sum(len(page["removed_fragments"]) for page in cleaned_pages),
        "removed_by_reason": dict(removed_counts),
        "normalizations": dict(normalization_counts),
    }
    cleaned["pages"] = cleaned_pages
    return cleaned, dict(removed_counts)
