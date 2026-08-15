"""Schema for manually verified retrieval evaluation cases."""

from dataclasses import dataclass
from typing import Any

QUESTION_TYPES = frozenset(
    ("factual", "error_code", "procedure", "symptom", "unanswerable")
)


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    category: str
    question_type: str
    manufacturer: str | None
    model: str | None
    expected_chunk_ids: frozenset[str]
    expected_pages: frozenset[int]
    should_abstain: bool
    notes: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "RetrievalCase":
        question_type = str(value["question_type"])
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"unsupported question_type: {question_type}")
        expected_pages = frozenset(
            int(page) for page in value.get("expected_pages", [])
        )
        should_abstain = bool(value["should_abstain"])
        if should_abstain != (question_type == "unanswerable"):
            raise ValueError(
                f"{value['case_id']}: abstention flag must match question type"
            )
        return cls(
            case_id=str(value["case_id"]),
            query=str(value["query"]),
            category=str(value["category"]),
            question_type=question_type,
            manufacturer=value.get("manufacturer"),
            model=value.get("model"),
            expected_chunk_ids=frozenset(
                str(item) for item in value.get("expected_chunk_ids", [])
            ),
            expected_pages=expected_pages,
            should_abstain=should_abstain,
            notes=str(value.get("notes", "")),
        )
