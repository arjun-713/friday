"""Derive DeepEval RAG goldens from the deterministic retrieval cases.

Reads ``eval/retrieval_cases.jsonl`` (manually verified, authoritative) and
writes ``eval/evals/datasets/rag_curated.json`` in DeepEval Golden format.

No LLM is involved and no expected answer text is invented: deterministic
cases carry evidence labels, not answers. LLM-judged metrics that need
``expected_output`` run on the synthetic dataset instead; here the golden
keeps the expected chunk IDs in metadata for deterministic retrieval scoring
against the ACTUAL chunks Friday returns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "eval" / "retrieval_cases.jsonl"
DEST = REPO_ROOT / "eval" / "evals" / "datasets" / "rag_curated.json"


def main() -> int:
    goldens: list[dict[str, object]] = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            goldens.append(
                {
                    "input": case["query"],
                    "expected_output": None,
                    "context": None,
                    "retrieval_context": None,
                    "additional_metadata": {
                        "case_id": case["case_id"],
                        "category": case.get("category"),
                        "question_type": case.get("question_type"),
                        "manufacturer": case.get("manufacturer"),
                        "model": case.get("model"),
                        "expected_chunk_ids": sorted(
                            case.get("expected_chunk_ids", [])
                        ),
                        "acceptable_chunk_ids": sorted(
                            case.get(
                                "acceptable_chunk_ids",
                                case.get("expected_chunk_ids", []),
                            )
                        ),
                        "expected_pages": sorted(case.get("expected_pages", [])),
                        "should_abstain": bool(case.get("should_abstain", False)),
                        "notes": case.get("notes", ""),
                        "provenance": "curated:eval/retrieval_cases.jsonl",
                    },
                }
            )
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(
        json.dumps(goldens, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(goldens)} goldens to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
