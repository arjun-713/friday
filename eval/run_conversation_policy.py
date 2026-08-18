"""Run deterministic conversation-policy regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .conversation_policy import ConversationCase, report


def load_cases(path: Path) -> list[ConversationCase]:
    cases = [
        ConversationCase.from_json(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"no conversation policy cases found in {path}")
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("conversation policy case IDs must be unique")
    return cases


def run(cases: list[ConversationCase]) -> dict[str, Any]:
    return report(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/conversation_cases.jsonl"),
        help="manually verified structured conversation scenarios",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON report path",
    )
    args = parser.parse_args()
    result = run(load_cases(args.cases))
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
