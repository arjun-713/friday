#!/usr/bin/env python
"""Create auditable cleaned JSON from the immutable native parser output."""

import json
from pathlib import Path

from copilot.ingestion.cleaning import clean_document


def main() -> None:
    source_root = Path("data/raw")
    output_root = Path("data/cleaned")
    report: list[dict[str, object]] = []

    for source_path in sorted(source_root.glob("*/*.json")):
        if source_path.name == "parse_report.json":
            continue
        document = json.loads(source_path.read_text(encoding="utf-8"))
        cleaned, removed = clean_document(document)
        category = source_path.parent.name
        output_path = output_root / category / source_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
        report.append({
            "source_file": str(source_path),
            "output_file": str(output_path),
            "pages": len(cleaned["pages"]),
            "removed_fragments": cleaned["cleaning"]["removed_fragment_count"],
            "removed_by_reason": removed,
            "normalizations": cleaned["cleaning"]["normalizations"],
            "excluded_pages": sum(page["excluded_from_chunking"] for page in cleaned["pages"]),
        })
        print(f"cleaned {source_path} -> {output_path}")

    report_path = output_root / "clean_report.json"
    report_path.write_text(json.dumps({"cleaner": "cleaned.v2", "documents": report}, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
