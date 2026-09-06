#!/usr/bin/env python
"""Parse only pdf-inspector text-based manuals into a mirrored output tree."""

import json
import time
from pathlib import Path

import pdf_inspector

from .pdf_inspector import PdfInspectorAdapter


def main() -> None:
    source_root = Path("data/manuals")
    output_root = Path("data/raw")
    adapter = PdfInspectorAdapter()
    report: list[dict[str, object]] = []

    for source_path in sorted(source_root.glob("**/*.pdf")):
        detection = pdf_inspector.detect_pdf(str(source_path))
        if str(detection.pdf_type).lower() != "text_based":
            continue

        started = time.perf_counter()
        kind = adapter.classify(str(source_path))
        pages = adapter.extract(str(source_path), kind)
        category = source_path.parent.name
        output_path = output_root / category / f"{source_path.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "source_file": str(source_path),
                    "pdf_type": kind.value,
                    "page_count": len(pages),
                    "pages": [page.model_dump() for page in pages],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        report.append(
            {
                "source_file": str(source_path),
                "output_file": str(output_path),
                "pdf_type": kind.value,
                "pages": len(pages),
                "characters": sum(len(page.text) for page in pages),
                "positioned_spans": sum(len(page.spans) for page in pages),
                "seconds": round(time.perf_counter() - started, 3),
                "preview": " ".join(next((page.text for page in pages if page.text.strip()), "").split())[:240],
            }
        )
        print(f"parsed {source_path} -> {output_path}")

    report_path = output_root / "parse_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "parser": "pdf-inspector",
                "input_root": str(source_root),
                "text_documents": len(report),
                "documents": report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
