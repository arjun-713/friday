#!/usr/bin/env python
"""Parse every manual with pdf-inspector's native, non-OCR path."""

import json

import pdf_inspector

from ...paths import manuals_dir, raw_dir, repo_relative
from .pdf_inspector import PdfInspectorAdapter


def main() -> None:
    source_root = manuals_dir()
    output_root = raw_dir()
    adapter = PdfInspectorAdapter()
    documents: list[dict[str, object]] = []

    for source_path in sorted(source_root.glob("**/*.pdf")):
        detection = pdf_inspector.detect_pdf(str(source_path))
        kind = adapter.classify(str(source_path))
        pages = adapter.extract(str(source_path), kind)
        output_path = output_root / source_path.parent.name / f"{source_path.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "source_file": repo_relative(source_path),
                    "pdf_type": kind.value,
                    "page_count": len(pages),
                    "ocr_required_pages": [page.page_number for page in pages if page.requires_ocr],
                    "pages": [page.model_dump() for page in pages],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        documents.append(
            {
                "source_file": repo_relative(source_path),
                "output_file": repo_relative(output_path),
                "pdf_type": kind.value,
                "page_count": len(pages),
                "ocr_required_pages": sum(page.requires_ocr for page in pages),
                "empty_pages": sum(not page.text.strip() for page in pages),
                "confidence": float(detection.confidence),
            }
        )
        print(f"parsed {source_path} -> {output_path}")

    report = {
        "parser": "pdf-inspector-native-no-ocr",
        "input_root": str(source_root),
        "documents": documents,
    }
    report_path = output_root / "parse_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
