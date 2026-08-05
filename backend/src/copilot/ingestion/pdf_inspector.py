"""Adapter for Firecrawl's local ``pdf-inspector`` Python bindings.

This module is intentionally not imported by the API at startup. Install the
optional native dependency and call the adapter from an ingestion job.
"""

from typing import Any

from .models import PageRecord, PdfKind, TextSpan


class PdfInspectorUnavailable(RuntimeError):
    """Raised when the native pdf-inspector binding is not installed."""


def _library() -> Any:
    try:
        import pdf_inspector
    except ImportError as exc:
        raise PdfInspectorUnavailable(
            "Install the backend dependencies to use pdf-inspector."
        ) from exc
    return pdf_inspector


def _pdf_kind(pdf_type: str) -> PdfKind:
    values = {
        "text_based": PdfKind.TEXT,
        "scanned": PdfKind.SCANNED,
        "image_based": PdfKind.IMAGE,
        "mixed": PdfKind.MIXED,
    }
    try:
        return values[pdf_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown pdf-inspector PDF type: {pdf_type!r}") from exc


class PdfInspectorAdapter:
    parser_name = "pdf-inspector"

    def classify(self, path: str) -> PdfKind:
        result = _library().detect_pdf(path)
        return _pdf_kind(result.pdf_type)

    def extract(self, path: str, kind: PdfKind) -> list[PageRecord]:
        """Extract per-page Markdown and positions without performing OCR.

        ``pdf-inspector`` reports OCR routing; this adapter records that route
        on each page for a later OCR stage rather than silently OCRing pages.
        Page numbers exposed by this application are 1-based.
        """
        library = _library()
        result = library.detect_pdf(path)
        pages = library.extract_pages_markdown(path)
        positioned = library.extract_text_with_positions(path)
        spans_by_page: dict[int, list[TextSpan]] = {}
        for item in positioned:
            page_number = int(item.page) + 1
            spans_by_page.setdefault(page_number, []).append(
                TextSpan(
                    text=item.text,
                    page=page_number,
                    x=item.x,
                    y=item.y,
                    width=item.width,
                    height=item.height,
                    font=item.font,
                    font_size=item.font_size,
                    is_bold=item.is_bold,
                    is_italic=item.is_italic,
                    item_type=item.item_type,
                )
            )
        pages_needing_ocr = {int(page) for page in result.pages_needing_ocr}
        return [
            PageRecord(
                page_number=int(page.page) + 1,
                text=page.markdown,
                parser=self.parser_name,
                confidence=float(result.confidence),
                requires_ocr=(int(page.page) + 1) in pages_needing_ocr,
                spans=spans_by_page.get(int(page.page) + 1, []),
            )
            for page in pages.pages
        ]
