from copilot.ingestion.models import ChunkKind, DocumentChunk, Evidence, OcrPage, OcrTextItem, PageRecord, SourceDocument, TextSpan


def test_chunk_requires_citation_evidence() -> None:
    document = SourceDocument(
        document_id="manual-001",
        title="Example Manual",
        manufacturer="Example",
        source_url="https://example.test/manual.pdf",
        retrieved_at="2026-08-03T00:00:00Z",
    )
    chunk = DocumentChunk(
        chunk_id="manual-001-p4",
        document=document,
        page=4,
        section="Troubleshooting > Wi-Fi",
        content="Restart the router.",
        kind=ChunkKind.PROCEDURE,
        parser="fixture",
        confidence=1.0,
        evidence=[Evidence(page=4, section="Troubleshooting > Wi-Fi", content="Restart the router.")],
    )
    assert chunk.evidence[0].page == chunk.page


def test_page_can_be_marked_for_selective_ocr() -> None:
    page = PageRecord(page_number=2, text="", parser="pdf-inspector", confidence=0.0, requires_ocr=True)
    assert page.requires_ocr is True


def test_positioned_text_keeps_one_based_citation_page() -> None:
    span = TextSpan(
        text="Warning", page=3, x=10, y=20, width=40, height=8,
        font="Arial-Bold", font_size=12, is_bold=True, item_type="text",
    )
    assert span.page == 3


def test_ocr_result_preserves_page_confidence_and_boxes() -> None:
    result = OcrPage(
        page=11,
        model="PP-OCRv6-small",
        image_path="tmp/page.png",
        text="Warning",
        confidence=0.98,
        items=[OcrTextItem(text="Warning", confidence=0.98, page=11, polygon=[(1, 2), (3, 2), (3, 4), (1, 4)])],
    )
    assert result.items[0].page == 11
    assert result.items[0].confidence == 0.98
