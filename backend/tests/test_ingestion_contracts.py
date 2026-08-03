from copilot.ingestion.models import ChunkKind, DocumentChunk, Evidence, PageRecord, SourceDocument


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
