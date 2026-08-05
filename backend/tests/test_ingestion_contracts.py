from copilot.ingestion.models import ChunkKind, DocumentChunk, Evidence, OcrPage, OcrTextItem, PageRecord, SourceDocument, TextSpan
from copilot.ingestion.cleaning import clean_document
from copilot.ingestion.chunking import extract_procedures, structure_document


def test_chunk_requires_citation_evidence() -> None:
    document = SourceDocument(
        document_id="manual-001",
        title="Example Manual",
        manufacturer="Example",
        model="Example 1",
        version="1.0",
        source_url="https://example.test/manual.pdf",
        retrieved_at="2026-08-03T00:00:00Z",
    )
    chunk = DocumentChunk(
        chunk_id="manual-001-p4",
        document=document,
        page=4,
        pages=[4],
        section="Troubleshooting > Wi-Fi",
        content="Restart the router.",
        kind=ChunkKind.PROCEDURE,
        parser="fixture",
        evidence=[Evidence(source_file="manual.pdf", page=4, section="Troubleshooting > Wi-Fi", content="Restart the router.")],
    )
    assert chunk.evidence[0].page == chunk.page
    assert chunk.pages == [4]


def test_chunk_contract_rejects_missing_resolved_document_metadata() -> None:
    from pydantic import ValidationError

    document = SourceDocument(
        document_id="manual-001",
        title="Example Manual",
        manufacturer="Example",
        source_url="https://example.test/manual.pdf",
        retrieved_at="2026-08-03T00:00:00Z",
    )
    try:
        DocumentChunk(
            chunk_id="manual-001-p4",
            document=document,
            page=4,
            pages=[4],
            section="Troubleshooting",
            content="Restart the router.",
            kind=ChunkKind.SECTION,
            parser="fixture",
            evidence=[Evidence(source_file="manual.pdf", page=4, section="Troubleshooting", content="Restart the router.")],
        )
    except ValidationError as error:
        assert "document.model" in str(error)
    else:
        raise AssertionError("chunk validation must reject unresolved model metadata")


def test_chunk_contract_keeps_multi_page_evidence_consistent() -> None:
    document = SourceDocument(
        document_id="manual-001",
        title="Example Manual",
        manufacturer="Example",
        model="Example 1",
        version="1.0",
        source_url="https://example.test/manual.pdf",
        retrieved_at="2026-08-03T00:00:00Z",
    )
    chunk = DocumentChunk(
        chunk_id="manual-001-p4-p5",
        document=document,
        page=4,
        pages=[4, 5],
        section="Troubleshooting > Wi-Fi",
        content="Restart the router, then verify the connection.",
        kind=ChunkKind.PROCEDURE,
        parser="fixture",
        evidence=[
            Evidence(source_file="manual.pdf", page=4, section="Troubleshooting > Wi-Fi", content="Restart the router."),
            Evidence(source_file="manual.pdf", page=5, section="Troubleshooting > Wi-Fi", content="Verify the connection."),
        ],
    )
    assert chunk.pages == [4, 5]


def test_structure_assigns_nested_sections_across_pages() -> None:
    structured = structure_document({
        "source_file": "manual.json",
        "pages": [
            {"page_number": 1, "parser": "pdf-inspector", "text": "# Troubleshooting\nIntro\n## Wi-Fi\nCheck the cable."},
            {"page_number": 2, "parser": "pdf-inspector", "text": "Continue checking.\n### Signal\nCheck the indicator."},
        ],
    })
    assert [heading.section for heading in structured.headings] == [
        "Troubleshooting",
        "Troubleshooting > Wi-Fi",
        "Troubleshooting > Wi-Fi > Signal",
    ]
    assert structured.pages[0].lines[1].section == "Troubleshooting"
    assert structured.pages[0].lines[3].section == "Troubleshooting > Wi-Fi"
    assert structured.pages[1].lines[0].section == "Troubleshooting > Wi-Fi"
    assert structured.pages[1].lines[1].is_heading is True


def test_structure_preserves_excluded_pages_without_creating_headings() -> None:
    structured = structure_document({
        "source_file": "manual.json",
        "pages": [
            {"page_number": 1, "parser": "pdf-inspector", "text": "# Contents", "excluded_from_chunking": True, "exclusion_reason": "table_of_contents"},
            {"page_number": 2, "parser": "pdf-inspector", "text": "# Setup\nConnect the cable."},
        ],
    })
    assert structured.pages[0].excluded_from_chunking is True
    assert structured.pages[0].lines == []
    assert [heading.title for heading in structured.headings] == ["Setup"]


def test_procedure_extraction_preserves_steps_prerequisites_and_warnings() -> None:
    document = {
        "source_file": "manual.json",
        "pages": [
            {
                "page_number": 1,
                "parser": "pdf-inspector",
                "text": "# Reset the router\n**Before you begin:** disconnect power.\n\n**1.** Press Reset.\n**2.** Hold it for ten seconds. **WARNING:** Do not remove power.",
            },
            {
                "page_number": 2,
                "parser": "pdf-inspector",
                "text": "3. Release Reset.\n\nVerify the status light.",
            },
        ],
    }
    procedures = extract_procedures(document)
    assert len(procedures) == 1
    procedure = procedures[0]
    assert procedure.section == "Reset the router"
    assert procedure.pages == [1, 2]
    assert [step.number for step in procedure.steps] == [1, 2, 3]
    assert "Before you begin" in procedure.prerequisites[0].content
    assert any("WARNING" in warning.content for warning in procedure.warnings)
    assert procedure.steps[2].content == "Release Reset."


def test_procedure_extraction_splits_numbering_resets_and_rejects_single_steps() -> None:
    document = {
        "source_file": "manual.json",
        "pages": [{
            "page_number": 1,
            "parser": "fixture",
            "text": "# Setup\n1. First.\n2. Second.\n1. Another procedure.\nOnly context.",
        }],
    }
    procedures = extract_procedures(document)
    assert len(procedures) == 1
    assert [step.number for step in procedures[0].steps] == [1, 2]


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


def test_cleaning_removes_repeated_boundary_and_page_number_but_keeps_warning() -> None:
    document = {
        "source_file": "manual.pdf",
        "pdf_type": "text",
        "page_count": 3,
        "ocr_required_pages": [],
        "pages": [
            {"page_number": 1, "text": "# Manual\n\n## Contents\n1", "spans": []},
            {"page_number": 2, "text": "## Contents\n\n**WARNING:** Do not open the cover.\n2", "spans": []},
            {"page_number": 3, "text": "## Contents\n\nFollow the procedure.\n3", "spans": []},
        ],
    }
    cleaned, counts = clean_document(document)
    assert "## Contents" not in cleaned["pages"][1]["text"]
    assert "WARNING" in cleaned["pages"][1]["text"]
    assert counts["page_number"] == 3
    assert "raw_text" not in cleaned["pages"][1]


def test_cleaning_compacts_table_of_contents_dot_leaders() -> None:
    document = {
        "pages": [{"page_number": 1, "text": "Safety precautions........................................................................7", "spans": []}]
    }
    cleaned, _ = clean_document(document)
    assert cleaned["pages"][0]["text"] == "Safety precautions 7"
    assert cleaned["pages"][0]["normalizations"]["dot_leaders"] == 1


def test_cleaning_removes_formatting_noise_and_keeps_semantic_text() -> None:
    document = {
        "source_file": "manual.pdf",
        "pages": [{
            "page_number": 1,
            "text": "<u>Reset</u> the router.\n**[Support**](https://example.test**)\n**WARNING:** Unplug it first.",
            "spans": [{"text": "Reset"}],
        }],
    }
    cleaned, _ = clean_document(document)
    page = cleaned["pages"][0]
    assert "<u>" not in page["text"]
    assert "**[Support](https://example.test)**" in page["text"]
    assert "WARNING" in page["text"]
    assert "spans" not in page
    assert page["span_count"] == 1


def test_cleaning_excludes_toc_and_duplicate_pages_without_dropping_page_identity() -> None:
    toc = "# Contents\nSafety ........ 7\nSetup ........ 9\nTroubleshooting ........ 12"
    document = {
        "pages": [
            {"page_number": 1, "text": toc, "spans": []},
            {"page_number": 2, "text": "# Reset\nPress and hold Reset.", "spans": []},
            {"page_number": 3, "text": "# Reset\nPress and hold Reset.", "spans": []},
        ]
    }
    cleaned, counts = clean_document(document)
    assert cleaned["pages"][0]["exclusion_reason"] == "table_of_contents"
    assert cleaned["pages"][0]["text"] == ""
    assert cleaned["pages"][2]["exclusion_reason"] == "duplicate_page"
    assert cleaned["pages"][2]["page_number"] == 3
    assert counts["duplicate_page"] == 1


def test_cleaning_removes_repeated_running_title_but_preserves_repeated_warning() -> None:
    pages = []
    for page_number in range(1, 4):
        pages.append({
            "page_number": page_number,
            "text": "**Router Model AX1 User Manual**\n**WARNING:** Disconnect power.\nBody %d" % page_number,
            "spans": [],
        })
    cleaned, _ = clean_document({"pages": pages})
    assert "Router Model AX1" in cleaned["pages"][0]["text"]
    assert all("Router Model AX1" not in page["text"] for page in cleaned["pages"][1:])
    assert all("WARNING" in page["text"] for page in cleaned["pages"])


def test_cleaning_does_not_mistake_late_cross_references_for_contents() -> None:
    references = "\n".join(f'- “Procedure {index}” on page {index}' for index in range(1, 10))
    document = {
        "pages": [
            {"page_number": 1, "text": "# Introduction\nManual body.", "spans": []},
            {"page_number": 96, "text": references, "spans": []},
        ]
    }
    cleaned, _ = clean_document(document)
    assert cleaned["pages"][1]["excluded_from_chunking"] is False
    assert "Procedure 1" in cleaned["pages"][1]["text"]


def test_cleaning_removes_underscore_leaders_and_layout_only_lines() -> None:
    document = {
        "pages": [{
            "page_number": 1,
            "text": "Heading________________20\n.\n--------------------\n•\nKeep this warning.",
            "spans": [],
        }]
    }
    cleaned, counts = clean_document(document)
    text = cleaned["pages"][0]["text"]
    assert "Heading 20" in text
    assert "________________" not in text
    assert "\n.\n" not in f"\n{text}\n"
    assert "----------------" not in text
    assert "•" in text
    assert counts["standalone_artifact"] == 1
    assert counts["layout_rule"] == 1
