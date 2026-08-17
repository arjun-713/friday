import io

from PIL import Image, ImageDraw

from copilot.ingestion.assets.filter import classify_image
from copilot.ingestion.assets.images import images_for_chunks


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_filter_rejects_hollow_border_asset() -> None:
    image = Image.new("RGB", (240, 140), "white")
    ImageDraw.Draw(image).rectangle((2, 2, 237, 137), outline="black", width=4)

    features = classify_image(_png(image))

    assert features["classification"] == "reject"
    assert "hollow_rectangle" in features["reasons"]


def test_images_for_chunks_returns_unique_content_addressed_references() -> None:
    manifest = {
        "version": 1,
        "assets": {
            "abc123": {
                "asset_id": "abc123",
                "path": "assets/images/abc123.png",
                "mime_type": "image/png",
                "features": {"classification": "valid", "quality_score": 4},
                "occurrences": [
                    {
                        "document_id": "manual",
                        "document_title": "Example Manual",
                        "source_file": "data/manuals/example.pdf",
                        "page": 4,
                        "chunk_ids": ["child-1"],
                    }
                ],
            }
        },
    }

    references = images_for_chunks(manifest, {"child-1"})

    assert references == [
        {
            "asset_id": "abc123",
            "path": "assets/images/abc123.png",
            "mime_type": "image/png",
            "document_title": "Example Manual",
            "page": 4,
            "source_file": "data/manuals/example.pdf",
        }
    ]


def test_images_for_chunks_does_not_serve_uncertain_assets() -> None:
    manifest = {
        "version": 1,
        "assets": {
            "uncertain": {
                "asset_id": "uncertain",
                "path": "assets/images/uncertain.png",
                "mime_type": "image/png",
                "features": {"classification": "uncertain", "quality_score": 1},
                "occurrences": [
                    {
                        "document_title": "Manual",
                        "source_file": "manual.pdf",
                        "page": 2,
                        "chunk_ids": ["chunk"],
                    }
                ],
            }
        },
    }

    assert images_for_chunks(manifest, {"chunk"}) == []
