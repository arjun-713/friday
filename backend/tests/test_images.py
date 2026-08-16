from copilot.ingestion.assets.images import images_for_chunks


def test_images_for_chunks_returns_unique_content_addressed_references() -> None:
    manifest = {
        "version": 1,
        "assets": {
            "abc123": {
                "asset_id": "abc123",
                "path": "assets/images/abc123.png",
                "mime_type": "image/png",
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
