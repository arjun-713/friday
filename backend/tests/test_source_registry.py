import json
from pathlib import Path

from copilot.ingestion.metadata.registry import build_registry, load_registry


def test_source_registry_resolves_hash_and_content_version(tmp_path: Path) -> None:
    manual_root = tmp_path / "manuals"
    manual_path = manual_root / "routers" / "example.pdf"
    manual_path.parent.mkdir(parents=True)
    manual_path.write_bytes(b"manual")
    registry_path = tmp_path / "source_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "routers/example.pdf": {
                    "title": "Example Manual",
                    "manufacturer": "Example",
                    "model": "Example 1",
                    "source_url": "https://example.test/manual.pdf",
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = build_registry(manual_root, registry_path, tmp_path / "resolved.json")

    assert len(resolved) == 1
    assert resolved[0].version == f"sha256:{resolved[0].sha256}"
    assert resolved[0].retrieved_at.endswith("+00:00")


def test_source_registry_rejects_unregistered_manual(tmp_path: Path) -> None:
    manual_root = tmp_path / "manuals"
    manual_path = manual_root / "routers" / "unregistered.pdf"
    manual_path.parent.mkdir(parents=True)
    manual_path.write_bytes(b"manual")
    registry_path = tmp_path / "source_registry.json"
    registry_path.write_text("{}", encoding="utf-8")

    try:
        build_registry(manual_root, registry_path, tmp_path / "resolved.json")
    except ValueError as error:
        assert "missing from source registry" in str(error)
    else:
        raise AssertionError("unregistered manuals must fail closed")


def test_source_registry_file_contains_all_expected_entries() -> None:
    registry = load_registry(Path(__file__).resolve().parents[2] / "config" / "source_registry.json")
    assert len(registry) == 21
