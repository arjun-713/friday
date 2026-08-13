"""Build auditable source metadata from the tracked manual registry."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, Field


class RegistryEntry(TypedDict):
    title: str
    manufacturer: str
    model: str
    source_url: str


class ResolvedSource(BaseModel):
    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    title: str = Field(min_length=1)
    manufacturer: str = Field(min_length=1)
    model: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_registry(path: Path) -> dict[str, RegistryEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source registry must contain an object")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retrieved_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def build_registry(
    manual_root: Path,
    registry_path: Path,
    output_path: Path,
) -> list[ResolvedSource]:
    registry = load_registry(registry_path)
    resolved: list[ResolvedSource] = []
    seen: set[str] = set()

    for manual_path in sorted(manual_root.glob("*/*.pdf")):
        relative_path = manual_path.relative_to(manual_root).as_posix()
        entry = registry.get(relative_path)
        if entry is None:
            raise ValueError(f"manual is missing from source registry: {relative_path}")
        seen.add(relative_path)
        digest = _sha256(manual_path)
        resolved.append(
            ResolvedSource(
                document_id=f"{manual_path.parent.name}-{manual_path.stem}",
                source_file=str(manual_path),
                title=entry["title"],
                manufacturer=entry["manufacturer"],
                model=entry["model"],
                version=f"sha256:{digest}",
                source_url=entry["source_url"],
                retrieved_at=_retrieved_at(manual_path),
                sha256=digest,
            )
        )

    unknown_entries = sorted(set(registry) - seen)
    if unknown_entries:
        raise ValueError(f"source registry has no matching local manual: {unknown_entries[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"schema_version": "source-registry.v1", "documents": [item.model_dump() for item in resolved]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return resolved


def main() -> None:
    sources = build_registry(
        manual_root=Path("data/manuals"),
        registry_path=Path("config/source_registry.json"),
        output_path=Path("data/raw/source_registry.json"),
    )
    print(f"resolved metadata for {len(sources)} manuals")
    print("wrote data/raw/source_registry.json")


if __name__ == "__main__":
    main()
