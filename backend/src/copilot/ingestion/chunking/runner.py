"""Generate validated JSONL chunks from cleaned manuals."""

import json
from pathlib import Path

from ..metadata.registry import ResolvedSource
from ..models import SourceDocument
from .generator import generate_chunks


def main() -> None:
    cleaned_root = Path("data/cleaned")
    manifest_path = Path("data/raw/source_registry.json")
    output_root = Path("data/chunks")
    if not manifest_path.exists():
        raise FileNotFoundError(
            "source manifest is missing; run PYTHONPATH=backend/src python -m copilot.ingestion.metadata.registry first"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {item["source_file"]: ResolvedSource.model_validate(item) for item in manifest["documents"]}
    report: list[dict[str, object]] = []

    for path in sorted(cleaned_root.glob("*/*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        source = sources.get(document["source_file"])
        if source is None:
            raise ValueError(f"cleaned document is missing from source manifest: {document['source_file']}")
        source_document = SourceDocument(
            document_id=source.document_id,
            title=source.title,
            manufacturer=source.manufacturer,
            model=source.model,
            version=source.version,
            source_url=source.source_url,
            retrieved_at=source.retrieved_at,
            sha256=source.sha256,
        )
        chunks = generate_chunks(document, source_document)
        output_path = output_root / path.parent.name / f"{path.stem}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n" for chunk in chunks),
            encoding="utf-8",
        )
        by_strategy: dict[str, int] = {}
        by_profile: dict[str, int] = {}
        for chunk in chunks:
            by_strategy[chunk.strategy.value] = by_strategy.get(chunk.strategy.value, 0) + 1
            for profile in chunk.retrieval_profiles:
                by_profile[profile.value] = by_profile.get(profile.value, 0) + 1
        report.append(
            {
                "source_file": document["source_file"],
                "output_file": str(output_path),
                "chunks": len(chunks),
                "by_strategy": by_strategy,
                "by_retrieval_profile": by_profile,
            }
        )
        print(f"chunked {path} -> {output_path} ({len(chunks)} chunks)")

    report_path = output_root / "chunk_report.json"
    report_path.write_text(json.dumps({"chunker": "chunking.v1", "documents": report}, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
