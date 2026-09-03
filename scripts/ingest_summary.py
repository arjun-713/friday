"""Summarize ingestion state so a fresh clone can verify its corpus version."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root

    manuals = (
        sorted((root / "data" / "manuals").glob("*/*.pdf"))
        if (root / "data" / "manuals").exists()
        else []
    )
    chunk_files = (
        sorted((root / "data" / "chunks").glob("*/*.jsonl"))
        if (root / "data" / "chunks").exists()
        else []
    )
    report_path = root / "data" / "chunks" / "chunk_report.json"

    chunk_total = 0
    chunker_versions: Counter[str] = Counter()
    for path in chunk_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_total += 1
            chunker_versions[str(payload.get("chunker", "unknown"))] += 1

    registry_path = root / "config" / "source_registry.json"
    registry_count = 0
    if registry_path.is_file():
        try:
            registry_count = len(json.loads(registry_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            registry_count = 0

    summary = {
        "documents_registered": registry_count,
        "manual_pdfs_present": len(manuals),
        "chunk_files": len(chunk_files),
        "chunks_emitted": chunk_total,
        "chunker_versions": dict(chunker_versions),
        "chunk_report_present": report_path.is_file(),
        "corpus_fingerprint": file_sha256(registry_path)
        if registry_path.is_file()
        else None,
        "ready_to_index": bool(chunk_files and chunk_total > 0),
    }
    print(json.dumps(summary, indent=2))
    if not summary["ready_to_index"]:
        print(
            "ingestion incomplete: run `make ingest` then `make index-vectors`",
            flush=True,
        )
        return 1
    if len(chunker_versions) > 1:
        print(
            "warning: mixed chunker versions detected; regenerate with `make chunk`",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
