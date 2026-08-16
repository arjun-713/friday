"""Extract embedded PDF images and build a content-addressed image registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def build_image_manifest(
    manuals_root: Path = Path("data/manuals"),
    chunks_root: Path = Path("data/chunks"),
    output_root: Path = Path("data/assets/images"),
    manifest_path: Path = Path("data/assets/image_manifest.json"),
) -> dict[str, Any]:
    chunks_by_document_page, document_metadata = _load_chunk_index(chunks_root)
    assets: dict[str, dict[str, Any]] = {}
    skipped_pages: list[dict[str, Any]] = []
    for pdf_path in sorted(manuals_root.rglob("*.pdf")):
        source_file = pdf_path.as_posix()
        metadata = document_metadata.get(source_file, {})
        document_id = str(metadata.get("document_id") or pdf_path.stem)
        try:
            extracted, skipped = _extract_pdf_images(pdf_path)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            skipped_pages.append({"source_file": source_file, "page": None, "reason": type(error).__name__})
            continue
        skipped_pages.extend({"source_file": source_file, "page": page, "reason": reason} for page, reason in skipped)
        for page, suffix, data in extracted:
            digest = hashlib.sha256(data).hexdigest()
            suffix = suffix or ".bin"
            destination = output_root / f"{digest}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(data)
            occurrence = {
                "document_id": document_id,
                "document_title": metadata.get("document_title", pdf_path.stem),
                "source_file": source_file,
                "page": page,
                "chunk_ids": sorted(chunks_by_document_page.get((document_id, page), set())),
            }
            asset = assets.setdefault(
                digest,
                {
                    "asset_id": digest,
                    "path": destination.relative_to(output_root.parent.parent).as_posix(),
                    "mime_type": mimetypes.guess_type(destination.name)[0] or "application/octet-stream",
                    "size_bytes": len(data),
                    "occurrences": [],
                },
            )
            if occurrence not in asset["occurrences"]:
                asset["occurrences"].append(occurrence)

    for asset in assets.values():
        asset["occurrences"].sort(key=lambda item: (item["document_id"], item["page"], item["source_file"]))
    manifest = {
        "version": MANIFEST_VERSION,
        "assets": dict(sorted(assets.items())),
        "skipped_pages": skipped_pages,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def images_for_chunks(manifest: dict[str, Any], chunk_ids: set[str]) -> list[dict[str, Any]]:
    """Return unique image references whose page occurrence belongs to retrieved chunks."""

    references: list[dict[str, Any]] = []
    for asset_id, asset in manifest.get("assets", {}).items():
        for occurrence in asset.get("occurrences", []):
            if chunk_ids.intersection(occurrence.get("chunk_ids", [])):
                references.append(
                    {
                        "asset_id": asset_id,
                        "path": asset["path"],
                        "mime_type": asset["mime_type"],
                        "document_title": occurrence["document_title"],
                        "page": occurrence["page"],
                        "source_file": occurrence["source_file"],
                    }
                )
    return sorted(references, key=lambda item: (item["document_title"], item["page"], item["asset_id"]))


def _extract_pdf_images(pdf_path: Path) -> tuple[list[tuple[int, str, bytes]], list[tuple[int, str]]]:
    extracted: list[tuple[int, str, bytes]] = []
    skipped: list[tuple[int, str]] = []
    with tempfile.TemporaryDirectory(prefix="friday-pdf-images-") as temporary:
        temporary_path = Path(temporary)
        page_count = _pdf_page_count(pdf_path)
        for page in range(1, page_count + 1):
            prefix = temporary_path / f"page-{page:05d}"
            try:
                subprocess.run(
                    ["pdfimages", "-all", "-f", str(page), "-l", str(page), str(pdf_path), str(prefix)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                skipped.append((page, type(error).__name__))
                continue
            for path in sorted(temporary_path.glob(f"page-{page:05d}-*")):
                extracted.append((page, path.suffix.lower(), path.read_bytes()))
    return extracted, skipped


def _pdf_page_count(pdf_path: Path) -> int:
    result = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True, timeout=20)
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"pdfinfo did not report a page count for {pdf_path}")


def _load_chunk_index(chunks_root: Path) -> tuple[dict[tuple[str, int], set[str]], dict[str, dict[str, Any]]]:
    chunks_by_document_page: dict[tuple[str, int], set[str]] = defaultdict(set)
    document_metadata: dict[str, dict[str, Any]] = {}
    for chunk_file in sorted(chunks_root.rglob("*.jsonl")):
        for line in chunk_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            document = chunk["document"]
            source_file = str(chunk["evidence"][0]["source_file"])
            document_id = str(document["document_id"])
            document_metadata[source_file] = {
                "document_id": document_id,
                "document_title": document.get("title", document_id),
            }
            for page in chunk.get("pages", [chunk["page"]]):
                chunks_by_document_page[(document_id, int(page))].add(str(chunk["chunk_id"]))
    return chunks_by_document_page, document_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuals-root", type=Path, default=Path("data/manuals"))
    parser.add_argument("--chunks-root", type=Path, default=Path("data/chunks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/assets/images"))
    parser.add_argument("--manifest", type=Path, default=Path("data/assets/image_manifest.json"))
    args = parser.parse_args()
    manifest = build_image_manifest(args.manuals_root, args.chunks_root, args.output_root, args.manifest)
    occurrence_count = sum(len(asset["occurrences"]) for asset in manifest["assets"].values())
    print(f"Indexed {len(manifest['assets'])} unique images across {occurrence_count} page occurrences")


if __name__ == "__main__":
    main()
