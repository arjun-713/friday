"""Extract embedded PDF images and build a content-addressed image registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .filter import classify_image, mark_perceptual_duplicates

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
    rejected_assets: list[dict[str, Any]] = []
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
        placements = _image_placements(pdf_path)
        for page, suffix, data in extracted:
            digest = hashlib.sha256(data).hexdigest()
            suffix = suffix or ".bin"
            occurrence = {
                "document_id": document_id,
                "document_title": metadata.get("document_title", pdf_path.stem),
                "source_file": source_file,
                "page": page,
                "chunk_ids": sorted(chunks_by_document_page.get((document_id, page), set())),
            }
            boxes = placements.get((page, hashlib.sha256(data).hexdigest()))
            if boxes:
                occurrence["image_boxes"] = boxes
            asset = assets.setdefault(
                digest,
                {
                    "asset_id": digest,
                    "suffix": suffix,
                    "mime_type": mimetypes.guess_type(f"asset{suffix}")[0] or "application/octet-stream",
                    "size_bytes": len(data),
                    "_data": data,
                    "occurrences": [],
                },
            )
            if occurrence not in asset["occurrences"]:
                asset["occurrences"].append(occurrence)

    for asset in assets.values():
        asset["occurrences"].sort(key=lambda item: (item["document_id"], item["page"], item["source_file"]))
        asset["features"] = classify_image(asset["_data"])
        if len(asset["occurrences"]) >= 3 and asset["features"]["width"] * asset["features"]["height"] < 50_000:
            asset["features"]["classification"] = "reject"
            asset["features"]["quality_score"] -= 4
            asset["features"]["reasons"].append("repeated_small_asset")
    mark_perceptual_duplicates(assets)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    kept_assets: dict[str, dict[str, Any]] = {}
    for asset_id, asset in assets.items():
        if asset["features"]["classification"] == "reject":
            rejected_assets.append(asset)
            continue
        data = asset["_data"]
        suffix = str(asset["suffix"])
        destination = output_root / f"{asset_id}{suffix}"
        destination.write_bytes(data)
        asset["path"] = destination.relative_to(output_root.parent.parent).as_posix()
        asset.pop("_data", None)
        asset.pop("suffix", None)
        kept_assets[asset_id] = asset
    manifest = {
        "version": MANIFEST_VERSION,
        "assets": dict(sorted(kept_assets.items())),
        "rejected_assets": [
            {key: value for key, value in asset.items() if key not in {"occurrences", "_data", "suffix"}}
            | {"occurrence_count": len(asset.get("occurrences", []))}
            for asset in rejected_assets
        ],
        "skipped_pages": skipped_pages,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def images_for_chunks(manifest: dict[str, Any], chunk_ids: set[str]) -> list[dict[str, Any]]:
    """Return at most one high-confidence image linked to retrieved chunks."""

    references: dict[str, dict[str, Any]] = {}
    for asset_id, asset in manifest.get("assets", {}).items():
        for occurrence in asset.get("occurrences", []):
            if chunk_ids.intersection(occurrence.get("chunk_ids", [])):
                reference = {
                    "asset_id": asset_id,
                    "path": asset["path"],
                    "mime_type": asset["mime_type"],
                    "document_title": occurrence["document_title"],
                    "page": occurrence["page"],
                    "source_file": occurrence["source_file"],
                }
                if asset.get("features", {}).get("classification") != "valid":
                    continue
                existing = references.get(asset_id)
                if existing is None or (reference["page"], reference["document_title"]) < (
                    existing["page"],
                    existing["document_title"],
                ):
                    references[asset_id] = reference
    ordered = sorted(references.values(), key=lambda item: (item["document_title"], item["page"], item["asset_id"]))
    return ordered[:1]


def _extract_pdf_images(pdf_path: Path) -> tuple[list[tuple[int, str, bytes]], list[tuple[int, str]]]:
    extracted: list[tuple[int, str, bytes]] = []
    skipped: list[tuple[int, str]] = []
    with tempfile.TemporaryDirectory(prefix="friday-pdf-images-") as temporary:
        temporary_path = Path(temporary)
        try:
            listing = subprocess.run(
                ["pdfimages", "-list", str(pdf_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            page_by_image_number = _parse_image_pages(listing.stdout)
            prefix = temporary_path / "asset"
            subprocess.run(
                ["pdfimages", "-all", str(pdf_path), str(prefix)],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as error:
            fallback = _extract_images_with_pymupdf(pdf_path)
            if fallback:
                return fallback, []
            return [], [(0, type(error).__name__)]
        except subprocess.CalledProcessError as error:
            return [], [(0, type(error).__name__)]
        for path in sorted(temporary_path.glob("asset-*")):
            match = re.search(r"-(\d+)(?:-\d+)?\.[^.]+$", path.name)
            image_number = int(match.group(1)) if match else len(extracted)
            page = page_by_image_number.get(image_number)
            if page is not None:
                extracted.append((page, path.suffix.lower(), path.read_bytes()))
    return extracted, skipped


def _extract_images_with_pymupdf(pdf_path: Path) -> list[tuple[int, str, bytes]]:
    """Fallback extractor for PDFs that exceed the native pdfimages timeout."""

    try:
        import pymupdf
    except ImportError:
        return []
    extracted: list[tuple[int, str, bytes]] = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            for image in page.get_images(full=True):
                try:
                    payload = document.extract_image(image[0])
                    data = payload["image"]
                    suffix = f".{payload.get('ext', 'bin')}"
                except (KeyError, RuntimeError, ValueError):
                    continue
                extracted.append((page_number, suffix, data))
    return extracted


def _parse_image_pages(listing: str) -> dict[int, int]:
    pages: dict[int, int] = {}
    for line in listing.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit() or len(parts) < 2:
            continue
        pages[int(parts[1])] = int(parts[0])
    return pages


def _image_placements(pdf_path: Path) -> dict[tuple[int, str], list[dict[str, float]]]:
    """Map embedded image content to its page coordinates for chunk-level relevance."""

    try:
        import pymupdf as fitz
    except ImportError:
        return {}
    placements: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            for image in page.get_images(full=True):
                xref = image[0]
                try:
                    extracted = document.extract_image(xref)
                    digest = hashlib.sha256(extracted["image"]).hexdigest()
                    rectangles = page.get_image_rects(xref)
                except (KeyError, RuntimeError, ValueError):
                    continue
                for rectangle in rectangles:
                    placements[(page_number, digest)].append(
                        {
                            "x0": round(float(rectangle.x0), 2),
                            "y0": round(float(rectangle.y0), 2),
                            "x1": round(float(rectangle.x1), 2),
                            "y1": round(float(rectangle.y1), 2),
                        }
                    )
    return dict(placements)


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
    print(
        f"Indexed {len(manifest['assets'])} kept images across {occurrence_count} page occurrences; "
        f"rejected {len(manifest['rejected_assets'])} assets"
    )


if __name__ == "__main__":
    main()
