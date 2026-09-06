#!/usr/bin/env python
"""Benchmark PP-OCRv6-small on 40 representative mixed-PDF pages."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pdf_inspector

from .ocr import PpOcrAdapter

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "tmp/paddlex")


SAMPLE = {
    "lenovo-thinkpad-t14-gen-2-p14s-gen-2-hardware-maintenance-manual.pdf": 14,
    "lenovo-thinkpad-t480-hardware-maintenance-manual.pdf": 13,
    "lenovo-thinkpad-t480s-hardware-maintenance-manual.pdf": 13,
}


def flagged_pages(path: Path) -> list[int]:
    result = pdf_inspector.detect_pdf(str(path))
    return [int(page) for page in result.pages_needing_ocr]


def render_page(pdf: Path, page: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{pdf.stem}-p{page:04d}"
    subprocess.run(
        ["pdftoppm", "-f", str(page), "-l", str(page), "-r", "300", "-png", "-singlefile", str(pdf), str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return output.with_suffix(".png")


def main() -> None:
    source_dir = Path("data/manuals/computers")
    image_dir = Path("tmp/pdfs/ocr_benchmark")
    result_dir = Path("data/raw/ocr_benchmark")
    result_dir.mkdir(parents=True, exist_ok=True)
    adapter = PpOcrAdapter(tier="small", device="cpu")
    fallback = PpOcrAdapter(tier="medium", device="cpu")
    records: list[dict[str, object]] = []

    for filename, requested_count in SAMPLE.items():
        pdf = source_dir / filename
        pages = flagged_pages(pdf)
        if not pages:
            raise RuntimeError(f"No OCR-routed pages found in {pdf}")
        stride = max(1, len(pages) // requested_count)
        selected = pages[::stride][:requested_count]
        for page in selected:
            started = time.perf_counter()
            image = render_page(pdf, page, image_dir)
            result = adapter.recognize(image, page)
            record = result.model_dump()
            if result.confidence < 0.85:
                medium_result = fallback.recognize(image, page)
                if medium_result.confidence > result.confidence:
                    result = medium_result
                    record = result.model_dump()
                    record["fallback_from"] = "PP-OCRv6-small"
                    record["fallback_reason"] = "small_confidence_below_0.85"
            record.update({"source_pdf": str(pdf), "seconds": round(time.perf_counter() - started, 3)})
            records.append(record)
            print(f"{pdf.name} page {page}: {len(result.items)} items, confidence={result.confidence:.3f}")

    output = result_dir / "pp_ocrv6_small_benchmark.json"
    output.write_text(json.dumps({"model": adapter.model_name, "pages": records}, indent=2), encoding="utf-8")
    print(f"wrote {output} ({len(records)} pages)")
    shutil.rmtree(image_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
