"""Offline synthetic dataset builder (run manually, never in CI).

Exports retrieval contexts from Friday's own chunk store, generates goldens
with ``deepeval generate --method contexts``, then validates and stamps them.

The generated file is a *candidate* set: synthetic expected outputs are not
ground truth. Ranking metrics may use them; correctness claims need human
review of the sampled cases first. Generated cases are always marked
``provenance: synthetic`` and kept separate from curated human cases.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTEXTS_FILE = REPO_ROOT / "eval" / "evals" / "datasets" / "_contexts_export.json"
OUTPUT_FILE = REPO_ROOT / "eval" / "evals" / "datasets" / "synthetic_rag.json"

TARGET_DEVICES = [
    ("TP-Link", "Archer C6"),
    ("ASUS", "RT-AX3000"),
    ("Brother", "HL-L2350DW"),
    ("Dell", "Latitude 7490"),
]
CONTEXTS_PER_DEVICE = 8


def max_contexts() -> int | None:
    """Cap exported contexts via FRIDAY_SYNTH_CONTEXTS (small runs validate
    the pipeline without burning the shared TPM budget)."""

    import os

    raw = os.getenv("FRIDAY_SYNTH_CONTEXTS")
    return max(1, int(raw)) if raw else None


def _is_boilerplate(chunk) -> bool:
    """Drop compliance/footer chunks: they generate nonsense QA pairs (e.g.
    troubleshooting advice from a restricted-substance exemption footer)."""

    text = f"{chunk.section}\n{chunk.content[:400]}".lower()
    markers = (
        "restricted substance",
        "rohs",
        "exemption",
        "certification",
        "compliance",
        "trademark",
        "fcc",
        "declaration of conformity",
        "recycl",
        "warranty",
    )
    return any(marker in text for marker in markers)


def export_contexts() -> list[list[str]]:
    from friday.paths import chunks_dir
    from friday.retrieval.indexer import load_all_chunks

    chunks = load_all_chunks(chunks_dir())
    by_device: dict[tuple[str, str], list] = defaultdict(list)
    for chunk in chunks:
        if len(chunk.content) < 200:
            continue
        if chunk.kind.value not in ("section", "procedure"):
            continue
        if _is_boilerplate(chunk):
            continue
        key = (chunk.document.manufacturer or "", chunk.document.model or "")
        by_device[key].append(chunk)
    contexts: list[list[str]] = []
    for manufacturer, model in TARGET_DEVICES:
        device_chunks = sorted(
            by_device.get((manufacturer, model), []),
            key=lambda chunk: (chunk.document.document_id, chunk.ordinal),
        )
        # Spread across documents/sections instead of one manual chapter.
        stride = max(1, len(device_chunks) // CONTEXTS_PER_DEVICE)
        for chunk in device_chunks[::stride][:CONTEXTS_PER_DEVICE]:
            header = f"[{manufacturer} {model} manual, p.{chunk.page} {chunk.section}]"
            contexts.append([f"{header}\n{chunk.content[:1500]}"])
    return contexts


def main() -> int:
    contexts = export_contexts()
    cap = max_contexts()
    if cap is not None:
        contexts = contexts[:cap]
    print(f"exported {len(contexts)} contexts")
    CONTEXTS_FILE.write_text(json.dumps(contexts, ensure_ascii=False), encoding="utf-8")
    cmd = [
        str(REPO_ROOT / "backend" / ".venv" / "bin" / "deepeval"),
        "generate",
        "--method",
        "contexts",
        "--variation",
        "single-turn",
        "--contexts-file",
        str(CONTEXTS_FILE),
        # NOTE: --num-goldens is scratch-only; contexts-method volume is driven
        # by --max-goldens-per-context (one golden per exported context here).
        "--max-goldens-per-context",
        "1",
        "--scenario",
        "Home users and office users troubleshooting routers, printers, and laptops with the Friday assistant",
        "--task",
        "Answer the troubleshooting question accurately using only the provided manual context",
        "--input-format",
        "A specific symptom, error indicator, or how-to question naming the device",
        "--expected-output-format",
        "One safe next diagnostic step grounded in the provided context",
        "--include-expected",
        "--model",
        "gpt-4o",
        "--output-dir",
        str(OUTPUT_FILE.parent),
        "--file-name",
        "_synthetic_raw",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    raw_path = OUTPUT_FILE.parent / "_synthetic_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    goldens = raw["goldens"] if isinstance(raw, dict) and "goldens" in raw else raw
    seen: set[str] = set()
    cleaned: list[dict] = []
    for golden in goldens:
        question = str(golden.get("input", "")).strip()
        answer = str(golden.get("expected_output", "")).strip()
        if len(question) < 20 or len(answer) < 20 or question in seen:
            continue
        seen.add(question)
        golden["additional_metadata"] = {
            **(golden.get("additional_metadata") or {}),
            "provenance": "synthetic:deepeval-generate-contexts (candidate, needs human review)",
        }
        cleaned.append(golden)
    OUTPUT_FILE.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    raw_path.unlink(missing_ok=True)
    CONTEXTS_FILE.unlink(missing_ok=True)
    print(f"kept {len(cleaned)}/{len(goldens)} synthetic goldens -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
