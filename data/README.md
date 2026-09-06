# `data/` — source inputs vs. rebuildable outputs

Only a few files under `data/` are versioned. Everything else is generated
locally by the ingestion pipeline and is ignored by Git (see the `data/`
section in the root `.gitignore`).

## Versioned (committed)

| Path | Purpose |
| --- | --- |
| `voice_benchmark/wifi-router-*.wav` | Original 44.1 kHz stereo field recording (`wifi-router.wav`) and the derived 16 kHz mono fixture used by `make smoke-voice` / `make benchmark-voice`. |
| `voice_benchmark/wifi-router-16khz-mono.pcm` | Raw PCM twin of the 16 kHz fixture. |

The tracked manual manifest lives outside `data/` at
`config/source_registry.json` (21 PDFs). The PDFs themselves are downloaded
with `scripts/download_manuals.sh` into `data/manuals/` and are **not**
committed.

## Generated (ignored, rebuildable)

| Path | Produced by | Rebuild |
| --- | --- | --- |
| `manuals/{computers,routers,printers}/*.pdf` | `scripts/download_manuals.sh` | Re-download |
| `raw/` | `make prepare` (parse + registry) | `make prepare` |
| `cleaned/` | `make prepare` (deterministic cleanup) | `make prepare` |
| `chunks/` | `make chunk` | `make chunk` |
| `assets/images/` + `image_manifest.json` | `make assets` | `make assets` |
| `index/` | eval/benchmark reports, `embedding_manifest.json`, and the runtime `diagnostic_sessions.sqlite3` | `make index-vectors` / re-run evals |
| `models/granite-small-r2-onnx/` | `make export-embedding` | `make export-embedding` |
| `models/huggingface/` | `HF_HOME`/`TRANSFORMERS_CACHE` model cache (see `compose.yml`) | Re-downloaded on demand |

Full bootstrap for a new contributor: `make bootstrap` (Qdrant up, ingest,
verify, index). `scripts/ingest_summary.py` (`make ingest-summary`) verifies
the generated state without rebuilding.

All repository-relative paths resolve through `backend/src/friday/paths.py`
(anchored at `$FRIDAY_ROOT`, defaulting to the checkout root), so the
pipeline works regardless of the invoking working directory.
