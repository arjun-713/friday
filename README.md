# Full-Duplex Troubleshooting Copilot

Scaffold for an evidence-grounded troubleshooting assistant. The repository is currently organized around the first delivery slice: document ingestion and the contracts required by later retrieval and voice phases.

## Layout

```text
backend/       FastAPI service and ingestion domain
frontend/      Next.js operator/query shell
data/          local runtime directories; source files are ignored
docs/          phase boundaries and ingestion contract
tests/         backend unit tests
```

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
uvicorn copilot.main:app --reload
```

The ingestion adapter uses Firecrawl's local `pdf-inspector` bindings for PDF classification, per-page Markdown, and positioned text. It records OCR-required pages but does not run OCR yet. Native parsing and deterministic text cleanup are implemented; chunking, BM25, vector search, and raw-page rendering remain later stages.

Parsed outputs mirror the source taxonomy under `data/raw/{computers,routers,printers}`. Re-run the text-only parser with:

```bash
PYTHONPATH=backend/src python scripts/parse_text_pdfs.py
```

The active corpus path is native-only parsing for all PDFs, including mixed PDFs. It does not run OCR:

```bash
PYTHONPATH=backend/src python scripts/parse_native_pdfs.py
```

Create auditable cleaned output without changing the raw JSON:

```bash
PYTHONPATH=backend/src python scripts/clean_raw_pdfs.py
```

Cleaned JSON is retrieval-oriented. It removes layout-only formatting and boilerplate, excludes contents/empty/duplicate pages from future chunking without renumbering them, and records every removal. Positioned spans remain in `data/raw` and are referenced by source file plus page number instead of being duplicated in `data/cleaned`.
