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

The current adapters are deterministic placeholders. They define the seams for `pdf-inspector`, OCR, persistent storage, BM25, vector search, and raw-page rendering without pretending those integrations are complete.
