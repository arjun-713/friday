# Text-only troubleshooting layer

The backend exposes `POST /v1/troubleshoot` as the first answer contract above
the hybrid retriever.

Example request:

```json
{
  "query": "The router cannot connect to Wi-Fi",
  "manufacturer": "TP-Link",
  "model": "Archer C6"
}
```

The response is either:

- `status: "ready"` with source-backed evidence, citations, and the
  deterministic evidence-only answer; or
- `status: "abstained"` with a reason and missing device observations.

The current answer generator returns the first verified parent context
verbatim. It does not paraphrase, invent a repair step, or call an LLM. This
keeps the evidence boundary testable before a DeepSeek answer generator is
introduced.

Each citation contains the document, model, document version, page, section,
and source URL. The service preserves retrieval timings so later LLM and voice
latency can be measured separately from retrieval.

Run the API from the repository root after Qdrant is available:

```bash
PYTHONPATH=backend/src backend/.venv/bin/uvicorn copilot.main:app --reload
```

The service reads `CHUNKS_ROOT` when set and defaults to `data/chunks`. The
embedding and Qdrant settings continue to use their existing environment
variables.
