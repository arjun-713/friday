# Text-only troubleshooting layer

The backend exposes `POST /v1/troubleshoot` and `POST /v1/troubleshoot/stream` above
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

- `status: "ready"` with LiteLLM-generated conversational guidance, structured
  `turn` (`solve`/`advance`/`clarify`/`abstain`), evidence, citations, and images; or
- `status: "abstained"` with a reason and missing device observations.

The default answer path uses LiteLLM (`groq/openai/gpt-oss-120b` via `GROQ_API_KEY`).
`EvidenceOnlyAnswerGenerator` remains only as a fallback when `llm.enabled` is false.
Streaming emits `retrieval`, `token`, and `complete` SSE events; citations are rendered
in a separate source row and stripped from chat/TTS text by the frontend.

Each citation contains the document, model, document version, page, section,
and source URL. The service preserves retrieval timings so later LLM and voice
latency can be measured separately from retrieval.

Supported providers (config-only switch in `backend/config.yml`):

- Groq OpenAI-compatible (`groq/openai/gpt-oss-120b`, `GROQ_API_KEY`) — default.
- Any LiteLLM OpenAI-compatible model via `llm.model`, `llm.api_base`, `llm.api_key_env`.
- Sarvam conversation models are supported but have no prompt-cache contract.

Run the API from the repository root after Qdrant is available:

```bash
PYTHONPATH=backend/src backend/.venv/bin/uvicorn friday.main:app --reload
```

The service reads `CHUNKS_ROOT` when set and defaults to `data/chunks`. The
embedding and Qdrant settings continue to use their existing environment
variables.
