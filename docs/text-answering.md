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

The default answer path uses LiteLLM (`openai/gpt-6-luna` via `OPENAI_API_KEY`, temperature `0.1`, reasoning off).
`EvidenceOnlyAnswerGenerator` remains only as a fallback when `llm.enabled` is false.
Streaming emits `retrieval`, optional `tool`, `token`, and `complete` SSE events; citations are rendered
in a separate source row and stripped from chat/TTS text by the frontend.

Both `/v1/troubleshoot` and `/v1/troubleshoot/stream` run the same planner-first flow:
state-aware retrieval → bounded tool-assisted planner (`solve`/`advance`/`clarify`/`abstain`)
→ validated `turn` with evidence-grounded `observation_request.options` rendered as clickable
buttons → streamed `turn.response` tokens → persisted session state. The stream never bypasses
the planner, so follow-ups advance instead of repeating and buttons always match the text.

The planner call itself streams (`stream_agent_turn`). A response gate buffers
the JSON, validates every non-`response` field the moment it is complete, and
only then releases decoded `response` characters, so text renders and voice
speaks while the turn is still generating without ever emitting unvalidated
prose. The planner schema orders `response` last for this reason; setting
`planner_response_position: first` with `streaming_gate: optimistic`
(opt-in experiment flags) speaks response bytes earlier at the cost of a
measured abandonment rate (see `docs/rag-performance.md`).

Every `complete` event carries an `llm_calls` array with per-provider-call
records (`ttft_ms`, `latency_ms`, token counts, `tokens_per_sec`,
`tool_names`, retry notes), so benchmarks can split queue/scheduling delay
from generation speed without log scraping. Abstained completions include the
calls made before giving up.

Each citation contains the document, model, document version, page, section,
and source URL. The service preserves retrieval timings so later LLM and voice
latency can be measured separately from retrieval.

Supported providers (config-only switch in `backend/config.yml`):

- OpenAI GPT-6 Luna (`openai/gpt-6-luna`, `OPENAI_API_KEY`) — default.
- Groq OpenAI-compatible models via `GROQ_API_KEY`.
- Any LiteLLM OpenAI-compatible model via `llm.model`, `llm.api_base`, `llm.api_key_env`.
- Sarvam conversation models are supported but have no prompt-cache contract.

Run the API from the repository root after Qdrant is available:

```bash
PYTHONPATH=backend/src backend/.venv/bin/uvicorn friday.main:app --reload
```

The service reads `CHUNKS_ROOT` when set and defaults to `data/chunks`. The
embedding and Qdrant settings continue to use their existing environment
variables.

## Bounded agentic retrieval

When `llm.agentic_enabled` is true, Luna can request up to two read-only tools
for a turn: `search_manual` (refined symptom when supplied evidence is generic),
`find_error_code` (exact code/message/LED pattern), and `open_manual_page`
(surrounding procedure and state tables). Tool results are scoped to the selected
manufacturer and model, merged into the planner evidence, and validated before
streaming. The capped diagnostic state already travels in every planner prompt,
so no state-fetch tool round trip is needed.
`observation_request.options` must be copied or closely paraphrased from retrieved
manual text (generic Yes/No/Not sure excepted); invented states are rejected.
The backend owns state mutation and citation validation; the model cannot execute
arbitrary code or device commands. Set `llm.max_tool_calls: 0` to disable the
agent loop while retaining planner-first streaming.
