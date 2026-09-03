# Friday

Friday is an evidence-grounded troubleshooting copilot for laptops, desktops,
Wi-Fi routers, and printers. A user describes what stopped working in natural
language; Friday searches the relevant manufacturer manuals, preserves the
diagnostic context, and returns the safest useful next check with a document,
page, and section citation.

It is designed for two people: an everyday device owner who needs clear,
practical guidance, and a new support technician who wants to learn a reliable,
source-backed troubleshooting process.

## Why this project exists

Troubleshooting advice is often either too generic or impossible to verify.
Friday treats the manual as the source of truth and makes the evidence visible
alongside the conversation. If the local corpus cannot support a safe answer,
the system abstains instead of inventing a repair step.

The core interaction is deliberately small:

```text
describe the symptom → find the matching evidence → take one safe check → report the result
```

## What is implemented

- Official-manual corpus covering 21 public manufacturer documents across
  computers, routers, and printers (see `config/source_registry.json`).
- PDF inspection and native text parsing with page-aware source metadata.
- Deterministic cleanup for repeated headers, footers, broken lines, and layout
  noise while retaining citation coordinates in the raw representation.
- Structure-aware chunk generation for sections, procedures, parent/child
  context, troubleshooting tables, and exact identifiers such as error codes.
- Hybrid retrieval combining Qdrant vector search, local Granite embeddings,
  BM25 lexical search, exact-identifier lookup, metadata filters, rank fusion,
  and parent-context expansion.
- Local persistent Qdrant storage behind a small vector-index abstraction.
- LiteLLM answer layer with provider-selectable streaming chat completions.
- Structured troubleshooting state that carries the selected device,
  observations, completed checks, and current diagnostic step across turns.
- Citation-aware answers and explicit unsupported-question handling.
- Next.js landing page and troubleshooting casebook interface.
- Sarvam Saaras realtime STT and Bulbul streaming TTS integration for the voice
  path, including cancellation hooks for interruption handling.
- Retrieval and conversation-policy evaluation suites with Recall@5, MRR,
  citation coverage, abstention accuracy, latency percentiles, and diagnostic
  interaction checks.

## Architecture

```text
                    public manufacturer manuals
                                  │
             inspect → parse → clean → structure-aware chunks
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
          BM25 / exact IDs                  Granite embeddings
                 │                                 │
                 └──────────────┬──────────────────┘
                                │
                 Qdrant + rank fusion + parent context
                                │
                    cited evidence or abstention
                                │
                    LiteLLM streaming answer layer
                                │
               Next.js text UI / Sarvam voice interface
```

The service boundaries are intentionally independent:

- `backend/src/copilot/ingestion` owns parsing, metadata, cleaning, assets, and
  chunk generation.
- `backend/src/copilot/retrieval` owns lexical search, embeddings, Qdrant,
  caching, indexing, and retrieval metrics.
- `backend/src/copilot/answering` owns provider calls, prompts, diagnostic
  state, tools, and citations.
- `backend/src/copilot/voice` owns the Sarvam STT/TTS bridge and timing hooks.
- `frontend/app` contains the public landing page and the interactive casebook.

## Technology

| Area | Implementation |
| --- | --- |
| Frontend | Next.js, React, TypeScript |
| API | FastAPI, Pydantic |
| Retrieval store | Qdrant with persistent local storage |
| Dense retrieval | Granite embedding model with optional AVX2 INT8 ONNX runtime |
| Lexical retrieval | BM25 and exact identifier matching |
| LLM gateway | LiteLLM with provider configuration in YAML |
| Voice | Sarvam Saaras realtime STT and Bulbul streaming TTS |
| Documents | Firecrawl `pdf-inspector`, native page-aware parsing |
| Runtime | Docker Compose or local Python/Node processes |
| Quality | pytest, mypy, Ruff, GitHub Actions, deterministic evals |

## Run locally

The repository keeps secrets in `backend/.env`; non-secret runtime settings are
in `backend/config.yml`.

```bash
cp backend/.env.example backend/.env
# Add the provider keys you want to use to backend/.env.

make qdrant-up
make backend-venv
```

Start the API in one terminal:

```bash
PYTHONPATH=backend/src backend/.venv/bin/uvicorn copilot.main:app --reload --port 8000
```

Start the frontend in another:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The casebook is available
at [http://localhost:3000/app](http://localhost:3000/app), the API health check
is at [http://localhost:8000/health](http://localhost:8000/health), and Qdrant
runs at `http://localhost:6333`.

### Docker Compose

```bash
cp backend/.env.example backend/.env
# Add API keys to backend/.env when using a hosted LLM or Sarvam voice.

docker compose up -d
```

This starts the frontend, FastAPI backend, and persistent Qdrant service on
ports `3000`, `8000`, and `6333`. The existing index is not rebuilt on startup.
To stop the stack:

```bash
docker compose down
```

After creating a new Qdrant volume or regenerating chunks, index the current
vector-retrieval chunks once:

```bash
docker compose run --rm backend python -m copilot.retrieval.indexer
```

## Rebuild the document pipeline

After adding or replacing PDFs under `data/manuals`, run:

```bash
make prepare   # native parse, metadata registry, deterministic cleanup
make chunk     # structure-aware retrieval chunks
make assets    # optional content-addressed manual figures
make index-vectors
```

Or run the complete workflow:

```bash
make ingest
```

The active path is text-only native parsing. Mixed PDFs are classified and
their OCR-required pages are recorded, but OCR is intentionally deferred so
the default workflow remains reliable on a CPU-only development laptop.

## Evaluation

Retrieval evaluation is deterministic and kept separate from application code:

```bash
make eval-retrieval
make eval-retrieval-optimized
make eval-conversation-policy
```

The retrieval benchmark reports Recall@5, MRR, citation-ready hit rate,
abstention accuracy, and P50/P70/P99/max retrieval latency. The conversation
policy benchmark checks that the assistant preserves durable observations,
avoids needless repeated checks, gives a concrete next action, explains why it
matters, and abstains when the corpus cannot support a safe answer.

Run the automated checks with:

```bash
pytest
mypy backend/src
ruff check .
ruff format --check .
```

## Design constraints

Friday follows a few non-negotiable rules:

1. Every technical instruction must be traceable to a document, page, and
   section.
2. The assistant asks for missing observations instead of silently assuming
   them.
3. It gives one safe, useful diagnostic move at a time, while allowing the
   model to decide whether the next turn should solve, advance, clarify, or
   abstain.
4. Warnings, prerequisites, and procedure order remain part of the evidence.
5. Direct lookups stay on hybrid retrieval; graph traversal is reserved for
   genuinely relational, multi-hop diagnosis.
6. Raw user audio is not stored by default.

Medical, automotive, aviation, high-voltage equipment, autonomous repair, and
unverified community advice are outside the project scope.

## Current status

The local ingestion, chunking, vector indexing, hybrid retrieval, cited text
answering, evaluation harness, Docker runtime, and frontend landing plus casebook are in
place. Voice transport and provider response quality still depend on the
configured external API keys and require representative end-to-end latency
measurement before production claims are made.

Deferred by design: OCR on mixed-PDF pages, production image filtering, speculative
retrieval on partial transcripts, and GraphRAG. See `docs/runtime-boundaries.md`.

The next engineering focus is hardening the end-to-end conversation and voice
experience while keeping retrieval local, observable, and independently
benchmarkable.
