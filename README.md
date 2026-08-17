# Full-Duplex Troubleshooting Copilot

Evidence-grounded troubleshooting for the manuals in `data/manuals`. Friday keeps
each troubleshooting session scoped to a supported device, retrieves manufacturer
evidence, gives one verified diagnostic check at a time, and retains confirmed
results for the next turn.

## Layout

```text
backend/       FastAPI service and ingestion domain
frontend/      Next.js conversational troubleshooting interface
data/          local runtime directories; source files are ignored
docs/          phase boundaries and ingestion contract
tests/         backend unit tests
```

## Run locally

```bash
make qdrant-up
make backend-venv

cp backend/.env.example backend/.env
# Add SARVAM_API_KEY to backend/.env when using Sarvam conversation or voice.

PYTHONPATH=backend/src backend/.venv/bin/uvicorn copilot.main:app --reload --port 8000
# In a second terminal:
cd frontend && npm install && npm run dev
```

The frontend is served on `http://localhost:3000`; the FastAPI service is served
on `http://localhost:8000`. `/health` and `/v1/devices` do not require an LLM
call. The latter is the authoritative supported-device catalog generated from
the source registry, so the interface cannot select a made-up model.

The ingestion adapter uses Firecrawl's local `pdf-inspector` bindings for PDF
classification, per-page Markdown, and positioned text. It records OCR-required
pages but does not run OCR yet. Native parsing, deterministic text cleanup,
structure-aware chunking, BM25, vector search, and raw-page rendering are in
place.

Parsed outputs mirror the source taxonomy under `data/raw/{computers,routers,printers}`. Re-run the text-only parser with:

```bash
PYTHONPATH=backend/src python -m copilot.ingestion.parsing.text_only
```

The active corpus path is native-only parsing for all PDFs, including mixed PDFs. It does not run OCR:

```bash
PYTHONPATH=backend/src python -m copilot.ingestion.parsing.native
```

## LiteLLM answer layer

The text endpoint uses the non-secret settings in `backend/config.yml` and reads API keys only from the ignored `backend/.env` file. Keep secrets out of YAML and Git. The default configuration targets Sarvam's OpenAI-compatible Conversations endpoint:

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and add SARVAM_API_KEY.
```

The backend sends only retrieved evidence to the model. Responses must cite a retrieved chunk; the server expands that marker into the document title, page, and section, and abstains when the model returns `UNSUPPORTED` or an unknown citation. Provider failures are returned as service-unavailable errors without logging credentials or prompt contents.

## Voice interaction

Voice is a persistent browser-to-backend WebSocket at `/v1/voice`. One click
opens microphone capture and keeps the session listening: Saaras v3 Realtime
returns partial and final transcripts, each final transcript starts the same
retrieval-and-answer turn used by typed chat, and Bulbul v3 streams the approved
diagnostic instruction as PCM audio. The transcript remains in the conversation
after playback. On `vad.speech_start`, Friday cancels the active answer/TTS and
clears queued browser audio before listening to the new turn.

The configured formats are 16 kHz PCM16 for microphone input and 24 kHz PCM16
for playback. API credentials remain backend-only in `backend/.env`; raw user
audio is forwarded for the live turn and is not stored by Friday.

Create auditable cleaned output without changing the raw JSON:

```bash
PYTHONPATH=backend/src python -m copilot.ingestion.cleaning.runner
```

Extract manual figures and build the local content-addressed image registry after chunking:

```bash
make assets
```

The registry is written to `data/assets/image_manifest.json` and the binaries to `data/assets/images/`. Each image is stored once by SHA-256 and records document, page, and matching chunk IDs. These generated assets are intentionally ignored by Git and can be recreated from the source PDFs.

Cleaned JSON is retrieval-oriented. It removes layout-only formatting and boilerplate, excludes contents/empty/duplicate pages from future chunking without renumbering them, and records every removal. Positioned spans remain in `data/raw` and are referenced by source file plus page number instead of being duplicated in `data/cleaned`.
