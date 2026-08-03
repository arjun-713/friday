# Full-Duplex Troubleshooting Copilot

## Goal

Build a voice-first technical troubleshooting assistant grounded in official manuals and support documents.

The user should be able to speak naturally, interrupt the assistant while it is talking, provide new observations, and receive one verified troubleshooting step at a time.

## Initial domain

- Laptops and desktops
- Wi-Fi routers
- Printers
- Public manufacturer manuals and support documents

## Core rules

- Every technical instruction must cite a document, page, and section.
- Never invent a repair step or silently fill missing information.
- Ask for missing observations instead of assuming them.
- Give one diagnostic step at a time.
- Preserve warnings, prerequisites, and procedure order.
- Prefer deterministic code over unnecessary LLM calls.
- Use GraphRAG only for genuinely relational or multi-hop questions.
- Measure end-to-end latency, not only model latency.
- Do not create a multi-agent swarm.

## Incremental phases

Build these phases in order. Do not introduce a later phase before the earlier phase is reliable.

1. **Document ingestion:** classify PDFs with `pdf-inspector`; OCR only required pages; preserve page numbers, headings, tables, warnings, captions, numbered procedures, coordinates, and source text; normalize repeated headers/footers, broken lines, and hyphenation; create structure-aware chunks with document, model, section, page, content type, parser, and confidence metadata; maintain BM25, vector, metadata-filter, and raw-document retrieval layers.
2. **Text-only base RAG:** implement query analysis, metadata filtering, BM25/vector retrieval, rank fusion, reranking, context assembly, cited answers, and abstention. Support factual, error-code, procedural, symptom, and unanswerable questions. Evaluate Recall@5, MRR, citation precision/completeness, procedure ordering, and unsupported-claim rate. Do not proceed until retrieval failures are understood and documented.
3. **Streaming voice:** add WebRTC audio, streaming speech-to-text, technical-vocabulary correction, end-of-turn detection, streaming LLM output, sentence-level streaming TTS, live transcript, visible citations, and timestamps for speech start, final transcript, retrieval, first LLM token, first TTS audio, and playback.
4. **Full duplex:** use states `IDLE`, `LISTENING`, `THINKING`, `SPEAKING`, `INTERRUPTING`, `RECOVERING`, and `ERROR`. Every turn has a unique `turn_id` and cancellation token. Interruptions stop playback, cancel TTS/LLM/retrieval, preserve completed state, and process the new utterance. Classify interruption as correction, new information, clarification, stop, repeat, or acknowledgement.
5. **Speculative retrieval:** retrieve from meaningful stable partial transcripts, cancel stale work, delay expensive reranking, reuse candidates only when appropriate, and never speak before adequate end-of-turn confidence. Compare final-only, fixed-interval, and confidence-gated retrieval by lead time, reuse, waste, Recall@5, and speech-end-to-first-audio latency.
6. **Agentic diagnosis:** maintain structured diagnostic state and use narrow tools: `search_manual`, `find_error_code`, `open_manual_page`, `find_procedure`, `get_procedure_step`, `get_safety_warnings`, `record_observation`, `mark_test_completed`, `compare_causes`, and `generate_session_summary`. Choose the safest, most informative next test, explain it briefly, then wait for the result.
7. **GraphRAG:** model `Device`, `Component`, `Symptom`, `ErrorCode`, `Cause`, `DiagnosticTest`, `Procedure`, `ProcedureStep`, `Warning`, and `ManualPage`, with evidence-backed relationships for components, indications, tests, causes, procedures, steps, warnings, ordering, and manual support. Use graph traversal only for multi-symptom diagnosis, test selection, procedure dependencies, and evidence-path explanations.

## Suggested stack

- Frontend: Next.js
- Backend: FastAPI
- Voice: LiveKit or native WebRTC
- Parsing: pdf-inspector
- OCR: PaddleOCR or Docling
- Vector search: Qdrant or pgvector
- Lexical search: BM25
- Graph: Neo4j
- Orchestration: custom state machine or LangGraph
- Observability: OpenTelemetry or Langfuse
- Deployment: Docker Compose

## Engineering standards

- Use typed schemas across service boundaries.
- Keep ingestion, retrieval, voice, orchestration, and graph modules independent.
- Version prompts, schemas, datasets, and evaluation results.
- Add unit tests for parsing, retrieval, routing, cancellation, and state updates.
- Add integration tests for complete troubleshooting conversations.
- Never hardcode credentials.
- Do not store raw user audio by default.
- Avoid abstractions not required by the current phase.

## Initial completion criteria

- Ingest at least 20 manuals.
- Achieve at least 90% Recall@5 on a manually verified benchmark.
- Provide citations for every technical instruction.
- Detect unsupported questions instead of hallucinating.
- Stop assistant audio within 200 ms median after interruption.
- Preserve state after user corrections.
- Complete at least 30 verified diagnostic scenarios.
- Demonstrate measurable latency improvement from speculative retrieval.
- Show GraphRAG gains specifically on multi-hop questions.

## Out of scope

- Medical, automotive, aviation, or high-voltage equipment
- Autonomous repair actions
- Unverified community advice
- Training a speech foundation model
- GraphRAG for simple factual questions
- Multiple agents debating each other
