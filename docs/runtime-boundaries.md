# Runtime boundaries

Friday's current runtime is intentionally narrower than the final research
roadmap:

- The active query path is hybrid retrieval (Qdrant dense search plus BM25)
  followed by one structured diagnostic agent.
- The agent chooses `solve`, `advance`, `clarify`, or `abstain` from the
  retrieved manufacturer evidence and persisted session state.
- Sarvam conversation, Saaras realtime STT, and Bulbul streaming TTS are
  optional provider integrations. Text mode remains usable without voice.
- OCR is an optional ingestion extra and is disabled for the current corpus.
  Native text parsing is the default, and OCR pages retain routing metadata
  for a future capable machine.
- Speculative retrieval is deferred until final-transcript retrieval has a
  stable latency and reuse benchmark.
- GraphRAG is deferred until direct hybrid RAG has documented failure cases;
  it will be used only for relational or multi-hop diagnostic questions.
- Authentication, raw-audio persistence, autonomous repair actions, and
  community advice are outside the current runtime boundary.

The next production-quality milestone is a verified text and voice vertical
slice with durable diagnostic state, cancellation-safe turns, citations, and
conversation-level regression cases. Later retrieval research must preserve
the same citation and abstention contracts.
