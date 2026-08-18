# Retrieval evaluation

This folder contains the deterministic, manually verified retrieval benchmark. It is deliberately separate from application code and does not use an LLM judge.

Run it from the repository root after the local Qdrant collection has been indexed:

```bash
PYTHONPATH=backend/src backend/.venv/bin/python -m eval.run_retrieval
```

The checked-in Make target runs the calibrated development configuration: 32 candidates per source, lexical RRF weight 1.5, rank constant 30, and a dense-confidence abstention threshold of 0.84. Tune these values only on a development split and validate on held-out manuals before changing the defaults.

For the live-demo deployment configuration, use:

```bash
make eval-retrieval-optimized
```

That target uses the locally exported AVX2 INT8 ONNX Granite artifact. The portable `make eval-retrieval` target uses the default PyTorch backend, so its dense ranking and latency can differ slightly.

The runner writes the full case-by-case report to `data/index/retrieval_eval.json`. That generated report is ignored by Git. It reports Recall@5, MRR, abstention accuracy, citation-ready evidence coverage, latency percentiles, metrics by question type, and individual retrieval failures.

The runner performs 20 uncached warm-up retrievals before measuring cases, clears the warm-up result cache, and reports the first warm-up latency separately as model cold-start time. Latency percentiles represent uncached warm serving behavior. Use `--warmup 0` only when intentionally measuring cold-start behavior.

The expected chunk IDs in `retrieval_cases.jsonl` are evidence labels, not answer text. Update them only after opening the referenced manual evidence and verifying the document, model, page, and section.

## Conversation policy evaluation

`conversation_cases.jsonl` contains manually verified, structured diagnostic traces. They exercise the agent policy without an LLM judge: valid advances need an action, a diagnosis-specific reason, and an explicit distinction between plausible causes; durable facts cannot be asked again without an action-backed recheck; transitions must be acknowledged; and an unsupported issue must terminate in abstention instead of an endless interview.

Run it with:

```bash
make eval-conversation-policy
```

The report includes pass rate, turns to first useful intervention, the longest clarification streak, state-transition acknowledgement rate, tool-call count, and each policy violation. Add real, manually reviewed conversations here as the agent is tested; do not rewrite a case merely to accommodate a behavior regression.
