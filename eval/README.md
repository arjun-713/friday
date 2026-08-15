# Retrieval evaluation

This folder contains the deterministic, manually verified retrieval benchmark. It is deliberately separate from application code and does not use an LLM judge.

Run it from the repository root after the local Qdrant collection has been indexed:

```bash
PYTHONPATH=backend/src backend/.venv/bin/python -m eval.run_retrieval
```

The checked-in Make target runs the calibrated development configuration: 32 candidates per source, lexical RRF weight 1.5, rank constant 30, and a dense-confidence abstention threshold of 0.84. Tune these values only on a development split and validate on held-out manuals before changing the defaults.

The runner writes the full case-by-case report to `data/index/retrieval_eval.json`. That generated report is ignored by Git. It reports Recall@5, MRR, abstention accuracy, citation-ready evidence coverage, latency percentiles, metrics by question type, and individual retrieval failures.

The runner performs one unscored warm-up retrieval before measuring cases, so latency percentiles represent warm serving behavior. Model cold-start time is measured separately by the embedding benchmark.

The expected chunk IDs in `retrieval_cases.jsonl` are evidence labels, not answer text. Update them only after opening the referenced manual evidence and verifying the document, model, page, and section.
