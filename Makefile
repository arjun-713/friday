PYTHON ?= python
BACKEND_PYTHONPATH := backend/src
MODULE := PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m

.PHONY: prepare chunk assets ingest ingest-summary bootstrap qdrant-up qdrant-down compose-up compose-down index-vectors benchmark-retrieval benchmark-retrieval-optimized benchmark-embedding benchmark-voice benchmark-conversation export-embedding eval-retrieval eval-retrieval-optimized eval-conversation-policy eval-deepeval-fast eval-deepeval-full eval-deepeval-agent eval-deepeval-conversation eval-deepeval-calibrate eval-deepeval-voice eval-deepeval-sim eval-deepeval-synthetic smoke-text smoke-voice backend-venv

# Run after adding or replacing manuals in data/manuals.
prepare:
	$(MODULE) friday.ingestion.parsing.native
	$(MODULE) friday.ingestion.metadata.registry
	$(MODULE) friday.ingestion.cleaning.runner

# Build all retrieval-oriented chunk representations from cleaned manuals.
chunk:
	$(MODULE) friday.ingestion.chunking.runner

# Extract content-addressed PDF images and map them to document pages/chunks.
assets:
	$(MODULE) friday.ingestion.assets.images

# Complete RAG ingestion workflow for the current manual corpus.
ingest: prepare chunk assets

# Verify generated-data state from a clean clone.
ingest-summary:
	$(PYTHON) scripts/ingest_summary.py

# Clean bootstrap for new contributors: ingest, verify, then index once Qdrant is up.
bootstrap: qdrant-up ingest ingest-summary index-vectors

qdrant-up:
	docker compose -f docker-compose.qdrant.yml up -d qdrant

qdrant-down:
	docker compose -f docker-compose.qdrant.yml down

compose-up:
	docker compose up --build

compose-down:
	docker compose down

index-vectors:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m friday.retrieval.indexer

benchmark-retrieval:
	PYTHONPATH=backend/src backend/.venv/bin/python -m friday.retrieval.real_benchmark

benchmark-retrieval-optimized:
	EMBEDDING_BACKEND=onnx EMBEDDING_MODEL=data/models/granite-small-r2-onnx EMBEDDING_MODEL_FILE=onnx/model_int8-avx2.onnx PYTHONPATH=backend/src backend/.venv/bin/python -m friday.retrieval.real_benchmark

eval-retrieval:
	PYTHONPATH=backend/src backend/.venv/bin/python -m eval.run_retrieval --candidate-limit 32 --dense-weight 1 --lexical-weight 1.5 --rrf-k 30 --abstention-dense-threshold 0.84

eval-retrieval-optimized:
	EMBEDDING_BACKEND=onnx EMBEDDING_MODEL=data/models/granite-small-r2-onnx EMBEDDING_MODEL_FILE=onnx/model_int8-avx2.onnx PYTHONPATH=backend/src backend/.venv/bin/python -m eval.run_retrieval --candidate-limit 32 --dense-weight 1 --lexical-weight 1.5 --rrf-k 30 --abstention-dense-threshold 0.84

eval-conversation-policy:
	PYTHONPATH=backend/src backend/.venv/bin/python -m eval.run_conversation_policy

# DeepEval harness (eval/evals/). Source backend/.env first so OPENAI_API_KEY
# (judge) is present; it is never committed. Tiers:
# - fast: PR gate (deterministic guards + 8-case RAG smoke + short
#   conversations + judge calibration). No simulator, voice, or synthetic.
# - full: all curated cases and conversations.
# - voice/sim/synthetic: separate heavyweight suites, run manually/nightly.
EVAL_ENV := EMBEDDING_BACKEND=onnx EMBEDDING_MODEL=data/models/granite-small-r2-onnx EMBEDDING_MODEL_FILE=onnx/model_int8-avx2.onnx

eval-deepeval-fast:
	$(EVAL_ENV) FRIDAY_EVAL_TIER=fast backend/.venv/bin/deepeval test run eval/evals/test_rag.py eval/evals/test_conversation.py eval/evals/test_calibration.py --identifier friday-fast

eval-deepeval-full:
	$(EVAL_ENV) FRIDAY_EVAL_TIER=full backend/.venv/bin/deepeval test run eval/evals/test_rag.py eval/evals/test_conversation.py eval/evals/test_agent.py eval/evals/test_calibration.py --identifier friday-full

eval-deepeval-agent:
	$(EVAL_ENV) backend/.venv/bin/deepeval test run eval/evals/test_agent.py --identifier friday-agent

eval-deepeval-conversation:
	$(EVAL_ENV) FRIDAY_EVAL_TIER=full backend/.venv/bin/deepeval test run eval/evals/test_conversation.py --identifier friday-conversation

eval-deepeval-calibrate:
	backend/.venv/bin/deepeval test run eval/evals/test_calibration.py --identifier friday-calibrate

eval-deepeval-voice:
	$(EVAL_ENV) backend/.venv/bin/deepeval test run eval/evals/test_voice.py --identifier friday-voice

eval-deepeval-sim:
	$(EVAL_ENV) FRIDAY_EVAL_SIM=1 backend/.venv/bin/deepeval test run eval/evals/test_simulator.py --identifier friday-sim

eval-deepeval-synthetic:
	PYTHONPATH=backend/src backend/.venv/bin/python eval/evals/datasets/generate_synthetic.py

smoke-text:
	PYTHONPATH=backend/src backend/.venv/bin/python scripts/smoke_text.py

smoke-voice:
	@test -n "$(AUDIO)" || (echo "Usage: make smoke-voice AUDIO=/path/to/16khz-mono-pcm" && exit 2)
	PYTHONPATH=backend/src backend/.venv/bin/python scripts/smoke_voice.py "$(AUDIO)"

benchmark-voice:
	@test -n "$(AUDIO)" || (echo "Usage: make benchmark-voice AUDIO=/path/to/16khz-mono-pcm [TRIALS=10]" && exit 2)
	PYTHONPATH=backend/src:. backend/.venv/bin/python scripts/benchmark_voice.py "$(AUDIO)" --trials "$(or $(TRIALS),10)"

benchmark-conversation:
	PYTHONPATH=backend/src backend/.venv/bin/python scripts/benchmark_conversation.py

benchmark-embedding:
	PYTHONPATH=backend/src backend/.venv/bin/python -m friday.retrieval.embedding_benchmark

export-embedding:
	PYTHONPATH=backend/src backend/.venv/bin/python -m friday.retrieval.export_embedding

backend-venv:
	uv venv --clear --python 3.11 backend/.venv
	uv pip install --index-strategy unsafe-best-match --python backend/.venv/bin/python -r backend/requirements-cpu.txt
