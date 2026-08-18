PYTHON ?= python
BACKEND_PYTHONPATH := backend/src
MODULE := PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m

.PHONY: prepare chunk assets ingest qdrant-up qdrant-down compose-up compose-down index-vectors benchmark-retrieval benchmark-retrieval-optimized benchmark-embedding export-embedding eval-retrieval eval-retrieval-optimized eval-conversation-policy backend-venv

# Run after adding or replacing manuals in data/manuals.
prepare:
	$(MODULE) copilot.ingestion.parsing.native
	$(MODULE) copilot.ingestion.metadata.registry
	$(MODULE) copilot.ingestion.cleaning.runner

# Build all retrieval-oriented chunk representations from cleaned manuals.
chunk:
	$(MODULE) copilot.ingestion.chunking.runner

# Extract content-addressed PDF images and map them to document pages/chunks.
assets:
	$(MODULE) copilot.ingestion.assets.images

# Complete RAG ingestion workflow for the current manual corpus.
ingest: prepare chunk assets

qdrant-up:
	docker compose -f docker-compose.qdrant.yml up -d qdrant

qdrant-down:
	docker compose -f docker-compose.qdrant.yml down

compose-up:
	docker compose up --build

compose-down:
	docker compose down

index-vectors:
	PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m copilot.retrieval.indexer

benchmark-retrieval:
	PYTHONPATH=backend/src backend/.venv/bin/python -m copilot.retrieval.real_benchmark

benchmark-retrieval-optimized:
	EMBEDDING_BACKEND=onnx EMBEDDING_MODEL=data/models/granite-small-r2-onnx EMBEDDING_MODEL_FILE=onnx/model_int8-avx2.onnx PYTHONPATH=backend/src backend/.venv/bin/python -m copilot.retrieval.real_benchmark

eval-retrieval:
	PYTHONPATH=backend/src backend/.venv/bin/python -m eval.run_retrieval --candidate-limit 32 --dense-weight 1 --lexical-weight 1.5 --rrf-k 30 --abstention-dense-threshold 0.84

eval-retrieval-optimized:
	EMBEDDING_BACKEND=onnx EMBEDDING_MODEL=data/models/granite-small-r2-onnx EMBEDDING_MODEL_FILE=onnx/model_int8-avx2.onnx PYTHONPATH=backend/src backend/.venv/bin/python -m eval.run_retrieval --candidate-limit 32 --dense-weight 1 --lexical-weight 1.5 --rrf-k 30 --abstention-dense-threshold 0.84

eval-conversation-policy:
	PYTHONPATH=backend/src backend/.venv/bin/python -m eval.run_conversation_policy

benchmark-embedding:
	PYTHONPATH=backend/src backend/.venv/bin/python -m copilot.retrieval.embedding_benchmark

export-embedding:
	PYTHONPATH=backend/src backend/.venv/bin/python -m copilot.retrieval.export_embedding

backend-venv:
	uv venv --clear --python 3.11 backend/.venv
	uv pip install --index-strategy unsafe-best-match --python backend/.venv/bin/python -r backend/requirements-cpu.txt
