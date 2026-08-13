PYTHON ?= python
BACKEND_PYTHONPATH := backend/src
MODULE := PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m

.PHONY: prepare chunk ingest qdrant-up qdrant-down

# Run after adding or replacing manuals in data/manuals.
prepare:
	$(MODULE) copilot.ingestion.parsing.native
	$(MODULE) copilot.ingestion.metadata.registry
	$(MODULE) copilot.ingestion.cleaning.runner

# Build all retrieval-oriented chunk representations from cleaned manuals.
chunk:
	$(MODULE) copilot.ingestion.chunking.runner

# Complete RAG ingestion workflow for the current manual corpus.
ingest: prepare chunk

qdrant-up:
	docker compose -f docker-compose.qdrant.yml up -d qdrant

qdrant-down:
	docker compose -f docker-compose.qdrant.yml down
