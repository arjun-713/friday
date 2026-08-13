PYTHON ?= python
BACKEND_PYTHONPATH := backend/src
MODULE := PYTHONPATH=$(BACKEND_PYTHONPATH) $(PYTHON) -m

.PHONY: parse metadata clean chunk ingest quality test lint format typecheck

parse:
	$(MODULE) copilot.ingestion.parsing.native

metadata:
	$(MODULE) copilot.ingestion.metadata.registry

clean:
	$(MODULE) copilot.ingestion.cleaning.runner

chunk:
	$(MODULE) copilot.ingestion.chunking.runner

ingest: parse metadata clean chunk

test:
	cd backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

lint:
	cd backend && ruff check src tests

format:
	cd backend && ruff format --check src tests

typecheck:
	cd backend && mypy

quality: lint format test typecheck
