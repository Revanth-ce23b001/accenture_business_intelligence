# CaseFile.ai — see CLAUDE.md §"Repository layout"
#
# Recipes shell out to python for filesystem work so the targets behave the
# same on Windows, macOS and Linux.

PY ?= python
PYTEST ?= $(PY) -m pytest

.PHONY: help install generate seed demo test reset

help:
	@echo "make install — install the project and dev extras"
	@echo "make generate — rebuild data/raw from the synthetic data generator"
	@echo "make seed    — build the DuckDB warehouse from data/raw"
	@echo "make demo    — serve the API offline, MOCK_LLM=true, on 127.0.0.1:8000"
	@echo "make test    — run the test suite"
	@echo "make reset   — delete generated artefacts and caches"

install:
	$(PY) -m pip install -e ".[dev]"

# Regenerate data/raw from the generator, then build the DuckDB warehouse.
# `generate` is the slow half and only needs re-running when the generator
# or its config changes; `seed` alone rebuilds the warehouse from the raw
# CSVs already on disk.
generate:
	$(PY) -m data.generator.build

seed:
	$(PY) -m engine.warehouse.load

# The offline demo (CLAUDE.md rule 8): MOCK_LLM=true, no network, every
# model call replayed from llm/fixtures/. One worker, because the case
# store is process-local - see api/store.py.
demo:
	@$(PY) -c "import os,subprocess,sys; subprocess.check_call([sys.executable,'-m','api.main'], env={**os.environ,'MOCK_LLM':'true'})"

test:
	$(PYTEST)

reset:
	@$(PY) -c "import pathlib,shutil; [p.unlink() for p in pathlib.Path('data').glob('*.duckdb*') if p.is_file()]; [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree('.pytest_cache',ignore_errors=True); [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('data/raw').iterdir() if p.is_dir()]; [p.unlink() for p in pathlib.Path('data/raw').rglob('*') if p.is_file() and p.name!='.gitkeep']; print('reset: warehouse, raw CSVs and caches cleared — run make generate && make seed')"
