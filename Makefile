# CaseFile.ai — see CLAUDE.md §"Repository layout"
#
# Recipes shell out to python for filesystem work so the targets behave the
# same on Windows, macOS and Linux.

PY ?= python
PYTEST ?= $(PY) -m pytest

.PHONY: help install seed demo test reset

help:
	@echo "make install — install the project and dev extras"
	@echo "make seed    — build the DuckDB warehouse from the generator (arrives at P2)"
	@echo "make demo    — run the full offline demo, MOCK_LLM=true (arrives at P8)"
	@echo "make test    — run the test suite"
	@echo "make reset   — delete generated artefacts and caches"

install:
	$(PY) -m pip install -e ".[dev]"

seed:
	@$(PY) -c "import pathlib,subprocess,sys; p=pathlib.Path('data/generator/base_series.py'); print('seed: not implemented yet - the generator arrives at P2.') if not p.exists() else subprocess.check_call([sys.executable,'-m','data.generator.base_series'])"

demo:
	@$(PY) -c "import pathlib,subprocess,sys,os; p=pathlib.Path('api/main.py'); print('demo: not implemented yet - the demo path arrives at P8.') if not p.exists() else subprocess.check_call([sys.executable,'-m','api.main'],env={**os.environ,'MOCK_LLM':'true'})"

test:
	$(PYTEST)

reset:
	@$(PY) -c "import pathlib,shutil; [p.unlink() for p in pathlib.Path('data').glob('*.duckdb*') if p.is_file()]; [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree('.pytest_cache',ignore_errors=True); [p.unlink() for p in pathlib.Path('data/raw').glob('*') if p.is_file() and p.name!='.gitkeep']; print('reset: warehouse, raw CSVs and caches cleared')"
