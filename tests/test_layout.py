"""The scaffold matches CLAUDE.md §"Repository layout"."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "semantic_layer",
    "semantic_layer/kpis",
    "semantic_layer/playbooks",
    "data",
    "data/generator",
    "data/generator/config",
    "data/generator/scenarios",
    "data/raw",
    "engine",
    "engine/validate",
    "engine/qualify",
    "engine/gather",
    "engine/adjudicate",
    "engine/verdict",
    "engine/contribution",
    "engine/confidence",
    "engine/abstain",
    "engine/recommend",
    "engine/learn",
    "llm",
    "llm/fixtures",
    "security",
    "telemetry",
    "api",
    "api/routes",
    "frontend",
    "tests",
    "docs",
]

EXPECTED_FILES = [
    "CLAUDE.md",
    "README.md",
    "Makefile",
    "docker-compose.yml",
    "pyproject.toml",
    "engine/contracts.py",
    "llm/provider.py",
    "api/main.py",
    "api/app.py",
    "docs/ARCHITECTURE.md",
    "docs/NUMBER_REGISTRY.md",
]


@pytest.mark.parametrize("relative", EXPECTED_DIRS)
def test_directory_exists(relative):
    assert (ROOT / relative).is_dir(), f"missing directory: {relative}"


@pytest.mark.parametrize("relative", EXPECTED_FILES)
def test_file_exists(relative):
    assert (ROOT / relative).is_file(), f"missing file: {relative}"


def test_engine_stage_packages_match_the_locked_stage_names():
    """All five locked stage names have engine packages.

    CLAUDE.md §"Architecture" calls the five names the module names. Four
    of them had packages from P5 to P14 and `verdict/` did not, which this
    test recorded as flagged-not-resolved. P15 resolved it: the stages had
    to be composed before the API could run one, and the composition is
    engine work, not transport.
    """
    for stage in ("validate", "qualify", "gather", "adjudicate", "verdict"):
        assert (ROOT / "engine" / stage).is_dir()
