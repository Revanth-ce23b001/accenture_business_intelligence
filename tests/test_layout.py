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
    "engine/contribution",
    "engine/confidence",
    "engine/abstain",
    "engine/recommend",
    "llm",
    "llm/fixtures",
    "security",
    "telemetry",
    "api",
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
    """VALIDATE, QUALIFY, GATHER and ADJUDICATE have engine packages.

    VERDICT deliberately has none yet — see the note in docs/ARCHITECTURE.md.
    CLAUDE.md calls the five stage names the module names, but the repository
    layout in the same file lists no `verdict/` package. Flagged, not resolved.
    """
    for stage in ("validate", "qualify", "gather", "adjudicate"):
        assert (ROOT / "engine" / stage).is_dir()
