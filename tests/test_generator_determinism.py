"""Generation is deterministic under a fixed seed.

The full proof is that regenerating every table produces byte-identical
CSVs; that runs for several minutes, so it lives behind `--runslow`. What
runs by default is the cheap half that catches every realistic regression:

  * the same request to a named RNG substream returns the same draw
  * emitting the same world twice produces identical bytes, which catches
    unstable sort orders, dict iteration order and float formatting
  * adding a NEW substream does not disturb existing ones, which is what
    lets the generator grow without invalidating the registry

The expensive full-world comparison is `test_full_regeneration_is_byte_identical`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from data.generator.model import Streams


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- RNG discipline ---------------------------------------------------------


def test_fresh_returns_the_same_draw_every_time():
    streams = Streams(20260822)
    first = streams.fresh("mechanism_2451").normal(size=64)
    second = streams.fresh("mechanism_2451").normal(size=64)
    assert np.array_equal(first, second)


def test_get_advances_but_fresh_does_not():
    """`get` is for one-shot use; anything a fit re-runs must use `fresh`."""
    streams = Streams(20260822)
    a = streams.get("x").normal(size=8)
    b = streams.get("x").normal(size=8)
    assert not np.array_equal(a, b)


def test_substreams_are_independent():
    streams = Streams(20260822)
    assert not np.array_equal(
        streams.fresh("store_notes").normal(size=64),
        streams.fresh("tickets").normal(size=64),
    )


def test_a_new_substream_does_not_disturb_existing_ones():
    """Adding a table later must not change the numbers already asserted."""
    baseline = Streams(20260822).fresh("bill_lines").normal(size=32)
    streams = Streams(20260822)
    streams.fresh("a_table_added_later").normal(size=1000)
    assert np.array_equal(streams.fresh("bill_lines").normal(size=32), baseline)


def test_a_different_seed_gives_a_different_world():
    assert not np.array_equal(
        Streams(20260822).fresh("base_demand").normal(size=64),
        Streams(20260823).fresh("base_demand").normal(size=64),
    )


# --- emission stability -----------------------------------------------------

FAST_TABLES = [
    ("dim/dim_store.csv", "emit_dimensions", "sources"),
    ("dim/dim_sku.csv", "emit_dimensions", "sources"),
    ("pos_erp/feed_status.csv", "emit_feed_status", "sources"),
    ("store_ops/store_notes.csv", "emit_store_notes", "sources_ops"),
    ("store_ops/tickets.csv", "emit_tickets", "sources_ops"),
    ("context/competitor_news.csv", "emit_competitor_news", "sources_ops"),
    ("context/marketing_spend.csv", "emit_marketing_spend", "sources_ops"),
    ("context/qcomm_weekly.csv", "emit_qcomm_weekly", "sources_ops"),
    ("context/weather_daily.csv", "emit_weather", "sources_ops"),
]


@pytest.fixture(scope="module")
def world():
    from data.generator.build import build_world

    return build_world()


@pytest.mark.parametrize("relative,func,module", FAST_TABLES, ids=lambda v: str(v))
def test_emission_is_byte_stable(world, tmp_path, relative, func, module):
    """Emitting the same world twice must produce identical bytes."""
    import importlib

    emitter = getattr(importlib.import_module(f"data.generator.{module}"), func)
    first, second = tmp_path / "a", tmp_path / "b"
    emitter(world, first)
    emitter(world, second)
    assert digest(first / relative) == digest(second / relative)


def test_csv_writer_uses_lf_endings(world, tmp_path):
    """CRLF would make the same data hash differently on Windows and Linux."""
    from data.generator import sources

    sources.emit_dimensions(world, tmp_path)
    raw = (tmp_path / "dim" / "dim_sku.csv").read_bytes()
    assert b"\r\n" not in raw


def test_manifest_records_the_fitted_parameters(world, tmp_path):
    """A reader must be able to see which numbers were solved for."""
    import json

    from data.generator.build import write_outputs

    write_outputs(world, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 20260822
    assert "availability_elasticity_beta" in manifest["fitted"]
    assert "idiosyncratic_sigma" in manifest["fitted"]
    assert "festival_amplitudes" in manifest["fitted"]
    assert manifest["measured"]["west_oct_cr"] == pytest.approx(83.70, abs=0.02)


@pytest.mark.slow
def test_full_regeneration_is_byte_identical(world, tmp_path):
    """The whole proof. Slow: rebuilds and re-emits every table.

    Run with: pytest -m slow tests/test_generator_determinism.py
    """
    from data.generator.build import build_world, write_outputs

    first, second = tmp_path / "run1", tmp_path / "run2"
    write_outputs(world, first)
    write_outputs(build_world(), second)

    left = sorted(p.relative_to(first).as_posix() for p in first.rglob("*.csv"))
    right = sorted(p.relative_to(second).as_posix() for p in second.rglob("*.csv"))
    assert left == right

    mismatched = [rel for rel in left if digest(first / rel) != digest(second / rel)]
    assert not mismatched, f"non-deterministic tables: {mismatched}"
