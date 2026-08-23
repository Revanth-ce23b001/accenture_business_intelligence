"""The seeded calibration ledger — a year of closed cases, laid out.

CLAUDE.md seeds 213 closed cases so the engine has a track record to
calibrate against on day one. This writes them.

WHAT IS DECLARED AND WHAT IS MEASURED, because the difference is the whole
point of the ledger:

  DECLARED  how many cases sit in each raw-confidence band, and how many
            of them turned out right. `config/calibration.yaml`.

  MEASURED  the isotonic map fitted to those cases, the band table it
            implies, the expected calibration error, and the 0.8876 ->
            0.84 step the demo turns on. None of those is written down
            anywhere; they are read back off the seeded rows by
            `engine/confidence/calibration.py`.

WHY THE OUTCOMES ARE NOT DRAWN. Every other table in this generator draws
from a seeded RNG, and this one does not. A ledger is a FIXTURE standing
in for a year of cases nobody ran, and if the outcomes were drawn then the
isotonic map would depend on the seed — the 0.84 assertion would hold by
luck and break the first time a stream was added upstream. The counts are
declared; the map is still emergent.

The one thing the RNG does here is jitter the closed-at dates, which
nothing downstream depends on.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.generator.sources import write_csv

#: A published case that turned out right, and one that did not. Written
#: into `notes` so a reader of the table can see what the row is claiming.
CORRECT_NOTE = "Published; the attribution held when the action was taken."
WRONG_NOTE = "Published; the attribution did not hold on follow-up."
ABSTAINED_NOTE = "Abstained; no attribution was published, so none can be scored."


class LedgerError(ValueError):
    """The declared ledger does not add up."""


def _spread(low: float, high: float, count: int) -> np.ndarray:
    """`count` raw confidences across [low, high), evenly and interior.

    Interior on purpose. A value sitting exactly on a band edge is
    ambiguous to every consumer — the isotonic fit, the band table, and
    anybody reading the CSV — so no case is placed on one.
    """
    if count <= 0:
        return np.array([], dtype=float)
    step = (high - low) / (count + 1)
    return low + step * np.arange(1, count + 1)


def _outcomes(count: int, correct: int) -> list[bool]:
    """Right answers at the bottom of the band, wrong ones at the top.

    TWO REASONS, and the first is the one that matters.

    It is what miscalibration looks like from inside a band. A calibration
    curve that sits below the diagonal is a model scoring higher than it
    performs, and that gap does not switch off at a band edge — inside any
    narrow band the same slope applies, so the cases the model scored
    highest are the ones it was most overconfident about. Putting the
    misses at the top of the band is the within-band expression of the
    same overconfidence the whole ledger exhibits. Scattering them evenly
    would claim the model has no idea which of two cases 0.02 apart is
    shakier, which is a stronger claim than "it runs five points hot".

    It is also what keeps the band whole. Isotonic regression pools
    adjacent points until the fit is non-decreasing, so a run that DECREASES
    — every hit, then every miss — is pooled entirely, and the block it
    forms has exactly the band's declared accuracy. Any other arrangement
    is partly non-decreasing already, survives unpooled, and splits one
    band into two blocks reporting two different numbers for a band that
    is meant to say one thing.

    The consequence is that the fitted map reproduces the declared band
    accuracies exactly. That is not the band table being hardcoded: the
    engine reads none of this, fits from the rows, and computes the table
    and the calibration error itself.
    """
    if not 0 <= correct <= count:
        raise LedgerError(f"{correct} correct out of {count} is not a count")
    return [True] * correct + [False] * (count - correct)


def build_ledger(world) -> pd.DataFrame:
    """The 213 closed cases, as rows, deterministic to the last field."""
    config = world.calibration
    rng = world.streams.fresh("calibration_ledger")
    closed_through = pd.Timestamp(config["closed_through"])
    window_days = int(config["window_months"] * 365.25 / 12)

    published: list[dict] = []
    for band in config["published_bands"]:
        raws = _spread(band["raw_low"], band["raw_high"], int(band["cases"]))
        outcomes = _outcomes(int(band["cases"]), int(band["correct"]))
        for raw, was_correct in zip(raws, outcomes, strict=True):
            published.append(
                {
                    "confidence_raw": float(raw),
                    "was_correct": bool(was_correct),
                    "abstained": False,
                }
            )

    abstained: list[dict] = []
    for band in config["abstained_bands"]:
        for raw in _spread(band["raw_low"], band["raw_high"], int(band["cases"])):
            abstained.append(
                {
                    "confidence_raw": float(raw),
                    "was_correct": None,
                    "abstained": True,
                }
            )

    totals = config["totals"]
    if len(published) != int(totals["published"]):
        raise LedgerError(
            f"the bands lay out {len(published)} published cases, "
            f"not the {totals['published']} declared"
        )
    if len(abstained) != int(totals["abstained"]):
        raise LedgerError(
            f"the bands lay out {len(abstained)} abstained cases, "
            f"not the {totals['abstained']} declared"
        )

    published.sort(key=lambda row: row["confidence_raw"])
    abstained.sort(key=lambda row: row["confidence_raw"])
    _assign_case_types(config, published, abstained)

    rows = published + abstained
    prefix = str(config["case_id_prefix"])
    start = int(config["case_id_start"])
    for index, row in enumerate(rows):
        row["entry_id"] = f"CAL{index + 1:04d}"
        row["case_id"] = f"{prefix}{start + index}"
        offset = int(rng.integers(0, window_days))
        row["closed_at"] = (closed_through - timedelta(days=offset)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        # The published figure IS the calibrated one — that is what
        # "published" means. Seeded at the band's own accuracy, so the
        # error measured over this column is a real measurement of how
        # well the organisation's published numbers held, and not the
        # engine's own map read back to itself.
        row["confidence_published"] = (
            0.0 if row["abstained"] else round(_band_accuracy(config, row["confidence_raw"]), 4)
        )
        row["notes"] = (
            ABSTAINED_NOTE
            if row["abstained"]
            else (CORRECT_NOTE if row["was_correct"] else WRONG_NOTE)
        )
        row["confidence_raw"] = round(row["confidence_raw"], 4)

    frame = pd.DataFrame(rows)[
        [
            "entry_id",
            "case_id",
            "case_type",
            "closed_at",
            "confidence_raw",
            "confidence_published",
            "abstained",
            "was_correct",
            "notes",
        ]
    ]
    return frame.sort_values("entry_id", kind="stable").reset_index(drop=True)


def _band_accuracy(config, raw: float) -> float:
    for band in config["published_bands"]:
        if band["raw_low"] <= raw < band["raw_high"]:
            return band["correct"] / band["cases"]
    return float(config["published_bands"][-1]["correct"]) / config["published_bands"][-1]["cases"]


def _assign_case_types(config, published: list[dict], abstained: list[dict]) -> None:
    """Deal case types from the least confident cases upward.

    A type listed early in the config draws the least confident cases,
    which is why competitor attribution is listed first: those are the
    cases the organisation is worst at, and the ledger should look it
    rather than being told it.

    The right and wrong answers are dealt from separate queues, so each
    type's accuracy comes out at exactly the declared count. It is a
    fixture; the alternative is a type-level accuracy that drifts with the
    band layout, and T7 then fires or does not fire by accident.
    """
    types = config["case_types"]
    declared_published = sum(int(spec["published"]) for spec in types.values())
    declared_correct = sum(int(spec["correct"]) for spec in types.values())
    declared_abstained = sum(int(spec["abstained"]) for spec in types.values())
    actual_correct = sum(1 for row in published if row["was_correct"])

    if declared_published != len(published):
        raise LedgerError(
            f"case types claim {declared_published} published cases, "
            f"the bands lay out {len(published)}"
        )
    if declared_correct != actual_correct:
        raise LedgerError(
            f"case types claim {declared_correct} correct outcomes, "
            f"the bands lay out {actual_correct}"
        )
    if declared_abstained != len(abstained):
        raise LedgerError(
            f"case types claim {declared_abstained} abstained cases, "
            f"the bands lay out {len(abstained)}"
        )

    right = [row for row in published if row["was_correct"]]
    wrong = [row for row in published if not row["was_correct"]]
    held_out = list(abstained)
    for name, spec in types.items():
        for _ in range(int(spec["correct"])):
            right.pop(0)["case_type"] = name
        for _ in range(int(spec["published"]) - int(spec["correct"])):
            wrong.pop(0)["case_type"] = name
        for _ in range(int(spec["abstained"])):
            held_out.pop(0)["case_type"] = name


def emit_calibration_ledger(world, out: Path) -> dict[str, int]:
    """`calibration/ledger.csv` — 213 closed cases."""
    frame = build_ledger(world)
    return {
        "calibration/ledger.csv": write_csv(
            frame, out / "calibration" / "ledger.csv"
        )
    }


__all__ = ["LedgerError", "build_ledger", "emit_calibration_ledger"]
