"""The isotonic map, fitted to the organisation's own track record.

    the model scored this 89%
    our record on cases like it says we run about five points hot
    so we publish 84%

That sentence is Accenture requirement 7, and this module is what makes it
a measurement rather than a slogan. Nothing here is written down in
advance: the map, the band table and the calibration error are all fitted
to `calibration_ledger` and reported.

WHY ISOTONIC. A calibration map has exactly one thing it must not do:
reorder cases. If the engine scores case A above case B, the published
figures have to keep that order, or the ranking a human reads is not the
ranking the engine produced. Isotonic regression is the least-squares fit
subject to that single constraint and nothing else — no functional form
assumed, no parameters to choose, no way to smuggle a prior in. Platt
scaling would impose a logistic shape the data has not been asked about.

WHEN NOT TO CALIBRATE. Below `min_cases_for_fit` closed cases the map is
the IDENTITY and the result is flagged `calibrating`, carrying n. A map
fitted to a dozen cases is a map fitted to noise, and shipping one is
worse than shipping the raw number, because it looks like a correction.

ABSTENTIONS ARE EXCLUDED. An abstained case published no attribution, so
there is nothing it could have been right or wrong about. Counting it as
a miss would punish the engine for the behaviour this project exists to
encourage; counting it as a hit would reward abstaining from everything.
It is counted in the abstention rate and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from engine.db import execute_metadata
from security.policy import User
from semantic_layer.schema import CalibrationSpec, SemanticLayer, get_semantic_layer

LEDGER_TABLE = "calibration_ledger"

LEDGER_SQL = """
SELECT case_type, confidence_raw, confidence_published, abstained, was_correct
FROM calibration_ledger
"""


class CalibrationError(RuntimeError):
    """The ledger cannot support a calibration."""


@dataclass(frozen=True)
class Band:
    """One row of the printed table. Computed, never declared."""

    low: float
    high: float
    cases: int
    mean_confidence: float
    accuracy: float | None
    thin: bool

    @property
    def label(self) -> str:
        return f"{self.low:.0%}-{self.high:.0%}"

    @property
    def gap(self) -> float | None:
        """How many points hot. Positive means the score ran above reality."""
        if self.accuracy is None:
            return None
        return self.mean_confidence - self.accuracy

    def render(self) -> str:
        if self.thin or self.accuracy is None:
            return (
                f"{self.label:>9} band   n={self.cases:<4} too thin to score "
                f"(fewer than the minimum this table will report on)"
            )
        return (
            f"{self.label:>9} band   n={self.cases:<4} "
            f"scored {self.mean_confidence:.0%}   was right {self.accuracy:.0%}   "
            f"{self.gap:+.1%} hot"
        )


@dataclass(frozen=True)
class Calibration:
    """A fitted map, and everything a reader needs to check it."""

    #: Closed cases with an outcome. Abstentions are not among them.
    scored_cases: int
    abstained_cases: int
    #: True when there were too few cases to fit and the map is the identity.
    calibrating: bool
    method: str
    bands: tuple[Band, ...]
    #: Expected calibration error of the RAW scores — the thing the map is
    #: there to correct.
    ece_raw: float
    #: The same measure applied to the map's own output.
    ece_calibrated: float
    _model: IsotonicRegression | None = None
    _floor: float = 0.0

    @property
    def total_cases(self) -> int:
        return self.scored_cases + self.abstained_cases

    @property
    def abstention_rate(self) -> float:
        return self.abstained_cases / self.total_cases if self.total_cases else 0.0

    def map(self, raw: float) -> float:
        """Raw confidence in, published confidence out.

        The identity while `calibrating`, and the fitted map otherwise.
        """
        if self.calibrating or self._model is None:
            return float(raw)
        return float(self._model.predict([float(raw)])[0])

    def shift(self, raw: float) -> float:
        """How far the map moves a score. Negative means it cools it."""
        return self.map(raw) - float(raw)

    def accuracy_for(self, case_type: str) -> tuple[float | None, int]:
        """Historical accuracy for one case type, and how many cases back it.

        This is what trigger T7 reads. `None` with a small n means the
        organisation has no track record on this kind of case, which is a
        different statement from a poor one and is reported as such.
        """
        return self._by_type.get(case_type, (None, 0))

    def render(self) -> str:
        lines = [
            f"calibration: {self.total_cases} closed cases "
            f"({self.scored_cases} published, {self.abstained_cases} abstained, "
            f"{self.abstention_rate:.0%} abstention rate)",
        ]
        if self.calibrating:
            lines.append(
                f"  CALIBRATING - {self.scored_cases} scored cases is below the "
                "minimum to fit a map. Raw scores are published unchanged."
            )
        lines.extend(f"  {band.render()}" for band in self.bands)
        lines.append(
            f"  expected calibration error: {self.ece_raw:.3f} raw, "
            f"{self.ece_calibrated:.3f} after the map"
        )
        return "\n".join(lines)


# `_by_type` is attached after construction so the dataclass stays frozen
# and printable. Declared here rather than as a field because it is an
# index, not part of what a calibration IS.
Calibration._by_type = {}  # type: ignore[attr-defined]


def fit_calibration(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    *,
    layer: SemanticLayer | None = None,
    case_type: str | None = None,
) -> Calibration:
    """Fit the map to the seeded ledger.

    `case_type` narrows the fit to one kind of case. Left None it fits
    across all of them, which is what a case with no track record of its
    own has to fall back on.
    """
    layer = layer or get_semantic_layer()
    spec = layer.adjudication.confidence.calibration
    rows = execute_metadata(
        user, LEDGER_TABLE, LEDGER_SQL,
        connection=connection, layer=layer, purpose="calibration.fit",
    )

    entries = [dict(row) for row in rows]
    if case_type is not None:
        entries = [row for row in entries if row["case_type"] == case_type]

    scored = [row for row in entries if not row["abstained"] and row["was_correct"] is not None]
    abstained = [row for row in entries if row["abstained"]]

    raw = np.array([float(row["confidence_raw"]) for row in scored])
    correct = np.array([1.0 if row["was_correct"] else 0.0 for row in scored])

    calibrating = len(scored) < spec.min_cases_for_fit
    model = None
    if not calibrating:
        model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(raw, correct)

    mapped = model.predict(raw) if model is not None else raw
    calibration = Calibration(
        scored_cases=len(scored),
        abstained_cases=len(abstained),
        calibrating=calibrating,
        method=spec.method,
        bands=_bands(raw, correct, spec),
        ece_raw=_ece(raw, correct, spec),
        ece_calibrated=_ece(np.asarray(mapped, dtype=float), correct, spec),
        _model=model,
        _floor=spec.publication_floor,
    )
    object.__setattr__(calibration, "_by_type", _by_type(entries))
    return calibration


def _by_type(entries) -> dict[str, tuple[float | None, int]]:
    """Accuracy per case type, over the cases that actually published."""
    buckets: dict[str, list[bool]] = {}
    for row in entries:
        if row["abstained"] or row["was_correct"] is None:
            continue
        buckets.setdefault(str(row["case_type"]), []).append(bool(row["was_correct"]))
    return {
        name: (sum(outcomes) / len(outcomes) if outcomes else None, len(outcomes))
        for name, outcomes in buckets.items()
    }


def _bands(raw: np.ndarray, correct: np.ndarray, spec: CalibrationSpec) -> tuple[Band, ...]:
    """The printed table. Every number in it is computed from the rows.

    Bands holding fewer than `min_cases_for_band` cases are reported as
    too thin rather than given an accuracy: three cases cannot tell you
    what a decile is worth, and printing 67% from two of three would
    invite somebody to act on it.
    """
    edges = spec.band_edges
    bands: list[Band] = []
    for low, high in zip(edges, edges[1:], strict=False):
        # The last band is closed at the top so a perfect score is counted.
        inside = (raw >= low) & ((raw < high) if high < edges[-1] else (raw <= high))
        count = int(inside.sum())
        if count == 0:
            continue
        thin = count < spec.min_cases_for_band
        bands.append(
            Band(
                low=low,
                high=high,
                cases=count,
                mean_confidence=float(raw[inside].mean()),
                accuracy=None if thin else float(correct[inside].mean()),
                thin=thin,
            )
        )
    return tuple(bands)


def _ece(scores: np.ndarray, correct: np.ndarray, spec: CalibrationSpec) -> float:
    """Expected calibration error: the case-weighted mean band gap.

    Every band counts, including the thin ones — a band too thin to PRINT
    an accuracy for is not too thin to be wrong, and dropping it from the
    error would flatter the number.
    """
    if not len(scores):
        return 0.0
    edges = spec.band_edges
    total = 0.0
    for low, high in zip(edges, edges[1:], strict=False):
        inside = (scores >= low) & ((scores < high) if high < edges[-1] else (scores <= high))
        count = int(inside.sum())
        if not count:
            continue
        total += count * abs(float(scores[inside].mean()) - float(correct[inside].mean()))
    return total / len(scores)


__all__ = [
    "LEDGER_SQL",
    "LEDGER_TABLE",
    "Band",
    "Calibration",
    "CalibrationError",
    "fit_calibration",
]
