"""VERDICT — the stage that turns four stages' output into one decision.

CLAUDE.md §"Architecture" names five stages. Four of them had packages;
this is the fifth, and `docs/ARCHITECTURE.md` has been carrying a note
since P4 saying so ("VERDICT deliberately has none yet ... Flagged, not
resolved"). This resolves it.

WHAT WAS ACTUALLY MISSING. Not the decision — `engine/abstain/verdict.py`
has held the decision table since P10, and it is tested against a truth
grid. What was missing is the WIRING: `TriggerInput` bundles fourteen
facts the eight abstention triggers read, and nothing built one from real
stage output. `tests/test_abstain.py` assembles it by hand and says so:
"That is a wiring job, and it is P11's." P11 built the recommendation
engine instead. Without this module the API can run every stage and still
not reach a verdict.

Three pieces:

    triggers.py   TriggerInput, assembled from what the stages measured
    casefile.py   the stage results, assembled into the frozen contracts
    pipeline.py   the five stages in order, emitting one event each

Nothing here computes a statistic. Every number it handles was measured
by an earlier stage; this module reads them off, puts them in the shape
the decision table and the case file expect, and records where each one
came from. That is deliberately dull work and it is the reason the
stages can stay ignorant of each other.
"""

from engine.verdict.casefile import CaseFile, assemble
from engine.verdict.pipeline import (
    STAGE_ORDER,
    CaseRun,
    PipelineError,
    StageEvent,
    run_case,
)
from engine.verdict.triggers import (
    build_trigger_input,
    hypothesis_confidences,
    leading_hypothesis,
)

__all__ = [
    "STAGE_ORDER",
    "CaseFile",
    "CaseRun",
    "PipelineError",
    "StageEvent",
    "assemble",
    "build_trigger_input",
    "hypothesis_confidences",
    "leading_hypothesis",
    "run_case",
]
