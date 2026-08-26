"""The learning loop — what the system does with being told it was wrong.

CLAUDE.md's pipeline ends "... VERDICT -> human decides -> outcome logged
-> recalibrate". This package is that last arrow, and it is the only one:
nothing else writes `calibration_ledger`, `hypothesis_prior` or a recovery
realisation. One door, for the same reason `execute_governed` is one door.

    feedback.py   what a reader said, at one of three levels
    outcomes.py   what actually happened, at D+14 and D+56
    priors.py     the causal graph's priors, and the runtime overlay
    curves.py     realised recovery against what was promised
    loop.py       one feedback event in, up to three things moved

THREE THINGS MOVE, FOR THREE DIFFERENT REASONS:

    the isotonic map        was the CONFIDENCE honest?
    the causal graph priors was this DRIVER the usual answer here?
    the recovery curves     did the ACTION recover what we said it would?

They are kept apart on purpose. A case whose driver was right and whose
recovery disappointed says something about the playbook and nothing about
the confidence, and one combined score would lose that distinction — which
is the distinction that tells you which part to fix.

EVERY UPDATE IS BOUNDED. A loop wired to a button can be moved by whoever
clicks most. Priors are Bayesian against a declared strength and capped in
absolute shift; curves are blended against the sample they were fitted on;
the map refuses to fit below a minimum. None of that makes the loop slow
to learn something real. All of it makes it slow to learn something loud.
"""

from engine.learn.curves import CurveUpdate, Realisation, record_realisation, update_for
from engine.learn.feedback import (
    ACTION_LEVEL,
    DRIVER_LEVEL,
    VERDICT_LEVEL,
    Feedback,
    FeedbackError,
    record,
)
from engine.learn.loop import LoopEffect, LoopError, apply, apply_outcome, settle
from engine.learn.outcomes import (
    Outcome,
    due,
    record_cause_check,
    record_recovery_check,
    record_unresolved,
)
from engine.learn.priors import (
    LearnedPrior,
    effective_prior,
    effective_priors,
    read_overlay,
    write_overlay,
)

__all__ = [
    "ACTION_LEVEL",
    "DRIVER_LEVEL",
    "VERDICT_LEVEL",
    "CurveUpdate",
    "Feedback",
    "FeedbackError",
    "LearnedPrior",
    "LoopEffect",
    "LoopError",
    "Outcome",
    "Realisation",
    "apply",
    "apply_outcome",
    "due",
    "effective_prior",
    "effective_priors",
    "read_overlay",
    "record",
    "record_cause_check",
    "record_realisation",
    "record_recovery_check",
    "record_unresolved",
    "settle",
    "update_for",
    "write_overlay",
]
