"""Abstention — eight named booleans and a three-condition decision table.

CLAUDE.md rule 6: abstention is architectural, not a prompt instruction.
Nothing here asks a model whether it is sure.

    from engine.abstain import evaluate_triggers, decide

    triggers = evaluate_triggers(inputs, layer)
    decision = decide(coverage=..., confidence=..., triggers=triggers,
                      held=..., layer=layer)
    decision.value          # EXPLAINED / PARTIALLY_EXPLAINED / INSUFFICIENT_EVIDENCE
    decision.reason_text    # rendered on the verdict chip
    decision.triggers_fired # every one, by name
"""

from engine.abstain.triggers import (
    IDS,
    Trigger,
    TriggerInput,
    evaluate_triggers,
    fired,
)
from engine.abstain.verdict import (
    BELOW_ALL_THRESHOLDS,
    COVERAGE_BELOW,
    EXPLAINED,
    INSUFFICIENT_EVIDENCE,
    LIVE_UNVERIFIABLE,
    PARTIALLY_EXPLAINED,
    TRIGGERS_FIRED,
    HeldResidual,
    VerdictDecision,
    decide,
    to_contract,
)

__all__ = [
    "BELOW_ALL_THRESHOLDS",
    "COVERAGE_BELOW",
    "EXPLAINED",
    "IDS",
    "INSUFFICIENT_EVIDENCE",
    "LIVE_UNVERIFIABLE",
    "PARTIALLY_EXPLAINED",
    "TRIGGERS_FIRED",
    "HeldResidual",
    "Trigger",
    "TriggerInput",
    "VerdictDecision",
    "decide",
    "evaluate_triggers",
    "fired",
    "to_contract",
]
