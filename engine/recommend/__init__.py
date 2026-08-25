"""RECOMMEND — a verdict turned into something a person does on Monday.

    from engine.recommend import RecommendRequest, recommend

    result = recommend(connection, user, RecommendRequest(...))
    result.cost.lakh(layer)          # playbook rate x measured quantity
    result.recovery.low_inr          # attributed money x the curve's p25
    result.owner.title               # from the KPI contract, never hardcoded
    result.investigations            # abstained cases, by value per rupee
    result.not_recommended           # and what we are advising against
    result.linked_case               # the cause of the cause, opened

Two shapes come out, and they are not the same object relabelled. A case
that reached a verdict gets an ACTION with a cost, a recovery band and an
owner. A case that abstained gets an INVESTIGATION LIST ranked by value
per rupee, plus at least one lever refused out loud with its price and
its error rate.

Nothing here supplies a number of its own (rule 2): rates come from the
playbooks, quartiles from the recovery curves, priors from the causal
graph, ownership from the KPI contract, and quantities from the case.
"""

from engine.recommend.cost import (
    CallDown,
    Cost,
    CostBasis,
    CostError,
    compute_cost,
    plan_call_down,
)
from engine.recommend.gate import (
    EVIDENCE_PREFIX,
    RecommendationResult,
    RecommendError,
    RecommendRequest,
    recommend,
    to_contract,
)
from engine.recommend.investigate import (
    Investigation,
    MatchError,
    NotRecommended,
    candidates,
    error_rate,
    is_informational,
    lead_time_days,
    match_playbook,
    rank_investigations,
    refuse,
)
from engine.recommend.linked import (
    LinkedCase,
    LinkedCaseError,
    SourceGap,
    acquisition_playbook,
    create_linked_case,
    linked_kpi,
    next_case_id,
    source_gaps,
    write_source_gaps,
)
from engine.recommend.ownership import Owner, escalation_owner, resolve_owner
from engine.recommend.recovery import (
    Recovery,
    RecoveryError,
    compute_recovery,
    confidence_label,
    curve_for,
)

__all__ = [
    "EVIDENCE_PREFIX",
    "CallDown",
    "Cost",
    "CostBasis",
    "CostError",
    "Investigation",
    "LinkedCase",
    "LinkedCaseError",
    "MatchError",
    "NotRecommended",
    "Owner",
    "RecommendError",
    "RecommendRequest",
    "RecommendationResult",
    "Recovery",
    "RecoveryError",
    "SourceGap",
    "acquisition_playbook",
    "candidates",
    "compute_cost",
    "compute_recovery",
    "confidence_label",
    "create_linked_case",
    "curve_for",
    "error_rate",
    "escalation_owner",
    "is_informational",
    "lead_time_days",
    "linked_kpi",
    "match_playbook",
    "next_case_id",
    "plan_call_down",
    "rank_investigations",
    "recommend",
    "refuse",
    "resolve_owner",
    "source_gaps",
    "to_contract",
    "write_source_gaps",
]
