"""Who signs this off.

One rule, and it is the whole module: the owner comes from the KPI
contract's driver ownership, keyed by the driver ADJUDICATION settled on.
Not from the playbook, not from a lookup table in Python, and not from
whoever raised the case.

A playbook does carry an `owner_role`, and it is not wrong — it is who
owns that lever in general. `availability_recovery` says
`supply_chain_director`, because expedited freight is a supply chain
lever wherever it is pulled. But the question a recommendation answers is
narrower than that: who answers for THIS movement in THIS KPI. Meridian
answered it in `net_revenue.yaml`, and the answer is the Regional Supply
Chain Lead, not the group director.

Where the two disagree the contract wins and the disagreement is
RECORDED. A silent override is how a system ends up addressing an action
to somebody who was never asked whether they own it.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_layer.schema import KpiContract, Playbook

#: Where the owner was resolved from, carried so the UI can say.
FROM_CONTRACT = "kpi_contract.driver_ownership"
FROM_CONTRACT_DEFAULT = "kpi_contract.driver_ownership.default"
FROM_OWNER_ROLE = "kpi_contract.owner_role"


@dataclass(frozen=True)
class Owner:
    """The person the action is addressed to, and how that was decided."""

    role: str
    title: str
    driver: str | None
    resolved_from: str
    #: What the playbook thought, when a playbook was involved at all.
    playbook_role: str | None = None

    @property
    def overrides_playbook(self) -> bool:
        return self.playbook_role is not None and self.playbook_role != self.role

    def render(self) -> str:
        line = f"owner: {self.title} ({self.role}) via {self.resolved_from}"
        if self.overrides_playbook:
            line += f"; playbook says {self.playbook_role} — contract wins"
        return line


def resolve_owner(
    contract: KpiContract, driver: str | None, *, playbook: Playbook | None = None
) -> Owner:
    """The owner for `driver` under `contract`.

    Three resolutions, and they are distinguished because they mean
    different things to a reader. A named driver owner is a decision. The
    contract default is a decision about everything nobody named. Falling
    through to `owner_role` means the contract has not been asked the
    question yet, and the UI should be able to say so rather than present
    a fallback as an answer.
    """
    role, title = contract.owner_for(driver)
    ownership = contract.driver_ownership
    if ownership is None:
        resolved_from = FROM_OWNER_ROLE
    elif driver is not None and driver in ownership.by_driver:
        resolved_from = FROM_CONTRACT
    else:
        resolved_from = FROM_CONTRACT_DEFAULT

    return Owner(
        role=role,
        title=title,
        driver=driver,
        resolved_from=resolved_from,
        playbook_role=playbook.owner_role if playbook is not None else None,
    )


def escalation_owner(contract: KpiContract, playbook: Playbook) -> Owner:
    """Who the monitoring plan escalates to when the kill criterion trips.

    This one IS the playbook's, and deliberately so. Escalation is a
    property of the lever — the person who can stop paying a freight
    premium is the person who authorised it — not of the KPI.
    """
    role = playbook.monitoring_plan.escalation_role
    ownership = contract.driver_ownership
    title = role
    if ownership is not None:
        for owner in (ownership.default, *ownership.by_driver.values()):
            if owner.role == role:
                title = owner.title
                break
    return Owner(
        role=role,
        title=title,
        driver=playbook.driver,
        resolved_from=f"playbook.{playbook.playbook}.monitoring_plan.escalation_role",
        playbook_role=role,
    )


__all__ = [
    "FROM_CONTRACT",
    "FROM_CONTRACT_DEFAULT",
    "FROM_OWNER_ROLE",
    "Owner",
    "escalation_owner",
    "resolve_owner",
]
