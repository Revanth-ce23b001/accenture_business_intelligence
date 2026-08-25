"""Who is asking, and what the KPI contract lets them see.

Authorization happens BEFORE retrieval (CLAUDE.md rule 5). This module
resolves a user against a KPI contract's `access_policy` and produces the
row predicate and the masked column list that `engine/db.py` injects. It
never touches a database and it holds no thresholds: every rule it applies
is read off the contract it was handed.

The one judgement call it makes is to fail loudly. A persona the contract
does not know, or a row predicate whose binding the user cannot supply, is
an error rather than a fallback to something permissive — a regional
manager with no region attached must not quietly become a CXO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from semantic_layer.schema import Governance, KpiContract

#: `:name` in a row predicate, as the KPI contracts write it.
BIND_PATTERN = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")


class PolicyError(PermissionError):
    """The caller cannot be resolved against this KPI's access policy."""


@dataclass(frozen=True)
class User:
    """An authenticated caller.

    `persona` is the key into the KPI contract's access policy. `region`
    and `store_id` are what a scoped predicate binds to; both are None for
    unscoped personas, and a predicate that needs one it does not have is
    refused rather than widened.
    """

    user_id: str
    persona: str
    region: str | None = None
    store_id: str | None = None
    display_name: str | None = None
    email: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> User:
        """Build from a `dim_user` row."""
        return cls(
            user_id=str(row["user_id"]),
            persona=str(row["persona"]),
            region=_optional(row.get("region")),
            store_id=_optional(row.get("store_id")),
            display_name=_optional(row.get("display_name")),
            email=_optional(row.get("email")),
        )


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class AppliedPolicy:
    """What the contract decided, ready for injection."""

    kpi: str
    persona: str
    row_predicate: str
    masked_columns: tuple[str, ...]
    bindings: Mapping[str, str]

    @property
    def unrestricted(self) -> bool:
        return not self.bindings and not self.masked_columns

    @classmethod
    def for_metadata(
        cls, table: str, persona: str, unrestricted_predicate: str
    ) -> AppliedPolicy:
        """The policy for reading a pipeline-metadata table.

        No row filter and no mask, because these tables carry the
        warehouse's account of itself — the calendar, the case registry,
        the definition log — and there is nothing in them a predicate
        would protect. The allow-list in `warehouse.yaml -> governance`
        is what keeps a business table off that path, and a load-time
        check refuses any entry starting `fact_`, `doc_` or `ext_`.

        It exists as a NAMED constructor so that this module remains the
        only place an `AppliedPolicy` is built. A caller assembling one
        inline would be choosing its own predicate and its own mask list,
        which is the definition of a bypass, and a static test asserts
        nothing does.
        """
        return cls(
            kpi=table,
            persona=persona,
            row_predicate=unrestricted_predicate,
            masked_columns=(),
            bindings={},
        )


def bind_names(sql: str) -> set[str]:
    """Every `:name` placeholder in a fragment of SQL.

    Casts (`::DATE`) are not placeholders, so a doubled colon is skipped
    before the scan rather than matched and then explained away.
    """
    return set(BIND_PATTERN.findall(sql.replace("::", "\x00")))


def resolve_policy(
    kpi: KpiContract, user: User, governance: Governance
) -> AppliedPolicy:
    """Resolve `user` against `kpi`'s access policy.

    Raises `PolicyError` when the persona is unknown to the contract, or
    when the persona's predicate needs a user attribute the user has not
    got. Both are refusals, never widenings.
    """
    policy = kpi.access_policy
    persona = user.persona
    if persona not in policy.personas:
        raise PolicyError(
            f"persona {persona!r} is not defined in the access policy for "
            f"{kpi.kpi!r} (known: {sorted(policy.personas)})"
        )

    rules = policy.personas[persona]
    predicate = rules.row_predicate

    bindings: dict[str, str] = {}
    for name in sorted(bind_names(predicate)):
        attribute = governance.reserved_bindings.get(name)
        if attribute is None:
            raise PolicyError(
                f"row predicate for {kpi.kpi!r}/{persona!r} binds {name!r}, which is not a "
                f"reserved binding ({sorted(governance.reserved_bindings)})"
            )
        value = getattr(user, attribute, None)
        if value is None:
            raise PolicyError(
                f"user {user.user_id!r} has persona {persona!r}, whose row predicate needs "
                f"{attribute!r}, and no {attribute} is set. Refusing rather than widening "
                "the query."
            )
        bindings[name] = str(value)

    return AppliedPolicy(
        kpi=kpi.kpi,
        persona=persona,
        row_predicate=predicate,
        masked_columns=tuple(rules.masked_columns),
        bindings=bindings,
    )


__all__ = ["AppliedPolicy", "PolicyError", "User", "bind_names", "resolve_policy"]
