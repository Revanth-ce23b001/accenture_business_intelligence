"""Gate 4 — is this the scope's problem, or the market's?

Every peer region is decomposed by exactly the same method and judged
against its own band. If three or more move the same way, the thing being
investigated is not West's stock-out — it is the market — and opening four
regional cases would give four teams the same wrong answer in parallel.

So it becomes ONE case, reclassified MARKET_CASE and escalated to the CCO,
because a market movement is nobody's region and everybody's problem.

The gate can also decline to rule. When the caller's row policy hides the
peers — a regional manager sees one region — it returns PEERS_NOT_VISIBLE
rather than concluding the movement was specific. "I could not see the
other regions" and "the other regions were fine" are different statements
and must not be printed as the same one.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.qualify.band import ResidualBand
from engine.qualify.calendar import CalendarDecomposition
from semantic_layer.schema import SpecificityGate


@dataclass(frozen=True)
class PeerResidual:
    """One region's residual, and whether it left its own band."""

    region: str
    residual_pt: float
    band_pt: float
    direction: int

    @property
    def breaching(self) -> bool:
        return self.direction != 0


@dataclass(frozen=True)
class SpecificityVerdict:
    """What the strip of regions says about who owns the movement."""

    subject: str
    peers: tuple[PeerResidual, ...]
    breaching: tuple[str, ...]
    direction: int
    is_market_case: bool
    peers_visible: bool
    escalate_to_role: str | None

    @property
    def strip(self) -> dict[str, float]:
        """The regional residual strip, for display."""
        return {peer.region: peer.residual_pt for peer in self.peers}

    def peer(self, region: str) -> PeerResidual | None:
        for candidate in self.peers:
            if candidate.region == region:
                return candidate
        return None


def assess_specificity(
    subject: str,
    decompositions: dict[str, CalendarDecomposition],
    bands: dict[str, ResidualBand],
    spec: SpecificityGate,
) -> SpecificityVerdict:
    """Compare the subject scope against every peer the caller can see."""
    peers: list[PeerResidual] = []
    for region in sorted(decompositions):
        band = bands.get(region)
        if band is None:
            continue
        residual = decompositions[region].residual_pt
        peers.append(
            PeerResidual(
                region=region,
                residual_pt=residual,
                band_pt=band.band_pt,
                direction=band.direction_of(residual),
            )
        )

    visible = len(peers) >= spec.min_peers_breaching
    breaching_down = [p for p in peers if p.direction < 0]
    breaching_up = [p for p in peers if p.direction > 0]

    if spec.same_direction_required:
        winner = max((breaching_down, breaching_up), key=len)
    else:
        winner = breaching_down + breaching_up

    direction = 0
    if winner:
        direction = winner[0].direction if spec.same_direction_required else 0

    is_market = visible and len(winner) >= spec.min_peers_breaching
    return SpecificityVerdict(
        subject=subject,
        peers=tuple(peers),
        breaching=tuple(sorted(p.region for p in winner)),
        direction=direction,
        is_market_case=is_market,
        peers_visible=visible,
        escalate_to_role=spec.escalate_to_role if is_market else None,
    )


__all__ = ["PeerResidual", "SpecificityVerdict", "assess_specificity"]
