"""The offline stand-in for the narrator. Not a model, and it says so.

A repository has to be buildable and testable without an API key
(CLAUDE.md rule 8), so `record_fixtures.py --offline` needs something to
record. This is that something for the `narrate` task, kept in its own
module for the same reason the classification stand-in is kept in one:
nothing the engine runs may reach it, and a reader should be able to tell
at a glance which side of the boundary a file is on.

WHAT IT IS. A template walker. It reads the frozen adjudication object,
picks the evidence records this persona cares about in the order this
persona cares about them, and writes one sentence per record from the
record's own label, value and unit. Every sentence therefore cites a real
evidence id and contains only numbers the object holds — by construction,
because it has no other source for either.

WHAT THAT PROVES, AND WHAT IT DOES NOT. It proves the pipeline: the
request builder, the fixture key, the parser, all four grounding rules and
the strip-regenerate-drop loop. It does NOT prove that Sonnet writes
grounded prose. A fixture written this way says what a template walker
produced, not what the model would produce, and every one of them is
labelled SYNTHETIC for that reason.

Re-record with `--live` before quoting a grounding rate to anybody. The
validator does not care which of the two wrote the sentences, which is
the entire point of putting a deterministic check between the model and
the screen.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from semantic_layer.schema import SemanticLayer

#: Which evidence records each persona reaches for, in order. Keys are
#: evidence ids minus the `case.` stage prefix; anything a given scenario
#: does not carry is skipped rather than faked.
BEATS: dict[str, tuple[str, ...]] = {
    "cxo": (
        "headline",
        "residual",
        "materiality",
        "confidence_published",
        "failed_feeds",
        "weekly_points",
        "calendar_expected",
    ),
    "regional_manager": (
        "treated_stores",
        "availability_before",
        "availability_after",
        "tickets",
        "store_notes",
        "east_stores",
        "road_closures",
        "failed_feeds",
        "west_stores",
        "dark_stores",
        "headline",
    ),
    "analyst": (
        "did",
        "dose_response",
        "coverage",
        "unattributed",
        "band",
        "affected_volume_share",
        "calibration_accuracy",
        "publication_floor",
        "marketing_staleness",
        "failed_share",
        "weekly_points_required",
        "residual",
        "calendar_expected",
    ),
}


def _value(record: Mapping[str, Any]) -> str:
    """Render a stored value the way prose renders it.

    A ratio becomes a percentage, because the object stores shares and
    people say percent. Everything else is printed as stored, trimmed of
    trailing zeros — the grounding check matches on the token's own
    precision, so printing 4.10 and printing 4.1 are both correct and the
    shorter one reads better.
    """
    value = record.get("value")
    unit = record.get("unit", "")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if unit == "ratio":
        scaled = value * 100
        return f"{scaled:g}%"
    return f"{value:g}"


def _unit(record: Mapping[str, Any]) -> str:
    unit = record.get("unit", "")
    return {
        "INR_CR": " Cr",
        "INR_L": " L",
        "pct": "%",
        "pt": " pt",
        "hours": " hours",
        "ratio": "",
        "count": "",
        "r": "",
    }.get(unit, f" {unit}")


def _prefix(record: Mapping[str, Any]) -> str:
    return "INR " if str(record.get("unit", "")).startswith("INR") else ""


def _statement(record: Mapping[str, Any]) -> str:
    """`Qualified residual: INR 4.1 Cr.`"""
    return (
        f"{record['label']} is {_prefix(record)}{_value(record)}{_unit(record)}."
    )


def _by_key(adjudication: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for record in adjudication.get("evidence", ()) or ():
        evidence_id = str(record.get("evidence_id", ""))
        _, _, key = evidence_id.partition(".")
        index[key] = record
    return index


def _claim(sentence: str, ids: Sequence[str], hypothesis: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"sentence": sentence, "evidence_ids": list(ids)}
    if hypothesis:
        payload["hypothesis_id"] = hypothesis
    return payload


def _verdict_claim(
    adjudication: Mapping[str, Any], index: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The opening line: what happened to this case.

    A gate kill quotes the gate's own detail, which already names the
    counts and is already evidence-linked. A live case states the coverage
    and the triggers by name.
    """
    gates = adjudication.get("gates") or ()
    if gates:
        gate = gates[0]
        ids = list(gate.get("evidence_ids") or ())
        if not ids:
            return None
        return _claim(f"{gate['outcome_code']}: {gate['detail']}", ids)

    triggers = adjudication.get("triggers_fired") or ()
    residual = index.get("residual")
    if triggers and residual is not None:
        named = ", ".join(triggers)
        return _claim(
            f"This case abstained: {named} fired, so no explanation was published.",
            [residual["evidence_id"]],
        )
    coverage = index.get("coverage")
    if coverage is not None:
        return _claim(
            f"The surviving hypotheses account for {_value(coverage)} of the "
            "qualified residual.",
            [coverage["evidence_id"]],
        )
    return None


def _hypothesis_claims(
    adjudication: Mapping[str, Any], index: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """One line per hypothesis, phrased by what it is allowed to claim.

    A supported hypothesis takes the required attribution form and says
    its share. An eliminated one says what it failed on and stops — no
    connective, because it did not earn one.
    """
    claims: list[dict[str, Any]] = []
    for entry in adjudication.get("hypotheses", ()) or ():
        label = entry.get("label", "")
        hypothesis_id = entry.get("hypothesis_id")
        ids = list(entry.get("evidence_ids") or ())
        if not ids:
            continue
        if entry.get("status") == "supported" and entry.get("attributed"):
            share = entry.get("attributed_share")
            attributed = entry["attributed"]
            share_text = f"{share * 100:g}%" if isinstance(share, (int, float)) else ""
            claims.append(
                _claim(
                    f"{label} is the best-supported explanation for {share_text} of the "
                    f"qualified residual, worth {_prefix(attributed)}"
                    f"{_value(attributed)}{_unit(attributed)}.",
                    [attributed["evidence_id"]],
                    hypothesis_id,
                )
            )
        elif entry.get("status") == "eliminated":
            claims.append(
                _claim(
                    f"{label} was eliminated on {entry.get('elimination_reason')}.",
                    ids[:1],
                    hypothesis_id,
                )
            )
        elif entry.get("status") == "live" and not entry.get("verifiable"):
            missing = ", ".join(entry.get("missing_sources") or ())
            claims.append(
                _claim(
                    f"{label} could not be tested: the organisation holds neither "
                    f"{missing}.",
                    ids[:1],
                    hypothesis_id,
                )
            )
    return claims


def synthesise_narrative(
    adjudication: Mapping[str, Any], persona: str, layer: SemanticLayer
) -> str:
    """The reply the stand-in would have given for one persona."""
    profile = layer.narrate.personas[persona]
    index = _by_key(adjudication)
    claims: list[dict[str, Any]] = []

    opening = _verdict_claim(adjudication, index)
    if opening is not None:
        claims.append(opening)

    if persona != "regional_manager":
        claims.extend(_hypothesis_claims(adjudication, index))

    for key in BEATS[persona]:
        record = index.get(key)
        if record is None:
            continue
        if any(record["evidence_id"] in claim["evidence_ids"] for claim in claims):
            continue
        claims.append(_claim(_statement(record), [record["evidence_id"]]))

    if not claims:
        # Every scenario carries a materiality threshold, so there is
        # always one honest sentence available. A narrative of length zero
        # would pass every grounding rule and tell the reader nothing.
        limit = adjudication.get("materiality")
        if limit:
            claims.append(_claim(_statement(limit), [limit["evidence_id"]]))

    return json.dumps(claims[: profile.max_sentences], ensure_ascii=False)


__all__ = ["BEATS", "synthesise_narrative"]
