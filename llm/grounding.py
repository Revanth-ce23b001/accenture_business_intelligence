"""The deterministic check between the model and the screen.

Nothing in this module asks a model anything. Every rule is a match
against the frozen adjudication object or against a fixed list, which is
the only reason the UI is allowed to print

    grounding: 14/14 claims linked · 0 stripped

as a measurement rather than a reassurance.

FOUR RULES, applied per sentence.

1. LINKAGE. Every sentence carries at least one evidence id, and every id
   it carries exists in the object. A sentence with nothing behind it
   cannot be clicked, and CLAUDE.md rule 3 says every number in the UI
   must be clickable to its evidence.

2. NUMBERS. Every numeric token in the sentence appears in the object.
   This is a WHITELIST, built from the object's own numeric fields and
   nothing else — not from numbers embedded in prose, which would let a
   fabricated figure launder itself through a caption. The tolerance is
   rounding and only rounding: a token matches when some stored value
   rounds to it at the TOKEN's precision, so a stored 1.8823 grounds
   "1.88" and "1.9" and refuses "1.87".

   It checks provenance, not relevance. A figure that rounds to some
   unrelated stored quantity will pass, and no whitelist can fix that —
   tying a number to the field it came from means letting the model
   declare the field, which it could misdeclare. This rule stops numbers
   being invented. The evidence id on the sentence is what lets a reader
   catch one being misapplied.

3. CAUSAL CONNECTIVES. "Because", "due to", "led to" and their relatives
   are allowed only in a sentence whose hypothesis passed BOTH hard
   gates. The hypothesis a sentence is about is not taken on trust: it is
   the union of what the model declared, what the sentence names, and
   what its cited evidence belongs to, so omitting the declaration widens
   the check rather than escaping it.

4. LANGUAGE. Two unconditional families. Bare causal claims — "revenue
   fell because of X" — assert the whole movement when the object only
   supports a share. Forbidden phrases — "root cause", "AI analysis
   shows", "the model found" — are each a way of sounding certain without
   being accountable to a method.

A sentence that breaks any rule is STRIPPED, and the caller offers one
regeneration naming exactly what failed. Anything still failing is
DROPPED. The narrative gets shorter; a claim that could not be supported
never gets a hedge bolted onto it and published anyway.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

from semantic_layer.schema import NarrateConfig, SemanticLayer

# --- violation codes, surfaced by name in the report and the UI ------------
NO_EVIDENCE = "no_evidence_id"
UNKNOWN_EVIDENCE = "unknown_evidence_id"
FABRICATED_NUMBER = "fabricated_number"
UNGATED_CAUSAL_CLAIM = "ungated_causal_claim"
BARE_CAUSAL_CLAIM = "bare_causal_claim"
FORBIDDEN_PHRASE = "forbidden_phrase"

#: A number as it appears in prose: optional sign, thousands separators,
#: optional decimal part. Currency symbols and units are matched around it
#: by the caller's own reading, not here — the token is the quantity.
#:
#: The lookbehind is load-bearing. Without it the hyphen inside a compound
#: word reads as a minus sign, "Top-20" scans as the number -20, and a
#: sentence that was telling the truth gets stripped for a quantity it
#: never claimed.
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+−]?\d[\d,]*(?:\.\d+)?")

#: Word boundaries that work for multi-word phrases too.
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(r"\b" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"\b", re.I)


class GroundingError(RuntimeError):
    """The narrative could not be checked as given."""


# ---------------------------------------------------------------------------
# What the model returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """One sentence and what the model says stands behind it."""

    sentence: str
    evidence_ids: tuple[str, ...]
    #: Optional. When absent the checker infers it, and infers WIDELY:
    #: leaving it off cannot be used to dodge the hard-gate rule.
    hypothesis_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sentence": self.sentence,
            "evidence_ids": list(self.evidence_ids),
        }
        if self.hypothesis_id:
            payload["hypothesis_id"] = self.hypothesis_id
        return payload


@dataclass(frozen=True)
class Violation:
    """Why one sentence was stripped, in terms the model can act on."""

    code: str
    detail: str
    token: str | None = None

    def render(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass(frozen=True)
class CheckedClaim:
    claim: Claim
    violations: tuple[Violation, ...]

    @property
    def grounded(self) -> bool:
        return not self.violations


@dataclass(frozen=True)
class GroundingReport:
    """What the UI prints, and what a test asserts.

    `claims_checked == claims_linked + claims_stripped` always holds, over
    the FINAL slate — a sentence that failed and was successfully replaced
    is counted once, as the replacement that survived. `claims_regenerated`
    is carried separately so that "we needed a second attempt" stays
    visible instead of being averaged away.
    """

    claims_checked: int
    claims_linked: int
    claims_stripped: int
    claims_regenerated: int = 0
    violations: tuple[Violation, ...] = ()

    def __post_init__(self) -> None:
        if self.claims_checked != self.claims_linked + self.claims_stripped:
            raise GroundingError(
                f"report does not balance: {self.claims_checked} checked, "
                f"{self.claims_linked} linked, {self.claims_stripped} stripped"
            )

    @property
    def clean(self) -> bool:
        return self.claims_stripped == 0

    def render(self) -> str:
        line = (
            f"grounding: {self.claims_linked}/{self.claims_checked} claims linked · "
            f"{self.claims_stripped} stripped"
        )
        if self.claims_regenerated:
            line += f" · {self.claims_regenerated} regenerated"
        return line

    def as_dict(self) -> dict[str, int]:
        """The three counts the brief names, and nothing else."""
        return {
            "claims_checked": self.claims_checked,
            "claims_linked": self.claims_linked,
            "claims_stripped": self.claims_stripped,
        }


# ---------------------------------------------------------------------------
# The whitelist
# ---------------------------------------------------------------------------


def evidence_ids(adjudication: Mapping[str, Any]) -> frozenset[str]:
    """Every citable id in the object.

    Walks the whole structure rather than reading one list, because ids
    appear in several places — the evidence index, a hypothesis's own
    `evidence_ids`, a confidence component's — and a sentence citing a
    real id from the wrong list is still citing a real id.
    """
    found: set[str] = set()
    for key, value in _walk(adjudication):
        if key == "evidence_id" and isinstance(value, str):
            found.add(value)
        elif key == "evidence_ids" and isinstance(value, (list, tuple)):
            found.update(str(item) for item in value)
    return frozenset(found)


def numeric_whitelist(
    adjudication: Mapping[str, Any], spec: NarrateConfig
) -> frozenset[Decimal]:
    """Every quantity the object actually holds.

    NUMERIC FIELDS ONLY. A number that appears inside a prose string — an
    evidence `description` reading "INR 53,000 per store x 34" — is
    deliberately NOT admitted. Admitting it would mean any figure could be
    grounded by first appearing in a caption, and captions are the part of
    the object nothing else validates.

    Two expansions, and only two. A ratio is admitted as its percentage
    (0.79 grounds "79%") because the object stores shares and prose says
    percent. A percentage is admitted as its ratio, for the same reason in
    reverse. Nothing else is scaled: a token is never multiplied around
    until something fits.
    """
    values: set[Decimal] = set()
    scale = spec.grounding.numeric.percent_expansion
    for _key, value in _walk(adjudication):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        number = _decimal(value)
        if number is None:
            continue
        values.add(number)
        if scale:
            values.add(number * Decimal(100))
            values.add(number / Decimal(100))

    # Digit-only string fields are identifiers — a case id, a store count
    # carried as text — and a sentence may name them.
    for _key, value in _walk(adjudication):
        if isinstance(value, str) and value.isdigit():
            number = _decimal(value)
            if number is not None:
                values.add(number)
    return frozenset(values)


def _walk(node: Any, key: str | None = None) -> Iterable[tuple[str | None, Any]]:
    """Every (key, value) pair in a nested JSON-ish structure."""
    yield key, node
    if isinstance(node, Mapping):
        for child_key, child in node.items():
            yield from _walk(child, str(child_key))
    elif isinstance(node, (list, tuple)):
        for child in node:
            yield from _walk(child, key)


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


# ---------------------------------------------------------------------------
# Numbers in prose
# ---------------------------------------------------------------------------


def numeric_tokens(sentence: str, spec: NarrateConfig) -> tuple[str, ...]:
    """The numbers a sentence claims, with identifiers removed first.

    Order matters. Dates, evidence ids, case references and trigger names
    are blanked BEFORE the number scan, so "2025-11" never becomes a
    quantity looking for something to ground against.
    """
    masked = sentence
    for pattern in spec.grounding.numeric.ignore_patterns.values():
        masked = re.sub(pattern, " ", masked)
    return tuple(match.group(0) for match in _NUMBER.finditer(masked))


def token_is_grounded(
    token: str, whitelist: frozenset[Decimal], spec: NarrateConfig
) -> bool:
    """Does any stored quantity round to this written number?

    Rounding at the TOKEN's precision, which is the honest reading of
    "with rounding tolerance". A sentence that writes 1.88 is claiming two
    decimal places and a stored 1.8823 satisfies it; one that writes 1.9
    is claiming one place, and 1.8823 satisfies that too. Fewer decimals
    is a wider window because that is what rounding means — "about 2
    crore" is a true thing to say about 1.88.

    What it refuses is a figure nothing rounds to at the precision it was
    written to: 1.87, 9.99, 41.7. Widening this to nearest-neighbour
    proximity would ground any number within a few percent of anything,
    which is most numbers.
    """
    numeric = spec.grounding.numeric
    written = _decimal(token.replace(",", "").replace("−", "-"))
    if written is None:
        return False

    magnitude = abs(written)
    if written == written.to_integral_value() and magnitude <= numeric.allow_small_integers_up_to:
        # "the two hard gates", "all six tests". A structural count, not a
        # claim about the business — and a fabricated figure is never 3.
        return True

    places = -written.as_tuple().exponent
    tolerance = numeric.tolerance_relative
    for stored in whitelist:
        if round(stored, places) == written:
            return True
        if magnitude and abs(stored - written) <= abs(written) * Decimal(str(tolerance)):
            return True
    return False


# ---------------------------------------------------------------------------
# Hypotheses and the hard gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisGates:
    """One hypothesis, and whether it may be spoken about causally."""

    hypothesis_id: str
    label: str
    status: str
    evidence_ids: frozenset[str]
    hard_gates_passed: bool
    hard_gates_run: tuple[str, ...]
    failed_gates: tuple[str, ...]


def hypothesis_gates(
    adjudication: Mapping[str, Any], layer: SemanticLayer
) -> dict[str, HypothesisGates]:
    """Which hypotheses cleared BOTH hard gates.

    "Both" is checked against the semantic layer's own list of hard-gate
    tests, not against whatever the object happens to carry. A hypothesis
    that was never given a hard gate has not passed one — an untested
    claim and a tested-and-passed claim must not read the same on screen.
    """
    names = tuple(layer.narrate.grounding.hard_gate_tests)
    by_id = {
        spec.test_id: name
        for name, spec in layer.adjudication.tests.items()
        if name in names
    }
    gates: dict[str, HypothesisGates] = {}
    for entry in adjudication.get("hypotheses", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        results: dict[str, bool] = {}
        for test in entry.get("tests", ()) or ():
            if not isinstance(test, Mapping):
                continue
            name = by_id.get(test.get("test_id"))
            if name is None and test.get("type") == "hard_gate":
                name = str(test.get("name", test.get("test_id")))
            if name is not None:
                results[name] = bool(test.get("passed"))
        failed = tuple(sorted(name for name in names if not results.get(name, False)))
        hypothesis_id = str(entry.get("hypothesis_id", ""))
        gates[hypothesis_id] = HypothesisGates(
            hypothesis_id=hypothesis_id,
            label=str(entry.get("label", "")),
            status=str(entry.get("status", "")),
            evidence_ids=frozenset(str(x) for x in entry.get("evidence_ids", ()) or ()),
            hard_gates_passed=not failed,
            hard_gates_run=tuple(sorted(results)),
            failed_gates=failed,
        )
    return gates


def hypotheses_for(
    claim: Claim, gates: Mapping[str, HypothesisGates]
) -> tuple[HypothesisGates, ...]:
    """Which hypotheses a sentence is about.

    THE UNION OF THREE READINGS, on purpose. What the model declared, what
    the sentence names, and what its cited evidence belongs to. Widening
    rather than narrowing is what stops the declaration from being a way
    out: dropping `hypothesis_id` from a sentence that names H4 and cites
    H4's evidence does not make the sentence unattributable, it makes it
    attributable twice.
    """
    lowered = claim.sentence.lower()
    cited = set(claim.evidence_ids)
    found: dict[str, HypothesisGates] = {}

    for hypothesis_id, gate in gates.items():
        if claim.hypothesis_id and claim.hypothesis_id == hypothesis_id:
            found[hypothesis_id] = gate
            continue
        if hypothesis_id and re.search(rf"\b{re.escape(hypothesis_id.lower())}\b", lowered):
            found[hypothesis_id] = gate
            continue
        if gate.label and gate.label.lower() in lowered:
            found[hypothesis_id] = gate
            continue
        if cited & gate.evidence_ids:
            found[hypothesis_id] = gate
    return tuple(found.values())


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def check_claim(
    claim: Claim,
    *,
    ids: frozenset[str],
    whitelist: frozenset[Decimal],
    gates: Mapping[str, HypothesisGates],
    layer: SemanticLayer,
) -> CheckedClaim:
    """Apply all four rules to one sentence. Reports every failure, not the first."""
    spec = layer.narrate
    language = spec.language
    violations: list[Violation] = []

    # --- 1. linkage --------------------------------------------------------
    minimum = spec.grounding.min_evidence_ids_per_sentence
    if len(claim.evidence_ids) < minimum:
        violations.append(
            Violation(
                NO_EVIDENCE,
                f"carries {len(claim.evidence_ids)} evidence ids, needs at least "
                f"{minimum}. Cite the id of the number this sentence rests on.",
            )
        )
    for evidence_id in claim.evidence_ids:
        if evidence_id not in ids:
            violations.append(
                Violation(
                    UNKNOWN_EVIDENCE,
                    f"cites {evidence_id!r}, which is not in the adjudication object.",
                    token=evidence_id,
                )
            )

    # --- 2. numbers --------------------------------------------------------
    for token in numeric_tokens(claim.sentence, spec):
        if not token_is_grounded(token, whitelist, spec):
            violations.append(
                Violation(
                    FABRICATED_NUMBER,
                    f"the number {token!r} does not appear in the adjudication object. "
                    "Use only figures from the object you were given.",
                    token=token,
                )
            )

    # --- 3. causal connectives, gated on the hard gates --------------------
    referenced = hypotheses_for(claim, gates)
    ungated = tuple(gate for gate in referenced if not gate.hard_gates_passed)
    if ungated:
        for connective in language.causal_connectives:
            if _phrase_pattern(connective).search(claim.sentence):
                names = ", ".join(
                    f"{gate.hypothesis_id} (failed {', '.join(gate.failed_gates)})"
                    for gate in ungated
                )
                violations.append(
                    Violation(
                        UNGATED_CAUSAL_CLAIM,
                        f"asserts cause with {connective!r} about {names}. A hypothesis "
                        "that did not pass both hard gates may be described, never "
                        "blamed.",
                        token=connective,
                    )
                )
                break

    # --- 4. language, unconditional ---------------------------------------
    for rule in language.bare_causal_claims:
        if re.search(rule.pattern, claim.sentence, re.IGNORECASE):
            violations.append(
                Violation(
                    BARE_CAUSAL_CLAIM,
                    f"{' '.join(rule.why.split())} Write it as: "
                    f"{' '.join(language.required_attribution_form.split())}.",
                )
            )
            break
    for rule in language.forbidden_phrases:
        if _phrase_pattern(rule.phrase).search(claim.sentence):
            violations.append(
                Violation(
                    FORBIDDEN_PHRASE,
                    f"uses {rule.phrase!r}. Write {rule.instead!r} instead — "
                    f"{' '.join(rule.why.split())}",
                    token=rule.phrase,
                )
            )

    return CheckedClaim(claim=claim, violations=tuple(violations))


def check(
    claims: Sequence[Claim],
    adjudication: Mapping[str, Any],
    layer: SemanticLayer,
) -> tuple[CheckedClaim, ...]:
    """Check every claim against one frozen adjudication object."""
    ids = evidence_ids(adjudication)
    whitelist = numeric_whitelist(adjudication, layer.narrate)
    gates = hypothesis_gates(adjudication, layer)
    return tuple(
        check_claim(claim, ids=ids, whitelist=whitelist, gates=gates, layer=layer)
        for claim in claims
    )


def report(
    checked: Sequence[CheckedClaim], *, regenerated: int = 0
) -> GroundingReport:
    """Turn a checked slate into the three counts the UI prints."""
    linked = [item for item in checked if item.grounded]
    stripped = [item for item in checked if not item.grounded]
    return GroundingReport(
        claims_checked=len(checked),
        claims_linked=len(linked),
        claims_stripped=len(stripped),
        claims_regenerated=regenerated,
        violations=tuple(v for item in stripped for v in item.violations),
    )


def feedback(checked: Sequence[CheckedClaim]) -> str:
    """What to tell the model about the sentences that failed.

    Quoted verbatim into the one regeneration it gets. Naming the rule and
    the offending token is the difference between a second attempt that
    fixes the problem and a second attempt that is merely shorter.
    """
    lines: list[str] = []
    for item in checked:
        if item.grounded:
            continue
        lines.append(f"- {item.claim.sentence.strip()}")
        lines.extend(f"    {violation.render()}" for violation in item.violations)
    return "\n".join(lines)


__all__ = [
    "BARE_CAUSAL_CLAIM",
    "FABRICATED_NUMBER",
    "FORBIDDEN_PHRASE",
    "NO_EVIDENCE",
    "UNGATED_CAUSAL_CLAIM",
    "UNKNOWN_EVIDENCE",
    "CheckedClaim",
    "Claim",
    "GroundingError",
    "GroundingReport",
    "HypothesisGates",
    "Violation",
    "check",
    "check_claim",
    "evidence_ids",
    "feedback",
    "hypotheses_for",
    "hypothesis_gates",
    "numeric_tokens",
    "numeric_whitelist",
    "report",
    "token_is_grounded",
]
