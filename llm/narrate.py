"""Narrative generation. The model writes prose; it never writes a number.

The input is a FROZEN adjudication object — the JSON form of
`engine/contracts.py::Adjudication`, exactly as VERDICT left it. The model
may render it and may not alter it (CLAUDE.md rule 1). It gets no
warehouse connection, no tools, and no second look at the data.

What comes back is not paragraphs. It is a list of

    [{"sentence": ..., "evidence_ids": [...], "hypothesis_id": ...}]

so that every sentence is separately checkable and separately droppable.
Asking for prose and then trying to attribute it afterwards does not work
— by the time it is one block of text, a fabricated figure in the third
sentence costs you the whole paragraph, and the usual fix for that is to
lower the standard.

THE LOOP, and it only goes round once:

    generate  ->  ground  ->  strip the failures
                              -> regenerate them, naming what was wrong
                                 ->  ground again  ->  drop what still fails

One regeneration, from `narrate.yaml`. A model that could not ground a
sentence when told precisely what was wrong with it does not do better on
a third attempt; it writes something vaguer that happens to pass. A
shorter narrative that is entirely checkable beats a complete one that is
not.

PER PERSONA. The same object, told to a chief executive, a regional
manager and an analyst. They differ in what they lead with rather than in
tone, because the three have different first questions and the first
sentence is the only one some of them read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from llm.grounding import (
    Claim,
    CheckedClaim,
    GroundingReport,
    check,
    feedback,
    hypothesis_gates,
    report,
)
from llm.provider import LLMMessage, LLMProvider, LLMRequest, get_provider
from semantic_layer.schema import PersonaNarrative, SemanticLayer

TASK = "narrate"

#: Generous per sentence: the sentence itself, a couple of evidence ids
#: and a hypothesis tag, as JSON.
_TOKENS_PER_SENTENCE = 160

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class NarrationError(RuntimeError):
    """The model's reply could not be read as claims."""


SYSTEM_PROMPT = """\
You write the case summary for one reader of a completed business
investigation. You are given the finished case as JSON. It is frozen.

Return a JSON array. Each element is one sentence:
  sentence       one complete sentence, plain text
  evidence_ids   ids from the case object that this sentence rests on
  hypothesis_id  the hypothesis the sentence is about, or null

ABSOLUTE RULES. A sentence breaking any of these is deleted before the
reader sees it, and you are not asked again.

  1. Every sentence carries at least one evidence id from the object.
  2. Every number you write appears in the object. Do not compute, round
     beyond the figure's own precision, convert units, or total anything.
     If a number is not in the object, do not write it.
  3. You may assert cause ONLY for a hypothesis marked
     `hard_gates_passed: true` below. For any other hypothesis, describe
     what was observed and say what it failed on. Never say a hypothesis
     caused, led to, resulted in or was responsible for anything unless
     it passed both gates.
  4. Say the SHARE, never the whole. A hypothesis explains a percentage
     of a residual; a sentence that omits the share claims more than the
     case does.

Write the fewest sentences that answer this reader's questions."""


@dataclass(frozen=True)
class Narrative:
    """One persona's summary, and the audit of how it got there."""

    case_id: str
    persona: str
    claims: tuple[Claim, ...]
    dropped: tuple[CheckedClaim, ...]
    report: GroundingReport
    regenerated: bool
    model: str
    from_fixture: bool

    @property
    def text(self) -> str:
        """The narrative as a reader sees it."""
        return " ".join(claim.sentence.strip() for claim in self.claims)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "persona": self.persona,
            "claims": [claim.as_dict() for claim in self.claims],
            "grounding": self.report.as_dict(),
        }

    def render(self) -> str:
        lines = [f"[{self.persona}] {self.text}", f"  {self.report.render()}"]
        for item in self.dropped:
            lines.append(f"  DROPPED: {item.claim.sentence.strip()}")
            lines.extend(f"    {v.render()}" for v in item.violations)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def freeze(adjudication: Mapping[str, Any]) -> str:
    """The object as the model sees it: canonical, sorted, stable.

    Stable because the request fingerprint is the fixture key. Two runs of
    the same case must produce byte-identical prompts or the offline demo
    stops replaying (rule 8).
    """
    return json.dumps(
        adjudication, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def gate_summary(adjudication: Mapping[str, Any], layer: SemanticLayer) -> str:
    """Which hypotheses may be spoken about causally, stated up front.

    Computed by the same function the validator uses, so the prompt and
    the check can never disagree about what passed. Telling the model the
    answer is not a substitute for checking it — it is how the check stops
    being expensive.
    """
    gates = hypothesis_gates(adjudication, layer)
    if not gates:
        return "  (no hypotheses on this case)"
    lines = []
    for gate in gates.values():
        status = (
            "hard_gates_passed: true — you may assert cause"
            if gate.hard_gates_passed
            else f"hard_gates_passed: false (failed {', '.join(gate.failed_gates)}) "
            "— describe only, never blame"
        )
        lines.append(f"  {gate.hypothesis_id} {gate.label}: {status}")
    return "\n".join(lines)


def build_request(
    adjudication: Mapping[str, Any],
    persona: str,
    layer: SemanticLayer,
    *,
    corrections: str | None = None,
    kept: Sequence[Claim] = (),
    model: str | None = None,
) -> LLMRequest:
    """One call, fully described and deterministic.

    `corrections` is the regeneration path: the sentences that failed and
    the rule each of them broke, verbatim from the validator. `kept` is
    what already grounded, listed so the second attempt replaces rather
    than repeats.
    """
    spec = layer.narrate
    profile = _persona(persona, layer)
    language = spec.language

    banned = "\n".join(
        f"  never {rule.phrase!r} — write {rule.instead!r} ({' '.join(rule.why.split())})"
        for rule in language.forbidden_phrases
    )
    style = "\n".join(f"  {line}" for line in language.house_style)

    sections = [
        f"Reader: {profile.display_name}",
        f"Audience: {' '.join(profile.audience.split())}",
        f"Lead with: {profile.leads_with}",
        f"Voice: {' '.join(profile.voice.split())}",
        f"Must cover: {', '.join(profile.must_cover)}",
        f"Omit: {', '.join(profile.omit) or 'nothing'}",
        f"At most {profile.max_sentences} sentences.",
        "",
        "Hypotheses and their hard gates:",
        gate_summary(adjudication, layer),
        "",
        "Banned phrasings:",
        banned,
        "",
        "House style:",
        style,
        "",
        "Attribution must take this form: "
        + " ".join(language.required_attribution_form.split()),
        "",
        "The case object:",
        freeze(adjudication),
    ]

    if corrections:
        sections += [
            "",
            "These sentences from your previous answer were DELETED, with the "
            "rule each of them broke:",
            corrections,
            "",
            "These grounded and are already published — do not repeat them:",
            "\n".join(f"  {claim.sentence.strip()}" for claim in kept) or "  (none)",
            "",
            "Write replacements for the deleted sentences only. Same JSON array "
            "format. Fewer sentences is an acceptable answer.",
        ]
    else:
        sections += ["", "Return the JSON array now."]

    return LLMRequest.for_task(
        TASK,
        system=SYSTEM_PROMPT,
        messages=(LLMMessage(role="user", content="\n".join(sections)),),
        max_tokens=profile.max_sentences * _TOKENS_PER_SENTENCE,
        temperature=0.0,
        **({"model": model} if model else {}),
    )


def _persona(persona: str, layer: SemanticLayer) -> PersonaNarrative:
    profile = layer.narrate.personas.get(persona)
    if profile is None:
        raise NarrationError(
            f"no narrative persona {persona!r}; narrate.yaml declares "
            f"{sorted(layer.narrate.personas)}"
        )
    return profile


# ---------------------------------------------------------------------------
# Reading the reply
# ---------------------------------------------------------------------------


def parse_response(text: str) -> tuple[Claim, ...]:
    """Read the reply as claims.

    Tolerates a fenced block or a line of preamble. Does not tolerate a
    missing `sentence` or a missing `evidence_ids` — a claim that arrives
    without either is not a claim with a default, it is an unreadable
    reply, and guessing at it here would defeat the point of asking for
    structure.
    """
    match = _JSON_ARRAY.search(text)
    if match is None:
        raise NarrationError(f"no JSON array in the reply: {text[:200]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise NarrationError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise NarrationError("reply is not a JSON array")

    claims: list[Claim] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise NarrationError(f"claim is not an object: {entry!r}")
        if "sentence" not in entry:
            raise NarrationError(f"claim has no sentence: {entry!r}")
        if "evidence_ids" not in entry:
            raise NarrationError(f"claim has no evidence_ids: {entry!r}")
        ids = entry["evidence_ids"] or []
        if isinstance(ids, str):
            ids = [ids]
        hypothesis = entry.get("hypothesis_id")
        claims.append(
            Claim(
                sentence=str(entry["sentence"]).strip(),
                evidence_ids=tuple(str(item) for item in ids),
                hypothesis_id=str(hypothesis) if hypothesis else None,
            )
        )
    return tuple(claims)


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def narrate(
    adjudication: Mapping[str, Any],
    persona: str,
    layer: SemanticLayer,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> Narrative:
    """Write, check, regenerate once, drop the rest.

    The counts in the returned report are over the FINAL slate: a sentence
    that failed and was successfully replaced is counted once, as the
    replacement that survived. That keeps `checked == linked + stripped`
    true, which is what makes `14/14 claims linked · 0 stripped` a
    sentence a reader can interpret without a footnote.
    """
    provider = provider or get_provider()
    profile = _persona(persona, layer)

    request = build_request(adjudication, persona, layer, model=model)
    response = provider.complete(request)
    claims = parse_response(response.text)[: profile.max_sentences]
    checked = check(claims, adjudication, layer)

    kept = [item.claim for item in checked if item.grounded]
    failed = [item for item in checked if not item.grounded]
    regenerated = 0

    if failed and layer.narrate.grounding.max_regenerations:
        retry = build_request(
            adjudication,
            persona,
            layer,
            corrections=feedback(checked),
            kept=kept,
            model=model,
        )
        retry_response = provider.complete(retry)
        replacements = parse_response(retry_response.text)[: len(failed)]
        regenerated = len(replacements)
        rechecked = check(replacements, adjudication, layer)
        checked = tuple(item for item in checked if item.grounded) + rechecked
        kept = [item.claim for item in checked if item.grounded]
        failed = [item for item in checked if not item.grounded]

    return Narrative(
        case_id=str(adjudication.get("case_id", "")),
        persona=persona,
        claims=tuple(kept),
        dropped=tuple(failed),
        report=report(checked, regenerated=regenerated),
        regenerated=bool(regenerated),
        model=response.model,
        from_fixture=response.from_fixture,
    )


def narrate_all(
    adjudication: Mapping[str, Any],
    layer: SemanticLayer,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> dict[str, Narrative]:
    """Every declared persona, over one frozen object."""
    provider = provider or get_provider()
    return {
        persona: narrate(
            adjudication, persona, layer, provider=provider, model=model
        )
        for persona in sorted(layer.narrate.personas)
    }


__all__ = [
    "SYSTEM_PROMPT",
    "TASK",
    "NarrationError",
    "Narrative",
    "build_request",
    "freeze",
    "gate_summary",
    "narrate",
    "narrate_all",
    "parse_response",
]
