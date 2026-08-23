"""Long-tail hypothesis generation. The third source, and the smallest.

The causal graph covers what the business has already thought of, and the
case registry covers what has happened here before. This covers neither:
it asks the model what else could move this KPI in this scope, and offers
the answers as candidates at a low prior.

Three things keep it honest.

It enters at a prior declared in `semantic_layer/gather.yaml`, below every
template in the graph. For a well-covered KPI like net revenue a generated
candidate will not make the top five, and that is the correct outcome —
the graph already has better answers. It earns its place on a KPI the
graph is thin on, which is exactly where a long tail is worth having.

It gets its own tag namespace (`lt_`), so a generated hypothesis can never
be mistaken for one the business declared, in a table or on a screen.

And it produces no number at all. A candidate is a label and a sentence.
Its prior comes from the config, its applicability from the same screening
every other candidate goes through, and its evidence from the same three
lanes. Nothing the model returns here is arithmetic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from llm.provider import LLMMessage, LLMProvider, LLMRequest, get_provider
from semantic_layer.schema import LongTailSource

TASK = "hypothesise"

SYSTEM_PROMPT = """\
You propose candidate explanations for a movement in a retail KPI.

Return a JSON array of objects, each with:
  label        a short name for the cause, 2 to 6 words
  description  one sentence saying what would have had to happen
  testable_by  one short phrase naming the data that would test it

Rules:
  - Do NOT repeat any of the causes already listed as known.
  - Do NOT return a probability, a percentage, an effect size, or any
    other number. You are proposing what to look at, not how big it is.
  - Propose causes that could be TESTED against retail data, not
    speculation about intent.
  - Fewer good candidates beats more weak ones."""


class HypothesisGenerationError(RuntimeError):
    """The model's reply could not be read as candidates."""


@dataclass(frozen=True)
class LongTailCandidate:
    """A hypothesis the graph does not contain.

    `prior` is the configured long-tail prior, not the model's opinion —
    the model is not asked for one and would not be believed if it
    volunteered one.
    """

    tag: str
    label: str
    description: str
    testable_by: str
    prior: float

    @property
    def generated(self) -> bool:
        return True


_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def build_request(
    kpi: str,
    scope: str,
    period: str,
    direction: str,
    known: Sequence[str],
    spec: LongTailSource,
    *,
    model: str | None = None,
) -> LLMRequest:
    """One call, deterministic. The known causes are listed so it cannot repeat them."""
    listed = "\n".join(f"  {name}" for name in sorted(known))
    prompt = (
        f"KPI: {kpi}\n"
        f"Scope: {scope}\n"
        f"Period: {period}\n"
        f"Direction of the unexplained movement: {direction}\n\n"
        f"Causes already under consideration:\n{listed}\n\n"
        f"Propose at most {spec.max_candidates} further candidates. "
        "Return the JSON array now."
    )
    return LLMRequest.for_task(
        TASK,
        system=SYSTEM_PROMPT,
        messages=(LLMMessage(role="user", content=prompt),),
        max_tokens=spec.max_candidates * _TOKENS_PER_CANDIDATE,
        temperature=0.0,
        **({"model": model} if model else {}),
    )


_TOKENS_PER_CANDIDATE = 96


def parse_response(text: str, spec: LongTailSource) -> list[dict[str, str]]:
    match = _JSON_ARRAY.search(text)
    if match is None:
        raise HypothesisGenerationError(f"no JSON array in the reply: {text[:200]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise HypothesisGenerationError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise HypothesisGenerationError("reply is not a JSON array")

    parsed: list[dict[str, str]] = []
    for entry in payload[: spec.max_candidates]:
        if not isinstance(entry, dict):
            raise HypothesisGenerationError(f"candidate is not an object: {entry!r}")
        missing = [
            name for name in ("label", "description", "testable_by") if name not in entry
        ]
        if missing:
            raise HypothesisGenerationError(f"candidate missing {missing}: {entry!r}")
        parsed.append({name: str(entry[name]) for name in entry})
    return parsed


def slug(label: str, prefix: str) -> str:
    """`Warehouse dispatch backlog` -> `lt_warehouse_dispatch_backlog`."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return f"{prefix}{cleaned}"


def generate(
    kpi: str,
    scope: str,
    period: str,
    direction: str,
    known: Sequence[str],
    spec: LongTailSource,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> tuple[LongTailCandidate, ...]:
    """Ask for candidates the graph does not already hold.

    A candidate whose slug collides with a known cause is dropped: the
    model was asked not to repeat, and quietly renaming a repeat would
    make the screening double-count it.
    """
    if not spec.enabled or spec.max_candidates <= 0:
        return ()

    provider = provider or get_provider()
    request = build_request(kpi, scope, period, direction, known, spec, model=model)
    response = provider.complete(request)

    taken = set(known)
    candidates: list[LongTailCandidate] = []
    for entry in parse_response(response.text, spec):
        tag = slug(entry["label"], spec.tag_prefix)
        if tag in taken or entry["label"] in taken:
            continue
        taken.add(tag)
        candidates.append(
            LongTailCandidate(
                tag=tag,
                label=entry["label"],
                description=entry["description"],
                testable_by=entry["testable_by"],
                prior=spec.prior,
            )
        )
    return tuple(candidates)


__all__ = [
    "SYSTEM_PROMPT",
    "TASK",
    "HypothesisGenerationError",
    "LongTailCandidate",
    "build_request",
    "generate",
    "parse_response",
    "slug",
]
