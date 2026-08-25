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

NOVELTY IS FLAGGED, NOT ASSUMED. A candidate the causal graph does not
contain carries `novel: true` all the way to the screen, so nobody ever
reads a model's suggestion as something the business declared.

The harder half is the other direction. A model told not to repeat the
known causes will not repeat the WORDING — it will repeat the idea under
a new name, and "Shelf replenishment shortfall" arriving as a novel
hypothesis beside the graph's own `stock_out` would double-count the same
explanation and split its evidence across two tags. So a proposal is
compared against every graph label on token overlap, and one that is a
graph hypothesis in different words is REJECTED with the match named.
What survives is either genuinely absent from the graph and flagged, or
it is not offered at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

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
    #: Always true for anything this module offers. Carried as a field
    #: rather than inferred from the tag prefix so that a caller cannot
    #: lose the flag by copying the label into another structure.
    novel: bool = True

    @property
    def generated(self) -> bool:
        return True


@dataclass(frozen=True)
class Rejected:
    """A proposal that was not offered, and why."""

    label: str
    reason: str
    matched: str | None = None

    def render(self) -> str:
        matched = f" (matches {self.matched})" if self.matched else ""
        return f"{self.label}: {self.reason}{matched}"


#: Reasons, named so a test can assert on one.
ALREADY_IN_GRAPH = "already in the causal graph under another name"
DUPLICATE_TAG = "collides with a cause already under consideration"


@dataclass(frozen=True)
class Proposal:
    """What the long tail offered, and what it did not."""

    candidates: tuple[LongTailCandidate, ...]
    rejected: tuple[Rejected, ...] = ()

    def render(self) -> str:
        lines = [f"long tail: {len(self.candidates)} offered, {len(self.rejected)} rejected"]
        lines += [f"  novel: {c.label}" for c in self.candidates]
        lines += [f"  rejected: {r.render()}" for r in self.rejected]
        return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def _tokens(text: str, stopwords: frozenset[str]) -> frozenset[str]:
    """Meaning-bearing words, lowercased. Stopwords removed first.

    Without the stopword pass, "a decline in store footfall" and "a
    decline in store staffing" share three words out of five and look more
    alike than they are.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(word for word in words if word not in stopwords)


def overlap(left: str, right: str, stopwords: frozenset[str]) -> float:
    """How much of the SHORTER phrase the longer one already contains.

    Deliberately not Jaccard. "Stock-out" against "Stock-out on ranged
    SKUs in the affected stores" scores 1.0 here and about 0.25 on
    Jaccard, and the first is the right answer: the longer phrase is the
    shorter one with detail added, which is exactly the restatement this
    check exists to catch.
    """
    a, b = _tokens(left, stopwords), _tokens(right, stopwords)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def graph_restatement(
    label: str, description: str, known_labels: Mapping[str, str], spec: LongTailSource
) -> str | None:
    """The graph hypothesis this proposal is a rewording of, if it is one.

    Compared on the label first and on the description second. A model
    that renames `stock_out` to "Shelf availability shortfall" shares no
    label tokens at all, and gives itself away in the sentence describing
    what would have had to happen.
    """
    stopwords = frozenset(spec.novelty_stopwords)
    threshold = spec.novelty_overlap_threshold
    for tag, graph_label in known_labels.items():
        for proposed in (label, description):
            if overlap(proposed, graph_label, stopwords) >= threshold:
                return tag
            if overlap(proposed, tag.replace("_", " "), stopwords) >= threshold:
                return tag
    return None


def graph_labels(layer) -> dict[str, str]:
    """Every hypothesis the causal graph declares: tag -> label."""
    return {
        tag: template.label for tag, template in layer.causal_graph.hypotheses.items()
    }


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
    layer=None,
) -> tuple[LongTailCandidate, ...]:
    """Ask for candidates the graph does not already hold.

    The offered candidates only. `propose` returns the same thing plus
    what was turned away, for a caller that wants to show its working.
    """
    return propose(
        kpi, scope, period, direction, known, spec,
        provider=provider, model=model, layer=layer,
    ).candidates


def propose(
    kpi: str,
    scope: str,
    period: str,
    direction: str,
    known: Sequence[str],
    spec: LongTailSource,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    layer=None,
) -> Proposal:
    """Ask for candidates, and screen every one for novelty.

    Two ways a proposal is turned away, and they are different failures.

    A slug COLLISION means the model returned something already under
    consideration by the same name. Quietly renaming it would make the
    screening count one explanation twice.

    A RESTATEMENT means it returned a graph hypothesis in different words.
    That is the one worth catching properly: admitted, it would enter as
    `novel: true` at the long-tail prior, split its evidence across two
    tags, and let the same cause compete against itself for the residual.

    Everything that survives is genuinely absent from the graph and is
    flagged `novel: true` — which is the whole contract of this module.
    Nothing leaves here unflagged.
    """
    if not spec.enabled or spec.max_candidates <= 0:
        return Proposal(candidates=())

    provider = provider or get_provider()
    request = build_request(kpi, scope, period, direction, known, spec, model=model)
    response = provider.complete(request)

    if layer is None:
        from semantic_layer.schema import get_semantic_layer

        layer = get_semantic_layer()
    known_labels = graph_labels(layer)

    taken = set(known)
    candidates: list[LongTailCandidate] = []
    rejected: list[Rejected] = []
    for entry in parse_response(response.text, spec):
        tag = slug(entry["label"], spec.tag_prefix)
        if tag in taken or entry["label"] in taken:
            rejected.append(Rejected(entry["label"], DUPLICATE_TAG, matched=tag))
            continue
        match = graph_restatement(
            entry["label"], entry.get("description", ""), known_labels, spec
        )
        if match is not None:
            rejected.append(Rejected(entry["label"], ALREADY_IN_GRAPH, matched=match))
            continue
        taken.add(tag)
        candidates.append(
            LongTailCandidate(
                tag=tag,
                label=entry["label"],
                description=entry["description"],
                testable_by=entry["testable_by"],
                prior=spec.prior,
                novel=True,
            )
        )
    return Proposal(candidates=tuple(candidates), rejected=tuple(rejected))


__all__ = [
    "ALREADY_IN_GRAPH",
    "DUPLICATE_TAG",
    "SYSTEM_PROMPT",
    "TASK",
    "HypothesisGenerationError",
    "LongTailCandidate",
    "Proposal",
    "Rejected",
    "build_request",
    "generate",
    "graph_labels",
    "graph_restatement",
    "overlap",
    "parse_response",
    "propose",
    "slug",
]
