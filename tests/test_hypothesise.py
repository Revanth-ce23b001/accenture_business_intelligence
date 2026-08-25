"""Long-tail hypothesis generation, and the novelty rule.

The brief: hypothesise.py "must not propose a hypothesis absent from the
causal graph without flagging it as novel".

Two halves, and the second is the one that bites. Flagging what IS novel
is a boolean. Recognising what is NOT — a graph hypothesis the model has
renamed — takes actual work, because a model told "do not repeat these
causes" complies with the letter by changing the words. Admitted, the
rename would enter at the long-tail prior under a fresh tag, split the
evidence for one explanation across two candidates, and let the same cause
compete against itself for the residual.
"""

from __future__ import annotations

import json

import pytest

from llm.hypothesise import (
    ALREADY_IN_GRAPH,
    DUPLICATE_TAG,
    LongTailCandidate,
    Proposal,
    graph_labels,
    graph_restatement,
    overlap,
    propose,
    slug,
)
from llm.provider import LLMResponse


@pytest.fixture(scope="module")
def spec(layer):
    return layer.gather.hypotheses.sources.long_tail


class _Replies:
    """A provider returning one scripted list of candidates."""

    name = "scripted"

    def __init__(self, candidates):
        self.candidates = candidates

    def complete(self, request):
        return LLMResponse(
            text=json.dumps(self.candidates),
            model=request.model,
            request_fingerprint=request.fingerprint(),
        )


def _candidate(label, description="Something happened.", testable_by="some feed"):
    return {"label": label, "description": description, "testable_by": testable_by}


def _propose(candidates, layer, spec, known=()):
    return propose(
        "net_revenue", "West", "2025-11", "down", list(known), spec,
        provider=_Replies(candidates), layer=layer,
    )


# ===========================================================================
# Flagging
# ===========================================================================


def test_a_genuinely_new_candidate_is_offered_and_flagged(layer, spec):
    result = _propose([_candidate("Payment terminal downtime")], layer, spec)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.novel is True
    assert candidate.generated is True
    assert candidate.tag == "lt_payment_terminal_downtime"


def test_nothing_leaves_this_module_unflagged(layer, spec):
    """The contract of the whole file, asserted over a mixed slate."""
    result = _propose(
        [
            _candidate("Payment terminal downtime"),
            _candidate("Warehouse dispatch backlog"),
            _candidate("Loyalty programme lapse"),
        ],
        layer,
        spec,
    )
    assert result.candidates
    assert all(candidate.novel for candidate in result.candidates)
    assert all(candidate.tag.startswith(spec.tag_prefix) for candidate in result.candidates)


def test_the_prior_is_the_configs_and_never_the_models(layer, spec):
    """The model is not asked for a prior and would not be believed."""
    result = _propose([_candidate("Payment terminal downtime")], layer, spec)
    assert result.candidates[0].prior == spec.prior


def test_a_generated_tag_can_never_be_read_as_a_declared_one(layer, spec):
    result = _propose([_candidate("Stock shortfall at till")], layer, spec)
    for candidate in result.candidates:
        assert candidate.tag not in layer.causal_graph.hypotheses


# ===========================================================================
# Restatement
# ===========================================================================


def test_a_graph_hypothesis_in_different_words_is_rejected(layer, spec):
    """The half that takes work."""
    result = _propose([_candidate("Stock-out on ranged lines")], layer, spec)
    assert result.candidates == ()
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == ALREADY_IN_GRAPH
    assert result.rejected[0].matched == "stock_out"


def test_a_restatement_is_caught_by_its_description_when_the_label_hides_it(
    layer, spec
):
    """A rename shares no label tokens and gives itself away in the sentence."""
    result = _propose(
        [
            _candidate(
                "Empty shelf syndrome",
                description="Stock-out on ranged SKUs across the treated stores.",
            )
        ],
        layer,
        spec,
    )
    assert result.candidates == ()
    assert result.rejected[0].matched == "stock_out"


def test_a_repeat_under_the_same_name_is_rejected_separately(layer, spec):
    """A different failure from a rename, and reported as one."""
    result = _propose(
        [_candidate("Payment terminal downtime")],
        layer,
        spec,
        known=["lt_payment_terminal_downtime"],
    )
    assert result.candidates == ()
    assert result.rejected[0].reason == DUPLICATE_TAG


def test_rejections_are_reported_not_silently_dropped(layer, spec):
    result = _propose(
        [_candidate("Adverse weather"), _candidate("Payment terminal downtime")],
        layer,
        spec,
    )
    assert len(result.candidates) == 1
    assert len(result.rejected) == 1
    assert "weather" in result.render()
    assert "novel" in result.render()


# ===========================================================================
# The comparison itself
# ===========================================================================


def test_overlap_is_against_the_shorter_phrase(layer, spec):
    """Deliberately not Jaccard.

    "Stock-out" against "Stock-out on ranged SKUs in the affected stores"
    is the shorter phrase with detail added — a restatement — and scores
    1.0 here against roughly 0.25 on Jaccard.
    """
    stopwords = frozenset(spec.novelty_stopwords)
    assert overlap("Stock-out", "Stock-out on ranged SKUs", stopwords) == 1.0


def test_stopwords_stop_filler_from_carrying_the_comparison(spec):
    """They drop the score; they are not the whole defence.

    "a decline in store footfall" against "a decline in store staffing"
    scores 0.8 on raw words — four of five match, and three of those four
    are filler. Removing the filler takes it to 0.67, which is the honest
    figure for two phrases that really do share two content words out of
    three. Short phrases stay close; the threshold is set where the real
    labels are, which the next test measures.
    """
    stopwords = frozenset(spec.novelty_stopwords)
    with_filler = overlap(
        "a decline in store footfall", "a decline in store staffing", frozenset()
    )
    without = overlap(
        "a decline in store footfall", "a decline in store staffing", stopwords
    )
    assert with_filler == pytest.approx(0.8)
    assert without < with_filler


def test_no_two_graph_hypotheses_read_as_restatements_of_each_other(layer, spec):
    """What justifies the threshold, measured rather than asserted.

    The fourteen declared hypotheses are the closest set of genuinely
    distinct causes this system holds. If the rule cannot tell two of THEM
    apart it would reject real novel candidates, and the number in
    gather.yaml would be wrong. The worst real pair scores 0.5 against a
    threshold of 0.6.
    """
    from itertools import combinations

    stopwords = frozenset(spec.novelty_stopwords)
    labels = graph_labels(layer)
    worst = 0.0
    for (tag_a, label_a), (tag_b, label_b) in combinations(labels.items(), 2):
        worst = max(
            worst,
            overlap(label_a, label_b, stopwords),
            overlap(label_a, tag_b.replace("_", " "), stopwords),
            overlap(tag_a.replace("_", " "), label_b, stopwords),
        )
    assert worst < spec.novelty_overlap_threshold, (
        f"two declared hypotheses score {worst:.2f}, at or above the "
        f"{spec.novelty_overlap_threshold} restatement threshold"
    )


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Adverse weather", "weather"),
        ("Store closure", "store_closure"),
        ("Payment terminal downtime", None),
        ("Loyalty programme lapse", None),
    ],
)
def test_restatement_detection(label, expected, layer, spec):
    assert graph_restatement(label, "", graph_labels(layer), spec) == expected


def test_graph_labels_covers_every_declared_hypothesis(layer):
    assert set(graph_labels(layer)) == set(layer.causal_graph.hypotheses)


# ===========================================================================
# Plumbing
# ===========================================================================


def test_the_long_tail_can_be_switched_off(layer, spec):
    disabled = spec.model_copy(update={"enabled": False})
    assert propose("net_revenue", "West", "2025-11", "down", [], disabled) == Proposal(())


def test_slug_namespaces_the_tag(spec):
    assert slug("Warehouse dispatch backlog!", spec.tag_prefix) == (
        "lt_warehouse_dispatch_backlog"
    )


def test_a_candidate_defaults_to_novel():
    candidate = LongTailCandidate(
        tag="lt_x", label="X", description="d", testable_by="t", prior=0.05
    )
    assert candidate.novel is True
