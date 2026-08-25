"""The deterministic check between the model and the screen.

Four rules, and the two the brief names explicitly:

    a deliberately injected fabricated number is stripped and counted
    a causal connective on an eliminated hypothesis is stripped

Nothing here calls a model. Every test below hand-builds the claims a
model might return — including the ones a careless or adversarial model
would return — and asserts what the validator does with them. That is the
point of putting a deterministic check in the path: its behaviour does not
depend on which model wrote the sentence, or on whether one wrote it at
all.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from llm.casefiles import frozen
from llm.grounding import (
    BARE_CAUSAL_CLAIM,
    FABRICATED_NUMBER,
    FORBIDDEN_PHRASE,
    NO_EVIDENCE,
    UNGATED_CAUSAL_CLAIM,
    UNKNOWN_EVIDENCE,
    Claim,
    GroundingError,
    GroundingReport,
    check,
    evidence_ids,
    feedback,
    hypotheses_for,
    hypothesis_gates,
    numeric_tokens,
    numeric_whitelist,
    report,
    token_is_grounded,
)


@pytest.fixture(scope="module")
def case(layer):
    """#2451, frozen. The case with a supported and four other hypotheses."""
    return frozen("2451", layer)


@pytest.fixture(scope="module")
def whitelist(case, layer):
    return numeric_whitelist(case, layer.narrate)


@pytest.fixture(scope="module")
def ids(case):
    return evidence_ids(case)


def codes(checked):
    return [violation.code for item in checked for violation in item.violations]


# ===========================================================================
# The whitelist
# ===========================================================================


def test_evidence_ids_are_collected_from_everywhere(case, ids):
    """A real id cited from the wrong list is still a real id."""
    assert "case.residual" in ids
    assert "case.did" in ids
    # Carried on a hypothesis rather than in the top-level index.
    hypothesis = next(h for h in case["hypotheses"] if h["hypothesis_id"] == "H1")
    assert set(hypothesis["evidence_ids"]) <= ids


def test_whitelist_holds_the_objects_own_quantities(whitelist):
    for value in ("3.24", "0.79", "4.1", "34", "0.84"):
        assert Decimal(value) in whitelist, value


def test_whitelist_expands_ratios_to_percentages(whitelist):
    """The object stores 0.79; prose says 79%."""
    assert Decimal("0.79") in whitelist
    assert Decimal("79") in whitelist


def test_whitelist_refuses_numbers_that_only_appear_in_prose(case, layer):
    """A figure in a caption must not be able to launder itself.

    Lineage descriptions are free text and nothing else validates them. If
    a number inside one could ground a sentence, the way to publish any
    figure would be to mention it in a description first.
    """
    prose = " ".join(
        str(step.get("description", ""))
        for record in case["evidence"]
        for step in record["lineage"]
    )
    assert "12:00" in prose  # a real number, in prose, in this object
    whitelist = numeric_whitelist(case, layer.narrate)
    assert not any(str(value) == "12:00" for value in whitelist)


# ===========================================================================
# Numbers in prose
# ===========================================================================


def test_rounding_is_at_the_tokens_own_precision(layer):
    """Fewer decimals is a WIDER window, because that is what rounding means.

    A stored 1.8823 grounds "1.88" (two places) and "1.9" (one place, and
    1.8823 does round to 1.9). It refuses "1.87", which nothing rounds to
    at two places. Reading this rule as "the shortest form only" would
    strip "about 1.9 crore", which is a true thing to say.
    """
    spec = layer.narrate
    tight = frozenset({Decimal("1.8823")})
    assert token_is_grounded("1.88", tight, spec)
    assert token_is_grounded("1.9", tight, spec)
    assert not token_is_grounded("1.87", tight, spec)
    assert not token_is_grounded("1.885", tight, spec)


def test_the_float_noise_guard_is_narrow(layer):
    """It absorbs float representation error, not a third decimal place.

    1.883 against a stored 1.8823 is inside the relative tolerance and
    grounds. 1.885 is outside it and does not. The guard exists so
    3.1104000000000003 can be written 3.11, and it has to stay tight
    enough that it is not a second, looser rounding rule.
    """
    tight = frozenset({Decimal("1.8823")})
    assert token_is_grounded("1.883", tight, layer.narrate)
    assert not token_is_grounded("1.885", tight, layer.narrate)


def test_a_number_no_stored_value_rounds_to_is_refused(whitelist, layer):
    for invented in ("9.99", "88.3", "41.7", "6412.5"):
        assert not token_is_grounded(invented, whitelist, layer.narrate), invented


def test_the_whitelist_checks_provenance_not_relevance(whitelist, layer):
    """A known limit, pinned rather than papered over.

    88.8 grounds against #2451 because the raw confidence is 0.8876 and
    88.76 rounds to 88.8 at one decimal place — even in a sentence about
    revenue. Fixing this would mean the model declaring which field each
    number came from, which it could equally misdeclare. The rule stops
    numbers being INVENTED; the evidence id on the sentence is what lets a
    reader catch one being misapplied.
    """
    assert token_is_grounded("88.8", whitelist, layer.narrate)
    assert Decimal("88.76") in whitelist


def test_float_noise_does_not_strip_a_true_sentence(layer):
    """3.1104000000000003 is 3.11 to anybody reading it."""
    noisy = frozenset({Decimal("3.1104000000000003")})
    assert token_is_grounded("3.11", noisy, layer.narrate)


def test_small_structural_integers_are_allowed(whitelist, layer):
    """"both hard gates", "all six tests" — counts of the method, not the business."""
    assert token_is_grounded("2", frozenset(), layer.narrate)
    assert token_is_grounded("6", frozenset(), layer.narrate)
    assert not token_is_grounded("7", frozenset(), layer.narrate)


@pytest.mark.parametrize(
    "sentence",
    [
        "The incident began on 2025-11-04.",
        "See case.residual for the derivation.",
        "Trigger T3 fired.",
        "Test 1 eliminated it.",
        "Top-20 availability fell.",
        "The period is 2025-11.",
        "Reported on 12 Nov.",
        "Reported on November 12.",
    ],
)
def test_identifiers_are_not_read_as_quantities(sentence, layer):
    assert numeric_tokens(sentence, layer.narrate) == ()


def test_a_real_quantity_beside_an_identifier_is_still_read(layer):
    tokens = numeric_tokens("On 2025-11-04 the residual was 4.1 Cr.", layer.narrate)
    assert tokens == ("4.1",)


# ===========================================================================
# Rule 1 — linkage
# ===========================================================================


def test_a_sentence_with_no_evidence_id_is_stripped(case, layer):
    checked = check((Claim("West revenue fell.", ()),), case, layer)
    assert codes(checked) == [NO_EVIDENCE]
    assert not checked[0].grounded


def test_a_sentence_citing_an_unknown_id_is_stripped(case, layer):
    checked = check((Claim("A true thing.", ("case.invented",)),), case, layer)
    assert UNKNOWN_EVIDENCE in codes(checked)


# ===========================================================================
# Rule 2 — the accept criterion on fabricated numbers
# ===========================================================================


def test_an_injected_fabricated_number_is_stripped_and_counted(case, layer):
    """Accept criterion.

    The sentence is otherwise perfect: real evidence id, real hypothesis,
    house phrasing, no connective. Only the number is invented, and that
    alone is enough.
    """
    honest = Claim(
        "Availability collapse on top SKUs is the best-supported explanation for "
        "79% of the qualified residual.",
        ("case.h1_attributed",),
        "H1",
    )
    fabricated = Claim(
        "Availability collapse on top SKUs is the best-supported explanation for "
        "INR 9.99 Cr of the qualified residual.",
        ("case.h1_attributed",),
        "H1",
    )
    checked = check((honest, fabricated), case, layer)
    assert checked[0].grounded
    assert not checked[1].grounded
    assert [v.code for v in checked[1].violations] == [FABRICATED_NUMBER]
    assert checked[1].violations[0].token == "9.99"

    counted = report(checked)
    assert counted.claims_checked == 2
    assert counted.claims_linked == 1
    assert counted.claims_stripped == 1
    assert counted.as_dict() == {
        "claims_checked": 2,
        "claims_linked": 1,
        "claims_stripped": 1,
    }


def test_a_fabricated_number_survives_nothing_about_its_neighbours(case, layer):
    """One bad sentence costs one sentence, not the narrative."""
    claims = (
        Claim("The qualified residual is INR 4.1 Cr.", ("case.residual",)),
        Claim("The decline was INR 41.7 Cr.", ("case.decline",)),
        Claim("The matched-control estimate is 4.3 pt.", ("case.did",), "H1"),
    )
    counted = report(check(claims, case, layer))
    assert (counted.claims_linked, counted.claims_stripped) == (2, 1)


# ===========================================================================
# Rule 3 — the accept criterion on causal connectives
# ===========================================================================


def test_gates_are_read_off_the_test_results(case, layer):
    gates = hypothesis_gates(case, layer)
    assert gates["H1"].hard_gates_passed is True
    assert gates["H3"].hard_gates_passed is False
    assert gates["H3"].failed_gates == ("sufficiency",)
    assert gates["H4"].hard_gates_passed is False
    assert gates["H4"].failed_gates == ("precedence", "sufficiency")


def test_a_causal_connective_on_an_eliminated_hypothesis_is_stripped(case, layer):
    """Accept criterion.

    H4 is the marketing cut — the strongest correlation in the whole data
    set and eliminated by Test 1, because the cut lands after the decline
    started. A sentence blaming it is exactly the sentence an LLM asked
    "why did revenue fall?" would write.
    """
    claim = Claim(
        "The marketing spend cut led to the decline in West revenue.",
        ("case.residual",),
        "H4",
    )
    checked = check((claim,), case, layer)
    assert not checked[0].grounded
    assert UNGATED_CAUSAL_CLAIM in codes(checked)
    violation = next(v for v in checked[0].violations if v.code == UNGATED_CAUSAL_CLAIM)
    assert "H4" in violation.detail
    assert "precedence" in violation.detail
    assert report(checked).claims_stripped == 1


def test_the_same_connective_is_allowed_on_a_hypothesis_that_passed_both_gates(
    case, layer
):
    """The rule is about the evidence, not about the word."""
    claim = Claim(
        "Treated stores lost transactions because shelf availability collapsed, "
        "which the matched-control estimate puts at 4.3 pt.",
        ("case.did",),
        "H1",
    )
    assert check((claim,), case, layer)[0].grounded


def test_omitting_the_hypothesis_id_does_not_dodge_the_rule(case, layer):
    """Inference is a UNION, so leaving the tag off widens the check.

    A sentence that names the hypothesis and cites its evidence is
    attributable twice over. Dropping the declaration makes it
    attributable, not anonymous.
    """
    claim = Claim(
        "Marketing spend cut led to the fall.", ("case.residual",), hypothesis_id=None
    )
    assert UNGATED_CAUSAL_CLAIM in codes(check((claim,), case, layer))


def test_hypotheses_are_inferred_from_the_evidence_a_sentence_cites(case, layer):
    gates = hypothesis_gates(case, layer)
    found = hypotheses_for(Claim("Anything at all.", ("case.did",)), gates)
    assert [gate.hypothesis_id for gate in found] == ["H1"]


# ===========================================================================
# Rule 4 — language, unconditional
# ===========================================================================


def test_a_bare_causal_claim_is_stripped_even_when_the_gates_passed(case, layer):
    """The hypothesis explains a SHARE. A sentence claiming the whole overstates it."""
    claim = Claim(
        "West revenue fell because of the availability collapse.",
        ("case.h1_attributed",),
        "H1",
    )
    checked = check((claim,), case, layer)
    assert BARE_CAUSAL_CLAIM in codes(checked)
    detail = next(v for v in checked[0].violations if v.code == BARE_CAUSAL_CLAIM).detail
    assert "best-supported explanation" in detail


@pytest.mark.parametrize(
    "phrase",
    [
        "The root cause is the allocation fault.",
        "AI analysis shows the shelf was empty.",
        "The model found a difference of 4.3 pt.",
        "Machine learning predicts a recovery.",
        "This definitely explains the movement.",
        "The evidence proves that availability fell.",
    ],
)
def test_forbidden_phrases_are_stripped(phrase, case, layer):
    checked = check((Claim(phrase, ("case.residual",), "H1"),), case, layer)
    assert FORBIDDEN_PHRASE in codes(checked)


def test_the_replacement_is_offered_not_just_the_refusal(case, layer):
    """A rule with no `instead` produces a second attempt that is only shorter."""
    checked = check(
        (Claim("The root cause is clear.", ("case.residual",)),), case, layer
    )
    detail = next(v for v in checked[0].violations if v.code == FORBIDDEN_PHRASE).detail
    assert "primary attributed driver" in detail


def test_language_rules_are_case_insensitive(case, layer):
    checked = check((Claim("The ROOT CAUSE is clear.", ("case.residual",)),), case, layer)
    assert FORBIDDEN_PHRASE in codes(checked)


# ===========================================================================
# The report
# ===========================================================================


def test_the_report_must_balance():
    with pytest.raises(GroundingError, match="does not balance"):
        GroundingReport(claims_checked=5, claims_linked=3, claims_stripped=1)


def test_the_report_renders_the_line_the_ui_prints(case, layer):
    claims = tuple(
        Claim(f"The qualified residual is INR 4.1 Cr.", ("case.residual",))
        for _ in range(4)
    )
    counted = report(check(claims, case, layer))
    assert counted.render() == "grounding: 4/4 claims linked · 0 stripped"
    assert counted.clean


def test_every_violation_is_reported_not_only_the_first(case, layer):
    """A sentence can be wrong in several ways, and the model needs all of them."""
    claim = Claim(
        "The root cause was a INR 9.99 Cr shortfall.", ("case.nonexistent",)
    )
    found = set(codes(check((claim,), case, layer)))
    assert {UNKNOWN_EVIDENCE, FABRICATED_NUMBER, FORBIDDEN_PHRASE} <= found


def test_feedback_names_the_rule_and_the_token(case, layer):
    claim = Claim("The decline was INR 41.7 Cr.", ("case.decline",))
    text = feedback(check((claim,), case, layer))
    assert "41.7" in text
    assert FABRICATED_NUMBER in text
