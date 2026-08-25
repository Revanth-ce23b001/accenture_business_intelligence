"""Narration, per persona, offline.

The accept criteria this file owns:

    0 ungrounded claims across 5 scenarios x 3 personas
    the whole suite passes with the network disabled

The second is asserted directly rather than assumed: one test replaces
`socket.socket` with something that raises, then narrates all fifteen. A
suite that merely happens not to make a call proves nothing about a suite
run on a plane.
"""

from __future__ import annotations

import json
import socket

import pytest

from llm.casefiles import CASE_IDS, frozen
from llm.grounding import Claim, FABRICATED_NUMBER, GroundingReport
from llm.narrate import (
    NarrationError,
    build_request,
    freeze,
    gate_summary,
    narrate,
    narrate_all,
    parse_response,
)
from llm.provider import LLMResponse, MockProvider

PERSONAS = ("analyst", "cxo", "regional_manager")


@pytest.fixture(scope="module")
def mock() -> MockProvider:
    """The recorded fixtures. No socket is opened by this provider, ever."""
    return MockProvider()


@pytest.fixture(scope="module")
def cases(layer):
    return {case_id: frozen(case_id, layer) for case_id in CASE_IDS}


@pytest.fixture(scope="module")
def narratives(cases, layer, mock):
    """Every scenario, every persona. Fifteen narratives, replayed."""
    return {
        case_id: narrate_all(adjudication, layer, provider=mock)
        for case_id, adjudication in cases.items()
    }


class _Scripted:
    """A provider that returns exactly what a test tells it to.

    Used to drive the strip-regenerate-drop loop, where the whole point is
    what happens when a reply is WRONG. A recorded fixture cannot express
    that, because the recordings are all of replies that ground.
    """

    name = "scripted"

    def __init__(self, *replies: list[dict]):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, request):
        payload = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMResponse(
            text=json.dumps(payload),
            model=request.model,
            request_fingerprint=request.fingerprint(),
        )


# ===========================================================================
# The accept criterion
# ===========================================================================


def test_five_scenarios_three_personas_zero_ungrounded(narratives):
    """Accept criterion: 0 ungrounded claims across 5 scenarios x 3 personas."""
    assert len(narratives) == 5
    stripped = 0
    checked = 0
    for case_id, per_persona in narratives.items():
        assert sorted(per_persona) == list(PERSONAS), case_id
        for persona, story in per_persona.items():
            assert story.report.clean, (
                f"{case_id}/{persona} dropped "
                f"{[v.render() for v in story.report.violations]}"
            )
            assert story.dropped == ()
            stripped += story.report.claims_stripped
            checked += story.report.claims_checked
    assert stripped == 0
    assert checked == 51


def test_every_narrative_says_something(narratives):
    """A narrative of length zero passes every rule and tells nobody anything."""
    for case_id, per_persona in narratives.items():
        for persona, story in per_persona.items():
            assert story.claims, f"{case_id}/{persona} is empty"
            assert story.text.strip()


def test_every_claim_carries_a_real_evidence_id(narratives, cases):
    from llm.grounding import evidence_ids

    for case_id, per_persona in narratives.items():
        valid = evidence_ids(cases[case_id])
        for story in per_persona.values():
            for claim in story.claims:
                assert claim.evidence_ids
                assert set(claim.evidence_ids) <= valid


def test_no_persona_exceeds_its_sentence_budget(narratives, layer):
    for per_persona in narratives.values():
        for persona, story in per_persona.items():
            assert len(story.claims) <= layer.narrate.personas[persona].max_sentences


# ===========================================================================
# Offline, and asserted as such
# ===========================================================================


def test_every_narrative_is_replayed_from_a_fixture(narratives):
    for per_persona in narratives.values():
        for story in per_persona.values():
            assert story.from_fixture


def test_the_whole_run_works_with_the_network_disabled(cases, layer, monkeypatch):
    """Accept criterion, asserted rather than assumed.

    `MockProvider` is documented as never opening a socket. This is the
    test that makes that a fact: any attempt to construct one raises, and
    all fifteen narratives still come back grounded.
    """

    def refuse(*args, **kwargs):
        raise OSError("the network is disabled for this test")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    provider = MockProvider()
    total = 0
    for adjudication in cases.values():
        for story in narrate_all(adjudication, layer, provider=provider).values():
            assert story.report.clean
            total += story.report.claims_linked
    assert total == 51


def test_the_anthropic_sdk_is_never_imported_on_the_offline_path():
    """Rule 8. The offline demo must not depend on the SDK being installed."""
    import sys

    assert "anthropic" not in sys.modules


# ===========================================================================
# The request
# ===========================================================================


def test_the_request_is_deterministic(cases, layer):
    """The fingerprint is the fixture key; a varying prompt never replays."""
    first = build_request(cases["2451"], "cxo", layer)
    second = build_request(cases["2451"], "cxo", layer)
    assert first.fingerprint() == second.fingerprint()


def test_the_frozen_object_is_canonical(cases):
    assert freeze(cases["2451"]) == freeze(cases["2451"])
    assert '"case_id":"2451"' in freeze(cases["2451"])


def test_the_prompt_states_which_hypotheses_may_be_blamed(cases, layer):
    summary = gate_summary(cases["2451"], layer)
    assert "H1" in summary and "hard_gates_passed: true" in summary
    assert "H4" in summary and "describe only, never blame" in summary
    assert "failed precedence" in summary


def test_the_prompt_carries_the_banned_phrasings(cases, layer):
    prompt = build_request(cases["2451"], "analyst", layer).messages[0].content
    assert "root cause" in prompt
    assert "primary attributed driver" in prompt
    assert "ai analysis shows" in prompt


def test_the_prompt_differs_by_persona(cases, layer):
    prompts = {
        persona: build_request(cases["2451"], persona, layer).messages[0].content
        for persona in PERSONAS
    }
    assert len(set(prompts.values())) == 3
    assert "Chief Executive" in prompts["cxo"]
    assert "Lead with: method" in prompts["analyst"]
    assert "Lead with: affected_stores" in prompts["regional_manager"]


def test_an_unknown_persona_is_refused(cases, layer):
    with pytest.raises(NarrationError, match="no narrative persona"):
        build_request(cases["2451"], "board_observer", layer)


def test_narration_routes_to_sonnet(cases, layer):
    assert build_request(cases["2451"], "cxo", layer).model == "claude-sonnet-5"


# ===========================================================================
# Reading the reply
# ===========================================================================


def test_a_fenced_reply_is_read(cases):
    claims = parse_response('```json\n[{"sentence": "A.", "evidence_ids": ["x"]}]\n```')
    assert claims[0].sentence == "A."


def test_a_claim_without_a_sentence_is_an_error():
    with pytest.raises(NarrationError, match="no sentence"):
        parse_response('[{"evidence_ids": ["x"]}]')


def test_a_claim_without_evidence_ids_is_an_error():
    with pytest.raises(NarrationError, match="no evidence_ids"):
        parse_response('[{"sentence": "A."}]')


def test_a_single_evidence_id_may_arrive_as_a_string():
    claims = parse_response('[{"sentence": "A.", "evidence_ids": "case.residual"}]')
    assert claims[0].evidence_ids == ("case.residual",)


def test_a_reply_that_is_not_a_json_array_is_an_error():
    with pytest.raises(NarrationError, match="no JSON array"):
        parse_response("I would rather write you a paragraph.")


# ===========================================================================
# Strip, regenerate once, drop
# ===========================================================================


def _sentence(text, ids=("case.residual",), hypothesis=None):
    payload = {"sentence": text, "evidence_ids": list(ids)}
    if hypothesis:
        payload["hypothesis_id"] = hypothesis
    return payload


GOOD = _sentence("The qualified residual is INR 4.1 Cr.")
ALSO_GOOD = _sentence("The materiality threshold is INR 0.5 Cr.", ("case.materiality",))
FABRICATED = _sentence("The qualified residual is INR 9.99 Cr.")


def test_a_failing_sentence_is_replaced_by_the_regeneration(cases, layer):
    provider = _Scripted([GOOD, FABRICATED], [ALSO_GOOD])
    story = narrate(cases["2451"], "cxo", layer, provider=provider)

    assert provider.calls == 2
    assert story.regenerated
    assert story.report.claims_regenerated == 1
    assert story.report.clean
    assert [claim.sentence for claim in story.claims] == [
        GOOD["sentence"],
        ALSO_GOOD["sentence"],
    ]


def test_a_sentence_that_fails_twice_is_dropped(cases, layer):
    """One regeneration, then it goes. The narrative gets shorter."""
    provider = _Scripted([GOOD, FABRICATED], [FABRICATED])
    story = narrate(cases["2451"], "cxo", layer, provider=provider)

    assert provider.calls == 2
    assert len(story.claims) == 1
    assert len(story.dropped) == 1
    assert story.report.claims_stripped == 1
    assert story.report.claims_linked == 1
    assert FABRICATED_NUMBER in {v.code for v in story.report.violations}


def test_a_clean_first_pass_never_asks_twice(cases, layer):
    provider = _Scripted([GOOD, ALSO_GOOD])
    story = narrate(cases["2451"], "cxo", layer, provider=provider)
    assert provider.calls == 1
    assert not story.regenerated


def test_the_regeneration_prompt_names_the_rule_that_was_broken(cases, layer):
    """A second attempt told only "try again" comes back vaguer, not truer."""
    captured: list[str] = []

    class _Capturing(_Scripted):
        def complete(self, request):
            captured.append(request.messages[0].content)
            return super().complete(request)

    narrate(
        cases["2451"], "cxo", layer, provider=_Capturing([GOOD, FABRICATED], [ALSO_GOOD])
    )
    retry = captured[1]
    assert "DELETED" in retry
    assert FABRICATED_NUMBER in retry
    assert "9.99" in retry
    assert GOOD["sentence"] in retry  # what already grounded, so it is not repeated


def test_the_report_always_balances(narratives):
    for per_persona in narratives.values():
        for story in per_persona.values():
            report = story.report
            assert report.claims_checked == report.claims_linked + report.claims_stripped


def test_the_narrative_serialises_with_its_grounding_counts(narratives):
    payload = narratives["2451"]["cxo"].as_dict()
    assert payload["case_id"] == "2451"
    assert set(payload["grounding"]) == {
        "claims_checked",
        "claims_linked",
        "claims_stripped",
    }
    assert all(claim["evidence_ids"] for claim in payload["claims"])


# ===========================================================================
# What the personas actually say
# ===========================================================================


def test_the_analyst_gets_the_method_and_the_cxo_does_not(narratives):
    analyst = narratives["2451"]["analyst"].text
    cxo = narratives["2451"]["cxo"].text
    assert "difference-in-differences" in analyst.lower()
    assert "difference-in-differences" not in cxo.lower()


def test_the_supported_hypothesis_takes_the_required_attribution_form(narratives):
    text = narratives["2451"]["cxo"].text
    assert "is the best-supported explanation for" in text


def test_an_eliminated_hypothesis_is_never_blamed(narratives):
    """H4 is the marketing cut. It is named and it is not blamed."""
    text = narratives["2451"]["analyst"].text
    assert "eliminated on precedence" in text
    assert "led to" not in text
    assert "because of" not in text


def test_the_abstained_case_says_it_abstained(narratives):
    text = narratives["2467"]["cxo"].text
    assert "abstained" in text
    for trigger in ("T3", "T4", "T7"):
        assert trigger in text


def test_the_gate_killed_case_names_the_gate_not_a_verdict(narratives):
    text = narratives["2470"]["cxo"].text
    assert "DATA_INCIDENT" in text
    assert "25 of 140" in text


def test_the_flat_case_reports_the_residual_not_the_headline(narratives):
    """#2472 is flat at 0.0% and holds a -12 pt residual. Both are said."""
    text = " ".join(narratives["2472"][persona].text for persona in PERSONAS)
    assert "12" in text


def test_no_narrative_contains_a_forbidden_phrase(narratives, layer):
    banned = [rule.phrase for rule in layer.narrate.language.forbidden_phrases]
    for case_id, per_persona in narratives.items():
        for persona, story in per_persona.items():
            lowered = story.text.lower()
            for phrase in banned:
                assert phrase not in lowered, f"{case_id}/{persona}: {phrase!r}"
