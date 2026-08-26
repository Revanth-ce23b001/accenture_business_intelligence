"""P17 — the API surface the two screens depend on.

The frontend is TypeScript and is not exercised here. What IS exercised is
the contract between them: the fields the Watchlist and the Case File
render, and — the load-bearing one — that every evidence id a screen
displays actually resolves.

WHY THIS TEST EXISTS AT ALL. Rule 3 says every number in the UI must be
clickable to its evidence. A component can only honour that if the id it
was given resolves, and an id that does not is invisible until somebody
clicks it in a demo. So the check is here, over every id the case object
carries, rather than in a component test that would only cover the ids
somebody remembered to write a case for.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.auth import HEADER, mint
from engine.verdict.canonical import CANONICAL

ANALYST = ("U008", "analyst")
WEST = ("U003", "regional_manager")

#: The Number Registry scenarios the API seeds. `/case/2451` must open.
SCENARIOS = ("2451", "2467", "2470", "2471", "2472")


@pytest.fixture(scope="module")
def wh(warehouse):
    from engine.warehouse.load import create_schema

    create_schema(warehouse)
    return warehouse


@pytest.fixture(scope="module")
def client(wh, layer):
    from api.app import create_app
    from api.store import CaseStore
    from llm.provider import MockProvider

    app = create_app(
        connection=wh,
        layer=layer,
        provider=MockProvider(),
        store=CaseStore(),
        canonical=True,
    )
    return TestClient(app)


def headers(user_id: str = ANALYST[0], persona: str = ANALYST[1]) -> dict[str, str]:
    return {HEADER: mint(user_id, persona)}


# ===========================================================================
# Screen 1 — the Watchlist
# ===========================================================================


def test_the_watchlist_carries_a_card_per_readable_kpi(client, layer):
    body = client.get("/api/watchlist?scan=false", headers=headers()).json()
    assert body["cards"], "the watchlist has no cards to render"
    assert {card["kpi"] for card in body["cards"]} <= set(layer.kpis)


def test_a_card_either_has_a_series_or_says_why_not(client):
    """A card that silently drew nothing would read as a KPI that had not
    moved, which is the same failure this product exists to prevent one
    level up."""
    body = client.get("/api/watchlist?scan=false", headers=headers()).json()
    for card in body["cards"]:
        if card["series_available"]:
            assert card["series"], card["kpi"]
        else:
            assert card["series_reason"], f"{card['kpi']} has no series and no reason"


def test_the_series_is_governed(client):
    """A regional manager's sparkline is their region's, and the card says
    how many rows the policy withheld."""
    west = client.get("/api/watchlist?scan=false", headers=headers(*WEST)).json()
    boardroom = client.get("/api/watchlist?scan=false", headers=headers()).json()

    def latest(body, kpi):
        card = next(c for c in body["cards"] if c["kpi"] == kpi)
        return card["latest"], card["rows_filtered"]

    west_value, west_filtered = latest(west, "net_revenue")
    all_value, all_filtered = latest(boardroom, "net_revenue")

    assert west_filtered > 0, "the row policy withheld nothing from a scoped reader"
    assert all_filtered == 0
    assert west_value < all_value, "a region should be smaller than the estate"


def test_the_suppressed_panel_names_the_gate_that_stopped_each(client):
    """ACCEPT, screen 1: the panel that matters most. A row that said only
    'suppressed' would be a status; the gate makes it an argument."""
    body = client.get("/api/watchlist?scan=false", headers=headers()).json()
    assert body["suppressed"], "no suppressed movements to render"
    for row in body["suppressed"]:
        assert row["stopped_by_gate"], row
        assert row["outcome_code"], row
        assert row["detail"], row
        assert row["headline"], "a row with no movement on it is not a row"


def test_the_data_incident_and_the_history_kill_are_both_there(client):
    """#2470 is a Gate 1 data incident; #2471 a Gate 3 history kill."""
    body = client.get("/api/watchlist?scan=false", headers=headers()).json()
    codes = {row["outcome_code"] for row in body["suppressed"]}
    assert "DATA_INCIDENT" in codes
    assert "INSUFFICIENT_HISTORY" in codes


def test_the_data_incident_row_says_how_many_feeds_failed(client):
    body = client.get("/api/watchlist?scan=false", headers=headers()).json()
    row = next(r for r in body["suppressed"] if r["outcome_code"] == "DATA_INCIDENT")
    assert "25 of 140" in row["detail"]
    assert row["headline"] == "-18.0%"


# ===========================================================================
# Screen 2 — the Case File
# ===========================================================================


@pytest.mark.parametrize("case_id", SCENARIOS)
def test_every_scenario_opens(client, case_id):
    response = client.get(f"/api/cases/{case_id}", headers=headers())
    assert response.status_code == 200, response.text
    assert response.json()["source"] == CANONICAL


def test_the_case_says_where_its_figures_came_from(client):
    """Rule 4: nothing on screen is a typed-in constant."""
    body = client.get("/api/cases/2451", headers=headers()).json()
    assert body["config_ref"] == "data/generator/config/scenario_2451.yaml"


def test_2451_reaches_the_verdict_the_registry_specifies(client):
    """The decision table, walked over the case's own figures.

    Not asserted as a literal anywhere in the engine: `decide()` reads the
    coverage, the confidence and the held residuals off the case and
    arrives here on its own.
    """
    body = client.get("/api/cases/2451", headers=headers()).json()
    verdict = body["verdict"]
    assert verdict["value"] == "PARTIALLY_EXPLAINED"
    assert verdict["reason_code"] == "live_unverifiable_above_materiality"
    assert "0.86 Cr" in verdict["reason_text"]
    assert "0.50 Cr" in verdict["reason_text"]


def test_the_decomposition_is_renderable(client):
    """The 8.1 / 3.2 / 4.9 bar needs three figures with units."""
    adjudication = client.get("/api/cases/2451", headers=headers()).json()[
        "adjudication"
    ]
    assert adjudication["headline_movement"]["value"] == pytest.approx(-8.1)
    assert adjudication["attributed"], "nothing to draw as calendar-expected"
    assert adjudication["qualified_residual"]["value"] == pytest.approx(4.1)
    assert adjudication["empirical_band"]["value"] == pytest.approx(1.8)


def test_the_region_strip_has_all_four_regions(client):
    """Gate 4's comparison is what makes a movement specific rather than
    the market, and the strip is how a reader sees it."""
    adjudication = client.get("/api/cases/2451", headers=headers()).json()[
        "adjudication"
    ]
    strip = {
        item["evidence_id"].rsplit(".", 1)[-1]: item["value"]
        for item in adjudication["evidence"]
        if "region_residual" in item["evidence_id"]
    }
    assert set(strip) == {"north", "south", "east", "west"}
    assert strip["west"] == pytest.approx(-4.9)
    assert strip["south"] == pytest.approx(1.1)


def test_a_case_that_opened_carries_five_gate_chips(client):
    adjudication = client.get("/api/cases/2451", headers=headers()).json()[
        "adjudication"
    ]
    gates = adjudication["gates"]
    assert len(gates) == 5
    assert all(gate["passed"] for gate in gates)
    assert all(gate["detail"] for gate in gates), "a chip with no reason behind it"


def test_a_killed_case_carries_only_the_gate_that_killed_it(client):
    """A rail showing five green ticks for a case that died at the first
    gate would be a lie about what the system did."""
    adjudication = client.get("/api/cases/2470", headers=headers()).json()[
        "adjudication"
    ]
    gates = adjudication["gates"]
    assert len(gates) == 1
    assert gates[0]["gate_id"] == 1
    assert gates[0]["passed"] is False


def test_a_killed_case_has_no_verdict(client):
    """It never reached one. Synthesising an abstention would report a
    decision nobody made."""
    for case_id in ("2470", "2471", "2472"):
        assert client.get(f"/api/cases/{case_id}", headers=headers()).json()[
            "verdict"
        ] is None


def test_the_ruled_out_hypotheses_carry_their_reasons(client):
    """CLAUDE.md calls the two hard gates the product. The ruled-out panel
    is where a reader sees that the strongest correlation in the data was
    eliminated on the dates alone."""
    adjudication = client.get("/api/cases/2451", headers=headers()).json()[
        "adjudication"
    ]
    eliminated = {
        h["hypothesis_id"]: h["elimination_reason"]
        for h in adjudication["hypotheses"]
        if h["status"] == "eliminated"
    }
    assert eliminated, "nothing was ruled out"
    assert all(reason for reason in eliminated.values())
    assert "sufficiency" in eliminated.values() or "precedence" in eliminated.values()


def test_the_unverifiable_hypothesis_is_marked_and_holds_money(client):
    adjudication = client.get("/api/cases/2451", headers=headers()).json()[
        "adjudication"
    ]
    unverifiable = [
        h
        for h in adjudication["hypotheses"]
        if h["status"] != "eliminated" and not h["verifiable"]
    ]
    assert unverifiable, "nothing is marked unverifiable"
    held = unverifiable[0]
    assert held["missing_sources"], "unverifiable, but nothing is named as missing"
    assert held["residual_held"], "it holds nothing, so it would block nothing"


def test_the_confidence_rail_has_s1_to_s6_and_the_published_step(client):
    confidence = client.get("/api/cases/2451", headers=headers()).json()[
        "adjudication"
    ]["confidence"]
    assert [c["key"] for c in confidence["components"]] == [
        "s1", "s2", "s3", "s4", "s5", "s6"
    ]
    assert confidence["raw"] == pytest.approx(0.8876, abs=1e-4)
    assert confidence["calibrated"] == pytest.approx(0.84, abs=1e-4)
    assert confidence["caps_applied"] == [], (
        "CLAUDE.md is explicit that no cap fires on #2451"
    )


def test_the_case_names_the_personas_the_narrator_can_write_for(client, layer):
    """The switcher is built from this, so it cannot offer a reader the
    narrator has no voice for."""
    body = client.get("/api/cases/2451", headers=headers()).json()
    assert set(body["narrative_personas"]) == set(layer.narrate.personas)


# ===========================================================================
# ACCEPT 2 — every displayed number opens its evidence
# ===========================================================================


@pytest.mark.parametrize("case_id", SCENARIOS)
def test_every_evidence_id_a_case_carries_resolves(client, case_id):
    """RULE 3, CHECKED EXHAUSTIVELY.

    Every id the case object carries — on the headline, the residual, the
    band, the materiality limit, each hypothesis's attribution, each test
    result, the region strip and the confidence components — is fetched
    through the endpoint the drawer uses. An id that does not resolve is a
    number published with nothing behind it, and it stays invisible until
    somebody clicks it in front of an audience.
    """
    body = client.get(f"/api/cases/{case_id}", headers=headers()).json()
    adjudication = body["adjudication"]

    ids: set[str] = {item["evidence_id"] for item in adjudication["evidence"]}
    for key in ("headline_movement", "qualified_residual", "materiality"):
        ids.add(adjudication[key]["evidence_id"])
    for item in adjudication["attributed"]:
        ids.add(item["evidence_id"])
    if adjudication["empirical_band"]:
        ids.add(adjudication["empirical_band"]["evidence_id"])
    for hypothesis in adjudication["hypotheses"]:
        for key in ("attributed", "residual_held"):
            if hypothesis[key]:
                ids.add(hypothesis[key]["evidence_id"])

    assert ids, f"case {case_id} publishes no evidence at all"

    unresolved = [
        evidence_id
        for evidence_id in sorted(ids)
        if client.get(f"/api/evidence/{evidence_id}", headers=headers()).status_code
        != 200
    ]
    assert not unresolved, (
        f"case {case_id} displays numbers whose evidence does not resolve: "
        f"{unresolved}"
    )


def test_an_evidence_record_carries_everything_the_drawer_shows(client):
    """Source, as-of, freshness, method, lineage, reliability."""
    record = client.get("/api/evidence/case.residual", headers=headers()).json()
    assert record["source_system"]
    assert record["method"]
    assert record["source_as_of"]
    assert record["reliability"] > 0
    assert record["lineage"], "rule 3: a record with no lineage is a number with nothing behind it"
    assert "freshness_hours" in record
    assert "completeness" in record


# ===========================================================================
# ACCEPT 3 — the persona switch re-renders prose, not the pipeline
# ===========================================================================


def test_switching_persona_changes_the_prose_and_nothing_else(client, layer):
    """ACCEPT: persona switch re-renders the narrative without re-running
    the pipeline.

    The case object is fetched once per persona and compared field by
    field: the verdict, the confidence and every evidence id must be
    identical. Only the narrative differs.
    """
    personas = sorted(layer.narrate.personas)
    assert len(personas) > 1, "one persona cannot demonstrate a switch"

    cases = {
        persona: client.get(
            f"/api/cases/2451?persona={persona}", headers=headers()
        ).json()
        for persona in personas
    }
    first = cases[personas[0]]
    for persona in personas[1:]:
        other = cases[persona]
        assert other["verdict"] == first["verdict"]
        assert other["adjudication"]["confidence"] == first["adjudication"]["confidence"]
        assert [e["evidence_id"] for e in other["adjudication"]["evidence"]] == [
            e["evidence_id"] for e in first["adjudication"]["evidence"]
        ]

    narratives = {}
    for persona in personas:
        response = client.get(
            f"/api/cases/2451/narrative?persona={persona}", headers=headers()
        )
        if response.status_code == 200:
            narratives[persona] = response.json()

    assert len(narratives) >= 2, (
        "at least two personas must render for a switch to be demonstrable; "
        f"got {sorted(narratives)}"
    )
    texts = {n["text"] for n in narratives.values()}
    assert len(texts) > 1, "every persona produced identical prose"


def test_every_narrative_is_grounded(client, layer):
    """A sentence that failed the check was stripped before it got here."""
    for persona in sorted(layer.narrate.personas):
        response = client.get(
            f"/api/cases/2451/narrative?persona={persona}", headers=headers()
        )
        if response.status_code != 200:
            continue
        body = response.json()
        assert body["claims_checked"] == (
            body["claims_linked"] + body["claims_stripped"]
        )
        assert "claims linked" in body["grounding"]
        for claim in body["claims"]:
            assert claim["evidence_ids"], (
                "a surviving sentence with no evidence behind it"
            )


def test_a_persona_the_narrator_does_not_declare_is_refused(client):
    response = client.get(
        "/api/cases/2451/narrative?persona=store_manager", headers=headers()
    )
    assert response.status_code == 404
