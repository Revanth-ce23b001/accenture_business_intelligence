"""The provider layer: routing, fingerprinting, and offline replay.

CLAUDE.md rule 8 — MOCK_LLM=true must always work, offline, from day one.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from llm.provider import (
    MODEL_HAIKU,
    MODEL_SONNET,
    TASK_MODEL,
    AnthropicProvider,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MissingFixtureError,
    MockProvider,
    get_provider,
    mock_enabled,
    record_fixture,
)
from tests.conftest import smoke_request


def _response_for(request: LLMRequest, text: str = "diagnose_kpi_movement") -> LLMResponse:
    return LLMResponse(
        text=text,
        model=request.model,
        stop_reason="end_turn",
        input_tokens=42,
        output_tokens=7,
        request_fingerprint=request.fingerprint(),
    )


# --- routing ----------------------------------------------------------------


def test_batched_work_routes_to_haiku():
    """CLAUDE.md §Architecture: Haiku 4.5 for batched classification and extraction."""
    assert TASK_MODEL["classify"] == MODEL_HAIKU
    assert TASK_MODEL["extract"] == MODEL_HAIKU


def test_generative_work_routes_to_sonnet():
    """CLAUDE.md §Architecture: Sonnet for hypothesis generation and narrative."""
    assert TASK_MODEL["hypothesise"] == MODEL_SONNET
    assert TASK_MODEL["narrate"] == MODEL_SONNET


def test_for_task_routes_the_model():
    request = LLMRequest.for_task("classify", messages=(LLMMessage(role="user", content="hi"),))
    assert request.model == MODEL_HAIKU


# --- fingerprinting ---------------------------------------------------------


def test_fingerprint_is_stable_across_instances():
    assert smoke_request().fingerprint() == smoke_request().fingerprint()


def test_fingerprint_changes_with_content():
    a = smoke_request()
    b = a.model_copy(update={"messages": (LLMMessage(role="user", content="different"),)})
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_changes_with_model():
    a = smoke_request()
    b = a.model_copy(update={"model": MODEL_SONNET})
    assert a.fingerprint() != b.fingerprint()


def test_requests_are_frozen():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        smoke_request().max_tokens = 99


# --- provider selection -----------------------------------------------------


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_mock_enabled_truthy(value):
    assert mock_enabled({"MOCK_LLM": value}) is True


@pytest.mark.parametrize("value", ["false", "0", "", "no"])
def test_mock_enabled_falsy(value):
    assert mock_enabled({"MOCK_LLM": value}) is False


def test_mock_llm_env_selects_the_mock(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "true")
    provider = get_provider()
    assert isinstance(provider, MockProvider)
    assert provider.name == "mock"


def test_providers_satisfy_the_protocol():
    assert isinstance(MockProvider(), LLMProvider)


# --- offline replay ---------------------------------------------------------


def test_record_then_replay(tmp_path):
    request = smoke_request()
    recorded = _response_for(request)
    path = record_fixture(request, recorded, fixtures_dir=tmp_path, note="unit test")
    assert path.exists()

    replayed = MockProvider(fixtures_dir=tmp_path).complete(request)
    assert replayed.text == recorded.text
    assert replayed.model == recorded.model
    assert replayed.from_fixture is True
    assert replayed.request_fingerprint == request.fingerprint()


def test_missing_fixture_names_the_fingerprint(tmp_path):
    request = smoke_request()
    with pytest.raises(MissingFixtureError) as excinfo:
        MockProvider(fixtures_dir=tmp_path).complete(request)
    assert request.fingerprint() in str(excinfo.value)


def test_committed_smoke_fixture_replays(monkeypatch):
    """The fixture committed in llm/fixtures/ must replay with MOCK_LLM=true.

    If this fails after editing tests/conftest.py::smoke_request, the request
    changed and the fixture needs re-recording — the fingerprint in the error
    message is the new filename.
    """
    monkeypatch.setenv("MOCK_LLM", "true")
    request = smoke_request()
    response = get_provider().complete(request)
    assert response.from_fixture is True
    assert response.text
    assert response.model == MODEL_HAIKU


def test_mock_provider_opens_no_socket(monkeypatch, tmp_path):
    """Rule 8, enforced: the offline path must not touch the network."""
    request = smoke_request()
    record_fixture(request, _response_for(request), fixtures_dir=tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("the mock provider attempted network access")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert MockProvider(fixtures_dir=tmp_path).complete(request).from_fixture is True


# --- live provider request shaping -----------------------------------------


class _FakeMessages:
    def __init__(self):
        self.captured: dict = {}

    def create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            model=kwargs["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=2,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_temperature_and_effort_are_omitted_when_unset():
    """`temperature` is rejected with a 400 on Sonnet 5, and `effort` errors on
    Haiku 4.5. Neither may be sent unless explicitly set."""
    client = _FakeClient()
    request = smoke_request()
    AnthropicProvider(client=client).complete(request)
    assert "temperature" not in client.messages.captured
    assert "output_config" not in client.messages.captured


def test_temperature_and_effort_are_sent_when_set():
    client = _FakeClient()
    request = smoke_request().model_copy(update={"temperature": 0.0, "effort": "low"})
    AnthropicProvider(client=client).complete(request)
    assert client.messages.captured["temperature"] == 0.0
    assert client.messages.captured["output_config"] == {"effort": "low"}


def test_live_response_is_mapped_onto_the_contract():
    client = _FakeClient()
    request = smoke_request()
    response = AnthropicProvider(client=client).complete(request)
    assert response.text == "ok"
    assert response.model == MODEL_HAIKU
    assert response.input_tokens == 10
    assert response.output_tokens == 2
    assert response.from_fixture is False
    assert response.request_fingerprint == request.fingerprint()
