"""LLM access for CaseFile.ai.

Every model call in this repository goes through `LLMProvider`. Nothing else
imports `anthropic`.

Two implementations:

  AnthropicProvider  live calls to the Claude API.
  MockProvider       replays recorded JSON from `llm/fixtures/`, keyed by a
                     content hash of the request. No network, no SDK import.

`MOCK_LLM=true` selects the mock (CLAUDE.md rule 8: the full demo runs offline
from recorded fixtures, and this must work from day one).

The fixture key is the same content hash CLAUDE.md §"Architecture" calls for
when it says classification and extraction are "content-hash cached" — one
fingerprint serves both the cache and the fixture store.

Note on rule 1: nothing in this module produces a number that reaches the UI.
Token counts and latencies here are telemetry, not case evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Model routing — CLAUDE.md §"Architecture" → Model routing
# ---------------------------------------------------------------------------

#: Batched classification and extraction. 200K context.
MODEL_HAIKU = "claude-haiku-4-5"

#: Hypothesis generation and narrative, 2-4 calls per case.
MODEL_SONNET = "claude-sonnet-5"

#: The six model-facing jobs listed in the LLM column of CLAUDE.md
#: §"Architecture".
LLMTask = Literal[
    "classify",       # intent classification
    "extract",        # document -> typed event extraction
    "hypothesise",    # hypothesis generation, long tail only
    "narrate",        # narrative generation, per persona
    "clarify",        # clarification-question wording
    "phrase_action",  # action phrasing, from playbook fields
]

TASK_MODEL: dict[str, str] = {
    "classify": MODEL_HAIKU,
    "extract": MODEL_HAIKU,
    "hypothesise": MODEL_SONNET,
    "narrate": MODEL_SONNET,
    "clarify": MODEL_SONNET,
    "phrase_action": MODEL_SONNET,
}

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """A model call, fully described.

    The fingerprint of this object is the cache key and the fixture key, so
    every field here must be deterministic. Do not put timestamps, UUIDs or
    anything else that varies per run into a request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: LLMTask
    model: str
    messages: tuple[LLMMessage, ...]
    system: str | None = None
    max_tokens: int = Field(default=4096, ge=1)

    # Omitted from the wire call unless explicitly set. `temperature` is
    # REJECTED with a 400 on Sonnet 5 (and on the Opus 4.7+ family); it is
    # still accepted on Haiku 4.5. Leaving it None is the safe default.
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)

    # Supported on Sonnet 5; errors on Haiku 4.5. None means "do not send".
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None

    stop_sequences: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        """Stable SHA-256 over the canonical JSON form of this request."""
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def for_task(cls, task: LLMTask, **kwargs: Any) -> LLMRequest:
        """Build a request with the model already routed for `task`."""
        return cls(task=task, model=kwargs.pop("model", TASK_MODEL[task]), **kwargs)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    model: str
    stop_reason: str | None = None

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)

    request_fingerprint: str
    from_fixture: bool = False
    latency_ms: float | None = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """The only interface the rest of the codebase may depend on."""

    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Base for provider failures."""


class MissingFixtureError(ProviderError):
    """MockProvider was asked for a request it has no recording of.

    The message carries the fingerprint so the fixture can be recorded with
    `record_fixture()` and committed.
    """


# ---------------------------------------------------------------------------
# Live provider
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Live Claude API calls.

    `anthropic` is imported lazily so that the offline path (rule 8) has no
    dependency on the SDK being installed or on credentials existing.
    """

    name = "anthropic"

    def __init__(self, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderError(
                "The `anthropic` package is not installed. Install it, or set MOCK_LLM=true "
                "to run offline from fixtures."
            ) from exc
        # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth
        # login` profile. Never hardcode a key.
        self._client = anthropic.Anthropic()

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.system is not None:
            kwargs["system"] = request.system
        if request.stop_sequences:
            kwargs["stop_sequences"] = list(request.stop_sequences)
        # Only send these when explicitly set — see the note on LLMRequest.
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.effort is not None:
            kwargs["output_config"] = {"effort": request.effort}

        started = time.perf_counter()
        message = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        usage = message.usage
        return LLMResponse(
            text=text,
            model=message.model,
            stop_reason=message.stop_reason,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            request_fingerprint=request.fingerprint(),
            from_fixture=False,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockProvider:
    """Replays recorded responses. Never opens a socket.

    Fixture layout — `llm/fixtures/<fingerprint>.json`:

        {
          "fingerprint": "<sha256 of the request>",
          "task": "classify",
          "model": "claude-haiku-4-5",
          "note": "optional human label",
          "response": { ...LLMResponse fields... }
        }
    """

    name = "mock"

    def __init__(self, fixtures_dir: Path | str | None = None) -> None:
        self.fixtures_dir = Path(fixtures_dir) if fixtures_dir else FIXTURE_DIR

    def _path_for(self, fingerprint: str) -> Path:
        return self.fixtures_dir / f"{fingerprint}.json"

    def complete(self, request: LLMRequest) -> LLMResponse:
        fingerprint = request.fingerprint()
        path = self._path_for(fingerprint)
        if not path.exists():
            raise MissingFixtureError(
                f"No fixture for task={request.task!r} model={request.model!r}.\n"
                f"  expected: {path}\n"
                f"  fingerprint: {fingerprint}\n"
                "Record it with llm.provider.record_fixture(request, response)."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = LLMResponse.model_validate(payload["response"])
        # The recording is authoritative for content; these two fields describe
        # *this* replay.
        return response.model_copy(
            update={"request_fingerprint": fingerprint, "from_fixture": True}
        )

    def has_fixture(self, request: LLMRequest) -> bool:
        return self._path_for(request.fingerprint()).exists()


def record_fixture(
    request: LLMRequest,
    response: LLMResponse,
    fixtures_dir: Path | str | None = None,
    note: str | None = None,
) -> Path:
    """Write a fixture for `request` so MockProvider can replay it offline."""
    directory = Path(fixtures_dir) if fixtures_dir else FIXTURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = request.fingerprint()
    path = directory / f"{fingerprint}.json"
    payload = {
        "fingerprint": fingerprint,
        "task": request.task,
        "model": request.model,
        "note": note,
        "response": response.model_dump(mode="json"),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}


def mock_enabled(env: dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get("MOCK_LLM", "").strip().lower() in _TRUTHY


def get_provider(env: dict[str, str] | None = None, **kwargs: Any) -> LLMProvider:
    """Return the provider selected by `MOCK_LLM`.

    MOCK_LLM=true -> MockProvider (offline, fixtures only)
    otherwise     -> AnthropicProvider (live)
    """
    if mock_enabled(env):
        return MockProvider(**kwargs)
    return AnthropicProvider(**kwargs)


__all__ = [
    "MODEL_HAIKU",
    "MODEL_SONNET",
    "TASK_MODEL",
    "AnthropicProvider",
    "LLMMessage",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMTask",
    "MissingFixtureError",
    "MockProvider",
    "ProviderError",
    "get_provider",
    "mock_enabled",
    "record_fixture",
]
