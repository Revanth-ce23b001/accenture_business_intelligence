"""Document classification. Tags in, no numbers out.

This is the one place a model touches a document, and the boundary is
drawn as narrowly as it can be drawn. For each document the classifier
returns three things:

    hypothesis_tag        which candidate the document is about, or `none`
    extracted_event_date  the date the document says something happened
    confidence            how sure it is ABOUT ITS OWN TAGGING

and nothing else. It never counts, never computes a rate, and never
returns a statistic. Case #2451's "21 of 34 treated stores, 2 of 34
controls, chi-square p < 0.001" is produced by
`engine/gather/unstructured.py` from these tags in Python — which is the
only reason the p-value means anything (CLAUDE.md rule 1).

`confidence` is worth being precise about, because it is a number and a
model produced it. It is not a fact about the business: it is the model's
report on its own tagging, used to decide whether a tag is accepted at
all. It never reaches an `Evidence.value` and never appears on a case.
`Evidence` refuses a model-produced numeric value outright, so this is
enforced and not merely intended.

CACHING. Documents repeat — "Routine day, footfall normal, no issues to
report" is one document however many stores filed it — so the cache key
is a hash of the DOCUMENT, not of the batch: the text, the candidate tag
set, the prompt version and the model. Batching is how the calls are made
efficient; hashing is how most of them stop being made at all.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, Iterable, Sequence

import duckdb

from llm.provider import LLMMessage, LLMProvider, LLMRequest, get_provider
from semantic_layer.schema import ClassificationSpec

#: What the model is asked to return, one object per document. Kept in the
#: prompt and in the parser so the two cannot drift.
RESPONSE_FIELDS = ("id", "hypothesis_tag", "event_date", "confidence")

SYSTEM_PROMPT = """\
You tag retail store documents against a fixed list of candidate causes.

For each document return exactly one JSON object with these fields:
  id              the document id, copied verbatim
  hypothesis_tag  one of the candidate tags, or "none"
  event_date      the date the document says something happened, as
                  YYYY-MM-DD, or null if it does not say
  confidence      your confidence in the TAG, 0.0 to 1.0

Rules:
  - Return a JSON array, one object per document, in the order given.
  - Never invent a count, a rate, a percentage or any other statistic.
    You are not being asked how many documents say something.
  - "none" is a real answer. Most documents are about nothing in
    particular, and tagging them anyway makes the tags worthless.
  - Tag what the document SAYS, not what you think caused it."""


#: A guard is anything that can be entered around a warehouse read, so a
#: caller running several lanes at once can serialise them. `llm/` does not
#: import `engine/` — the caller passes one in, and the default is none.
Guard = Callable[[], AbstractContextManager]
NO_GUARD: Guard = nullcontext


class ClassificationError(RuntimeError):
    """The model's reply could not be read as tags."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """One document offered for tagging."""

    document_id: str
    text: str
    corpus: str
    store_id: str | None = None
    document_date: date | None = None

    def content_hash(self, tags: Sequence[str], spec: ClassificationSpec, model: str) -> str:
        """The cache key: everything that could change the answer.

        The text, the candidate tags, the prompt version and the model.
        Not the document id, and not the batch it happened to land in —
        the same note filed by forty stores is one classification.
        """
        payload = json.dumps(
            {
                "text": self.text,
                "tags": sorted(tags),
                "prompt_version": spec.prompt_version,
                "model": model,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DocumentTag:
    """What the classifier said about one document.

    Deliberately NOT an `Evidence`. A tag is an input to the counting, and
    the counting is what becomes evidence.
    """

    document_id: str
    corpus: str
    hypothesis_tag: str
    confidence: float
    extracted_event_date: date | None
    store_id: str | None
    content_hash: str
    from_cache: bool
    model: str

    def accepted(self, spec: ClassificationSpec) -> bool:
        """A tag below the confidence floor is recorded and not counted."""
        return (
            self.hypothesis_tag != spec.null_tag
            and self.confidence >= spec.min_confidence
        )


@dataclass
class ClassificationRun:
    """Everything one classification pass did, including what it did not do."""

    tags: list[DocumentTag] = field(default_factory=list)
    lookups: int = 0
    cache_hits: int = 0
    batches: int = 0
    documents: int = 0
    distinct_documents: int = 0
    model: str = ""

    @property
    def cache_hit_rate(self) -> float:
        """Share of document lookups served without a model call."""
        return self.cache_hits / self.lookups if self.lookups else 0.0

    @property
    def deduplication_factor(self) -> float:
        """Documents per distinct document. The corpora repeat heavily."""
        return self.documents / self.distinct_documents if self.distinct_documents else 1.0

    def by_id(self) -> dict[str, DocumentTag]:
        return {tag.document_id: tag for tag in self.tags}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_INSERT = (
    "INSERT INTO llm_cache (content_hash, task, model, prompt_version, response_json, "
    "created_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (content_hash) DO NOTHING"
)


class ClassificationCache:
    """Content-hash cache, in the warehouse.

    In the warehouse rather than in memory so it survives the process: the
    second run of a demo should not pay for the first one again, and
    CLAUDE.md's cost target is per case, not per process.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection, spec: ClassificationSpec):
        self.connection = connection
        self.spec = spec

    def get_many(self, hashes: Iterable[str]) -> dict[str, dict[str, Any]]:
        wanted = list(dict.fromkeys(hashes))
        if not wanted:
            return {}
        placeholders = ", ".join("?" for _ in wanted)
        rows = self.connection.execute(
            f"SELECT content_hash, response_json FROM {self.spec.cache_table} "
            f"WHERE content_hash IN ({placeholders})",
            wanted,
        ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def put_many(
        self, entries: dict[str, dict[str, Any]], model: str, now: datetime
    ) -> int:
        from engine.db import as_stored_timestamp

        rows = [
            (
                content_hash,
                self.spec.task,
                model,
                self.spec.prompt_version,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
                as_stored_timestamp(now),
            )
            for content_hash, payload in entries.items()
        ]
        if rows:
            self.connection.executemany(CACHE_INSERT, rows)
        return len(rows)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def build_request(
    documents: Sequence[Document],
    tags: Sequence[str],
    spec: ClassificationSpec,
    *,
    model: str | None = None,
) -> LLMRequest:
    """One batch, fully described.

    Deterministic by construction — the request fingerprint is both the
    fixture key and, upstream of it, part of what makes the offline demo
    reproducible. Nothing time-varying goes in.
    """
    catalogue = "\n".join(f"  {tag}" for tag in sorted(tags))
    body = "\n".join(
        f'{{"id": "{document.document_id}", "text": {json.dumps(document.text, ensure_ascii=False)}}}'
        for document in documents
    )
    prompt = (
        f"Candidate tags:\n{catalogue}\n  {spec.null_tag}\n\n"
        f"Documents:\n{body}\n\n"
        "Return the JSON array now."
    )
    return LLMRequest.for_task(
        spec.task,
        system=SYSTEM_PROMPT,
        messages=(LLMMessage(role="user", content=prompt),),
        max_tokens=len(documents) * _TOKENS_PER_DOCUMENT,
        temperature=0.0,
        **({"model": model} if model else {}),
    )


#: Generous per document: an id, a tag, a date and a float.
_TOKENS_PER_DOCUMENT = 64

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def parse_response(text: str) -> list[dict[str, Any]]:
    """Read the model's reply as a list of tag objects.

    Tolerates a fenced code block or a sentence of preamble, because a
    model occasionally adds one and failing the whole batch over it would
    be brittle. Does not tolerate a missing field: a tag object that
    cannot be read is an error, not a default.
    """
    match = _JSON_ARRAY.search(text)
    if match is None:
        raise ClassificationError(f"no JSON array in the reply: {text[:200]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ClassificationError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ClassificationError("reply is not a JSON array")

    parsed: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ClassificationError(f"tag entry is not an object: {entry!r}")
        missing = [name for name in RESPONSE_FIELDS if name not in entry]
        if missing:
            raise ClassificationError(f"tag entry missing {missing}: {entry!r}")
        parsed.append(entry)
    return parsed


def _as_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def classify_documents(
    documents: Sequence[Document],
    tags: Sequence[str],
    spec: ClassificationSpec,
    *,
    connection: duckdb.DuckDBPyConnection,
    provider: LLMProvider | None = None,
    now: datetime | None = None,
    model: str | None = None,
    guard: Guard = NO_GUARD,
) -> ClassificationRun:
    """Tag every document, using the cache and batching what is left.

    Order of operations, and each step is there for a reason:

      1. hash every document. Identical text is one classification.
      2. look the hashes up. Most of them, on a second run all of them.
      3. batch the misses, and only the misses.
      4. write every new classification back under its own hash, so a
         batch that contained one new document does not have to be
         repeated for the other twenty-four.
    """
    provider = provider or get_provider()
    now = now or datetime.now()
    cache = ClassificationCache(connection, spec)

    resolved_model = model or LLMRequest.for_task(spec.task, system="", messages=()).model
    hashes = {
        document.document_id: document.content_hash(tags, spec, resolved_model)
        for document in documents
    }
    distinct: dict[str, Document] = {}
    for document in documents:
        distinct.setdefault(hashes[document.document_id], document)

    run = ClassificationRun(
        lookups=len(documents),
        documents=len(documents),
        distinct_documents=len(distinct),
        model=resolved_model,
    )

    with guard():
        cached = cache.get_many(distinct)
    run.cache_hits = sum(
        1 for document in documents if hashes[document.document_id] in cached
    )

    pending = [
        document
        for content_hash, document in distinct.items()
        if content_hash not in cached
    ]
    fresh: dict[str, dict[str, Any]] = {}
    for start in range(0, len(pending), spec.batch_size):
        batch = pending[start : start + spec.batch_size]
        request = build_request(batch, tags, spec, model=model)
        response = provider.complete(request)
        run.batches += 1

        by_id = {str(entry["id"]): entry for entry in parse_response(response.text)}
        for document in batch:
            entry = by_id.get(document.document_id)
            if entry is None:
                raise ClassificationError(
                    f"the reply has no tag for document {document.document_id!r}; "
                    f"a batch of {len(batch)} came back with {len(by_id)} tags"
                )
            fresh[hashes[document.document_id]] = {
                "hypothesis_tag": str(entry["hypothesis_tag"]),
                "event_date": entry["event_date"],
                "confidence": float(entry["confidence"]),
            }

    if fresh:
        with guard():
            cache.put_many(fresh, resolved_model, now)
    resolved = {**cached, **fresh}

    for document in documents:
        content_hash = hashes[document.document_id]
        payload = resolved[content_hash]
        run.tags.append(
            DocumentTag(
                document_id=document.document_id,
                corpus=document.corpus,
                hypothesis_tag=str(payload["hypothesis_tag"]),
                confidence=float(payload["confidence"]),
                extracted_event_date=_as_date(payload.get("event_date")),
                store_id=document.store_id,
                content_hash=content_hash,
                from_cache=content_hash in cached,
                model=resolved_model,
            )
        )
    return run


__all__ = [
    "CACHE_INSERT",
    "NO_GUARD",
    "RESPONSE_FIELDS",
    "SYSTEM_PROMPT",
    "ClassificationCache",
    "ClassificationError",
    "ClassificationRun",
    "Document",
    "DocumentTag",
    "Guard",
    "build_request",
    "classify_documents",
    "parse_response",
]
