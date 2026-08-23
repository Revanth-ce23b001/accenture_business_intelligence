"""Lane 2 — the unstructured lane, and the line the model does not cross.

Retrieval ranks. The classifier tags. Python counts.

That third sentence is the whole design. Case #2451's headline
unstructured finding is "21 of 34 treated stores filed a note about
unavailability against 2 of 34 controls, chi-square p < 0.001", and every
number in it is computed here, in Python, from tags the model assigned.
The model was never asked how many. It was asked, twenty-odd times, what
one note is about.

The distinction is not pedantry. A model asked to count will produce a
number that looks exactly like a count and is not one, and no downstream
test can tell the difference. A model asked to tag produces something that
can be checked document by document, and the count that follows is
arithmetic.

WHAT IS CLASSIFIED. Every in-scope document, not the top-ranked ones. A
store whose note the ranker happened to miss would otherwise count as a
store that filed nothing, and the chi-square is a statement about 68
stores rather than about the twelve documents that scored highest.
Ranking decides what is SHOWN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import duckdb
from scipy import stats

from engine.contracts import Evidence
from engine.db import GovernanceError, execute_governed
from engine.evidence import EvidenceFactory
from engine.gather.guard import NO_GUARD, Guard
from engine.gather.retrieval import HybridIndex, Ranked
from llm.classify import Document, DocumentTag, classify_documents
from llm.provider import LLMProvider
from security.policy import User
from semantic_layer.schema import Corpus, SemanticLayer

DERIVED_KIND = "derived_estimate"
DERIVED = "derived"

#: The two store groups case #2451's chi-square compares.
TREATED = "treated"
CONTROL = "control"


@dataclass(frozen=True)
class GroupCount:
    """Stores in one group, and how many of them filed a tagged document."""

    group: str
    stores: int
    flagged: int

    @property
    def rate(self) -> float:
        return self.flagged / self.stores if self.stores else 0.0


@dataclass(frozen=True)
class ChiSquare:
    """A 2x2 test computed in Python, from tags a model assigned."""

    statistic: float
    p_value: float
    degrees_of_freedom: int
    yates: bool
    table: tuple[tuple[int, int], tuple[int, int]]
    min_expected: float
    #: From `gather.yaml -> unstructured.statistics.min_expected_cell`,
    #: carried on the result so `reliable` is answerable without the config.
    min_expected_required: float

    @property
    def reliable(self) -> bool:
        """False when a cell's expected count is too small for the test."""
        return self.min_expected >= self.min_expected_required


@dataclass(frozen=True)
class CorpusResult:
    """One corpus, tagged and counted."""

    corpus: str
    documents: int
    ranked: tuple[Ranked, ...]
    tags: tuple[DocumentTag, ...]
    tag_counts: dict[str, int]

    def documents_for(self, tag: str) -> tuple[DocumentTag, ...]:
        return tuple(item for item in self.tags if item.hypothesis_tag == tag)


@dataclass
class UnstructuredResult:
    """Everything lane 2 found, and how much of it came from the cache."""

    corpora: dict[str, CorpusResult] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()
    #: Keyed by (corpus, tag), never pooled across corpora. A shift note is
    #: the store's own report and a support ticket is a customer's
    #: complaint: they have different base rates and different meanings,
    #: and adding them together produces a number about neither. Case
    #: #2451's "21 of 34 treated stores" is a statement about NOTES.
    group_counts: dict[tuple[str, str], tuple[GroupCount, GroupCount]] = field(
        default_factory=dict
    )
    chi_square: dict[tuple[str, str], ChiSquare] = field(default_factory=dict)
    cache_hits: int = 0
    cache_lookups: int = 0
    batches: int = 0
    distinct_documents: int = 0
    unavailable: dict[str, str] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.cache_lookups if self.cache_lookups else 0.0

    @property
    def all_tags(self) -> tuple[DocumentTag, ...]:
        return tuple(tag for result in self.corpora.values() for tag in result.tags)

    def comparison(
        self, corpus: str, tag: str
    ) -> tuple[GroupCount, GroupCount, ChiSquare] | None:
        """The treated-against-control test for one corpus and one tag."""
        counts = self.group_counts.get((corpus, tag))
        test = self.chi_square.get((corpus, tag))
        if counts is None or test is None:
            return None
        return counts[0], counts[1], test


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def load_documents(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    corpus: Corpus,
    *,
    kpi: str,
    scope: str,
    period_start: date,
    period_end: date,
    guard: Guard = NO_GUARD,
) -> tuple[Document, ...]:
    """Every in-scope document from one corpus, governed."""
    with guard():
        rows, _filtered, _masked = execute_governed(
            user,
            kpi,
            f"""
            SELECT
                c."{corpus.id_column}"   AS document_id,
                c."{corpus.text_column}" AS text,
                c."{corpus.date_column}" AS document_date,
                c.store_id,
                d.region
            FROM "{corpus.table}" AS c
            JOIN dim_store AS d USING (store_id)
            WHERE c."{corpus.date_column}" BETWEEN $period_start AND $period_end
              AND d.region = $scope
            """,
            {"period_start": period_start, "period_end": period_end, "scope": scope},
            connection=connection,
            layer=layer,
            purpose=f"gather.unstructured.{corpus.table}",
        )
    return tuple(
        Document(
            document_id=str(row["document_id"]),
            text=str(row["text"]),
            corpus=corpus.table,
            store_id=str(row["store_id"]) if row.get("store_id") else None,
            document_date=row.get("document_date"),
        )
        for row in rows
    )


def _store_groups(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    *,
    kpi: str,
    scope: str,
    guard: Guard = NO_GUARD,
) -> dict[str, set[str]]:
    """The treated and matched-control stores, from the store master.

    Named in `dim_store` because the generator planted the mechanism
    there; a real deployment would read them from the intervention log.
    """
    with guard():
        rows, _filtered, _masked = execute_governed(
            user,
            kpi,
            """
            SELECT region, store_id, is_treated_2451, is_matched_control_2451
            FROM dim_store
            WHERE region = $scope
            """,
            {"scope": scope},
            connection=connection,
            layer=layer,
            purpose="gather.unstructured.store_groups",
        )
    return {
        TREATED: {str(r["store_id"]) for r in rows if r.get("is_treated_2451")},
        CONTROL: {str(r["store_id"]) for r in rows if r.get("is_matched_control_2451")},
    }


# ---------------------------------------------------------------------------
# Counting — all Python, from here down
# ---------------------------------------------------------------------------


def count_by_group(
    tags: tuple[DocumentTag, ...],
    groups: dict[str, set[str]],
    tag: str,
    spec,
) -> tuple[GroupCount, GroupCount]:
    """Stores in each group with at least one accepted document of this tag.

    Per STORE, not per document: a store that filed three notes about the
    same empty shelf is one store with a problem, and counting the notes
    would make a talkative manager look like an outbreak.
    """
    flagged = {
        item.store_id
        for item in tags
        if item.hypothesis_tag == tag and item.accepted(spec) and item.store_id
    }
    return (
        GroupCount(TREATED, len(groups[TREATED]), len(flagged & groups[TREATED])),
        GroupCount(CONTROL, len(groups[CONTROL]), len(flagged & groups[CONTROL])),
    )


def chi_square(
    treated: GroupCount, control: GroupCount, spec
) -> ChiSquare:
    """A 2x2 chi-square with Yates' correction, in Python.

    Yates because the counts are small — 21 of 34 against 2 of 34 — and
    the uncorrected statistic overstates significance on a table this
    size. The correction makes the p-value larger, which is the direction
    a claim should err in.
    """
    table = (
        (treated.flagged, treated.stores - treated.flagged),
        (control.flagged, control.stores - control.flagged),
    )
    statistic, p_value, degrees, expected = stats.chi2_contingency(
        table, correction=spec.yates_correction
    )
    return ChiSquare(
        statistic=float(statistic),
        p_value=float(p_value),
        degrees_of_freedom=int(degrees),
        yates=spec.yates_correction,
        table=table,
        min_expected=float(expected.min()),
        min_expected_required=float(spec.min_expected_cell),
    )


# ---------------------------------------------------------------------------
# The lane
# ---------------------------------------------------------------------------


def gather_unstructured(
    connection: duckdb.DuckDBPyConnection,
    user: User,
    layer: SemanticLayer,
    factory: EvidenceFactory,
    *,
    kpi: str,
    scope: str,
    period_start: date,
    period_end: date,
    hypotheses: tuple[str, ...],
    provider: LLMProvider | None = None,
    now: datetime | None = None,
    guard: Guard = NO_GUARD,
) -> UnstructuredResult:
    """Rank, tag and count the three document corpora."""
    spec = layer.gather.unstructured
    result = UnstructuredResult()

    try:
        groups = _store_groups(connection, user, layer, kpi=kpi, scope=scope, guard=guard)
    except GovernanceError as exc:
        groups = {TREATED: set(), CONTROL: set()}
        result.unavailable["store_groups"] = str(exc)

    evidence: list[Evidence] = []
    for name, corpus in sorted(spec.corpora.items()):
        try:
            documents = load_documents(
                connection, user, layer, corpus,
                kpi=kpi, scope=scope,
                period_start=period_start, period_end=period_end, guard=guard,
            )
        except GovernanceError as exc:
            result.unavailable[name] = f"not readable at this persona's scope: {exc}"
            continue
        if not documents:
            result.unavailable[name] = (
                f"no {name} in {scope} between {period_start} and {period_end}"
            )
            continue

        index = HybridIndex(
            [document.document_id for document in documents],
            [document.text for document in documents],
            spec.retrieval,
        )
        query = " ".join(
            candidate for candidate in hypotheses if candidate
        ).replace("_", " ")
        ranked = index.search(query)

        run = classify_documents(
            documents,
            hypotheses,
            spec.classification,
            connection=connection,
            provider=provider,
            now=now,
            guard=guard,
        )
        result.cache_hits += run.cache_hits
        result.cache_lookups += run.lookups
        result.batches += run.batches
        result.distinct_documents += run.distinct_documents

        counts: dict[str, int] = {}
        for tag in run.tags:
            if tag.accepted(spec.classification):
                counts[tag.hypothesis_tag] = counts.get(tag.hypothesis_tag, 0) + 1
        result.corpora[name] = CorpusResult(
            corpus=name,
            documents=len(documents),
            ranked=ranked,
            tags=tuple(run.tags),
            tag_counts=counts,
        )

        for tag in sorted(counts):
            evidence.append(
                factory.emit(
                    f"unstructured.{name}.{tag}",
                    kind=corpus.reliability_kind,
                    label=f"{name} tagged {tag} — {scope}",
                    value=counts[tag],
                    unit="count",
                    source_system=corpus.source_system,
                    method="count",
                    description=(
                        f"documents from {corpus.table} whose classifier tag is {tag} "
                        f"at confidence >= {spec.classification.min_confidence}, counted "
                        "in Python"
                    ),
                    ref="semantic_layer/gather.yaml::unstructured",
                    inputs=(corpus.table,),
                    notes=(
                        "The classifier assigned the tags. This count is arithmetic over "
                        "them; the model was never asked how many."
                    ),
                )
            )

    # --- the treated-versus-control comparison ---------------------------
    #
    # Per corpus, never pooled. Case #2451's finding is that 21 of 34
    # treated stores FILED A NOTE about unavailability against 2 of 34
    # controls; a support ticket is a customer complaining, which is a
    # different act with a different base rate, and adding the two
    # together gives a number about neither.
    for corpus_name, corpus_result in sorted(result.corpora.items()):
        for tag in hypotheses:
            if not groups[TREATED] or not groups[CONTROL]:
                continue
            treated, control = count_by_group(
                corpus_result.tags, groups, tag, spec.classification
            )
            if not treated.flagged and not control.flagged:
                continue
            test = chi_square(treated, control, spec.statistics)
            result.group_counts[(corpus_name, tag)] = (treated, control)
            result.chi_square[(corpus_name, tag)] = test

            evidence.extend(
                (
                    factory.emit(
                        f"unstructured.{corpus_name}.{tag}.treated_stores",
                        kind=DERIVED_KIND,
                        label=f"Treated stores with a {tag} {corpus_name} document",
                        value=treated.flagged,
                        unit="count",
                        source_system=DERIVED,
                        method="count",
                        description=(
                            f"{treated.flagged} of {treated.stores} treated stores filed "
                            f"at least one {corpus_name} document tagged {tag}"
                        ),
                        ref="semantic_layer/gather.yaml::unstructured.statistics",
                        inputs=(f"gather.unstructured.{corpus_name}.{tag}",),
                        notes=(
                            "Counted per STORE, not per document: a store that filed "
                            "three notes about the same empty shelf is one store with a "
                            "problem, and counting notes would make a talkative manager "
                            "look like an outbreak."
                        ),
                    ),
                    factory.emit(
                        f"unstructured.{corpus_name}.{tag}.control_stores",
                        kind=DERIVED_KIND,
                        label=f"Control stores with a {tag} {corpus_name} document",
                        value=control.flagged,
                        unit="count",
                        source_system=DERIVED,
                        method="count",
                        description=(
                            f"{control.flagged} of {control.stores} matched-control "
                            f"stores filed at least one {corpus_name} document tagged "
                            f"{tag}"
                        ),
                        ref="semantic_layer/gather.yaml::unstructured.statistics",
                    ),
                    factory.emit(
                        f"unstructured.{corpus_name}.{tag}.chi_square_p",
                        kind=DERIVED_KIND,
                        label=(
                            f"Chi-square p-value, treated against control, {tag} in "
                            f"{corpus_name}"
                        ),
                        value=test.p_value,
                        unit="p_value",
                        source_system=DERIVED,
                        method="compare",
                        description=(
                            "2x2 chi-square"
                            + (" with Yates correction" if test.yates else "")
                            + f" on {test.table}; statistic {test.statistic:.3f}, "
                            f"{test.degrees_of_freedom} d.f."
                        ),
                        ref="semantic_layer/gather.yaml::unstructured.statistics",
                        inputs=(
                            f"gather.unstructured.{corpus_name}.{tag}.treated_stores",
                            f"gather.unstructured.{corpus_name}.{tag}.control_stores",
                        ),
                        notes=(
                            "Computed by scipy from the tag counts. The classifier "
                            "produced the tags and no part of this number."
                        ),
                    ),
                )
            )

    result.evidence = tuple(evidence)
    return result


__all__ = [
    "CONTROL",
    "TREATED",
    "ChiSquare",
    "CorpusResult",
    "GroupCount",
    "UnstructuredResult",
    "chi_square",
    "count_by_group",
    "gather_unstructured",
    "load_documents",
]
