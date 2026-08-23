"""BM25 and in-memory embeddings, blended.

~2,000 documents does not need a vector database, and pretending it does
would be dishonest (CLAUDE.md §Stack lists one under "explicitly not
used"). What it needs is a lexical ranker that finds "size 8 not
available", a semantic one that also finds "popular sizes stock mein nahi
hai", and a way to combine them.

BM25 is implemented here rather than pulled in, because it is forty lines
and the alternative is a dependency whose tokeniser we would then have to
match to the embedding one. The embeddings are TF-IDF reduced by truncated
SVD — classical LSA, from scikit-learn, deterministic under a fixed seed,
and entirely in memory.

WHAT RANKING IS FOR. It decides what is SHOWN, never what is COUNTED. The
counting in `unstructured.py` classifies every in-scope document, because
a store whose note the ranker happened to miss would otherwise count as a
store that filed nothing — and case #2451's chi-square is a statement
about 68 stores, not about the 12 documents that scored highest.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from semantic_layer.schema import RetrievalSpec

#: Words, numbers and Devanagari-transliterated tokens alike. Deliberately
#: simple: the corpora are short shift notes in English and Hinglish, and a
#: heavier tokeniser would have to be matched in two places.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Ranked:
    """One document's score against one query."""

    document_id: str
    score: float
    bm25: float
    embedding: float


class Bm25:
    """Okapi BM25 over a fixed corpus.

    `k1` controls how quickly term frequency saturates and `b` how much
    document length is penalised; both come from the semantic layer.
    """

    def __init__(self, documents: Sequence[str], spec: RetrievalSpec):
        self.spec = spec
        self.tokens = [tokenise(text) for text in documents]
        self.lengths = np.array([len(document) for document in self.tokens], dtype=float)
        self.average_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        self.frequencies = [Counter(document) for document in self.tokens]

        containing: Counter[str] = Counter()
        for document in self.frequencies:
            containing.update(document.keys())
        total = max(len(self.tokens), 1)
        # Robertson/Sparck Jones idf, floored at zero so a term in almost
        # every document cannot score negative. The smoothing term comes
        # from the semantic layer with the rest of the formula.
        smoothing = spec.bm25.idf_smoothing
        self.idf = {
            term: max(
                math.log(1.0 + (total - count + smoothing) / (count + smoothing)), 0.0
            )
            for term, count in containing.items()
        }

    def scores(self, query: str) -> np.ndarray:
        terms = tokenise(query)
        out = np.zeros(len(self.tokens), dtype=float)
        if not terms or not self.average_length:
            return out
        k1, b = self.spec.bm25.k1, self.spec.bm25.b
        for index, frequency in enumerate(self.frequencies):
            length_ratio = self.lengths[index] / self.average_length
            total = 0.0
            for term in terms:
                count = frequency.get(term, 0)
                if not count:
                    continue
                total += self.idf.get(term, 0.0) * (
                    count * (k1 + 1.0) / (count + k1 * (1.0 - b + b * length_ratio))
                )
            out[index] = total
        return out


class Embeddings:
    """TF-IDF reduced by truncated SVD. In memory, deterministic, no server.

    A corpus smaller than the SVD's requested rank is fitted at the rank
    it can support rather than raising — a scope with nine documents is a
    real case and should still rank.
    """

    def __init__(self, documents: Sequence[str], spec: RetrievalSpec):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.spec = spec
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenise,
            lowercase=True,
            min_df=spec.embeddings.min_df,
            token_pattern=None,
        )
        matrix = self.vectorizer.fit_transform(documents)
        components = max(
            min(spec.embeddings.components, min(matrix.shape) - 1), 1
        )
        self.svd = TruncatedSVD(
            n_components=components, random_state=spec.embeddings.random_state
        )
        self.vectors = _unit(self.svd.fit_transform(matrix))

    def scores(self, query: str) -> np.ndarray:
        vector = _unit(self.svd.transform(self.vectorizer.transform([query])))
        return np.asarray(self.vectors @ vector[0], dtype=float)


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def _normalise(scores: np.ndarray, spec: RetrievalSpec) -> np.ndarray:
    """Min-max to [0, 1] so two different scales can be blended.

    BM25 is unbounded and cosine is in [-1, 1]; adding them raw would let
    whichever happens to be larger decide every ranking. A flat score set
    normalises to zero rather than to noise.
    """
    if not len(scores):
        return scores
    low, high = float(scores.min()), float(scores.max())
    if high - low < spec.min_score_range:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


class HybridIndex:
    """Both rankers over one corpus, blended by the configured weights."""

    def __init__(self, document_ids: Sequence[str], documents: Sequence[str], spec: RetrievalSpec):
        if len(document_ids) != len(documents):
            raise ValueError("document ids and documents must be the same length")
        self.document_ids = list(document_ids)
        self.spec = spec
        self.bm25 = Bm25(documents, spec) if documents else None
        self.embeddings = Embeddings(documents, spec) if documents else None

    def __len__(self) -> int:
        return len(self.document_ids)

    def search(self, query: str, top_k: int | None = None) -> tuple[Ranked, ...]:
        if not self.document_ids or self.bm25 is None or self.embeddings is None:
            return ()
        lexical = self.bm25.scores(query)
        semantic = self.embeddings.scores(query)
        blended = (
            self.spec.bm25_weight * _normalise(lexical, self.spec)
            + self.spec.embedding_weight * _normalise(semantic, self.spec)
        )
        order = np.argsort(-blended, kind="stable")
        limit = top_k if top_k is not None else self.spec.top_k_per_corpus
        return tuple(
            Ranked(
                document_id=self.document_ids[index],
                score=float(blended[index]),
                bm25=float(lexical[index]),
                embedding=float(semantic[index]),
            )
            for index in order[:limit]
        )


__all__ = ["Bm25", "Embeddings", "HybridIndex", "Ranked", "tokenise"]
