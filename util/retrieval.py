from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SearchHit:
    score: float
    chunk: dict


class TfidfRetriever:
    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 50_000,
    ) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self._X: Any | None = None
        self._chunks: list[dict] = []

    def fit(self, chunks: list[dict]) -> None:
        self._chunks = list(chunks)
        texts = [f"{c.get('header','')}\n{c.get('text','')}" for c in self._chunks]
        self._X = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if self._X is None:
            raise RuntimeError("Retriever is not fitted. Call fit(chunks) first.")

        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._X).ravel()
        idxs = sims.argsort()[::-1][:top_k]

        hits: list[SearchHit] = []
        for idx in idxs:
            hits.append(SearchHit(score=float(sims[int(idx)]), chunk=self._chunks[int(idx)]))
        return hits
