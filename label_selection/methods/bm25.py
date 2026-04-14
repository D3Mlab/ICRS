from __future__ import annotations

from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .base import MethodResult, RelevanceMethod


class BM25SnippetRanker(RelevanceMethod):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.snippet_ids: List[str] = []
        self.tokenized: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def fit(self, snippet_texts: Dict[str, str]) -> None:
        self.snippet_ids = sorted(snippet_texts.keys())
        self.tokenized = [snippet_texts[i].split() for i in self.snippet_ids]
        self.bm25 = BM25Okapi(self.tokenized, k1=self.k1, b=self.b)

    def score(self, query_text: str, topk: int) -> MethodResult:
        assert self.bm25 is not None
        q_tokens = query_text.split()
        scores = self.bm25.get_scores(q_tokens)
        pairs: List[Tuple[str, float]] = list(zip(self.snippet_ids, map(float, scores)))
        # sort by (score desc, snippet_id asc) for determinism
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return MethodResult(scores=pairs[:topk])


