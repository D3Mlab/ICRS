from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import cohere  # type: ignore

    _HAS_COHERE = True
except ImportError:  # pragma: no cover - dependency guard
    _HAS_COHERE = False
    cohere = None  # type: ignore

import numpy as np

from .base import MethodResult, RelevanceMethod
from .bm25 import BM25SnippetRanker


DEFAULT_RERANK_MODEL = "cohere/rerank-v3.5"


class DenseCohereRerankSnippetRanker(RelevanceMethod):
    """BM25 retrieval followed by Cohere rerank on the top-K snippets."""

    def __init__(
        self,
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        initial_k: int = 5,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_top_n: int = 5,
        rerank_api_key: Optional[str] = None,
        rerank_api_key_env: str = "COHERE_API_KEY",
        request_timeout: float = 30.0,
    ) -> None:
        self.bm25 = BM25SnippetRanker(k1=bm25_k1, b=bm25_b)
        self.initial_k = max(1, int(initial_k))
        self.rerank_model = 'rerank-v3.5'
        self.rerank_top_n = max(1, int(rerank_top_n))
        self.rerank_api_key = rerank_api_key
        self.rerank_api_key_env = rerank_api_key_env or "COHERE_API_KEY"
        self.request_timeout = max(1.0, float(request_timeout))
        self.snippet_texts: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def fit(self, snippet_texts: Dict[str, str]) -> None:
        self.snippet_texts = dict(snippet_texts)
        self.bm25.fit(snippet_texts)

    def _resolve_key(self) -> str:
        key = self.rerank_api_key or os.getenv(self.rerank_api_key_env, "")
        if not key:
            raise ValueError(
                "DenseCohereRerankSnippetRanker requires a Cohere API key. "
                f"Set {self.rerank_api_key_env} or provide rerank_api_key."
            )
        return key

    def _cohere_rerank(self, query: str, docs: List[Tuple[str, str]]) -> Optional[Dict[str, float]]:
        if not docs:
            return {}
        if not _HAS_COHERE:  # pragma: no cover - dependency guard
            raise RuntimeError("cohere package is required for rerank. Install with: pip install cohere")
        client = cohere.ClientV2(api_key=self._resolve_key())

        def _do_call():
            return client.rerank(
                model=self.rerank_model,
                query=query,
                documents=[{"text": text, "id": sid} for sid, text in docs],
                top_n=min(len(docs), self.rerank_top_n),
            )

        resp = None
        for attempt in range(2):  # one retry on timeout or 429
            try:
                resp = _do_call()
                break
            except Exception as exc:  # pragma: no cover - network
                msg = str(exc).lower()
                if "timeout" in msg or "timed out" in msg or "429" in msg or "rate" in msg:
                    wait_s = 60
                    print(f"[DenseCohereRerankSnippetRanker] Cohere rate/timeout, sleeping {wait_s}s then retrying...")
                    import time

                    time.sleep(wait_s)
                    continue
                print(f"[DenseCohereRerankSnippetRanker] Cohere rerank failed: {exc}")
                return None
        if resp is None:
            return None

        scores: Dict[str, float] = {}
        results = getattr(resp, "results", []) or []

        for entry in results:
            # Cohere SDK v2 sometimes returns only index/relevance_score; use index fallback when no doc id
            doc = getattr(entry, "document", None)
            doc_id = None
            if isinstance(doc, dict):
                doc_id = doc.get("id")
            elif doc is not None:
                doc_id = getattr(doc, "id", None)
            rel = getattr(entry, "relevance_score", None)
            if doc_id is None:
                idx = getattr(entry, "index", None)
                if idx is not None and 0 <= int(idx) < len(docs):
                    doc_id = docs[int(idx)][0]
            if doc_id is None or rel is None:
                continue
            scores[str(doc_id)] = float(rel)
        return scores

    # ------------------------------------------------------------------
    def score(self, query_text: str, topk: int) -> MethodResult:
        if not self.snippet_texts:
            return MethodResult(scores=[])

        # Dense retrieval to get initial ranking
        bm25_topk = max(topk, self.initial_k)
        bm25_result = self.bm25.score(query_text, topk=bm25_topk).scores
        if not bm25_result:
            return MethodResult(scores=[])

        candidate_ids = [sid for sid, _ in bm25_result[: self.initial_k]]
        docs = [(sid, self.snippet_texts.get(sid, "")) for sid in candidate_ids if self.snippet_texts.get(sid)]
        
        rerank_scores = self._cohere_rerank(query_text, docs)
        if not rerank_scores:
            return MethodResult(scores=bm25_result[:topk])

        # Offset rerank scores so top-k remain above others while preserving rerank ordering
        bm25_top_max = max(score for _, score in bm25_result[: self.initial_k])
        score_map: Dict[str, float] = {sid: float(score) for sid, score in bm25_result}
        for sid, rel in rerank_scores.items():
            score_map[sid] = bm25_top_max + float(rel)

        pairs = list(score_map.items())
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return MethodResult(scores=pairs[: min(topk, len(pairs))])


