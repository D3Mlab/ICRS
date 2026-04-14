from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import time
from collections import deque

try:
    import cohere  # type: ignore

    _HAS_COHERE = True
except ImportError:  # pragma: no cover - dependency guard
    _HAS_COHERE = False
    cohere = None  # type: ignore

from item_recommendation.helpers import top_level_metadata_fields

from .base import BatchResult, ObjectRankingMethod
from .bm25 import BM25ObjectRanker


DEFAULT_RERANK_MODEL = "cohere/rerank-3.5"


class RerankObjectRanker(ObjectRankingMethod):
    """Two-stage retrieval: BM25 filter then Cohere rerank."""

    _call_timestamps: deque = deque(maxlen=20)  # shared simple limiter

    def __init__(
        self,
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        initial_k: int = 100,
        rerank_api_key: Optional[str] = None,
        rerank_api_key_env: str = "COHERE_API_KEY",
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_top_n: int = 50,
        request_timeout: float = 30.0,
    ) -> None:
        self.bm25 = BM25ObjectRanker(k1=bm25_k1, b=bm25_b)
        self.initial_k = max(1, int(initial_k))
        self.rerank_api_key = rerank_api_key
        self.rerank_api_key_env = rerank_api_key_env or "COHERE_API_KEY"
        self.rerank_model = rerank_model or DEFAULT_RERANK_MODEL
        self.rerank_top_n = 5
        self.request_timeout = max(1.0, float(request_timeout))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_rerank_key(self) -> str:
        key = self.rerank_api_key or os.getenv(self.rerank_api_key_env, "")
        if not key:
            raise ValueError(
                "RerankObjectRanker requires a Cohere API key. "
                f"Set {self.rerank_api_key_env} in the environment."
            )
        return key

    @staticmethod
    def _build_summary(item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}
        parts = [f"{name}: {value}" for name, value in top_level_metadata_fields(metadata)]
        summary = "; ".join(parts)
        if summary:
            return summary
        return str(item.get("text") or "")

    def _cohere_rerank(self, query: str, docs: List[Dict[str, str]]) -> Optional[Dict[str, float]]:
        if not docs:
            return {}
        if not _HAS_COHERE:  # pragma: no cover - dependency guard
            raise RuntimeError("cohere package is required for RerankObjectRanker. Install with: pip install cohere")
        # Simple rate limit: max 10 calls per 60 seconds
        now = time.time()
        while len(self._call_timestamps) >= 10 and now - self._call_timestamps[0] < 60:
            sleep_for = 60 - (now - self._call_timestamps[0])
            time.sleep(max(0.0, sleep_for))
            now = time.time()
        self._call_timestamps.append(now)
        client = cohere.ClientV2(api_key=self._resolve_rerank_key())
        def _do_call() -> Any:
            scores = client.rerank(
                model=self.rerank_model,
                query=query,
                documents=[{"text": d["text"], "id": d["id"]} for d in docs],
                top_n=min(len(docs), self.rerank_top_n),
            )
            return scores

        resp = None
        for attempt in range(2):  # one retry on 429
            try:
                resp = _do_call()
                break
            except Exception as exc:  # pragma: no cover - network
                msg = str(exc)
                if "429" in msg or "rate" in msg.lower():
                    wait_s = 60
                    print(f"[RerankObjectRanker] Rate limited, sleeping {wait_s}s then retrying...")
                    time.sleep(wait_s)
                    continue
                print(f"[RerankObjectRanker] Cohere rerank failed: {exc}")
                return None
        if resp is None:
            return None

        scores: Dict[str, float] = {}
        results = getattr(resp, "results", []) or []

        # If results contain document refs, use them; otherwise use index mapping
        for idx, entry in enumerate(results):
            doc = getattr(entry, "document", None)
            doc_id = None
            if isinstance(doc, dict):
                doc_id = doc.get("id")
            elif doc is not None:
                doc_id = getattr(doc, "id", None)
            rel = getattr(entry, "relevance_score", None)
            if doc_id is None and doc is None:
                # Fallback: use index to map back to docs list
                doc_id = docs[idx]["id"] if idx < len(docs) else None
            if doc_id is None or rel is None:
                continue
            scores[str(doc_id)] = float(max(0.0, min(3.0, float(rel) * 3.0)))
        return scores

    # ------------------------------------------------------------------
    # ObjectRankingMethod interface
    # ------------------------------------------------------------------
    def score_batch(
        self,
        user_query: str,
        batch: List[Dict[str, Any]],
        temperature: float,
        run_index: int,
    ) -> BatchResult:
        if not batch:
            return BatchResult(scores={})

        # 1) BM25 filter to get initial candidates
        bm25_result = self.bm25.score_batch(
            user_query=user_query,
            batch=batch,
            temperature=temperature,
            run_index=run_index,
        )

        # Select top-k by bm25 score
        ranked_bm25 = sorted(
            bm25_result.parsed_items or [],
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )
        candidate_ids = [entry.get("id") for entry in ranked_bm25[: self.initial_k] if entry.get("id")]
        bm25_top_max = 0.0
        if ranked_bm25:
            bm25_top_max = max(float(entry.get("score", 0.0)) for entry in ranked_bm25[: self.initial_k])
        id_to_item = {str(item.get("id")): item for item in batch if item.get("id")}

        docs: List[Dict[str, str]] = []
        for cid in candidate_ids:
            item = id_to_item.get(str(cid))
            if not item:
                continue
            docs.append({"id": str(cid), "text": self._build_summary(item)})

        rerank_scores = self._cohere_rerank(user_query, docs)
        print(rerank_scores)
        if not rerank_scores:
            # fallback to bm25 only
            return bm25_result

        scores: Dict[str, float] = {}
        parsed: List[Dict[str, Any]] = []

        # Start from bm25 scores for all
        scores.update(bm25_result.scores or {})

        # Override candidates with rerank scores, offset by top-k bm25 max to keep them above others
        for sid, score in rerank_scores.items():
            scores[sid] = bm25_top_max + float(score)

        # Build parsed list
        for item in batch:
            sid = str(item.get("id") or "")
            if not sid:
                continue
            score = float(scores.get(sid, 0.0))
            parsed.append(
                {
                    "id": sid,
                    "score": score,
                    "rationale": "cohere_rerank" if sid in rerank_scores else "bm25_fallback",
                    "doc_id": item.get("doc_id"),
                }
            )

        parsed.sort(key=lambda entry: float(entry.get("score", 0.0)), reverse=True)

        return BatchResult(
            scores=scores,
            parsed_items=parsed,
            field_scores={},
            raw_response=None,
            temperature=None,
            prompt=None,
        )

