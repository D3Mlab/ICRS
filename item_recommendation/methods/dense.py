from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests

from item_recommendation.helpers import top_level_metadata_fields

from .base import BatchResult, ObjectRankingMethod


DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_DIM = 1536


class DenseObjectRanker(ObjectRankingMethod):
    """Dense-retrieval alternative to UmbrellaLLM that scores metadata summaries."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        normalize_embeddings: bool = True,
        request_timeout: float = 60.0,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.api_base = (api_base or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_API_BASE).rstrip("/")
        self.api_key_env = api_key_env
        self._explicit_api_key = api_key
        self.normalize_embeddings = normalize_embeddings
        self.request_timeout = max(1.0, float(request_timeout))
        self.batch_size = max(1, int(batch_size))
        self.embedding_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_api_key(self) -> str:
        key = (
            self._explicit_api_key
            or os.getenv(self.api_key_env or "")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError(
                "DenseObjectRanker requires an OpenRouter API key. "
                f"Set {self.api_key_env or 'OPENROUTER_API_KEY'} or OPENAI_API_KEY."
            )
        return key

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        referer = os.getenv("OPENROUTER_SITE_URL")
        if referer:
            headers["HTTP-Referer"] = referer
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if app_name:
            headers["X-Title"] = app_name
        return headers

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dim = self.embedding_dim or DEFAULT_DIM
            return np.zeros((0, dim), dtype=np.float32)

        endpoint = f"{self.api_base}/embeddings"
        vectors: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            response = requests.post(
                endpoint,
                headers=self._build_headers(),
                json={"model": self.model_name, "input": list(chunk)},
                timeout=self.request_timeout,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:  # pragma: no cover - network
                raise RuntimeError(
                    f"DenseObjectRanker embedding request failed ({response.status_code}): {response.text}"
                ) from exc
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError("OpenRouter embedding response missing 'data'")
            chunk_vecs = [
                np.asarray(entry.get("embedding"), dtype=np.float32)
                for entry in data
                if entry.get("embedding") is not None
            ]
            if chunk_vecs:
                vectors.append(np.stack(chunk_vecs, axis=0))

        if not vectors:
            dim = self.embedding_dim or DEFAULT_DIM
            return np.zeros((len(texts), dim), dtype=np.float32)

        mat = np.vstack(vectors)
        self.embedding_dim = mat.shape[1]
        if self.normalize_embeddings:
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            mat = mat / norms
        return mat.astype(np.float32)

    @staticmethod
    def _similarity_to_score(sim: float) -> float:
        # Scale cosine similarity [-1, 1] into 0-3 range
        scaled = (sim + 1.0) * 1.5
        return float(max(0.0, min(3.0, scaled)))

    @staticmethod
    def _build_summary(item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}
        parts = [f"{name}: {value}" for name, value in top_level_metadata_fields(metadata)]
        summary = "; ".join(parts)
        if summary:
            return summary
        return str(item.get("text") or "")

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

        query_vec = self._embed_texts([user_query])[0]
        summaries = [self._build_summary(item) for item in batch]
        doc_vecs = self._embed_texts(summaries)

        sims = doc_vecs @ query_vec.astype(np.float32)
        scores: Dict[str, float] = {}
        parsed: List[Dict[str, Any]] = []

        for item, sim in zip(batch, sims):
            sid = str(item.get("id") or "")
            if not sid:
                continue
            score = self._similarity_to_score(float(sim))
            scores[sid] = score
            parsed.append(
                {
                    "id": sid,
                    "score": score,
                    "rationale": "dense_similarity",
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

