from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from tqdm import tqdm

from item_recommendation.helpers import top_level_metadata_fields

from .base import BatchResult, ObjectRankingMethod


DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_DIM = 1536


class FusionDenseObjectRanker(ObjectRankingMethod):
    """Dense-retrieval variant of the fusion scorer using OpenRouter embeddings."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        aggregation: str = "mean",
        normalize_embeddings: bool = True,
        request_timeout: float = 60.0,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.api_base = (api_base or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_API_BASE).rstrip("/")
        self.api_key_env = api_key_env or "OPENROUTER_API_KEY"
        self._explicit_api_key = api_key
        self.aggregation = aggregation.lower()
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
            or os.getenv(self.api_key_env)
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError(
                "FusionDenseObjectRanker requires an OpenRouter API key. "
                f"Set {self.api_key_env} or OPENROUTER_API_KEY in the environment."
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
        headers = self._build_headers()
        vectors: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            response = requests.post(
                endpoint,
                headers=headers,
                json={"model": self.model_name, "input": chunk},
                timeout=self.request_timeout,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:  # pragma: no cover - network
                raise RuntimeError(
                    f"OpenRouter embeddings request failed ({response.status_code}): {response.text}"
                ) from exc
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError("OpenRouter response missing 'data' entries")
            chunk_vecs: List[np.ndarray] = []
            for entry in data:
                emb = entry.get("embedding")
                if emb is None:
                    raise ValueError("OpenRouter response missing 'embedding'")
                chunk_vecs.append(np.asarray(emb, dtype=np.float32))
            if not chunk_vecs:
                continue
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

    def _aggregate(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        agg = self.aggregation
        if agg == "sum":
            return float(sum(scores))
        if agg == "max":
            return float(max(scores))
        if agg == "min":
            return float(min(scores))
        return float(sum(scores) / len(scores))

    @staticmethod
    def _similarity_to_score(sim: float) -> float:
        # Convert cosine similarity [-1, 1] to 0-3
        scaled = (sim + 1.0) * 1.5
        return float(max(0.0, min(3.0, scaled)))

    # ------------------------------------------------------------------
    # ObjectRankingMethod API
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

        field_texts: List[str] = []
        owners: List[Tuple[str, str]] = []

        for item in tqdm(batch, desc="FusionDense collecting fields", unit="segment", disable=False):
            sid = str(item.get("id") or "")
            if not sid:
                continue
            metadata = item.get("metadata") or {}
            field_entries = top_level_metadata_fields(metadata)
            fields = [{"name": name, "value": text} for name, text in field_entries]
            had_field = False
            for field in fields:
                if not isinstance(field, dict):
                    continue
                value = str(field.get("value") or "").strip()
                if not value:
                    continue
                name = str(field.get("name") or "field")
                text = f"{name}: {value}"
                field_texts.append(text)
                owners.append((sid, name))
                had_field = True
            if not had_field:
                summary = str(item.get("text") or "").strip()
                if summary:
                    field_texts.append(summary)
                    owners.append((sid, "summary"))

        embeddings = self._embed_texts(field_texts)
        sims = embeddings @ query_vec.astype(np.float32) if embeddings.size else np.array([], dtype=np.float32)

        per_segment_scores: Dict[str, List[float]] = defaultdict(list)
        per_segment_fields: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for (sid, name), sim in zip(owners, sims):
            score = self._similarity_to_score(float(sim))
            per_segment_scores[sid].append(score)
            per_segment_fields[sid].append(
                {
                    "name": name,
                    "score": score,
                    "similarity": float(sim),
                }
            )

        scores: Dict[str, float] = {}
        parsed_items: List[Dict[str, Any]] = []

        for item in batch:
            sid = str(item.get("id") or "")
            if not sid:
                continue
            agg_score = self._aggregate(per_segment_scores.get(sid, []))
            scores[sid] = agg_score
            parsed_items.append(
                {
                    "id": sid,
                    "score": agg_score,
                    "rationale": "dense_similarity",
                    "field_scores": per_segment_fields.get(sid, []),
                }
            )

        parsed_items.sort(key=lambda entry: float(entry.get("score", 0.0)), reverse=True)

        return BatchResult(
            scores=scores,
            parsed_items=parsed_items,
            field_scores=dict(per_segment_fields),
            raw_response=None,
            temperature=None,
            prompt=None,
        )

