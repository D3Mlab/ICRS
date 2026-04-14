from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np
import os
import requests

from cache_manager import CacheManager
from .base import MethodResult, RelevanceMethod
from ..utils.io import stable_hash


DEFAULT_EMBED_DIM = 1536
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class EmbeddingCache:
    """Small helper around CacheManager for snippet/query embeddings."""

    def __init__(self, cache_dir: Optional[Path], model_name: str, dataset_root: Optional[Path] = None) -> None:
        self.cache_base = cache_dir
        self.dataset_root = CacheManager.infer_dataset_root(dataset_root, cache_dir)
        self.model_name = model_name
        self.enabled = cache_dir is not None
        self.module = "label_selection"
        self.method = "dense"

    def _load_records(self, variant: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.enabled:
            return {}, {}
        records, meta = CacheManager.load_embedding_cache(
            module=self.module,
            method=self.method,
            model_name=self.model_name,
            dataset=self.dataset_root,
            base_dirs=[self.cache_base],
            variant=variant,
        )
        if not isinstance(records, dict):
            records = {}
        return records, meta or {}

    def _save_records(self, records: Dict[str, Any], variant: str, config: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        CacheManager.save_embedding_cache(
            module=self.module,
            method=self.method,
            model_name=self.model_name,
            records=records,
            meta={"config": config},
            dataset=self.dataset_root,
            base_dirs=[self.cache_base],
            variant=variant,
        )

    def get_embeddings(
        self,
        *,
        keys: List[str],
        texts: List[str],
        variant: str,
        config: Dict[str, Any],
        encode_fn: Callable[[List[str]], np.ndarray],
    ) -> np.ndarray:
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)

        if not self.enabled:
            return encode_fn(texts)

        records, meta = self._load_records(variant)
        if meta.get("config") != config:
            records = {}

        ordered: List[Optional[np.ndarray]] = [None] * len(keys)
        missing_indices: List[int] = []

        for idx, (key, text) in enumerate(zip(keys, texts)):
            rec = records.get(key)
            if rec and rec.get("text") == text and rec.get("embedding") is not None:
                ordered[idx] = np.asarray(rec["embedding"], dtype=np.float32)
            else:
                missing_indices.append(idx)

        if missing_indices:
            need_texts = [texts[i] for i in missing_indices]
            embeddings = encode_fn(need_texts)
            for local_idx, emb in zip(missing_indices, embeddings):
                vec = np.asarray(emb, dtype=np.float32)
                ordered[local_idx] = vec
                records[keys[local_idx]] = {"text": texts[local_idx], "embedding": vec}

        for idx, vec in enumerate(ordered):
            if vec is None:
                emb = encode_fn([texts[idx]])[0]
                vec = np.asarray(emb, dtype=np.float32)
                ordered[idx] = vec
                records[keys[idx]] = {"text": texts[idx], "embedding": vec}

        stacked = np.stack([np.asarray(vec, dtype=np.float32) for vec in ordered], axis=0)
        self._save_records(records, variant, config)
        return stacked.astype(np.float32)


class DenseSnippetRanker(RelevanceMethod):
    def __init__(
        self,
        model_name: str = "qwen-2.5-8b",
        normalize_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        request_timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.cache_dir = cache_dir
        self.dataset_root = dataset_root
        self.api_base = (api_base or os.getenv("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
        self.api_key_env = api_key_env or "OPENROUTER_API_KEY"
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.request_timeout = request_timeout
        self.batch_size = 16
        self.embedding_dim: Optional[int] = None
        self.cache = EmbeddingCache(self.cache_dir, self.model_name, dataset_root=self.dataset_root)
        self.snippet_ids: List[str] = []
        self.snippet_embs: Optional[np.ndarray] = None

    def _cache_config(self) -> Dict[str, Any]:
        return {
            "normalize_embeddings": bool(self.normalize_embeddings),
            "model_name": self.model_name,
            "api_base": self.api_base,
        }

    def _resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        raise ValueError(
            f"DenseSnippetRanker requires an OpenRouter API key. "
            f"Set {self.api_key_env} or OPENROUTER_API_KEY in the environment."
        )

    def _build_headers(self) -> Dict[str, str]:
        api_key = self._resolve_api_key()
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
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

    def _embed_online(self, texts: List[str]) -> np.ndarray:
        if not texts:
            dim = self.embedding_dim or DEFAULT_EMBED_DIM
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
                    f"DenseSnippetRanker failed to fetch embeddings from OpenRouter: {response.text}"
                ) from exc
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError("OpenRouter embeddings response missing 'data' entries")
            chunk_vecs: List[np.ndarray] = []
            for item in data:
                embedding = item.get("embedding")
                if embedding is None:
                    raise ValueError("OpenRouter embeddings response missing 'embedding'")
                chunk_vecs.append(np.asarray(embedding, dtype=np.float32))
            if not chunk_vecs:
                raise ValueError("OpenRouter returned empty embeddings list")
            vectors.append(np.stack(chunk_vecs, axis=0))

        return np.vstack(vectors)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            dim = self.embedding_dim or DEFAULT_EMBED_DIM
            return np.zeros((0, dim), dtype=np.float32)

        embs = self._embed_online(texts)
        if embs.size == 0:
            dim = self.embedding_dim or DEFAULT_EMBED_DIM
            return np.zeros((len(texts), dim), dtype=np.float32)
        self.embedding_dim = embs.shape[1]
        if self.normalize_embeddings:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            embs = embs / norms
        return embs.astype(np.float32)

    def fit(self, snippet_texts: Dict[str, str]) -> None:
        self.snippet_ids = sorted(snippet_texts.keys())
        if not self.snippet_ids:
            dim = self.embedding_dim or DEFAULT_EMBED_DIM
            self.snippet_embs = np.zeros((0, dim), dtype=np.float32)
            return
        texts = [snippet_texts[sid] for sid in self.snippet_ids]
        self.snippet_embs = self.cache.get_embeddings(
            keys=self.snippet_ids,
            texts=texts,
            variant="snippets",
            config=self._cache_config(),
            encode_fn=self._encode_texts,
        )

    def score(self, query_text: str, topk: int) -> MethodResult:
        if not self.snippet_ids or self.snippet_embs is None or self.snippet_embs.size == 0:
            return MethodResult(scores=[])

        query_key = f"query::{stable_hash(query_text)}"
        query_vec = self.cache.get_embeddings(
            keys=[query_key],
            texts=[query_text],
            variant="queries",
            config=self._cache_config(),
            encode_fn=self._encode_texts,
        )[0]

        scores = self.snippet_embs @ query_vec.astype(np.float32)
        pairs: List[Tuple[str, float]] = list(zip(self.snippet_ids, map(float, scores)))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        if topk <= 0:
            return MethodResult(scores=[])
        return MethodResult(scores=pairs[: min(topk, len(pairs))])