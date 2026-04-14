from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os
import sys
from pathlib import Path

import numpy as np

from utils.cache_manager import CacheManager


class EmbedderBase:
    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = True,
        provider: str = "local",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.provider = (provider or "local").lower()
        self.api_key = api_key
        self.base_url = base_url
        self._local_model = None

    def _ensure_local_model(self):
        if self._local_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            print(f"[EmbedderBase] sentence_transformers not available: {e}", file=sys.stderr)
            raise
        self._local_model = SentenceTransformer(self.model_name)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        self._ensure_local_model()
        embs = self._local_model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=1)
        return embs.astype(np.float32)

    def _resolve_api_key(self, base_url: str) -> Optional[str]:
        # Prefer provider-specific env vars if base_url matches known hosts
        if "deepinfra.com" in base_url.lower():
            return os.getenv("DEEPINFRA_API_KEY") or self.api_key or os.getenv("OPENAI_API_KEY")
        if "openrouter.ai" in base_url.lower():
            return os.getenv("OPENROUTER_API_KEY") or self.api_key or os.getenv("OPENAI_API_KEY")
        return self.api_key or os.getenv("OPENAI_API_KEY")

    def _embed_online(self, texts: List[str]) -> np.ndarray:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            print(f"[EmbedderBase] OpenAI client not available: {e}", file=sys.stderr)
            raise
        # Choose default base URL based on provider/model
        if self.base_url:
            base_url = self.base_url
        elif self.provider == "openrouter" or "openrouter" in (self.base_url or "").lower() or "qwen" in self.model_name.lower():
            base_url = "https://openrouter.ai/api/v1"
        else:
            base_url = "https://api.deepinfra.com/v1/openai"
        api_key = self._resolve_api_key(base_url)
        if not api_key:
            raise ValueError("EmbedderBase online embedding requires API key (DEEPINFRA_TOKEN/OPENROUTER_API_KEY/OPENAI_API_KEY)")
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            resp = client.embeddings.create(model=self.model_name, input=texts, encoding_format="float")
            vecs: List[List[float]] = [d.embedding for d in resp.data]  # type: ignore[attr-defined]
            return np.array(vecs, dtype=np.float32)
        except Exception as e:
            print(f"[EmbedderBase] Online embedding failed: {e}", file=sys.stderr)
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.provider in {"openai", "openrouter"}:
            embs = self._embed_online(texts)
        else:
            embs = self._embed_local(texts)
        if self.normalize_embeddings:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            embs = embs / norms
        return embs.astype(np.float32)

    def encode_with_cache(
        self,
        *,
        module: str,
        method: str,
        dataset_root: Optional[Path],
        cache_base: Optional[Path],
        variant: str,
        keys: List[str],
        texts: List[str],
        meta: Dict[str, Any],
    ) -> np.ndarray:
        if len(keys) != len(texts):
            raise ValueError("keys and texts must have the same length")

        cache_meta = {
            "model_name": self.model_name,
            "normalize_embeddings": bool(self.normalize_embeddings),
            "provider": self.provider,
        }
        cache_meta.update(meta)

        records, stored_meta = CacheManager.load_embedding_cache(
            module=module,
            method=method,
            model_name=self.model_name,
            dataset=dataset_root,
            base_dirs=[cache_base],
            variant=variant,
        )
        if stored_meta.get("config") != cache_meta:
            records = {}
        if not isinstance(records, dict):
            records = {}

        ordered: List[Optional[np.ndarray]] = [None] * len(keys)
        to_compute: List[Tuple[int, str, str]] = []

        for idx, (key, text) in enumerate(zip(keys, texts)):
            rec = records.get(key)
            if rec and rec.get("text") == text and rec.get("embedding") is not None:
                ordered[idx] = np.asarray(rec["embedding"], dtype=np.float32)
            else:
                to_compute.append((idx, key, text))

        if to_compute:
            batch_texts = [item[2] for item in to_compute]
            embeddings = self.embed(batch_texts)
            for (idx, key, text), emb in zip(to_compute, embeddings):
                vec = np.asarray(emb, dtype=np.float32)
                ordered[idx] = vec
                records[key] = {"text": text, "embedding": vec}

        for idx, maybe_vec in enumerate(ordered):
            if maybe_vec is None:
                vec = np.asarray(self.embed([texts[idx]])[0], dtype=np.float32)
                ordered[idx] = vec
                records[keys[idx]] = {"text": texts[idx], "embedding": vec}

        stacked = np.stack([np.asarray(vec, dtype=np.float32) for vec in ordered], axis=0)

        CacheManager.save_embedding_cache(
            module=module,
            method=method,
            model_name=self.model_name,
            records=records,
            meta={"config": cache_meta},
            dataset=dataset_root,
            base_dirs=[cache_base],
            variant=variant,
        )

        return stacked.astype(np.float32)


