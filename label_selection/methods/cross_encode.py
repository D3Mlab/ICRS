from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from utils.cache_manager import CacheManager
from .base import MethodResult, RelevanceMethod
from ..utils.io import stable_hash

try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    CrossEncoder = None  # type: ignore


class CrossEncoderCache:
    """Small helper around CacheManager for cross-encoder scores."""

    def __init__(
        self,
        cache_dir: Optional[Path],
        model_name: str,
        dataset_root: Optional[Path] = None,
    ) -> None:
        self.cache_base = cache_dir
        self.dataset_root = CacheManager.infer_dataset_root(dataset_root, cache_dir)
        self.model_name = model_name
        self.enabled = cache_dir is not None
        self.module = "label_selection"
        self.method = "cross_encode"

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

    def get_scores(
        self,
        *,
        keys: List[str],
        pairs: List[Tuple[str, str]],  # (query, snippet) pairs
        variant: str,
        config: Dict[str, Any],
        score_fn: Callable[[List[Tuple[str, str]]], np.ndarray],
    ) -> np.ndarray:
        """Get cross-encoder scores with caching."""
        if not keys:
            return np.zeros((0,), dtype=np.float32)

        if not self.enabled:
            scores = score_fn(pairs)
            return np.asarray(scores, dtype=np.float32)

        records, meta = self._load_records(variant)
        if meta.get("config") != config:
            records = {}

        ordered: List[Optional[float]] = [None] * len(keys)
        missing_indices: List[int] = []

        for idx, (key, (query, snippet)) in enumerate(zip(keys, pairs)):
            rec = records.get(key)
            if (
                rec
                and rec.get("query") == query
                and rec.get("snippet") == snippet
                and rec.get("score") is not None
            ):
                ordered[idx] = float(rec["score"])
            else:
                missing_indices.append(idx)

        if missing_indices:
            need_pairs = [pairs[i] for i in missing_indices]
            scores = score_fn(need_pairs)
            scores_array = np.asarray(scores, dtype=np.float32)
            for local_idx, score_val in zip(missing_indices, scores_array):
                ordered[local_idx] = float(score_val)
                records[keys[local_idx]] = {
                    "query": pairs[local_idx][0],
                    "snippet": pairs[local_idx][1],
                    "score": float(score_val),
                }

        for idx, score_val in enumerate(ordered):
            if score_val is None:
                score = score_fn([pairs[idx]])[0]
                score_val = float(score)
                ordered[idx] = score_val
                records[keys[idx]] = {
                    "query": pairs[idx][0],
                    "snippet": pairs[idx][1],
                    "score": score_val,
                }

        scores_array = np.asarray([s for s in ordered], dtype=np.float32)
        self._save_records(records, variant, config)
        return scores_array


class CrossEncoderSnippetRanker(RelevanceMethod):
    """Cross-encoder based snippet ranker using sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError(
                "sentence-transformers package is required for CrossEncoderSnippetRanker. "
                "Install it with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.device = device
        self.cache = CrossEncoderCache(self.cache_dir, self.model_name, dataset_root=self.dataset_root)
        self.model: Optional[CrossEncoder] = None
        self.snippet_ids: List[str] = []
        self.snippet_texts: Dict[str, str] = {}

    def _get_model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self.model is None:
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
        return self.model

    def _cache_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
        }

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Score query-snippet pairs using the cross-encoder model."""
        if not pairs:
            return np.zeros((0,), dtype=np.float32)

        model = self._get_model()
        # CrossEncoder.predict expects a list of (text1, text2) tuples
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        # Convert to numpy array (handles both single value and array)
        if isinstance(scores, (int, float)):
            scores = np.array([scores], dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32)
        return scores

    def fit(self, snippet_texts: Dict[str, str]) -> None:
        """Store snippet texts for later scoring."""
        self.snippet_ids = sorted(snippet_texts.keys())
        self.snippet_texts = snippet_texts

    def score(self, query_text: str, topk: int) -> MethodResult:
        """Score snippets against the query using cross-encoder."""
        if not self.snippet_ids or not self.snippet_texts:
            return MethodResult(scores=[])

        query_hash = stable_hash(query_text)
        variant = "queries"

        # Build keys and pairs for caching, maintaining alignment with snippet_ids
        keys: List[str] = []
        pairs: List[Tuple[str, str]] = []
        valid_snippet_ids: List[str] = []
        for sid in self.snippet_ids:
            snippet_text = self.snippet_texts.get(sid, "")
            if not snippet_text:
                continue
            key = f"{query_hash}::{sid}"
            keys.append(key)
            pairs.append((query_text, snippet_text))
            valid_snippet_ids.append(sid)

        if not pairs:
            return MethodResult(scores=[])

        # Get scores with caching
        scores = self.cache.get_scores(
            keys=keys,
            pairs=pairs,
            variant=variant,
            config=self._cache_config(),
            score_fn=self._score_pairs,
        )

        # Build (snippet_id, score) pairs - scores align with valid_snippet_ids
        score_map: Dict[str, float] = {}
        for idx, sid in enumerate(valid_snippet_ids):
            if idx < len(scores):
                score_map[sid] = float(scores[idx])

        # Sort by score (descending)
        pairs_list: List[Tuple[str, float]] = list(score_map.items())
        pairs_list.sort(key=lambda x: (-x[1], x[0]))

        # Return top-k
        if topk <= 0:
            return MethodResult(scores=[])
        return MethodResult(scores=pairs_list[: min(topk, len(pairs_list))])

