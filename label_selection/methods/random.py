from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np

from .base import MethodResult, RelevanceMethod

class RandomSnippetRanker(RelevanceMethod):
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
    ):
        self.cache_dir = cache_dir
        self.dataset_root = dataset_root
        self.snippet_ids: List[str] = []

    def _cache_config(self) -> Dict[str, Any]:
        return {
            "normalize_embeddings": bool(self.normalize_embeddings),
            "model_name": self.model_name,
            "api_base": self.api_base,
        }

    def fit(self, snippet_texts: Dict[str, str]) -> None:
        self.snippet_ids = sorted(snippet_texts.keys())

    def score(self, query_text: str, topk: int) -> MethodResult:
        scores = np.random.random(len(self.snippet_ids))
        pairs: List[Tuple[str, float]] = list(zip(self.snippet_ids, map(float, scores)))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        if topk <= 0:
            return MethodResult(scores=[])
        return MethodResult(scores=pairs[: min(topk, len(pairs))])