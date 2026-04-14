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


class RandomObjectRanker(ObjectRankingMethod):
    """Dense-retrieval alternative to UmbrellaLLM that scores metadata summaries."""

    def __init__(
        self,
    ) -> None:
        self.object_ids: List[str] = []

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
        scores: Dict[str, float] = {}
        parsed: List[Dict[str, Any]] = []

        for item in batch:
            sid = str(item.get("id") or "")
            if not sid:
                continue
            score = np.random.random()
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

