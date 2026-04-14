from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pipeline import SegmentRankingResult, rank_from_meta_json, rank_segments


@dataclass
class ObjectRelevanceState:
    last_result: Optional[SegmentRankingResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObjectRelevanceFilter:
    """High-level wrapper around the object ranking pipeline."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        provider: str = "openai",
        method_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for ObjectRelevanceFilter")
        base_cfg = copy.deepcopy(method_cfg) if method_cfg else {}
        base_cfg.setdefault("method", "UMBRELLA_LLM")
        llm_cfg = dict(base_cfg.get("llm", {}))
        llm_cfg.setdefault("api_key", api_key)
        llm_cfg.setdefault("model", model)
        llm_cfg.setdefault("temperature", temperature)
        llm_cfg.setdefault("provider", provider)
        base_cfg["llm"] = llm_cfg
        self._base_cfg = base_cfg
        self.state = ObjectRelevanceState()

    def _merge_cfg(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not overrides:
            return copy.deepcopy(self._base_cfg)
        merged = copy.deepcopy(self._base_cfg)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def rank_segments(
        self,
        user_query: str,
        segments: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        self_consistency_k: int = 1,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        method_overrides: Optional[Dict[str, Any]] = None,
    ) -> SegmentRankingResult:
        cfg = self._merge_cfg(method_overrides)
        effective_batch_size = int(cfg.get("batch_size", 40)) if batch_size is None else int(batch_size)
        effective_self_k = int(cfg.get("self_consistency_k", self_consistency_k))
        effective_cache_dir = cache_dir
        if effective_cache_dir is None and cfg.get("cache_dir"):
            effective_cache_dir = Path(str(cfg.get("cache_dir")))
        if isinstance(effective_cache_dir, str):
            effective_cache_dir = Path(effective_cache_dir)
        result = rank_segments(
            user_query=user_query,
            segments=segments,
            batch_size=effective_batch_size,
            self_consistency_k=effective_self_k,
            cache_dir=effective_cache_dir,
            verbose=verbose,
            method_cfg=cfg,
        )
        self.state.last_result = result
        return result

    def rank_from_meta_json(
        self,
        user_query: str,
        meta_json_path: Path,
        batch_size: Optional[int] = None,
        self_consistency_k: int = 1,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        method_overrides: Optional[Dict[str, Any]] = None,
    ) -> SegmentRankingResult:
        cfg = self._merge_cfg(method_overrides)
        effective_batch_size = int(cfg.get("batch_size", 40)) if batch_size is None else int(batch_size)
        effective_self_k = int(cfg.get("self_consistency_k", self_consistency_k))
        effective_cache_dir = cache_dir
        if effective_cache_dir is None and cfg.get("cache_dir"):
            effective_cache_dir = Path(str(cfg.get("cache_dir")))
        if isinstance(effective_cache_dir, str):
            effective_cache_dir = Path(effective_cache_dir)
        result = rank_from_meta_json(
            user_query=user_query,
            meta_json_path=meta_json_path,
            batch_size=effective_batch_size,
            self_consistency_k=effective_self_k,
            cache_dir=effective_cache_dir,
            verbose=verbose,
            method_cfg=cfg,
        )
        self.state.last_result = result
        return result

