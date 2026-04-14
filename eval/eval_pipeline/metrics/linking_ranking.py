
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import safe_ratio
from .utils import load_link_data as _load_link_data
from .helpers import ndcg_at_k as _ndcg_at_k, map_at_k as _map_at_k


def _ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel <= 0:
            continue
        dcg += (2 ** rel - 1) / math.log2(idx + 1)
    ideal = sorted(relevances, reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal[:k], start=1):
        if rel <= 0:
            continue
        idcg += (2 ** rel - 1) / math.log2(idx + 1)
    return safe_ratio(dcg, idcg)


def _map_at_k(relevances: Sequence[int], k: int) -> float:
    hits = 0
    ap = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            hits += 1
            ap += hits / idx
    return safe_ratio(ap, hits) if hits > 0 else 0.0

def _hit_at_k(relevances: Sequence[int], k: int) -> float:
    hits = 0
    for rel in relevances[:k]:
        if rel > 0:
            hits += 1
    return hits / len(relevances)


class LinkingRankingMetric(EvaluationMetric):
    slug = "linking_ranking"
    description = "MAP/nDCG for ranked catalog matches per segment"
    default_config = {
        "dataset": "vogue",
        "predictions": "data/{dataset}/links/segment_links.json",
        "ground_truth": "data/{dataset}/meta_data/ground_truth.json",
        "k_values": [5, 10, 30],
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        _, per_segment, _, segment_truth, _ = _load_link_data(base=ctx.data_root, cfg=config)

        k_values = config.get("k_values", [1, 3, 5, 10])
        if not isinstance(k_values, (list, tuple)) or not k_values:
            raise MetricError("k_values must be a non-empty list")
        k_values = sorted({int(k) for k in k_values if int(k) > 0})

        agg_map: Dict[int, List[float]] = {k: [] for k in k_values}
        agg_ndcg: Dict[int, List[float]] = {k: [] for k in k_values}
        agg_hit: Dict[int, List[float]] = {k: [] for k in k_values}
        evaluated_segments = 0

        for seg_id, ranked_docs in per_segment.items():
            relevant_docs = set(segment_truth.get(f'{seg_id}.png', []))
            if not relevant_docs:
                continue
            relevances = [1 if doc_id in relevant_docs else 0 for doc_id, _ in ranked_docs]
            if not relevances:
                continue
            evaluated_segments += 1
            for k in k_values:
                agg_map[k].append(_map_at_k(relevances, k))
                agg_ndcg[k].append(_ndcg_at_k(relevances, k))
                agg_hit[k].append(_hit_at_k(relevances, k))
        if evaluated_segments == 0:
            raise MetricError("No overlapping segment rankings with ground truth")

        result: Dict[str, Any] = {"evaluated_segments": evaluated_segments}
        for k in k_values:
            result[f"MAP@{k}"] = sum(agg_map[k]) / len(agg_map[k]) if agg_map[k] else 0.0
            result[f"nDCG@{k}"] = sum(agg_ndcg[k]) / len(agg_ndcg[k]) if agg_ndcg[k] else 0.0
            result[f"Hit@{k}"] = sum(agg_hit[k]) / len(agg_hit[k]) if agg_hit[k] else 0.0
        return result

    def load_data(self, gt_path: Path, pred_path: Path) -> None:  # type: ignore[override]
        self._override_paths = {"ground_truth": str(gt_path), "predictions": str(pred_path)}

    def eval(self, ctx: MetricContext, config: Dict[str, Any], gt_path: Path = None, pred_path: Path = None, iou_type: str = "segm") -> Dict[str, Any]:  # type: ignore[override]
        cfg = dict(self.default_config)
        cfg.update(config or {})
        if gt_path is not None:
            cfg["ground_truth"] = str(gt_path)
        if pred_path is not None:
            cfg["predictions"] = str(pred_path)
        return self._compute(ctx, cfg)


