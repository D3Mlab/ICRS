from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Set

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import (
    safe_ratio,
    load_object_ranking_scores,
    load_object_ranking_truth,
)

class NoiseReductionMetric(EvaluationMetric):
    slug = "noise_reduction"
    description = "Drop rate of irrelevant items without harming recall"
    default_config = {
        "dataset": "vogue",
        "object_ranker_method": "FUSION_LLM",
        "predictions": "",
        "ground_truth": "data/{dataset}/ground_truth_object_ranking.json",
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        pred_map = load_object_ranking_scores(ctx.data_root, config)
        gt_map = load_object_ranking_truth(ctx.data_root, config)
        if not gt_map:
            raise MetricError("Ground truth rankings are empty or missing")

        total_dropped = 0
        total_irrelevant = 0
        recall_hits = 0
        recall_total = 0
        target_query_id = config.get("query_id")
        if target_query_id is not None:
            target_query_id = str(target_query_id)
        evaluated_any = False

        for query_id, gt_scores in gt_map.items():
            if target_query_id and str(query_id) != target_query_id:
                continue
            positives: Set[str] = {doc for doc, score in gt_scores.items() if score > 0}
            negatives: Set[str] = {doc for doc, score in gt_scores.items() if score <= 0}
            if not negatives:
                continue
            score_map = pred_map.get(query_id, {})
            pred_ids: Set[str] = {str(doc_id) for doc_id in score_map.keys()}

            total_irrelevant += len(negatives)
            total_dropped += len(negatives - pred_ids)

            if positives:
                recall_hits += len(pred_ids & positives)
                recall_total += len(positives)
            evaluated_any = True

        if not evaluated_any:
            if target_query_id:
                raise MetricError(f"No rankings found for query_id '{target_query_id}'.")
            raise MetricError("Ground truth relevance list is empty")
        if total_irrelevant == 0:
            raise MetricError("Ground truth relevance list is empty")
        drop_rate = safe_ratio(total_dropped, total_irrelevant)
        recall = safe_ratio(recall_hits, recall_total)
        result = {"drop_rate": drop_rate, "recall_preserved": recall}
        if target_query_id is not None:
            result["query_id"] = target_query_id
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

