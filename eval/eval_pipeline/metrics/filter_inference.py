from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import load_json, precision_recall_f1, safe_ratio


class FilterInferenceMetric(EvaluationMetric):
    slug = "step5_filter_accuracy"
    description = "Accuracy of inferred filters/attributes per query"
    default_config = {
        "inference": "data/preds/object_filters.json",
        "ground_truth": "data/gt/object_filters.json",
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        inf_path = (ctx.data_root / config.get("inference", "data/preds/object_filters.json")).resolve()
        gt_path = (ctx.data_root / config.get("ground_truth", "data/gt/object_filters.json")).resolve()
        if not inf_path.exists() or not gt_path.exists():
            raise MetricError("Filter files not found")
        inferred = load_json(inf_path)
        ground = load_json(gt_path)
        tp = fp = fn = 0
        for query_id, filters in inferred.items():
            pred_set = set(filters)
            gt_set = set(ground.get(query_id, []))
            tp += len(pred_set & gt_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)
        stats = precision_recall_f1(tp, fp, fn)
        stats.update({"tp": tp, "fp": fp, "fn": fn})
        return stats

    def load_data(self, gt_path: Path, pred_path: Path) -> None:  # type: ignore[override]
        self._override_paths = {"ground_truth": str(gt_path), "inference": str(pred_path)}

    def eval(self, ctx: MetricContext, config: Dict[str, Any], gt_path: Path = None, pred_path: Path = None, iou_type: str = "segm") -> Dict[str, Any]:  # type: ignore[override]
        cfg = dict(self.default_config)
        cfg.update(config or {})
        if gt_path is not None:
            cfg["ground_truth"] = str(gt_path)
        if pred_path is not None:
            cfg["inference"] = str(pred_path)
        return self._compute(ctx, cfg)



