from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import precision_recall_f1
from .utils import load_snippet_results as _load_snippet_results


class AspectCoverageMetric(EvaluationMetric):
    slug = "aspect_coverage"
    description = "Aspect coverage precision/recall"
    default_config = {
        "predictions": "reports/snippets.json",
        "ground_truth": "data/gt/snippet_relevance.json",
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        pred_map, gt_map = _load_snippet_results(ctx, config)
        tp = fp = fn = 0
        for key, pred_snippets in pred_map.items():
            gt_entry = gt_map.get(key, [])
            gt_aspects = {str(item.get("aspect")) for item in gt_entry if isinstance(item, dict) and item.get("aspect")}
            pred_aspects = {str(item.get("evidence", {}).get("field")) for item in pred_snippets if isinstance(item, dict)}
            tp += len(pred_aspects & gt_aspects)
            fp += len(pred_aspects - gt_aspects)
            fn += len(gt_aspects - pred_aspects)
        if tp + fp + fn == 0:
            raise MetricError("No aspect annotations found")
        stats = precision_recall_f1(tp, fp, fn)
        stats.update({"tp": tp, "fp": fp, "fn": fn})
        return stats

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



