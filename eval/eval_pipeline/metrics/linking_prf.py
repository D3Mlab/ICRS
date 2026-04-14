from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

from .base import EvaluationMetric, MetricContext
from .utils import precision_recall_f1, safe_ratio
from .utils import load_link_data as _load_link_data


class LinkingPRFMetric(EvaluationMetric):
    slug = "linking_prf"
    description = "Precision / Recall / F1 for catalog linking"
    default_config = {
        "dataset": "vogue",
        "predictions": "data/{dataset}/links/segment_links.json",
        "ground_truth": "data/{dataset}/meta_data/ground_truth.json",
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        doc_best, _, gt_map, _, _ = _load_link_data(base=ctx.data_root, cfg=config)

        tp = fp = fn = 0
        for doc_id, (pred_seg, _) in doc_best.items():
            true_seg = gt_map.get(doc_id)
            if true_seg is None:
                fp += 1
            elif pred_seg == true_seg:
                tp += 1
            else:
                fp += 1

        for doc_id, true_seg in gt_map.items():
            pred = doc_best.get(doc_id)
            if pred is None:
                fn += 1
            elif pred[0] != true_seg:
                fn += 1

        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        accuracy = safe_ratio(tp, len(gt_map))

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_ground_truth": len(gt_map),
            "total_predictions": len(doc_best),
        }

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


