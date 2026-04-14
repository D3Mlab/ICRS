from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .base import EvaluationMetric, MetricContext
from .utils import precision_recall_f1
from .utils import load_caption_pairs_map as _load_caption_pairs


class AttributeExtractionMetric(EvaluationMetric):
    slug = "attribute_f1"
    description = "Attribute extraction precision/recall/F1 over schema keys"
    default_config = {
        "references": "data/gt/descriptions.json",
        "predictions": "data/preds/descriptions_pred.json",
        "schema_keys": ["color", "brand", "material", "pattern"],
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        gt_map, pred_map = _load_caption_pairs(ctx, config)
        keys = config.get("schema_keys") or []

        tp = 0
        fp = 0
        fn = 0
        for seg_id, gt_entry in gt_map.items():
            gt_attrs = gt_entry.get("attributes") or {}
            pred_attrs = (pred_map.get(seg_id, {}) or {}).get("attributes", {})
            if not isinstance(gt_attrs, dict):
                gt_attrs = {}
            if not isinstance(pred_attrs, dict):
                pred_attrs = {}

            for key in keys or gt_attrs.keys():
                gt_val = gt_attrs.get(key)
                pred_val = pred_attrs.get(key)
                if pred_val is None:
                    if gt_val is not None:
                        fn += 1
                    continue
                if gt_val is None:
                    fp += 1
                    continue
                if str(gt_val).strip().lower() == str(pred_val).strip().lower():
                    tp += 1
                else:
                    fp += 1

            # Count missing GT values predicted elsewhere
            for key in keys or gt_attrs.keys():
                if key not in pred_attrs and key in gt_attrs and gt_attrs[key] is not None:
                    fn += 1

        stats = precision_recall_f1(tp, fp, fn)
        stats.update({"tp": tp, "fp": fp, "fn": fn})
        return stats

    def load_data(self, gt_path: Path, pred_path: Path) -> None:  # type: ignore[override]
        self._override_paths = {"references": str(gt_path), "predictions": str(pred_path)}

    def eval(self, ctx: MetricContext, config: Dict[str, Any], gt_path: Path = None, pred_path: Path = None, iou_type: str = "segm") -> Dict[str, Any]:  # type: ignore[override]
        cfg = dict(self.default_config)
        cfg.update(config or {})
        if gt_path is not None:
            cfg["references"] = str(gt_path)
        if pred_path is not None:
            cfg["predictions"] = str(pred_path)
        return self._compute(ctx, cfg)




