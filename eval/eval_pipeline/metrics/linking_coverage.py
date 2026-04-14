
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import EvaluationMetric, MetricContext
from .utils import safe_ratio
from .linking_prf import LinkingPRFMetric
from .utils import load_link_data as _load_link_data

class LinkingCoverageMetric(EvaluationMetric):
    slug = "linking_coverage"
    description = "Coverage of catalog documents with linked segments"
    default_config = LinkingPRFMetric.default_config

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        doc_best, _, gt_map, _, segments_with_predictions = _load_link_data(base=ctx.data_root, cfg=config)

        matched_docs = sum(1 for doc_id, (pred_seg, _) in doc_best.items() if gt_map.get(doc_id) == pred_seg)
        docs_with_predictions = sum(1 for doc_id in doc_best.keys() if doc_id in gt_map)

        coverage = safe_ratio(docs_with_predictions, len(gt_map))
        hit_rate = safe_ratio(matched_docs, len(gt_map))

        return {
            "doc_coverage": coverage,
            "correct_match_rate": hit_rate,
            "docs_with_predictions": docs_with_predictions,
            "matched_docs": matched_docs,
            "segments_with_predictions": segments_with_predictions,
            "total_docs": len(gt_map),
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



