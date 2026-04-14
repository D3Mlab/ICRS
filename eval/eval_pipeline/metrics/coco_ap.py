from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import EvaluationMetric, MetricContext, MetricError
from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore

class CocoAPMetric(EvaluationMetric):
    def __init__(self) -> None:
        super().__init__()
        self.slug = "COCO_AP"
        self.description = "COCO-style AP metrics (AP, AP50, AP75)"
        self.default_config = {
            "gt": "data/gt/coco_instances.json",
            "pred": "data/preds/segments_pred.json",
            "iou_type": "segm",
        }
        self.COCO = COCO

    def load_data(self, gt_path: Path, pred_path: Path) -> Dict[str, Any]:
        try:
            self.gt_path = gt_path
            self.pred_path = pred_path
            if not self.gt_path.exists() or not self.pred_path.exists():
                raise MetricError(f"Missing inputs: {self.gt_path} or {self.pred_path}")
            self.coco_gt = self.COCO(str(self.gt_path))
            self.coco_dt = self.coco_gt.loadRes(str(self.pred_path))
        except Exception as exc:
            raise MetricError(f"Failed to load data: {exc}") from exc

    def _compute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, iouType=config.get("iou_type", "segm"))
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()
        return {
            "AP": float(self.coco_eval.stats[0]),
            "AP50": float(self.coco_eval.stats[1]),
            "AP75": float(self.coco_eval.stats[2]),
        }

    def eval(self, ctx: MetricContext, config: Dict[str, Any], gt_path: Path = None, pred_path: Path = None, iou_type: str = "segm") -> Dict[str, Any]:
        """Evaluate COCO metrics for a given ground truth and prediction paths."""
        if gt_path is None or pred_path is None:
            print('[Eval] Using default paths for COCO metrics')
            self.load_data(ctx.data_root / config["gt"], ctx.data_root / config["pred"])
        else:
            self.load_data(gt_path, pred_path)

        return self._compute(config)