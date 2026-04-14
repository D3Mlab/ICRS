"""
Step 1 evaluator (Segmentation):
- Inputs: data/gt/coco_instances.json, data/preds/segments_pred.json
- Artifacts: step1.matches.json, step1.miou_per_instance.json
- Metrics: COCO AP@[.5:.95], AP50, AP75; mIoU macro over matched instances
"""

from __future__ import annotations

from pathlib import Path

from .base_step import StepEvaluator
from ..metrics.coco_ap import CocoAPMetric
from ..metrics.mean_iou import MeanIoUMetric


class Step1Evaluator(StepEvaluator):
    step_id = 1
    metrics = [CocoAPMetric, MeanIoUMetric]

    def _ensure_dependencies(self) -> None:
        return

    def _prepare(self) -> None:
        return

    def _report_method_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("segmenter_method")

    def _report_model_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("segmenter_model")

    def _report_data_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("data")



