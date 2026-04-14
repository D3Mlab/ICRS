from __future__ import annotations

from typing import Any, Dict
from .base import EvaluationMetric, MetricContext, MetricError
from .utils import load_json

class MeanIoUMetric(EvaluationMetric):
    slug = "mean_iou"
    description = "Mean IoU aggregated from per-instance IoU reports"
    required_artifacts = ("step1.miou_per_instance",)

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        path = ctx.artifacts.get_path("step1.miou_per_instance")
        data = load_json(path)
        per_instance = data.get("per_instance") or data
        if not isinstance(per_instance, dict) or not per_instance:
            raise MetricError("Invalid or empty mIoU artifact")

        values = list(per_instance.values())
        try:
            values = [float(v) for v in values]
        except Exception as exc:
            raise MetricError(f"Invalid mIoU values: {exc}") from exc

        mean_iou = sum(values) / len(values)
        per_class: Dict[str, float] = {}
        if "per_class" in data and isinstance(data["per_class"], dict):
            for cls, cls_vals in data["per_class"].items():
                try:
                    per_class[cls] = float(sum(cls_vals) / len(cls_vals))
                except Exception:
                    continue

        result = {"mIoU": mean_iou}
        for cls, val in per_class.items():
            result[f"mIoU_{cls}"] = val
        return result


