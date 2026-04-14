
from __future__ import annotations

from typing import Any, Dict

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import safe_ratio

class ABUpliftMetric(EvaluationMetric):
    slug = "uplift"
    description = "Conversion uplift from A/B experiments"
    default_config = {
        "control_rate": None,
        "treatment_rate": None,
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        control = config.get("control_rate")
        treatment = config.get("treatment_rate")
        if control is None or treatment is None:
            raise MetricError("control_rate and treatment_rate must be provided in metrics_cfg")
        control = float(control)
        treatment = float(treatment)
        uplift = safe_ratio(treatment - control, control) * 100
        return {"uplift_percent": uplift, "control": control, "treatment": treatment}




