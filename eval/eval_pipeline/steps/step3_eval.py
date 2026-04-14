"""
Step 3 evaluator (Initial Object Filtering):
- Inputs: data/gt/relevance.json, data/preds/filtering_pred.json
- Artifacts: step3.confusion.json
- Metrics: Retained-Relevant (recall), Irrelevant-Removal (specificity), FNR, FPR, F1_relevant, Noise-Reduction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base_step import StepEvaluator
from ..stores.io import write_json_safe


class Step3Evaluator(StepEvaluator):
    step_id = 3
    metrics = []

    def _ensure_dependencies(self) -> None:
        # No hard deps besides inputs; artifacts enforced inside metrics
        return

    def _prepare(self) -> None:
        # Placeholder: in a real implementation, generate step3 specific artifacts
        return

    def _report_method_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("step3_method")

    def _report_model_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("step3_model")

    def _report_data_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("data")



