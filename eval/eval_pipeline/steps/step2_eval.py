"""
Step 2 evaluator (Object Description):
- Inputs: data/gt/regions.json, data/preds/descriptions_pred.json
- Artifacts: step2.caption_scores.json, step2.listener_cache.npz, step2.attr_compare.json
- Metrics: CIDEr, BLEU-4, METEOR, ROUGE-L, SPICE; attribute P/R/F1; listener@1
"""

from __future__ import annotations

from pathlib import Path

from .base_step import StepEvaluator
from ..stores.io import write_json_safe
from ..metrics.caption_quality import CaptionQualityMetric
from ..metrics.attribute_extraction import AttributeExtractionMetric
from ..metrics.listener_at_one import ListenerAtOneMetric


class Step2Evaluator(StepEvaluator):
    step_id = 2
    metrics = [CaptionQualityMetric, AttributeExtractionMetric, ListenerAtOneMetric]

    def _ensure_dependencies(self) -> None:
        return

    def _prepare(self) -> None:
        cap = self.cfg.artifacts_dir / "step2" / "step2.caption_scores.json"
        lst = self.cfg.artifacts_dir / "step2" / "step2.listener_cache.npz"
        attr = self.cfg.artifacts_dir / "step2" / "step2.attr_compare.json"
        write_json_safe(cap, {"placeholder": True})
        write_json_safe(attr, {"placeholder": True})
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_bytes(b"")
        self.store.put("step2.caption_scores", cap)
        self.store.put("step2.listener_cache", lst)
        self.store.put("step2.attr_compare", attr)

    def _report_method_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("describer_method")

    def _report_model_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("describer_model")

    def _report_data_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("data")



