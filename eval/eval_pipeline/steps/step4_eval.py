"""
Step 4 evaluator (Linking to Catalog & Metadata):
- Inputs: data/gt/catalog.json, data/gt/links_gt.json (optional), data/preds/segment_links.json
- Artifacts: step4.topk_candidates.json, step4.attr_prf.json, step4.status.json
- Metrics: Attribute precision/recall/F1, coverage, hallucination rate
"""

from __future__ import annotations

from .base_step import StepEvaluator
from ..stores.io import write_json_safe
from ..metrics.linking_coverage import LinkingCoverageMetric
from ..metrics.linking_prf import LinkingPRFMetric
from ..metrics.linking_ranking import LinkingRankingMetric


class Step4Evaluator(StepEvaluator):
    step_id = 4
    metrics = [LinkingPRFMetric, LinkingCoverageMetric, LinkingRankingMetric]

    def _ensure_dependencies(self) -> None:
        return

    def _prepare(self) -> None:
        topk = self.cfg.artifacts_dir / "step4" / "step4.topk_candidates.json"
        attr = self.cfg.artifacts_dir / "step4" / "step4.attr_prf.json"
        status = self.cfg.artifacts_dir / "step4" / "step4.status.json"
        write_json_safe(topk, {"placeholder": True})
        write_json_safe(attr, {"placeholder": True})
        write_json_safe(status, {"placeholder": True})
        self.store.put("step4.topk_candidates", topk)
        self.store.put("step4.attr_prf", attr)
        self.store.put("step4.status", status)

    def _report_method_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("linker_method")

    def _report_model_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("linker_model")

    def _report_data_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("data")



