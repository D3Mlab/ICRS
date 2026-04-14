"""
Step 5 evaluator (Object Ranking):
- Inputs: object ranking predictions and ground truth relevance judgments
- Metrics: Precision/Recall/MAP/nDCG, filter accuracy, noise reduction
"""

from __future__ import annotations

from pathlib import Path

from .base_step import StepEvaluator
from ..stores.io import write_json_safe
from ..stores.reporters import write_step_csv
from ..metrics.ranking_quality import RankingQualityMetric
from ..metrics.filter_inference import FilterInferenceMetric
from ..metrics.noise_reduction import NoiseReductionMetric


class Step5Evaluator(StepEvaluator):
    step_id = 5
    metrics = [RankingQualityMetric]

    def _ensure_dependencies(self) -> None:
        # Requires Step 4 artifacts (segment_links.json, segment_metadata.json)
        return

    def _prepare(self) -> None:
        report_dir = self.cfg.artifacts_dir / "step5"
        report_dir.mkdir(parents=True, exist_ok=True)
        placeholder = report_dir / "step5.summary.json"
        write_json_safe(placeholder, {"placeholder": True})
        self.store.put("step5.summary", placeholder)

        rankings_path = report_dir / "step5.object_rankings.json"
        write_json_safe(rankings_path, {"placeholder": True})
        self.store.put("step5.object_rankings", rankings_path)

    def _report_method_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("object_ranker_method")

    def _report_model_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("object_ranker_model")

    def _report_data_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("data")

    def _write_reports(self, results):
        query_id = self._resolve_query_id()
        if query_id:
            for res in results:
                if isinstance(res.values, dict) and not res.values.get("query_id"):
                    res.values["query_id"] = query_id
        step_csv = self.cfg.reports_dir / f"step{self.step_id}_report.csv"
        method_name = self._report_method_name()
        rows = [{"name": r.name, "values": r.values} for r in results]
        model_name = self._report_model_name()
        data_name = self._report_data_name()
        write_step_csv(step_csv, rows, method_name=method_name, model_name=model_name, data_name=data_name)

        metadata = self._collect_metadata()
        if metadata:
            meta_path = self.cfg.reports_dir / f"step{self.step_id}_metadata.json"
            write_json_safe(meta_path, metadata)

    def _resolve_query_id(self) -> str | None:
        cfg = self.cfg.metrics_cfg if isinstance(self.cfg.metrics_cfg, dict) else {}
        value = cfg.get("query_id")
        if value:
            return str(value)
        metric_cfg = cfg.get("step5_ranking_quality")
        if isinstance(metric_cfg, dict):
            metric_value = metric_cfg.get("query_id")
            if metric_value:
                return str(metric_value)
        return None

