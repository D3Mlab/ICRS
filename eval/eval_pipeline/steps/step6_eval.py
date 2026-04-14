"""
Step 6 evaluator (Metadata Snippet Ranking):
- Inputs: snippet ranking outputs and relevance judgments
- Metrics: P@K, MAP, nDCG, aspect coverage, faithfulness, diversity, uplift
"""

from __future__ import annotations

from .base_step import StepEvaluator
from ..stores.io import write_json_safe
from ..stores.reporters import write_step_csv
from ..metrics.snippet_ranking import SnippetBertScoreMetric


class Step6Evaluator(StepEvaluator):
    step_id = 6
    metrics = [
        ("snippet_ranking", SnippetBertScoreMetric),
    ]

    def _ensure_dependencies(self) -> None:
        # Requires Step 4 linking outputs and snippet ranking artifacts
        return

    def _prepare(self) -> None:
        report_dir = self.cfg.artifacts_dir / "step6"
        report_dir.mkdir(parents=True, exist_ok=True)
        placeholder = report_dir / "step6.summary.json"
        write_json_safe(placeholder, {"placeholder": True})
        self.store.put("step6.summary", placeholder)

        snippets_path = report_dir / "step6.snippets.json"
        write_json_safe(snippets_path, {"placeholder": True})
        self.store.put("step6.snippets", snippets_path)

    def _report_method_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("snippet_ranker_method")

    def _report_model_name(self) -> str | None:
        return self.cfg.metrics_cfg.get("snippet_ranker_model")

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
        metric_cfg = cfg.get("snippet_ranking")
        if isinstance(metric_cfg, dict):
            metric_value = metric_cfg.get("query_id")
            if metric_value:
                return str(metric_value)
        return None

