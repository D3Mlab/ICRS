"""
StepEvaluator (Template Method):
- evaluate():
  1) _ensure_dependencies()
  2) _prepare()  # produce artifacts for this step only
  3) _compute_metrics()
  4) _write_reports()
  5) _commit_artifacts()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type, Union

from ..metrics.base import EvaluationMetric, MetricContext, MetricError
from ..stores.artifacts import ArtifactStore
from ..stores.io import write_json_safe
from ..stores.reporters import write_step_csv
from ..types import EvalConfig, MetricResult


MetricSpec = Union[Type[EvaluationMetric], Tuple[str, Type[EvaluationMetric]]]


class StepEvaluator(ABC):
    step_id: int
    metrics: List[MetricSpec] = []

    def __init__(self, cfg: EvalConfig, store: ArtifactStore) -> None:
        self.cfg = cfg
        self.store = store

    def evaluate(self) -> List[MetricResult]:
        self._ensure_dependencies()
        self._prepare()
        results = self._compute_metrics()
        self._write_reports(results)
        self._commit_artifacts()
        return results

    @abstractmethod
    def _ensure_dependencies(self) -> None:
        ...

    @abstractmethod
    def _prepare(self) -> None:
        ...

    def _compute_metrics(self) -> List[MetricResult]:
        results: List[MetricResult] = []
        for metric_spec in self.metrics:
            cfg_key: Optional[str] = None
            if isinstance(metric_spec, tuple):
                cfg_key, metric_cls = metric_spec
            else:
                metric_cls = metric_spec
            metric = metric_cls()
            base_cfg: dict = {}
            if isinstance(self.cfg.metrics_cfg, dict):
                for key, value in self.cfg.metrics_cfg.items():
                    target_key = cfg_key or metric.slug
                    if key == target_key:
                        continue
                    if key == "__global__" and isinstance(value, dict):
                        base_cfg.update(value)
                    elif not isinstance(value, dict):
                        base_cfg[key] = value
            per_metric_key = cfg_key or metric.slug
            per_metric = self.cfg.metrics_cfg.get(per_metric_key, {}) if isinstance(self.cfg.metrics_cfg, dict) else {}
            if isinstance(per_metric, dict):
                base_cfg.update(per_metric)
            context = MetricContext(cfg=base_cfg, data_root=self.cfg.data_root, artifacts=self.store)
            try:
                results.append(metric.compute(context))
            except MetricError as exc:
                results.append(MetricResult(name=metric.slug, values={"error": str(exc)}))
        return results

    def _write_reports(self, results: List[MetricResult]) -> None:
        step_csv = self.cfg.reports_dir / f"step{self.step_id}_report.csv"
        method_name = self._report_method_name()
        rows = [{"name": r.name, "values": r.values} for r in results]
        model_name = self._report_model_name()
        data_name = self._report_data_name()
        print(f"[Eval] Writing step {self.step_id} report to {step_csv} with method {method_name}, model {model_name}, data {data_name}")
        write_step_csv(step_csv, rows, method_name=method_name, model_name=model_name, data_name=data_name)

        metadata = self._collect_metadata()
        if metadata:
            meta_path = self.cfg.reports_dir / f"step{self.step_id}_metadata.json"
            write_json_safe(meta_path, metadata)

    def _commit_artifacts(self) -> None:
        self.store.save()

    def _report_method_name(self) -> str | None:
        return None

    def _report_model_name(self) -> str | None:
        return None

    def _report_data_name(self) -> str | None:
        return None

    def _collect_metadata(self) -> Optional[Dict[str, str]]:
        key_prefix = f"step{self.step_id}_"
        meta: Dict[str, str] = {}
        cfg = self.cfg.metrics_cfg if isinstance(self.cfg.metrics_cfg, dict) else {}
        for key, value in cfg.items():
            if isinstance(key, str) and key.startswith(key_prefix) and isinstance(value, str) and value:
                meta_key = key[len(key_prefix) :]
                meta[meta_key] = value
        return meta or None

