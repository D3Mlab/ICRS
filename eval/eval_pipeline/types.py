"""
Dataclasses and small types for evaluation pipeline.

Contracts:
- EvalConfig: per-run configuration including paths for artifacts and reports, data roots, plugin settings.
- EvalResult: step-level summary of metrics written to CSV and scorecard.
- MetricResult: name + flat dict of computed values for one metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class EvalConfig:
    data_root: Path
    artifacts_dir: Path
    reports_dir: Path
    manifest_path: Path
    plugins_dir: Optional[Path] = None
    extra_metrics: List[str] = field(default_factory=list)
    metrics_cfg: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    values: Dict[str, Any]


@dataclass
class EvalResult:
    step: int
    metric_results: List[MetricResult] = field(default_factory=list)


