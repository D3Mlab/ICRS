"""
Abstract Metric interface (Strategy pattern).

Implementations must declare:
- name: unique string
- step: 1..4
- requires: set of artifact keys that must exist in the ArtifactStore manifest

compute(artifacts, cfg) -> MetricResult-like dict { name: str, values: {k: v} }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Set


class BaseMetric(ABC):
    name: str
    step: int
    requires: Set[str]

    @abstractmethod
    def compute(self, artifacts: Dict[str, str], cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metric.

        artifacts: mapping from artifact key to absolute/relative path (string)
        cfg: per-metric configuration dict
        Returns: { 'name': <metric_name>, 'values': { 'metric_key': value, ... } }
        """
        raise NotImplementedError



