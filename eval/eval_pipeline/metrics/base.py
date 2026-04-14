from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from abc import ABC, abstractmethod
from ..stores.artifacts import ArtifactStore
from ..types import MetricResult


class MetricError(Exception):
    """Raised when a metric cannot be computed because of missing data or configuration."""


@dataclass
class MetricContext:
    cfg: Dict[str, Any]
    data_root: Path
    artifacts: ArtifactStore


class EvaluationMetric(ABC):
    """Base class for all evaluation metrics."""

    def __init__(self) -> None:
        self.slug: str = "metric"
        self.description: str = ""
        self.required_artifacts: tuple[str, ...] = ()
        self.default_config: Dict[str, Any] = {}

    def compute(self, ctx: MetricContext) -> MetricResult:
        if not isinstance(ctx.data_root, Path):
            raise MetricError("MetricContext.data_root must be a pathlib.Path instance")

        # Ensure required artifacts are present
        for key in self.required_artifacts:
            try:
                ctx.artifacts.require([key])
            except Exception as exc:  # pragma: no cover - defensive
                raise MetricError(str(exc)) from exc

        # Merge default config with provided overrides
        config = dict(self.default_config)
        config.update(ctx.cfg or {})

        # Delegate to eval(), which may call load_data() and _compute()
        try:
            values = self.eval(ctx, config)
        except MetricError:
            raise
        except Exception as exc:  # pragma: no cover - metric specific errors
            raise MetricError(str(exc)) from exc

        return MetricResult(name=self.slug, values=values)
    
    # Optional: subclasses can override to load files from explicit paths
    def load_data(self, gt_path: Path, pred_path: Path) -> None:
        return None

    @abstractmethod
    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    # Default eval simply computes with the merged config; subclasses may override
    def eval(self, ctx: MetricContext, config: Dict[str, Any], gt_path: Path = None, pred_path: Path = None, iou_type: str = "segm") -> Dict[str, Any]:
        return self._compute(ctx, config)


