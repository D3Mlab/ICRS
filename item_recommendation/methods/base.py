from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BatchResult:
    """Normalized output for a single method batch invocation."""

    scores: Dict[str, float]
    parsed_items: List[Dict[str, Any]] = field(default_factory=list)
    field_scores: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[str] = None
    temperature: Optional[float] = None
    prompt: Optional[str] = None


class ObjectRankingMethod(ABC):
    """Abstract interface for object relevance ranking methods."""

    def prepare(self, objects: List[Dict[str, Any]]) -> None:  # pragma: no cover - default no-op
        """Optional hook executed once before processing any batches."""

    def temperature_for_run(self, base_temperature: float, run_index: int) -> float:
        """Return the sampling temperature for a given run (can apply jitter)."""

        return base_temperature

    @abstractmethod
    def score_batch(
        self,
        user_query: str,
        batch: List[Dict[str, Any]],
        temperature: float,
        run_index: int,
    ) -> BatchResult:
        """Return relevance scores for the provided batch."""


