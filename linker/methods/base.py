from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class LinkMethod(ABC):
    @abstractmethod
    def prepare(self, docs: List[Dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def score_segment(self, segment: Dict[str, Any], top_k: int) -> List[Tuple[str, float, str]]:
        """
        Returns list of (doc_id, score, rationale) sorted by score desc.
        """
        ...


