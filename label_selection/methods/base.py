from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MethodResult:
    scores: List[Tuple[str, float]]  # list of (snippet_id, score)


class RelevanceMethod(ABC):
    @abstractmethod
    def fit(self, snippet_texts: Dict[str, str]) -> None:
        ...

    @abstractmethod
    def score(self, query_text: str, topk: int) -> MethodResult:
        ...


