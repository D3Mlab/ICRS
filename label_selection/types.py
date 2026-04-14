from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Query:
    query_id: str
    text: str
    context: Dict[str, object] = field(default_factory=dict)
    candidate_segment_ids: Optional[List[str]] = None


@dataclass
class Snippet:
    snippet_id: str
    doc_id: str
    field: str
    text: str

@dataclass
class RankedSnippet:
    rank: int
    snippet_id: str
    doc_id: str
    segment_ids: List[str]
    score: float
    text: str
    content: str
    evidence: Dict[str, object]


