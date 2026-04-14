from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rapidfuzz import fuzz

from .base import LinkMethod
from label_selection.utils.text import normalize


def extract_keys(segment: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    desc = segment.get("description", "")
    if isinstance(desc, str):
        keys.append(desc)
    name = segment.get("name") or segment.get("title")
    if isinstance(name, str):
        keys.append(name)
    return keys


def flatten_doc(doc: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    for f in fields:
        val = doc.get(f)
        if isinstance(val, str) and val.strip():
            parts.append(val)
        elif isinstance(val, list):
            parts.append(" ".join([v if isinstance(v, str) else str(v) for v in val]))
    return " ".join(parts)


class TextExactFuzzy(LinkMethod):
    def __init__(self, fields: List[str]):
        self.fields = fields
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []

    def prepare(self, docs: List[Dict[str, Any]]) -> None:
        self.doc_ids = [str(d.get("doc_id")) for d in docs]
        self.doc_texts = [normalize(flatten_doc(d, self.fields)) for d in docs]

    def score_segment(self, segment: Dict[str, Any], top_k: int) -> List[Tuple[str, float, str]]:
        query = " ".join([normalize(x) for x in extract_keys(segment) if x])
        if not query:
            return []
        scores: List[Tuple[str, float, str]] = []
        for doc_id, text in zip(self.doc_ids, self.doc_texts):
            score = fuzz.token_set_ratio(query, text) / 100.0
            scores.append((doc_id, float(score), "fuzzy"))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]


