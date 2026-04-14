from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .base import LinkMethod
from label_selection.utils.text import normalize


def flatten_text_fields(doc: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    for f in fields:
        val = doc.get(f)
        if isinstance(val, str) and val.strip():
            parts.append(f"[{f}] {val}")
        elif isinstance(val, list):
            joined = " ".join([v if isinstance(v, str) else str(v) for v in val])
            if joined:
                parts.append(f"[{f}] {joined}")
        elif isinstance(val, dict):
            for k, v in val.items():
                parts.append(f"[{f}] {k}: {v}")
    return " \n ".join(parts)


class BM25Linker(LinkMethod):
    def __init__(self, fields: List[str], k1: float = 1.5, b: float = 0.75):
        self.fields = fields
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def prepare(self, docs: List[Dict[str, Any]]) -> None:
        self.doc_ids = []
        corpus: List[List[str]] = []
        for d in docs:
            doc_id = str(d.get("doc_id"))
            text = normalize(flatten_text_fields(d, self.fields))
            tokens = [tok for tok in text.split() if tok]
            if not tokens:
                continue
            self.doc_ids.append(doc_id)
            corpus.append(tokens)
        self.corpus_tokens = corpus
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        else:
            self.bm25 = None

    def score_segment(self, segment: Dict[str, Any], top_k: int) -> List[Tuple[str, float, str]]:
        if self.bm25 is None or not self.doc_ids:
            return []
        q = normalize(segment.get("description", ""))
        if not q:
            return []
        scores = self.bm25.get_scores(q.split())
        pairs = list(zip(self.doc_ids, map(float, scores)))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return [(doc_id, score, "bm25") for doc_id, score in pairs[:top_k]]


