from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple


_sent_split_re = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str, lower: bool = True, deaccent: bool = True, strip_punct: bool = True) -> str:
    t = text
    if deaccent:
        t = unicodedata.normalize("NFKD", t)
        t = "".join([c for c in t if not unicodedata.combining(c)])
    if lower:
        t = t.lower()
    if strip_punct:
        t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_sentences(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    # Simple deterministic sentence splitter returning spans
    spans: List[Tuple[str, Tuple[int, int]]] = []
    start = 0
    for m in _sent_split_re.finditer(text):
        end = m.start()
        seg = text[start:end]
        if seg.strip():
            spans.append((seg, (start, end)))
        start = m.end()
    if start < len(text):
        seg = text[start:]
        if seg.strip():
            spans.append((seg, (start, len(text))))
    return spans


