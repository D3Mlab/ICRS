from __future__ import annotations

from typing import Optional, Tuple


def snippet_id_for_text(doc_id: str, field: str, span: Tuple[int, int]) -> str:
    return f"{doc_id}#{field}#{span[0]}_{span[1]}"


def snippet_id_for_kv(doc_id: str, field: str, key: str) -> str:
    return f"{doc_id}#{field}#{key}"


