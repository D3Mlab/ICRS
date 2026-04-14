from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SnippetRecord:
    snippet_id: str
    doc_id: str
    field: str
    text: str

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def validate_doc(obj: Dict[str, Any]) -> None:
    if "doc_id" not in obj:
        raise ValueError("catalog doc missing doc_id")


