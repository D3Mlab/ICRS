from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class QueryReformatter:
    """
    Reformat a queries JSON into a single string for snippet ranking.
    Current strategy: take the first query, append simple context if present.
    """

    @staticmethod
    def reformat_file_to_text(path: Path) -> str:
        try:
            obj: Dict[str, Any] = json.loads(path.read_text())
        except Exception:
            return ""
        queries: List[Dict[str, Any]] = obj.get("queries", []) if isinstance(obj, dict) else []
        if not queries:
            return ""
        q0 = queries[0]
        parts: List[str] = []
        text = str(q0.get("text", "")).strip()
        if text:
            parts.append(text)
        ctx = q0.get("context") or {}
        if isinstance(ctx, dict):
            constraints = ctx.get("constraints")
            if isinstance(constraints, list) and constraints:
                parts.append("Constraints: " + ", ".join(map(str, constraints)))
        return " | ".join(parts).strip()


