from __future__ import annotations

import json
from typing import List, Dict, Any


def llm_consistency_stub(segment_desc: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Local stub: mark contradictions if obvious token conflicts, and supply brief rationales.
    Real LLM integration can replace this behind a flag.
    """
    verdicts = []
    seg = segment_desc.lower()
    for c in candidates:
        doc_id = c["doc_id"]
        text = (c.get("text") or "").lower()
        attrs = c.get("attrs") or {}
        contradict = False
        rationale = ""
        # Heuristics: gender mismatch
        seg_is_men = "men" in seg or "men's" in seg
        seg_is_women = "women" in seg or "women's" in seg
        attr_gender = str(attrs.get("gender", "")).lower()
        if seg_is_men and attr_gender and attr_gender not in ("men", "male"):
            contradict = True
            rationale = "gender mismatch"
        if seg_is_women and attr_gender and attr_gender not in ("women", "female"):
            contradict = True
            rationale = "gender mismatch"
        verdicts.append({"doc_id": doc_id, "contradict": contradict, "rationale": rationale or "ok"})
    return verdicts


