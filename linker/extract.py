from __future__ import annotations

import re
from typing import Dict, Any


_brand_re = re.compile(r"\b(\w+?)\b(?:\s+(?:brand|co\.|inc\.|ltd\.))?", re.IGNORECASE)
_material_re = re.compile(r"\b(mesh|leather|suede|cotton|poly\w*|nylon|wool)\b", re.IGNORECASE)
_gender_re = re.compile(r"\b(men|women|male|female|unisex)\b", re.IGNORECASE)
_color_re = re.compile(r"\b(black|white|red|green|blue|gray|grey|brown|beige|yellow|orange|pink|purple)\b", re.IGNORECASE)


def rule_extract(doc_text: str) -> Dict[str, Any]:
    text = doc_text or ""
    out: Dict[str, Any] = {
        "brand": None,
        "category": None,
        "subtype": None,
        "material": None,
        "terrain": None,
        "gender": None,
        "color": None,
        "size": None,
    }
    m = _material_re.search(text)
    if m:
        out["material"] = m.group(1).lower()
    g = _gender_re.search(text)
    if g:
        val = g.group(1).lower()
        out["gender"] = {"men": "men", "male": "men", "women": "women", "female": "women", "unisex": "unisex"}.get(val, val)
    c = _color_re.search(text)
    if c:
        out["color"] = c.group(1).lower()
    return out


