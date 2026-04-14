from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_CONTROL_CHARS = re.compile(r"[\u0000-\u001F]")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify_value(v) for v in value if v is not None)
    return normalize_text(str(value))


def flatten_metadata(metadata: Any, prefix: str = "") -> List[Tuple[str, str]]:
    fields: List[Tuple[str, str]] = []
    if isinstance(metadata, dict):
        for key in sorted(metadata.keys()):
            value = metadata[key]
            name = f"{prefix}.{key}" if prefix else str(key)
            fields.extend(flatten_metadata(value, name))
    elif isinstance(metadata, list):
        if not metadata:
            return []
        # If list is simple scalars, capture as one field
        if all(not isinstance(item, (dict, list)) for item in metadata):
            fields.append((prefix or "value", _stringify_value(metadata)))
        else:
            for idx, item in enumerate(metadata):
                name = f"{prefix}[{idx}]" if prefix else f"item[{idx}]"
                fields.extend(flatten_metadata(item, name))
    else:
        fields.append((prefix or "value", _stringify_value(metadata)))
    return fields


def build_segment_entry(segment: Dict[str, Any]) -> Dict[str, Any]:
    segment_id = str(segment.get("segment_id") or segment.get("id") or segment.get("doc_id") or "")
    doc_id = str(segment.get("doc_id") or "") or None
    metadata = segment.get("metadata") or {}

    fields = flatten_metadata(metadata)
    if doc_id:
        fields.append(("doc_id", doc_id))
    text = "; ".join([f"{name}: {value}" for name, value in fields if value])

    return {
        "id": segment_id,
        "doc_id": doc_id,
        "text": text,
        "fields": [{"name": name, "value": value} for name, value in fields if value],
        "metadata": metadata,
    }


def stringify_field_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key in sorted(value.keys()):
            child = stringify_field_value(value[key])
            if child:
                parts.append(f"{key}: {child}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = [stringify_field_value(v) for v in value if v is not None]
        return "; ".join(part for part in parts if part)
    if isinstance(value, (int, float)):
        return str(value)
    return normalize_text(str(value))


def top_level_metadata_fields(metadata: Any) -> List[Tuple[str, str]]:
    if not isinstance(metadata, dict):
        return []
    entries: List[Tuple[str, str]] = []
    for key in sorted(metadata.keys()):
        value_text = stringify_field_value(metadata.get(key))
        if value_text:
            entries.append((str(key), value_text))
    return entries

