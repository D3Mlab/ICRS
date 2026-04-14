from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dataclasses import asdict
from ..config import Step6Config
from ..utils.io import read_json, write_json, stable_hash
from ..utils.text import normalize, split_sentences
from ..utils.idgen import snippet_id_for_kv, snippet_id_for_text
from .schema import SnippetRecord, validate_doc


def _extract_text_fields(
    doc: Dict, 
    fields_include: List[str], 
    fields_exclude: List[str]
) -> List[Tuple[str, str]]:
    """
    Recursively extract all text fields from the document, excluding specified fields.
    Formats nested fields as 'field_name:content' or 'parent.child:content'.
    Splits content into sentences and returns (field_path, formatted_sentence) tuples
    where formatted_sentence is "field_name:sentence".
    """
    out: List[Tuple[str, str]] = []
    exclude_set = set(fields_exclude) if fields_exclude else set()
    
    def _extract_recursive(obj: any, field_path: str = "", depth: int = 0) -> None:
        """Recursively extract text from nested structures and split into sentences."""
        if depth > 10:  # Prevent infinite recursion
            return
        
        # Skip excluded fields by checking if current field path ends with excluded key
        if field_path:
            path_parts = field_path.split(".")
            if any(part in exclude_set for part in path_parts):
                return
        
        if isinstance(obj, str):
            # String content - split into sentences and format each as "field_name:sentence"
            if obj.strip():
                sents = split_sentences(obj)
                for sent, _ in sents:
                    if sent.strip():
                        # Format as field_name:content (or just content if no field_path)
                        formatted = f"{field_path}:{sent}" if field_path else sent
                        out.append((field_path if field_path else "text", formatted))
        elif isinstance(obj, dict):
            # Dictionary - recurse into each key-value pair
            for key, value in obj.items():
                # Skip excluded keys
                if key in exclude_set:
                    continue
                # Build field path
                new_path = f"{field_path}.{key}" if field_path else key
                _extract_recursive(value, new_path, depth + 1)
        elif isinstance(obj, list):
            # List - recurse into each item
            for item in obj:
                if isinstance(item, str):
                    # Direct string in list - split into sentences
                    if item.strip():
                        sents = split_sentences(item)
                        for sent, _ in sents:
                            if sent.strip():
                                formatted = f"{field_path}:{sent}" if field_path else sent
                                out.append((field_path if field_path else "text", formatted))
                elif isinstance(item, dict):
                    # Dictionary in list - recurse into it
                    _extract_recursive(item, field_path, depth + 1)
                else:
                    # Other types - convert to string (one sentence)
                    str_val = str(item).strip()
                    if str_val:
                        formatted = f"{field_path}:{str_val}" if field_path else str_val
                        out.append((field_path if field_path else "text", formatted))
        else:
            # Other types (numbers, bools, etc.) - convert to string (one sentence)
            str_val = str(obj).strip()
            if str_val:
                formatted = f"{field_path}:{str_val}" if field_path else str_val
                out.append((field_path if field_path else "text", formatted))
    
    # If fields_include is specified and not empty, only process those fields
    if fields_include:
        for f in fields_include:
            if f in exclude_set:
                continue
            val = doc.get(f)
            if val is not None:
                _extract_recursive(val, f, depth=0)
    else:
        # Process all fields recursively, skipping excluded top-level keys
        for key, value in doc.items():
            if key in exclude_set:
                continue
            _extract_recursive(value, key, depth=0)
    
    return out


def _extract_kv_fields(doc: Dict, fields_include: List[str]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for f in fields_include:
        if f in {"attributes", "specs"} and isinstance(doc.get(f), dict):
            for k, v in doc[f].items():
                out.append((f, str(k), str(v)))
    return out


def build_snippet_index(catalog_path: Path, artifacts_dir: Path, cfg: Step6Config) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    docs: List[Dict] = []
    if catalog_path.is_dir():
        for fp in sorted(catalog_path.glob("*.json")):
            try:
                obj = read_json(fp)
            except Exception:
                obj = {}
            if "doc_id" not in obj:
                obj["doc_id"] = fp.name  # use filename with extension for consistency
            docs.append(obj)
    else:
        docs_obj = read_json(catalog_path)
        docs = docs_obj.get("docs", docs_obj if isinstance(docs_obj, list) else [])

    records: List[SnippetRecord] = []
    for doc in docs:
        validate_doc(doc)
        doc_id = str(doc["doc_id"])
        # text fields → sentences / windows
        # raw_text is already formatted as "field_name:sentence" and split into sentences
        for field, formatted_text in _extract_text_fields(
            doc, 
            cfg.index.fields_include,
            getattr(cfg.index, "fields_exclude", [])
        ):
            # formatted_text is already "field_name:sentence" format
            # Normalize the formatted text
            norm = normalize(formatted_text, **cfg.index.normalize.__dict__)
            if not norm:
                continue
            # Generate snippet ID using field path and content hash
            content_hash = stable_hash(formatted_text)[:8]
            sid = snippet_id_for_text(doc_id, field, (0, len(formatted_text)))
            records.append(SnippetRecord(
                snippet_id=sid, doc_id=doc_id, field=field, text=norm))
        # kv pairs
        for field, key, value in _extract_kv_fields(doc, cfg.index.fields_include):
            txt = normalize(f"{key}: {value}", **cfg.index.normalize.__dict__)
            sid = snippet_id_for_kv(doc_id, field, key)
            records.append(SnippetRecord(
                snippet_id=sid, doc_id=doc_id, field=field, text=txt))

    # Cap per doc
    if cfg.limits.max_snippets_per_doc > 0:
        by_doc: Dict[str, List[SnippetRecord]] = {}
        for r in records:
            by_doc.setdefault(r.doc_id, []).append(r)
        trimmed: List[SnippetRecord] = []
        for doc_id, recs in by_doc.items():
            trimmed.extend(recs[: cfg.limits.max_snippets_per_doc])
        records = trimmed

    # Write JSONL store
    store_path = artifacts_dir / "snippet_store.jsonl"
    with store_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")

    # Compute a stable hash over the catalog source
    if catalog_path.is_dir():
        concat = "".join(
            [fp.name + "\n" + (fp.read_text() if fp.exists() else "") for fp in sorted(catalog_path.glob("*.json"))]
        )
        catalog_hash = stable_hash(concat)
    else:
        catalog_hash = stable_hash(Path(catalog_path).read_text())

    manifest = {
        "catalog_hash": catalog_hash,
        "num_snippets": len(records),
        "config": {
            "index": asdict(cfg.index),
            "limits": asdict(cfg.limits),
        },
        "store_path": str(store_path),
    }
    manifest_path = artifacts_dir / "snippet_index_manifest.json"
    write_json(manifest_path, manifest)
    return store_path


