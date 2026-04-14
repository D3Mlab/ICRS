from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

from ..utils.io import read_json


def candidate_docs_for_query(queries_path: Path, segment_links_path: Path) -> Dict[str, Set[str]]:
    queries = read_json(queries_path).get("queries", [])
    links = read_json(segment_links_path).get("links", [])

    seg_to_doc: Dict[str, str] = {}
    for row in links:
        if row.get("status") == "matched" and row.get("chosen"):
            seg_to_doc[str(row["segment_id"])] = str(row["chosen"])  # chosen must be exact filename

    out: Dict[str, Set[str]] = {}
    for q in queries:
        qid = str(q["query_id"])
        cands = q.get("candidate_segment_ids")
        if cands:
            docs = {seg_to_doc[s] for s in cands if s in seg_to_doc}
        else:
            docs = set(seg_to_doc.values())
        out[qid] = docs
    return out


