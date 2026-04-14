"""
Report writers: per-step CSV and overall scorecard.json (aggregating step CSVs and plugin metrics).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from .io import write_csv_flat


def write_step_csv(
    report_path: Path,
    metrics: List[Dict[str, Any]],
    method_name: str | None = None,
    model_name: str | None = None,
    data_name: str | None = None,
) -> None:
    new_rows: List[Dict[str, Any]] = []
    for m in metrics:
        name = m.get("name")
        values = m.get("values", {})
        flat = {"metric": name or ""}
        flat["method"] = method_name or ""
        flat["model"] = model_name or ""
        flat["data"] = data_name or ""
        flat.update(values)
        new_rows.append(flat)

    current_query_id: str | None = None
    current_gt: str | None = None
    for row in new_rows:
        if current_query_id is None:
            qid = row.get("query_id")
            if qid not in (None, ""):
                current_query_id = str(qid)
        if current_gt is None:
            gt_val = row.get("gt")
            if gt_val not in (None, ""):
                current_gt = str(gt_val)
        # Break early if we've found both
        if current_query_id is not None and current_gt is not None:
            break

    rows: List[Dict[str, Any]] = new_rows
    # Merge with existing if present: replace rows matching current method_name only
    if report_path.exists():
        try:
            import csv
            existing: List[Dict[str, Any]] = []
            with report_path.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    existing.append(row)
            if method_name is not None:
                filtered: List[Dict[str, Any]] = []
                for r in existing:
                    same_method = r.get("method") == method_name
                    same_model = model_name is None or r.get("model", "") == (model_name or "")
                    same_data = data_name is None or r.get("data", "") == (data_name or "")
                    same_query = True
                    if current_query_id not in (None, ""):
                        same_query = r.get("query_id") == current_query_id
                    same_gt = True
                    if current_gt not in (None, ""):
                        same_gt = r.get("gt", "") == current_gt
                    if same_method and same_model and same_data and same_query and same_gt:
                        continue
                    filtered.append(r)
                existing = filtered
            rows = existing + rows
        except Exception:
            pass
    # Deduplicate keeping latest entries per (method, model, data, metric, query_id, gt)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in reversed(rows):
        key = (
            row.get("method") or "",
            row.get("model") or "",
            row.get("data") or "",
            row.get("metric") or "",
            row.get("query_id") or "",
            row.get("gt") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    rows = list(reversed(deduped))
    write_csv_flat(report_path, rows)


def write_scorecard(scorecard_path: Path, per_step_csvs: Dict[str, Path], plugins: Dict[str, Any]) -> None:
    obj = {
        "steps": {k: str(v) for k, v in per_step_csvs.items()},
        "plugins": plugins or {},
    }
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))



