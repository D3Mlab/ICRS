from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k}: {flatten_text(v)}" for k, v in value.items())
    return str(value)


def jaccard_similarity(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in text.split() if tok]


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ------------
# Path helpers
# ------------
def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / value).resolve()
    return path


def format_path(template: str, *, dataset: str) -> str:
    try:
        return template.format(dataset=dataset)
    except KeyError as exc:
        raise ValueError(f"Invalid path template '{template}': missing key {exc}") from exc


def select_links_object(data: Any, method: Optional[str]) -> Iterable[Dict[str, Any]]:
    if isinstance(data, dict):
        if "links" in data:
            pred_method = data.get("method")
            if method and pred_method and pred_method != method:
                raise ValueError(
                    f"Expected linker method '{method}' but found '{pred_method}' in predictions"
                )
            return data.get("links", [])
        if "results" in data and isinstance(data["results"], list):
            return select_links_object(data["results"], method)
    if isinstance(data, list):
        if method:
            for item in data:
                if isinstance(item, dict) and item.get("method") == method:
                    return item.get("links", []) or item.get("results", [])
            raise ValueError(f"Could not find predictions for linker method '{method}'")
        for item in data:
            if isinstance(item, dict) and ("links" in item or "results" in item):
                return item.get("links", []) or item.get("results", [])
    raise ValueError("Unsupported prediction JSON structure for segment links")


# --------------------
# Load helper routines
# --------------------
def load_link_data(
    *,
    base: Path,
    cfg: Dict[str, Any],
) -> Tuple[
    Dict[str, Tuple[str, float]],
    Dict[str, List[Tuple[str, float]]],
    Dict[str, str],
    Dict[str, Sequence[str]],
    int,
]:
    dataset = cfg.get("dataset", "vogue")
    predictions_rel = format_path(cfg.get("predictions", "data/{dataset}/links/segment_links.json"), dataset=dataset)
    ground_truth_rel = format_path(cfg.get("ground_truth", "data/{dataset}/meta_data/ground_truth.json"), dataset=dataset)

    pred_path = resolve_path(base, predictions_rel)
    gt_path = resolve_path(base, ground_truth_rel)

    method = cfg.get("method") or cfg.get("linker_method")

    predictions = load_json(pred_path)
    links_iter = select_links_object(predictions, method)

    doc_best: Dict[str, Tuple[str, float]] = {}
    per_segment: Dict[str, List[Tuple[str, float]]] = {}
    segments_with_predictions = 0

    for entry in links_iter:
        if not isinstance(entry, dict):
            continue
        seg_id = str(entry.get("segment_id")) if entry.get("segment_id") is not None else None
        matches = entry.get("matches")
        if seg_id is None or not isinstance(matches, list):
            continue
        had_match = False
        ranked: List[Tuple[str, float]] = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            doc_id = match.get("doc_id")
            if doc_id is None:
                continue
            doc_id = str(doc_id)
            score_val = match.get("score", 0.0)
            try:
                score = float(score_val)
            except (TypeError, ValueError):
                score = 0.0
            prev = doc_best.get(doc_id)
            if prev is None or score > prev[1]:
                doc_best[doc_id] = (seg_id, score)
            ranked.append((doc_id, score))
            had_match = True
        if had_match:
            segments_with_predictions += 1
            per_segment[seg_id] = ranked

    gt_raw = load_json(gt_path)
    if not isinstance(gt_raw, dict):
        raise ValueError("Ground truth metadata file must be a JSON object mapping doc_id to segment_id")
    gt_map = {str(doc): str(seg) for doc, seg in gt_raw.items() if seg is not None}
    segment_truth: Dict[str, List[str]] = {}
    for doc_id, seg_id in gt_map.items():
        segment_truth.setdefault(seg_id, []).append(doc_id)

    return doc_best, per_segment, gt_map, segment_truth, segments_with_predictions


def load_caption_pairs_map(base: Path, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    gt_path = (base / cfg.get("references", "data/gt/descriptions.json")).resolve()
    pred_path = (base / cfg.get("predictions", "data/preds/descriptions_pred.json")).resolve()
    gt = load_json(gt_path)
    pred = load_json(pred_path)

    def to_map(data: Any) -> Dict[str, Any]:
        if isinstance(data, dict) and "descriptions" in data:
            data = data["descriptions"]
        if isinstance(data, list):
            return {str(item.get("id")): item for item in data if isinstance(item, dict) and item.get("id")}
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
        raise ValueError("Unexpected caption JSON format")

    return to_map(gt), to_map(pred)


def load_object_rankings(base: Path, cfg: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    pred_path = (base / cfg.get("predictions", "data/preds/object_rankings.json")).resolve()
    gt_path = (base / cfg.get("ground_truth", "data/gt/object_relevance.json")).resolve()
    preds = load_json(pred_path)
    gts = load_json(gt_path)
    if isinstance(preds, dict) and "results" in preds:
        preds = preds["results"]
    if isinstance(gts, dict) and "results" in gts:
        gts = gts["results"]
    if not isinstance(preds, list) or not isinstance(gts, list):
        raise ValueError("Ranking JSON must contain a 'results' list")
    pred_map = {str(item.get("segment_id")): item for item in preds if isinstance(item, dict) and item.get("segment_id")}
    gt_map = {str(item.get("segment_id")): item for item in gts if isinstance(item, dict) and item.get("segment_id")}
    return pred_map, gt_map


def load_snippet_results(base: Path, cfg: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    pred_path = (base / cfg.get("predictions", "data/vogue/rankings/segment_rankings.json")).resolve()
    gt_path = (base / cfg.get("ground_truth", "data/gt/snippet_relevance.json")).resolve()
    preds = load_json(pred_path)
    gts = load_json(gt_path)
    pred_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in preds.get("results", []):
        query_id = str(item.get("query_id"))
        segment_id = str(item.get("segment_id"))
        key = (query_id, segment_id)
        pred_map[key] = item.get("ranked_snippets", [])
    gt_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in gts.get("results", []):
        query_id = str(item.get("query_id"))
        segment_id = str(item.get("segment_id"))
        key = (query_id, segment_id)
        gt_map[key] = item.get("relevant_snippets", [])
    return pred_map, gt_map


def _format_ranker_path(template: str, dataset: str, method: str, query_slug: Optional[str] = None) -> str:
    try:
        return template.format(
            dataset=dataset,
            object_ranker_method=method,
            method=method,
            query_id=query_slug if query_slug is not None else "",
        )
    except KeyError as exc:
        if "query_id" in template and query_slug is None:
            raise ValueError("Predictions path template requires a query_id but none was provided") from exc
        return str(template)


def load_object_ranking_scores(base: Path, cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    dataset = str(cfg.get("dataset", "vogue"))
    method = str(cfg.get("object_ranker_method") or cfg.get("method") or "FUSION_LLM")
    query_id = cfg.get("query_id")
    query_slug = _slugify_query_id(str(query_id)) if query_id else None

    predictions_cfg = cfg.get("predictions")
    path: Path
    
    # If predictions is an absolute path, use it directly
    if predictions_cfg:
        pred_path = Path(str(predictions_cfg))
        if pred_path.is_absolute() and pred_path.exists():
            path = pred_path
        elif pred_path.is_absolute():
            # Absolute path but doesn't exist - try to use it anyway
            path = pred_path
        else:
            # Relative path - use template formatting
            template = str(predictions_cfg)
            if "{query_id" in template and not query_slug:
                raise ValueError("Predictions path template references {query_id} but no query_id provided in config")
            rel_path = _format_ranker_path(template, dataset, method, query_slug)
            path = resolve_path(base, rel_path)
    else:
        # No predictions provided - use default template
        default_template = (
            "results/{dataset}/object_ranker/{object_ranker_method}/{query_id}_rankings.json"
            if query_slug
            else "results/{dataset}/object_ranker/{object_ranker_method}_rankings.json"
        )
        rel_path = _format_ranker_path(default_template, dataset, method, query_slug)
        path = resolve_path(base, rel_path)

    # Fallback: if path doesn't exist and we have query_id, try method-level file
    if not path.exists() and not predictions_cfg and query_slug:
        fallback_rel = _format_ranker_path(
            "results/{dataset}/object_ranker/{object_ranker_method}_rankings.json",
            dataset,
            method,
            None,
        )
        fallback_path = resolve_path(base, fallback_rel)
        if fallback_path.exists():
            path = fallback_path

    data = load_json(path)
    scores_by_query: Dict[str, Dict[str, float]] = {}

    def _coerce_scores(entry: Dict[str, Any]) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        scores = entry.get("scores")
        if isinstance(scores, dict):
            for doc_id, raw_val in scores.items():
                try:
                    cleaned[str(doc_id)] = float(raw_val)
                except (TypeError, ValueError):
                    continue
        elif isinstance(scores, list):
            # handle list of tuples or objects
            for item in scores:
                if isinstance(item, dict):
                    doc_id = item.get("doc_id") or item.get("segment_id") or item.get("id")
                    if doc_id is None:
                        continue
                    raw_val = item.get("score")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    doc_id, raw_val = item[:2]
                else:
                    continue
                if doc_id is None:
                    continue
                try:
                    cleaned[str(doc_id)] = float(raw_val)
                except (TypeError, ValueError):
                    continue
        return cleaned

    def _collect(entry: Dict[str, Any]) -> None:
        query_id = entry.get("query_id")
        if query_id is None:
            return
        cleaned = _coerce_scores(entry)
        if not cleaned:
            rankings = entry.get("rankings") or entry.get("results")
            if isinstance(rankings, list):
                for idx, row in enumerate(rankings):
                    if not isinstance(row, dict):
                        continue
                    doc_id = row.get("segment_id") or row.get("doc_id") or row.get("id")
                    if doc_id is None:
                        continue
                    score_val = row.get("score")
                    if score_val is None:
                        score = float(len(rankings) - idx)
                    else:
                        try:
                            score = float(score_val)
                        except (TypeError, ValueError):
                            score = float(len(rankings) - idx)
                    cleaned[str(doc_id)] = score
        if cleaned:
            scores_by_query[str(query_id)] = cleaned

    if isinstance(data, dict):
        if data.get("query_id") is not None:
            _collect(data)
        elif isinstance(data.get("results"), list):
            for item in data["results"]:
                if isinstance(item, dict):
                    _collect(item)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _collect(item)
    else:
        raise ValueError(f"Unsupported prediction JSON structure: {path}")

    return scores_by_query


def load_object_ranking_truth(base: Path, cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    dataset = str(cfg.get("dataset", "vogue"))
    method = str(cfg.get("object_ranker_method") or cfg.get("method") or "FUSION_LLM")
    template = cfg.get("ground_truth") or "data/{dataset}/ground_truth_object_ranking.json"
    rel_path = _format_ranker_path(str(template), dataset, method)
    path = resolve_path(base, rel_path)
    data = load_json(path)
    truth: Dict[str, Dict[str, float]] = {}
    if isinstance(data, dict):
        for query_id, doc_scores in data.items():
            if isinstance(doc_scores, dict):
                truth[str(query_id)] = {str(doc): float(score) for doc, score in doc_scores.items()}
    return truth


def _slugify_query_id(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return ""
    cleaned = []
    for ch in trimmed:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or ""


