from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple, Optional, Set

import torch
try:
    from bert_score import BERTScorer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BERTScorer = None

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import load_json, resolve_path, _slugify_query_id


class SnippetBertScoreMetric(EvaluationMetric):
    slug = "snippet_ranking"
    description = "Average BERTScore@K between retrieved snippets and ground-truth snippets."
    default_config = {
        "dataset": "vogue",
        "predictions": "results/{dataset}/snippet_ranker/DENSE_snippets_json.json",
        "ground_truth": "data/{dataset}/groud_truth_snippet/ground_truth_snippet_by_tag_with_item_number.json",
        "k": [1, 5, 10],
        "model_type": "microsoft/deberta-base-mnli",
        "lang": "en",
        "rescale_with_baseline": False,
        "device": None,
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        dataset = str(config.get("dataset", "vogue"))
        method = str(config.get("snippet_ranker_method") or config.get("method") or "DENSE").upper()
        raw_query_id = config.get("query_id")
        query_slug = _slugify_query_id(str(raw_query_id)) if raw_query_id else ""
        pred_tpl = config.get("predictions")
        gt_tpl = config.get("ground_truth")
        if not pred_tpl:
            raise MetricError("SnippetBertScoreMetric requires 'predictions' path in config")
        if not gt_tpl:
            raise MetricError("SnippetBertScoreMetric requires 'ground_truth' path in config")
        if pred_tpl:
            try:
                pred_rel = pred_tpl.format(
                    dataset=dataset,
                    snippet_ranker_method=method,
                    method=method,
                    query_id=query_slug or raw_query_id or "",
                )
            except Exception:
                pred_rel = str(pred_tpl)
        else:
            if query_slug:
                default_pred = "results/{dataset}/snippet_ranker/{method}/{query_id}_snippets_json.json"
                pred_rel = default_pred.format(dataset=dataset, method=method, query_id=query_slug)
            else:
                default_pred = "results/{dataset}/snippet_ranker/{method}_snippets_json.json"
                pred_rel = default_pred.format(dataset=dataset, method=method)
        try:
            gt_rel = gt_tpl.format(dataset=dataset)
        except Exception:
            gt_rel = str(gt_tpl)

        pred_path = resolve_path(ctx.data_root, pred_rel)
        gt_path = resolve_path(ctx.data_root, gt_rel)

        predictions = load_json(pred_path)
        results = predictions.get("results", [])
        if not isinstance(results, list):
            raise MetricError(f"Prediction JSON must contain a 'results' list: {pred_path}")

        gt_raw = load_json(gt_path)
        if not isinstance(gt_raw, dict):
            raise MetricError(f"Ground truth JSON must be an object mapping query_id to snippets: {gt_path}")

        if BERTScorer is None:
            raise MetricError("bert_score package is required for SnippetRankingMetric")

        scorer = BERTScorer(
            model_type=config.get("model_type", "microsoft/deberta-base-mnli"),
            lang=config.get("lang", "en"),
            rescale_with_baseline=bool(config.get("rescale_with_baseline", False)),
            device=config.get("device"),
        )

        k_cfg = config.get("k", [1, 5, 10])
        if isinstance(k_cfg, (list, tuple)):
            k_values = sorted({max(1, int(k)) for k in k_cfg if isinstance(k, (int, float, str)) and int(k) > 0})
        else:
            k_values = [max(1, int(k_cfg))]
        if not k_values:
            raise MetricError("SnippetBertScoreMetric requires at least one positive k value.")
        max_k = max(k_values)

        aggregates: Dict[int, List[float]] = {k_val: [] for k_val in k_values}
        target_query_id = config.get("query_id")
        if target_query_id is not None:
            target_query_id = str(target_query_id)

        # Group entries by query_id to aggregate snippets across segments
        entries_by_query: Dict[str, List[Dict[str, Any]]] = {}
        for entry in results:
            if not isinstance(entry, dict):
                continue
            query_id = entry.get("query_id")
            if query_id is None:
                continue
            query_id = str(query_id)
            if target_query_id and query_id != target_query_id:
                continue
            if query_id not in entries_by_query:
                entries_by_query[query_id] = []
            entries_by_query[query_id].append(entry)

        # Process each query, aggregating snippets across all segments
        for query_id, query_entries in entries_by_query.items():
            # Load ground truth for this query
            gt_entry = gt_raw.get(query_id)
            if not gt_entry:
                continue
            
            # Parse ground truth format (new format with sentences and mention_item_number)
            gt_sentences: List[str] = []
            mention_item_numbers: Optional[List[str]] = None
            
            if isinstance(gt_entry, dict):
                # New format: {"sentences": [...], "mention_item_number": [...]}
                gt_sentences = gt_entry.get("sentences", [])
                if isinstance(gt_sentences, list):
                    gt_sentences = [str(s).strip() for s in gt_sentences if str(s).strip()]
                mention_item_numbers = gt_entry.get("mention_item_number")
                if isinstance(mention_item_numbers, list):
                    mention_item_numbers = [str(item).strip() for item in mention_item_numbers if str(item).strip()]
            elif isinstance(gt_entry, list):
                # Old format: just a list of sentences
                gt_sentences = [str(s).strip() for s in gt_entry if str(s).strip()]
            
            if not gt_sentences:
                continue
            
            # Convert mention_item_number to doc_ids (e.g., "Item 27" -> "item_27.json")
            mentioned_doc_ids: Optional[Set[str]] = None
            if mention_item_numbers:
                mentioned_doc_ids = self._normalize_item_numbers_to_doc_ids(mention_item_numbers)

            # Aggregate all ranked snippets across all segments for this query
            all_ranked_snippets: List[Dict[str, Any]] = []
            for entry in query_entries:
                ranked_snippets = entry.get("ranked_snippets") or []
                all_ranked_snippets.extend(ranked_snippets)
            
            if not all_ranked_snippets:
                continue
            
            # Filter ranked snippets by mentioned items if mention_item_number is provided
            filtered_snippets = self._filter_snippets_by_doc_ids(all_ranked_snippets, mentioned_doc_ids)
            
            if not filtered_snippets:
                # No snippets match the mentioned items, skip this query
                continue
            
            # Re-rank filtered snippets by score (highest first)
            filtered_snippets.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
            
            # Assign ranks to filtered snippets
            for rank_idx, snippet in enumerate(filtered_snippets, start=1):
                snippet["rank"] = rank_idx
            
            ranked_texts = self._select_topk_texts(filtered_snippets, max_k)
            if not ranked_texts:
                continue

            for k_val in k_values:
                portion = ranked_texts[:k_val]
                if not portion:
                    continue
                segment_score = self._average_bert_for_segment(portion, gt_sentences, scorer)
                if segment_score is not None:
                    aggregates[k_val].append(segment_score)

        if not any(aggregates.values()):
            raise MetricError("No overlapping query IDs between predictions and ground truth.")

        results_dict: Dict[str, Any] = {
            "query_id": config.get("query_id", ""),
        }
        for k_val in k_values:
            scores = aggregates.get(k_val) or []
            if scores:
                results_dict[f"Bert@{k_val}"] = sum(scores) / len(scores)
            else:
                results_dict[f"Bert@{k_val}"] = 0.0
        return results_dict

    @staticmethod
    def _normalize_item_numbers_to_doc_ids(item_numbers: List[str]) -> Set[str]:
        """Convert item numbers like 'Item 27' to doc_ids like 'item_27.json'."""
        return item_numbers
    
    @staticmethod
    def _filter_snippets_by_doc_ids(
        snippets: Sequence[Dict[str, Any]], 
        doc_ids: Optional[Set[str]]
    ) -> List[Dict[str, Any]]:
        """Filter snippets to only include those with doc_ids in the provided set."""
        if doc_ids is None:
            # No filtering if no doc_ids provided (backward compatibility)
            return [s for s in snippets if isinstance(s, dict)]
        
        filtered: List[Dict[str, Any]] = []
        for snip in snippets:
            if not isinstance(snip, dict):
                continue
            doc_id = str(snip.get("doc_id") or "")
            if doc_id in doc_ids:
                filtered.append(snip)
        return filtered

    @staticmethod
    def _select_topk_texts(snippets: Sequence[Dict[str, Any]], k: int) -> List[str]:
        normalized: List[Tuple[int, str]] = []
        for snip in snippets:
            if not isinstance(snip, dict):
                continue
            text = str(snip.get("text") or "").strip()
            if not text:
                continue
            rank_val = snip.get("rank")
            try:
                rank = int(rank_val)
            except (TypeError, ValueError):
                rank = len(normalized) + 1
            normalized.append((rank, text))
        normalized.sort(key=lambda item: item[0])
        return [text for _, text in normalized[:k]]

    @staticmethod
    def _average_bert_for_segment(
        predicted_snippets: Sequence[str],
        ground_truth_snippets: Sequence[Any],
        scorer: BERTScorer,
    ) -> Optional[float]:
        gt_texts = [str(text).strip() for text in ground_truth_snippets if str(text).strip()]
        if not gt_texts:
            return None

        per_snippet_scores: List[float] = []
        for pred in predicted_snippets:
            candidates = [pred] * len(gt_texts)
            _, _, f1 = scorer.score(candidates, gt_texts, verbose=False)
            best = torch.max(f1).item() if f1.numel() else 0.0
            per_snippet_scores.append(float(best))

        if not per_snippet_scores:
            return None
        return sum(per_snippet_scores) / len(per_snippet_scores)

