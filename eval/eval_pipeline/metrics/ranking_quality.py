from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import (
    safe_ratio,
    load_object_ranking_scores,
    load_object_ranking_truth,
)
from .helpers import ndcg_at_k as _ndcg_at_k, map_at_k as _map_at_k


class RankingQualityMetric(EvaluationMetric):
    slug = "step5_ranking_quality"
    description = "Precision/Recall/MAP/nDCG for object ranking"
    default_config = {
        "dataset": "vogue",
        "predictions": "",
        "ground_truth": "data/{dataset}/ground_truth_object_ranking.json",
        "object_ranker_method": "FUSION_LLM",
        "k": [5, 10, 20, 50],
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        pred_scores = load_object_ranking_scores(ctx.data_root, config)
        if not pred_scores:
            raise MetricError("No predicted rankings found")

        gt_map = load_object_ranking_truth(ctx.data_root, config)
        if not gt_map:
            raise MetricError("Ground truth rankings are empty or missing")

        raw_k = config.get("k", [5, 10, 20, 50])
        if isinstance(raw_k, int):
            k_values = [raw_k]
        elif isinstance(raw_k, (list, tuple, set)):
            k_values = sorted({int(k) for k in raw_k if int(k) > 0})
        else:
            raise MetricError("k must be an int or a sequence of ints")
        if not k_values:
            raise MetricError("k must contain at least one positive integer")

        target_query_id = config.get("query_id")
        if target_query_id is not None:
            target_query_id = str(target_query_id)

        agg: Dict[int, Dict[str, float]] = {
            k_val: {"precision": 0.0, "recall": 0.0, "map": 0.0, "ndcg": 0.0} for k_val in k_values
        }
        counts: Dict[int, int] = {k_val: 0 for k_val in k_values}
        evaluated_any = False
        spearman_scores: List[float] = []
        kendall_scores: List[float] = []

        for query_id, score_map in pred_scores.items():
            if target_query_id and str(query_id) != target_query_id:
                continue
            gt_scores = gt_map.get(query_id)
            if not gt_scores:
                continue
            positive_docs = [doc for doc, score in gt_scores.items() if score > 0]
            if not positive_docs:
                continue

            binary_rels: List[int] = []
            gain_rels: List[float] = []
            hits = 0
            ranked_docs = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
            for doc_id, _score in ranked_docs:
                doc_id = str(doc_id)
                gain = max(float(gt_scores.get(doc_id, 0.0)), 0.0)
                gain_rels.append(gain)
                if gain >= 5:
                    binary_rels.append(1)
                    hits += 1
                else:
                    binary_rels.append(0)

            if not binary_rels:
                continue

            for k_val in k_values:
                top_binary = binary_rels[:k_val]
                top_gain = gain_rels[:k_val]
                if not top_binary:
                    continue
                hits_k = sum(top_binary)
                precision_k = safe_ratio(hits_k, len(top_binary))
                recall_k = safe_ratio(hits_k, len(positive_docs))
                map_k = _map_at_k(binary_rels, k_val)
                ndcg_k = _ndcg_at_k(gain_rels, k_val)
                agg[k_val]["precision"] += precision_k
                agg[k_val]["recall"] += recall_k
                agg[k_val]["map"] += map_k
                agg[k_val]["ndcg"] += ndcg_k
                counts[k_val] += 1
                evaluated_any = True

        if not evaluated_any:
            if target_query_id:
                raise MetricError(f"No overlapping rankings found for query_id '{target_query_id}'.")
            raise MetricError("No overlapping query IDs between predictions and ground truth.")

        results: Dict[str, Any] = {}
        if target_query_id is not None:
            results["query_id"] = target_query_id

            spearman = self._spearman_correlation(ranked_docs, gt_scores)
            if spearman is not None:
                spearman_scores.append(spearman)
            kendall = self._kendall_tau_b(ranked_docs, gt_scores)
            if kendall is not None:
                kendall_scores.append(kendall)


        for k_val in k_values:
            count = counts[k_val]
            if count == 0:
                continue
            results[f"precision@{k_val}"] = np.round(agg[k_val]["precision"] / count, 4)
            results[f"recall@{k_val}"] = np.round(agg[k_val]["recall"] / count, 4)
            results[f"map@{k_val}"] = np.round(agg[k_val]["map"] / count, 4)
            results[f"ndcg@{k_val}"] = np.round(agg[k_val]["ndcg"] / count, 4)
        if spearman_scores:
            results["spearman"] = float(np.mean(spearman_scores))
        if kendall_scores:
            results["kendall_tau_b"] = float(np.mean(kendall_scores))
        return results

    def load_data(self, gt_path: Path, pred_path: Path) -> None:  # type: ignore[override]
        self._override_paths = {"ground_truth": str(gt_path), "predictions": str(pred_path)}

    def eval(self, ctx: MetricContext, config: Dict[str, Any], gt_path: Path = None, pred_path: Path = None, iou_type: str = "segm") -> Dict[str, Any]:  # type: ignore[override]
        cfg = dict(self.default_config)
        cfg.update(config or {})
        if gt_path is not None:
            cfg["ground_truth"] = str(gt_path)
        if pred_path is not None:
            cfg["predictions"] = str(pred_path)
        return self._compute(ctx, cfg)

    @staticmethod
    def _spearman_correlation(ranked_docs: List[tuple[str, float]], gt_scores: Dict[str, float]) -> Optional[float]:
        if not ranked_docs:
            return None
        ids = []
        preds = []
        trues = []
        for doc_id, score in ranked_docs:
            doc_id = str(doc_id)
            if doc_id not in gt_scores:
                continue
            ids.append(doc_id)
            preds.append(float(score))
            trues.append(float(gt_scores[doc_id]))
        if len(ids) < 2:
            return None
        pred_ranks = np.argsort(np.argsort(preds))
        true_ranks = np.argsort(np.argsort(trues))
        pred_ranks = pred_ranks.astype(np.float64)
        true_ranks = true_ranks.astype(np.float64)
        pred_ranks += 1
        true_ranks += 1
        d = pred_ranks - true_ranks
        n = len(pred_ranks)
        numerator = 6 * np.sum(d * d)
        denom = n * (n * n - 1)
        if denom == 0:
            return None
        return float(1 - numerator / denom)

    @staticmethod
    def _kendall_tau_b(ranked_docs: List[tuple[str, float]], gt_scores: Dict[str, float]) -> Optional[float]:
        if not ranked_docs:
            return None
        ids = []
        preds = []
        trues = []
        for doc_id, score in ranked_docs:
            doc_id = str(doc_id)
            if doc_id not in gt_scores:
                continue
            ids.append(doc_id)
            preds.append(float(score))
            trues.append(float(gt_scores[doc_id]))
        n = len(ids)
        if n < 2:
            return None
        concordant = 0
        discordant = 0
        ties_pred = 0
        ties_true = 0
        for i in range(n):
            for j in range(i + 1, n):
                pred_diff = preds[i] - preds[j]
                true_diff = trues[i] - trues[j]
                if pred_diff == 0 and true_diff == 0:
                    continue
                if pred_diff == 0:
                    ties_pred += 1
                    continue
                if true_diff == 0:
                    ties_true += 1
                    continue
                concordant += int(pred_diff * true_diff > 0)
                discordant += int(pred_diff * true_diff < 0)
        denom = np.sqrt((concordant + discordant + ties_pred) * (concordant + discordant + ties_true))
        if denom == 0:
            return None
        return float((concordant - discordant) / denom)


