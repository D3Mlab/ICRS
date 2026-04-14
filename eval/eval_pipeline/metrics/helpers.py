from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

from .base import MetricContext, MetricError
from .utils import tokenize, jaccard_similarity, safe_ratio


# ----------------------------
# Common path helpers (linking)
# ----------------------------


# ---------------------------------
# Caption/description metrics helpers
# ---------------------------------


def nltk_bleu(refs: Iterable[List[str]], hyp: List[str]) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
    except Exception as exc:
        raise MetricError(f"nltk is required for BLEU computation: {exc}") from exc
    smoothie = SmoothingFunction().method4
    try:
        return float(sentence_bleu(list(refs), hyp, smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25)))
    except ZeroDivisionError:
        return 0.0


def nltk_meteor(ref: str, hyp: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score  # type: ignore
    except Exception as exc:
        raise MetricError(f"nltk is required for METEOR computation: {exc}") from exc
    return float(meteor_score([ref], hyp))


def rouge_l(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if ref_len == 0 or hyp_len == 0:
        return 0.0
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(ref_len):
        for j in range(hyp_len):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[ref_len][hyp_len]
    prec = lcs / hyp_len
    rec = lcs / ref_len
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def simple_cider(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not ref_tokens or not hyp_tokens:
        return 0.0
    from collections import Counter

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    overlap = 0.0
    norm = 0.0
    for token, h_count in hyp_counts.items():
        norm += h_count
        overlap += min(h_count, ref_counts.get(token, 0))
    if norm == 0:
        return 0.0
    return overlap / norm


# ------------------
# Ranking helpers
# ------------------
def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel <= 0:
            continue
        dcg += (2 ** rel - 1) / math.log2(idx + 1)
    ideal = sorted(relevances, reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal[:k], start=1):
        if rel <= 0:
            continue
        idcg += (2 ** rel - 1) / math.log2(idx + 1)
    return safe_ratio(dcg, idcg)


# def map_at_k(relevances: Sequence[int], k: int) -> float:
#     hits = 0
#     ap = 0.0
#     for idx, rel in enumerate(relevances[:k], start=1):
#         if rel > 0:
#             hits += 1
#             ap += hits / idx
#     return safe_ratio(ap, hits) if hits > 0 else 0.0

def map_at_k(relevances: Sequence[int], k: int) -> float:
    ap = 0.0
    hits = 0
    total_relevant = sum(1 for r in relevances if r > 0)

    if total_relevant == 0:
        return 0.0

    for idx, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            hits += 1
            ap += hits / idx

    return ap / total_relevant

# ------------------
# Snippet helpers
# ------------------


def mean_average_precision(binary: Sequence[int], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for idx, rel in enumerate(binary[:k], start=1):
        if rel:
            hits += 1
            precision_sum += hits / idx
    total_relevant = sum(binary)
    return safe_ratio(precision_sum, min(k, total_relevant))


def ndcg(binary: Sequence[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(binary[:k], start=1):
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    ideal = sorted(binary, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k], start=1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)
    return safe_ratio(dcg, idcg)


