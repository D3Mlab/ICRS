from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import flatten_text, tokenize, jaccard_similarity
from .helpers import nltk_bleu as _nltk_bleu, nltk_meteor as _nltk_meteor, rouge_l as _rouge_l, simple_cider as _simple_cider
from .utils import load_caption_pairs_map as _load_caption_pairs


class CaptionQualityMetric(EvaluationMetric):
    slug = "caption_quality"
    description = "Caption similarity metrics: BLEU-4, ROUGE-L, METEOR, CIDEr (approx), SPICE (token-F1)"
    default_config = {
        "references": "data/gt/descriptions.json",
        "predictions": "data/preds/descriptions_pred.json",
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        gt_map, pred_map = _load_caption_pairs(ctx, config)
        if not gt_map:
            raise MetricError("No ground truth descriptions available")
        bleu_scores: List[float] = []
        rouge_scores: List[float] = []
        meteor_scores: List[float] = []
        cider_scores: List[float] = []
        spice_scores: List[float] = []

        for seg_id, gt_entry in gt_map.items():
            ref_text = flatten_text(gt_entry.get("description") or gt_entry.get("text") or "")
            hyp_text = flatten_text(pred_map.get(seg_id, {}).get("description") or pred_map.get(seg_id, {}).get("text") or "")
            ref_tokens = tokenize(ref_text)
            hyp_tokens = tokenize(hyp_text)
            if not hyp_tokens:
                continue
            if ref_tokens:
                bleu_scores.append(_nltk_bleu([ref_tokens], hyp_tokens))
                rouge_scores.append(_rouge_l(ref_tokens, hyp_tokens))
                meteor_scores.append(_nltk_meteor(ref_text, hyp_text))
                cider_scores.append(_simple_cider(ref_tokens, hyp_tokens))
                spice_scores.append(jaccard_similarity(ref_tokens, hyp_tokens))

        if not bleu_scores:
            raise MetricError("No overlapping captions between predictions and references")

        def mean(values: List[float]) -> float:
            return sum(values) / len(values)

        return {
            "BLEU4": mean(bleu_scores),
            "ROUGE_L": mean(rouge_scores),
            "METEOR": mean(meteor_scores),
            "CIDEr_approx": mean(cider_scores),
            "SPICE_token_f1": mean(spice_scores),
        }

    # Align to abstract interface (no-op loaders; paths are read inside _compute)
    def load_data(self, gt_path: Path, pred_path: Path) -> None:  # type: ignore[override]
        self._override_paths = {"references": str(gt_path), "predictions": str(pred_path)}

    def eval(self, ctx: MetricContext, config: Dict[str, Any], gt_path: Path = None, pred_path: Path = None, iou_type: str = "segm") -> Dict[str, Any]:  # type: ignore[override]
        cfg = dict(self.default_config)
        cfg.update(config or {})
        if gt_path is not None:
            cfg["references"] = str(gt_path)
        if pred_path is not None:
            cfg["predictions"] = str(pred_path)
        return self._compute(ctx, cfg)
