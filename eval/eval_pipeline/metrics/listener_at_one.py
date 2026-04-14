from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .base import EvaluationMetric, MetricContext, MetricError
from .utils import flatten_text, jaccard_similarity, tokenize
from .utils import load_caption_pairs_map as _load_caption_pairs
class ListenerAtOneMetric(EvaluationMetric):
    slug = "listener_at_1"
    description = "Listener accuracy@1 based on token-overlap retrieval"
    default_config = {
        "references": "data/gt/descriptions.json",
        "predictions": "data/preds/descriptions_pred.json",
        "candidate_subset": 50,
    }

    def _compute(self, ctx: MetricContext, config: Dict[str, Any]) -> Dict[str, Any]:
        gt_map, pred_map = _load_caption_pairs(ctx, config)
        if not gt_map:
            raise MetricError("No ground truth descriptions available")

        ids = list(gt_map.keys())
        candidate_subset = int(config.get("candidate_subset", len(ids)))
        success = 0
        total = 0
        for seg_id, pred_entry in pred_map.items():
            hyp_text = flatten_text(pred_entry.get("description") or pred_entry.get("text") or "")
            hyp_tokens = tokenize(hyp_text)
            if not hyp_tokens:
                continue
            total += 1
            best_id = None
            best_score = -1.0
            candidates = ids
            if candidate_subset and candidate_subset < len(ids):
                candidates = ids[:candidate_subset]
            for cand_id in candidates:
                ref_text = flatten_text(gt_map[cand_id].get("description") or gt_map[cand_id].get("text") or "")
                score = jaccard_similarity(hyp_tokens, tokenize(ref_text))
                if score > best_score:
                    best_score = score
                    best_id = cand_id
            if best_id == seg_id:
                success += 1

        if total == 0:
            raise MetricError("No predicted descriptions available for listener evaluation")
        return {"listener_at_1": success / total, "evaluated": total}

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

