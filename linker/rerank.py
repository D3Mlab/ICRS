from __future__ import annotations

import numpy as np
from typing import List


def fuse_scores(hybrid: np.ndarray, cross: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if hybrid.size == 0:
        return cross
    if cross.size == 0:
        return hybrid
    # z-score normalize both
    def z(x: np.ndarray) -> np.ndarray:
        mu = x.mean() if x.size else 0.0
        sd = x.std() if x.size else 1.0
        sd = sd if sd > 1e-6 else 1.0
        return (x - mu) / sd

    return alpha * z(hybrid) + (1 - alpha) * z(cross)


def dummy_cross_encoder_scores(query: str, doc_texts: List[str]) -> np.ndarray:
    # Simple token overlap baseline
    q_tokens = set(query.lower().split())
    scores = []
    for t in doc_texts:
        dt = set(t.lower().split())
        inter = len(q_tokens & dt)
        union = len(q_tokens | dt) + 1e-6
        scores.append(inter / union)
    return np.array(scores, dtype=np.float32)


