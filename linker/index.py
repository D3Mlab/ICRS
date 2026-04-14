from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SimpleIndex:
    """A tiny cosine similarity index using numpy; FAISS can replace this later."""

    vectors: np.ndarray  # shape (N, D)

    def search(self, query: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize
        A = self.vectors
        if query.ndim == 1:
            q = query[None, :]
        else:
            q = query
        # cosine similarity
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        sims = (A_norm @ q_norm.T)[:, 0]
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx]


