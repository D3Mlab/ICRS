from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


_word_re = re.compile(r"[A-Za-z0-9_#\-]+")


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return _word_re.findall(text)


def flatten_doc_text(title: str, description: str, attrs: dict | None) -> str:
    parts: List[str] = []
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    if attrs:
        for k, v in attrs.items():
            if v is None:
                continue
            parts.append(f"{k}:{v}")
    return "\n".join(parts)


@dataclass
class TextEmbedder:
    max_features: int = 20000

    def __post_init__(self):
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, tokenizer=_tokenize)

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return normalize(X, norm="l2")


def image_histogram_embedding(image_path: Optional[str], bins: int = 64) -> np.ndarray:
    if not image_path:
        return np.zeros((bins * 3,), dtype=np.float32)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((bins * 3,), dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(hsv)
    hists = []
    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [bins], [0, 256]).flatten()
        if hist.sum() > 0:
            hist = hist / hist.sum()
        hists.append(hist)
    return np.concatenate(hists).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    return float((a @ b.T)[0, 0] / denom)


