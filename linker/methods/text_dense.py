from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from pathlib import Path

from utils.cache_manager import CacheManager
from .base import LinkMethod
from label_selection.utils.text import normalize
from linker.embedder_base import EmbedderBase


def flatten_text_fields(doc: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    use_fields: List[str]
    if not fields:
        # Use all keys except non-textual identifiers/paths if fields is empty
        use_fields = [k for k in doc.keys() if k not in {"id", "image_path", "reviews"}]
    else:
        use_fields = fields
    for f in use_fields:
        val = doc.get(f)
        if isinstance(val, str) and val.strip():
            parts.append(val)
        elif isinstance(val, list):
            parts.append(" ".join([v if isinstance(v, str) else str(v) for v in val]))
        elif isinstance(val, dict):
            parts.append(" ".join([f"{k}: {v}" for k, v in val.items()]))
    return " \n ".join(parts)


class DenseLinker(LinkMethod):
    def __init__(
        self,
        fields: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
        embed_provider: str = "local",
        embed_api_key: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        dataset_root: Optional[Path] = None,
    ):
        self.fields = fields
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.doc_ids: List[str] = []
        self.doc_embs: np.ndarray | None = None
        inferred_root = CacheManager.infer_dataset_root(dataset_root, cache_dir)
        self.dataset_root: Path = inferred_root
        self.cache_base = cache_dir
        self.embed_provider = (embed_provider or "local").lower()
        self.embed_api_key = embed_api_key
        self.embed_base_url = embed_base_url
        self.embedder = EmbedderBase(
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
            provider=self.embed_provider,
            api_key=self.embed_api_key,
            base_url=self.embed_base_url,
        )
        self.method_name: str = "TEXT_DENSE"

    def prepare(self, docs: List[Dict[str, Any]]) -> None:
        self.doc_ids = [str(d.get("doc_id")) for d in docs]
        texts = [normalize(flatten_text_fields(d, self.fields)) for d in docs]
        cfg_sig: Dict[str, object] = {
            "model_name": self.model_name,
            "normalize_embeddings": bool(self.normalize_embeddings),
            "provider": self.embed_provider,
            "fields": list(self.fields),
        }
        variant = self._doc_cache_variant()
        self.doc_embs = self.embedder.encode_with_cache(
            module="linker",
            method=self.method_name.lower(),
            dataset_root=self.dataset_root,
            cache_base=self.cache_base,
            variant=variant,
            keys=self.doc_ids,
            texts=texts,
            meta=cfg_sig,
        )

    def _build_query_text(self, segment: Dict[str, Any]) -> str:
        # Default query: normalized description
        return normalize(segment.get("description", ""))

    def score_segment(self, segment: Dict[str, Any], top_k: int) -> List[Tuple[str, float, str]]:
        assert self.doc_embs is not None
        q = self._build_query_text(segment)
        if not q:
            return []
        qv: np.ndarray
        cfg_sig = {
            "model_name": self.model_name,
            "normalize_embeddings": bool(self.normalize_embeddings),
            "provider": self.embed_provider,
        }
        seg_variant = self._segment_cache_variant()
        seg_id = str(segment.get("id"))
        qv = self.embedder.encode_with_cache(
            module="linker",
            method=self.method_name.lower(),
            dataset_root=self.dataset_root,
            cache_base=self.cache_base,
            variant=seg_variant,
            keys=[seg_id],
            texts=[q],
            meta=cfg_sig,
        )[0]
        sims = (self.doc_embs @ qv).astype(np.float32)
        pairs = list(zip(self.doc_ids, map(float, sims)))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return [(doc_id, score, "dense") for doc_id, score in pairs[:top_k]]

    @staticmethod
    def _sanitize_label(s: str) -> str:
        # Keep it readable: replace path separators and common specials with underscore
        return (
            str(s)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
            .replace("|", "_")
            .replace(",", "_")
        )

    def _doc_cache_variant(self) -> str:
        fields_label = self._sanitize_label("+".join(self.fields)) if self.fields else "ALL"
        norm_tag = "norm1" if self.normalize_embeddings else "norm0"
        method = self._sanitize_label(self.method_name.lower())
        return f"{method}_docs_{fields_label}_{norm_tag}"

    def _segment_cache_variant(self) -> str:
        norm_tag = "norm1" if self.normalize_embeddings else "norm0"
        method = self._sanitize_label(self.method_name.lower())
        return f"{method}_segments_{norm_tag}"


