from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from cache_manager import CacheManager
from utils.io_manager import IOManager, OutputSpec
from linker.embed import TextEmbedder, flatten_doc_text, image_histogram_embedding, cosine_sim
from linker.rerank import dummy_cross_encoder_scores, fuse_scores
from linker.methods.base import LinkMethod
from linker.methods.text_bm25 import BM25Linker
from linker.methods.text_dense import DenseLinker
from linker.methods.text_dense_expansion import QueryExpansionLinker
from linker.methods.text_fuzzy import TextExactFuzzy
from linker.methods.llm_judge import LLMLinkJudge
from linker.llm_check import llm_consistency_stub
from linker.extract import rule_extract
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_segments(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        segments: List[Dict[str, Any]] = []
        for fp in sorted(path.glob("*.json")):
            try:
                obj = json.loads(fp.read_text())
            except Exception as exc:
                print(f"[linker.pipeline] Failed to parse segment JSON {fp}: {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                segments.append(obj)
            elif isinstance(obj, list):
                segments.extend([item for item in obj if isinstance(item, dict)])
        return segments

    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    segs = data.get("segments") or []
    return segs


def _coerce_doc_from_file(fp: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(fp.read_text())
    except Exception as e:
        print(f"[linker.pipeline] Failed to parse JSON file {fp}: {e}", file=sys.stderr)
        obj = {}

    # Force doc_id to exact filename (with extension) when loading from directory
    doc_id = fp.name
    title = obj.get("title") or obj.get("name") or obj.get("product_name") or ""
    desc = obj.get("description") or obj.get("desc") or obj.get("text") or obj.get("details") or ""
    if isinstance(desc, list):
        desc = "\n".join(map(str, desc))
    desc = str(desc)

    attrs = obj.get("attributes")
    if not isinstance(attrs, dict):
        known = {"doc_id", "id", "title", "name", "product_name", "description", "desc", "text", "details", "image", "image_path", "img", "thumbnail", "photo"}
        attrs = {k: v for k, v in obj.items() if k not in known}

    img = obj.get("image_path") or obj.get("image") or obj.get("img") or obj.get("thumbnail") or obj.get("photo")
    if isinstance(img, str) and img:
        img_path = (fp.parent / img) if not Path(img).is_absolute() else Path(img)
        image_path = str(img_path)
    else:
        image_path = None

    return {
        "doc_id": str(doc_id),
        "title": str(title),
        "description": desc,
        "attributes": attrs or {},
        "image_path": image_path,
    }


def load_docs(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        docs: List[Dict[str, Any]] = []
        for fp in sorted(path.glob("*.json")):
            if fp.name == "ground_truth.json":
                continue
            docs.append(_coerce_doc_from_file(fp))
        return docs
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    docs = data.get("docs") or []
    return docs

class MetaDataLinker:
    """High-level wrapper around linker pipeline with IO-managed caching.

    Usage:
        linker = MetaDataLinker(io_manager, config_path)
        result = linker.link(segments_path, docs_path, out_links, out_meta, **kwargs)
    """

    def __init__(self, io_manager: Optional[IOManager] = None, config_path: Optional[Path] = None) -> None:
        self.io: IOManager = io_manager or IOManager()
        self.config_path: Optional[Path] = config_path
        self._config: Optional[Dict[str, Any]] = None

    def _load_config(self) -> Dict[str, Any]:
        if self._config is not None:
            return self._config
        cfg: Dict[str, Any] = {}
        path = self.config_path or Path("configs/linker.yaml")
        if path and path.exists():
            try:
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
                    import yaml  # type: ignore
                    cfg = yaml.safe_load(path.read_text()) or {}
                elif path.is_file():
                    cfg = json.loads(path.read_text() or "{}")
            except Exception:
                cfg = {}
        self._config = cfg if isinstance(cfg, dict) else {}
        return self._config

    def _select_requested_method_and_model(self, cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        requested = str((cfg or {}).get("method", "TEXT_BM25")).upper()
        link_model: Optional[str] = None
        if isinstance(cfg, dict):
            dense_cfg = cfg.get("dense") if isinstance(cfg.get("dense"), dict) else None
            llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else None
            if requested in {"TEXT_DENSE", "TEXT_DENSE_EXPANSION"} and isinstance(dense_cfg, dict):
                link_model = dense_cfg.get("model_name")
            if requested == "TEXT_DENSE_EXPANSION" and isinstance(llm_cfg, dict):
                extra = llm_cfg.get("model")
                if extra:
                    link_model = f"{link_model}+{extra}" if link_model else str(extra)
            if requested == "LLM_JUDGE" and isinstance(llm_cfg, dict):
                link_model = llm_cfg.get("model")
        return requested, link_model

    def link(
        self,
        *,
        segments_path: Path,
        docs_path: Path,
        out_links: Path,
        out_meta: Path,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cfg = self._load_config()
        requested, link_model = self._select_requested_method_and_model(cfg)

        dataset_root = CacheManager.infer_dataset_root(segments_path, docs_path)
        dataset_name = dataset_root.name if dataset_root.name else (dataset_root.parts[-1] if dataset_root.parts else "default")

        def _generate() -> Dict[str, Any]:
            out_links.parent.mkdir(parents=True, exist_ok=True)
            out_meta.parent.mkdir(parents=True, exist_ok=True)
            run_pipeline(
                segments_path=segments_path,
                docs_path=docs_path,
                out_links=out_links,
                out_meta=out_meta,
                w_text=float(kwargs.get("w_text", 0.7)),
                w_image=float(kwargs.get("w_image", 0.3)),
                k=int(kwargs.get("k", 20)),
                m=int(kwargs.get("m", 5)),
                min_match_threshold=float(kwargs.get("min_match_threshold", 0.55)),
                min_margin=float(kwargs.get("min_margin", 0.05)),
                tfidf_max_features=int(kwargs.get("tfidf_max_features", 20000)),
                image_hist_bins=int(kwargs.get("image_hist_bins", 64)),
                llm_enable=bool(kwargs.get("llm_enable", False)),
                seed=int(kwargs.get("seed", 42)),
                dry_run=bool(kwargs.get("dry_run", False)),
                method_cfg=cfg,
            )
            links_payload = json.loads(out_links.read_text()) if out_links.exists() else {}
            meta_payload = json.loads(out_meta.read_text()) if out_meta.exists() else {}
            return {"links": links_payload, "metadata": meta_payload}

        self.io.get_or_generate(
            module="linker",
            method=requested,
            dataset=dataset_name,
            specs={
                "links": OutputSpec(
                    format="json",
                    mirror_path=out_links,
                    metadata={
                        "segments_json": str(segments_path),
                        "docs_json": str(docs_path),
                    },
                ),
                "metadata": OutputSpec(
                    format="json",
                    mirror_path=out_meta,
                    metadata={
                        "segments_json": str(segments_path),
                        "docs_json": str(docs_path),
                    },
                ),
            },
            generator=_generate,
        )

        return {
            "method": requested,
            "model": link_model,
            "links_path": out_links,
            "meta_path": out_meta,
            "dataset": dataset_name,
        }


def build_text_embeddings(segments: List[Dict[str, Any]], docs: List[Dict[str, Any]], max_features: int) -> Tuple[np.ndarray, np.ndarray, TextEmbedder, List[str], List[str]]:
    seg_texts = [s.get("description", "") for s in segments]
    doc_texts = [flatten_doc_text(d.get("title", ""), d.get("description", ""), d.get("attributes")) for d in docs]
    emb = TextEmbedder(max_features=max_features)
    emb.fit(seg_texts + doc_texts)
    seg_vecs = emb.transform(seg_texts).toarray().astype(np.float32)
    doc_vecs = emb.transform(doc_texts).toarray().astype(np.float32)
    return seg_vecs, doc_vecs, emb, seg_texts, doc_texts


def build_image_embeddings(segments: List[Dict[str, Any]], docs: List[Dict[str, Any]], bins: int) -> Tuple[np.ndarray, np.ndarray]:
    seg_img_vecs = np.stack([image_histogram_embedding(s.get("image_path"), bins=bins) for s in segments], axis=0)
    doc_img_vecs = np.stack([image_histogram_embedding(d.get("image_path"), bins=bins) for d in docs], axis=0)
    return seg_img_vecs, doc_img_vecs


def hybrid_search(
    seg_idx: int,
    seg_text_vecs: np.ndarray,
    doc_text_vecs: np.ndarray,
    seg_img_vecs: np.ndarray,
    doc_img_vecs: np.ndarray,
    w_text: float,
    w_image: float,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # combine sims per doc
    q_t = seg_text_vecs[seg_idx]
    q_i = seg_img_vecs[seg_idx]
    text_sims = (doc_text_vecs @ (q_t / (np.linalg.norm(q_t) + 1e-8))).astype(np.float32)
    image_sims = (doc_img_vecs @ (q_i / (np.linalg.norm(q_i) + 1e-8))).astype(np.float32)
    hybrid = w_text * text_sims + w_image * image_sims
    idx = np.argsort(-hybrid)[:k]
    return idx, hybrid[idx]


def maybe_llm_check(enable: bool, seg_desc: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not enable:
        return [{"doc_id": c["doc_id"], "contradict": False, "rationale": "stub"} for c in candidates]
    return llm_consistency_stub(seg_desc, candidates)


def _select_method(cfg: Dict[str, Any], dataset_root: Optional[Path] = None) -> LinkMethod:
    name = str(cfg.get("method", "TEXT_BM25")).upper()
    default_fields = ["title", "description", "attributes", "specs"]
    fields_cfg = cfg.get("text_fields")
    if isinstance(fields_cfg, list) and fields_cfg:
        fields = fields_cfg
    elif isinstance(fields_cfg, str) and fields_cfg.strip():
        fields = [fields_cfg.strip()]
    else:
        fields = default_fields

    dense_cfg = cfg.get("dense", {}) if isinstance(cfg.get("dense"), dict) else {}
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm"), dict) else {}
    dense_cache_base = dense_cfg.get("cache_dir")
    llm_cache_base = llm_cfg.get("cache_dir")

    if name == "TEXT_BM25":
        k1 = float(cfg.get("bm25", {}).get("k1", 1.5))
        b = float(cfg.get("bm25", {}).get("b", 0.75))
        return BM25Linker(fields=fields, k1=k1, b=b)

    if name == "TEXT_DENSE":
        provider = str(dense_cfg.get("provider", "local")).lower()
        api_key = dense_cfg.get("api_key")
        base_url = dense_cfg.get("base_url")
        return DenseLinker(
            fields=fields,
            model_name=dense_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            normalize_embeddings=bool(dense_cfg.get("normalize_embeddings", True)),
            cache_dir=Path(dense_cache_base) if dense_cache_base else None,
            embed_provider=provider,
            embed_api_key=str(api_key) if isinstance(api_key, str) else None,
            embed_base_url=str(base_url) if isinstance(base_url, str) else None,
            dataset_root=dataset_root,
        )

    if name == "TEXT_DENSE_EXPANSION":
        provider = str(dense_cfg.get("provider", "local")).lower()
        api_key_dense = dense_cfg.get("api_key")
        base_url = dense_cfg.get("base_url")
        api_key_llm = llm_cfg.get("api_key", os.getenv("OPENAI_API_KEY"))
        llm_model = llm_cfg.get("model", "gpt-4o-mini")
        temp = float(llm_cfg.get("temperature", 0.2))
        prov = str(llm_cfg.get("provider", "openai"))
        return QueryExpansionLinker(
            fields=fields,
            model_name=dense_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            normalize_embeddings=bool(dense_cfg.get("normalize_embeddings", True)),
            cache_dir=Path(dense_cache_base) if dense_cache_base else None,
            embed_provider=provider,
            embed_api_key=str(api_key_dense) if isinstance(api_key_dense, str) else None,
            embed_base_url=str(base_url) if isinstance(base_url, str) else None,
            llm_api_key=str(api_key_llm or ""),
            llm_provider=prov,
            llm_model=str(llm_model),
            llm_temperature=temp,
            variant_suffix="qe",
            llm_cache_dir=Path(llm_cache_base) if llm_cache_base else None,
            dataset_root=dataset_root,
        )

    if name == "TEXT_EXACT_FUZZY":
        return TextExactFuzzy(fields=fields)

    if name == "LLM_JUDGE":
        prov = llm_cfg.get("provider", "hf-local")
        model = llm_cfg.get("model", "stub")
        api_key = llm_cfg.get("api_key", "")
        cache_base = Path(llm_cache_base) if llm_cache_base else None
        return LLMLinkJudge(provider=prov, model=model, cache_dir=cache_base, api_key=str(api_key), dataset_root=dataset_root)

    return BM25Linker(fields=fields)


def run_pipeline(
    segments_path: Path,
    docs_path: Path,
    out_links: Path,
    out_meta: Path,
    w_text: float,
    w_image: float,
    k: int,
    m: int,
    min_match_threshold: float,
    min_margin: float,
    tfidf_max_features: int,
    image_hist_bins: int,
    llm_enable: bool,
    seed: int,
    dry_run: bool = False,
    method_cfg: Optional[Dict[str, Any]] = None,
):
    set_seed(seed)
    segments = load_segments(segments_path)
    docs = load_docs(docs_path)

    dataset_root = CacheManager.infer_dataset_root(
        segments_path,
        docs_path,
        (method_cfg or {}).get("dense", {}).get("cache_dir") if isinstance((method_cfg or {}).get("dense"), dict) else None,
    )

    seg_text_vecs, doc_text_vecs, text_emb, seg_texts, doc_texts = build_text_embeddings(segments, docs, tfidf_max_features)
    seg_img_vecs, doc_img_vecs = build_image_embeddings(segments, docs, image_hist_bins)

    links: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []

    method: Optional[LinkMethod] = None
    method_name_used: str = ""
    if method_cfg is not None:
        method_name_used = str(method_cfg.get("method", "TEXT_BM25")).upper()
        method = _select_method(method_cfg, dataset_root=dataset_root)
        method.prepare(docs)

    for si, seg in enumerate(tqdm(segments, desc="Linking segments", unit="seg")):
        seg_id = seg.get("id")
        seg_desc = seg.get("description", "")

        if method is not None:
            pairs = method.score_segment(seg, top_k=k)
            candidates: List[Dict[str, Any]] = []
            for doc_id, score, rationale in pairs:
                d = next((x for x in docs if str(x.get("doc_id")) == doc_id), None)
                candidates.append({
                    "doc_id": doc_id,
                    "score_hybrid": float(score),
                    "score_cross": 0.0,
                    "score_fused": float(score),
                    "text": "",
                    "attrs": d.get("attributes", {}) if d else {},
                    "rationale": rationale,
                })
        else:
            idx, hybrid_scores = hybrid_search(
                si, seg_text_vecs, doc_text_vecs, seg_img_vecs, doc_img_vecs, w_text, w_image, k
            )
            cand_doc_texts = [doc_texts[j] for j in idx]
            cross_scores = dummy_cross_encoder_scores(seg_desc, cand_doc_texts)
            fused = fuse_scores(hybrid_scores, cross_scores, alpha=0.5)
            candidates = []
            for rank, (j, s_h, s_c, s_f) in enumerate(zip(idx, hybrid_scores, cross_scores, fused), start=1):
                d = docs[j]
                candidates.append({
                    "doc_id": d.get("doc_id"),
                    "score_hybrid": float(s_h),
                    "score_cross": float(s_c),
                    "score_fused": float(s_f),
                    "text": cand_doc_texts[rank-1],
                    "attrs": d.get("attributes", {}),
                })

        # LLM consistency (optional) for top-m
        top_m = candidates[:m]
        verdicts = maybe_llm_check(llm_enable, seg_desc, top_m)
        veto_ids = {v["doc_id"] for v in verdicts if v.get("contradict")}

        # Choose best non-vetoed by fused score
        chosen = None
        rationale = None
        status = "unmatched"
        sorted_cands = sorted(candidates, key=lambda x: -x["score_fused"])
        if sorted_cands:
            best = sorted_cands[0]
            second = sorted_cands[1] if len(sorted_cands) > 1 else None
            margin = (best["score_fused"] - (second["score_fused"] if second else 0.0))
            if (best["doc_id"] not in veto_ids):
                chosen = best["doc_id"]
                status = "matched"
                rationale = "ok"
            else:
                status = "unmatched"
                rationale = "low confidence or ambiguity"

        links.append({
            "segment_id": seg_id,
            "matches": [
                {
                    "doc_id": c["doc_id"],
                    "score": round(float(c["score_fused"]), 4),
                    "rationale": next((v["rationale"] for v in verdicts if v["doc_id"] == c["doc_id"]), ""),
                } for c in sorted_cands[:k]
            ],
            "chosen": chosen,
            "status": status,
        })

        if status == "matched" and chosen is not None:
            # find doc
            doc = next((d for d in docs if d.get("doc_id") == chosen), None)
            meta_obj = doc.get("attributes", {}) if doc else {}
            # Rule-based enrich from doc text
            extracted = rule_extract(flatten_doc_text(doc.get("title", ""), doc.get("description", ""), doc.get("attributes"))) if doc else {}
            # merge
            merged = {**meta_obj, **{k: v for k, v in extracted.items() if v is not None}}
            meta_rows.append({
                "segment_id": seg_id,
                "doc_id": chosen,
                "metadata": merged,
            })

        if dry_run:
            break

    out_links.parent.mkdir(parents=True, exist_ok=True)
    out_links.write_text(json.dumps({
        "method": method_name_used,
        "links": links,
    }, indent=2, ensure_ascii=False))
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps({"segments": meta_rows}, indent=2, ensure_ascii=False))