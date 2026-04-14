from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
from tqdm import tqdm

from utils.cache_manager import CacheManager
from ..config import Step6Config
from ..types import RankedSnippet
from ..utils.io import read_json, write_json, write_csv
from ..indexing.builder import build_snippet_index
from ..indexing.snippetizer import Snippetizer
from ..methods.bm25 import BM25SnippetRanker
from ..methods.dense import DenseSnippetRanker
from ..methods.llm_judge import LLMJudgeLinkerMethod
from ..methods.llm_list import LLMListRankerMethod
from ..methods.llm_expansion import LLMExpansionMethod
from ..methods.random import RandomSnippetRanker
from ..methods.cross_encode import CrossEncoderSnippetRanker
from ..methods.rerank import DenseCohereRerankSnippetRanker
from ..methods.clip import ClipSnippetRanker
from ..methods.llm_rerank import DenseLLMRerankSnippetRanker

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def _load_snippet_store(store_path: Path) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    # returns: snippet_id -> record, and snippet_id -> text
    recs: Dict[str, Dict] = {}
    texts: Dict[str, str] = {}
    with store_path.open() as f:
        for line in f:
            obj = json.loads(line)
            recs[obj["snippet_id"]] = obj
            texts[obj["snippet_id"]] = obj["text"]
    return recs, texts


def build_index(catalog: Path, artifacts_dir: Path, config: Step6Config) -> Path:
    return build_snippet_index(catalog, artifacts_dir, config)


def _find_image_path(dataset_root: Path, item_id: str) -> Optional[Path]:
    """Find the image path for an item in segments/crops directory."""
    if dataset_root is None:
        return None
    
    # Extract dataset name from dataset_root (e.g., data/fashion -> fashion)
    # dataset_root is typically already inferred by CacheManager.infer_dataset_root
    # which returns paths like data/fashion, data/movie, etc.
    dataset_name = dataset_root.name or (dataset_root.parts[-1] if dataset_root.parts else None)
    
    if not dataset_name:
        return None
    
    # Construct path: dataset_root/segments/crops/{item_id}.png
    # Since dataset_root is already data/{dataset_name}, we just need segments/crops
    if '.json' in item_id:
        item_id = item_id.split('.json')[0]
    image_path = dataset_root / "segments" / "crops" / f"{item_id}.png"
    if image_path.exists():
        return image_path
    
    # Fallback: try to find any segments/crops directory
    for segments_dir in dataset_root.rglob("segments/crops"):
        image_path = segments_dir / f"{item_id}.png"
        if image_path.exists():
            return image_path
    
    return None


def _collect_snippets_from_doc(
    doc: Dict[str, Any],
    segment_id: str,
    precomputed_snippets: Optional[Dict[str, List[Any]]] = None,
    snippetizer: Optional[Snippetizer] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Collect snippets from a document.
    Returns:
        snippet_texts: snippet_id -> content (for indexing)
        snippet_labels: snippet_id -> text (field + summary for display)
        snippet_metadata: snippet_id -> metadata dict (doc_id, segment_id, field, etc.)
    """
    snippet_texts: Dict[str, str] = {}
    snippet_labels: Dict[str, str] = {}
    snippet_metadata: Dict[str, Dict[str, Any]] = {}
    doc_id = str(doc.get("doc_id"))
    if precomputed_snippets and doc_id in precomputed_snippets:
        for idx, entry in enumerate(precomputed_snippets[doc_id]):
            if isinstance(entry, dict):
                sid = str(entry.get("id") or entry.get("snippet_id") or f"{doc_id}#{idx}")
                # Use content for indexing, text (field + summary) for display
                # Backward compat: if content not present, use text for both
                content = str(entry.get("content", entry.get("text", "")))
                label = str(entry.get("text", content))
                snippet_texts[sid] = content
                snippet_labels[sid] = label
            else:
                sid = f"{doc_id}#{idx}"
                txt = str(entry)
                snippet_texts[sid] = txt
                snippet_labels[sid] = txt
            # Store metadata
            try:
                field_part = sid.split("#", 2)[1] if "#" in sid else ""
            except Exception:
                field_part = ""
            snippet_metadata[sid] = {
                "doc_id": doc_id,
                "segment_id": segment_id,
                "field": field_part,
            }
    else:
        if snippetizer is None:
            raise ValueError("snippetizer must be provided if precomputed_snippets is not available")
        for idx, sp in enumerate(snippetizer.snippetize_doc(doc)):
            sid = sp.snippet_id or f"{doc_id}#{idx}"
            # Use content for indexing/searching, text (field + summary) for display
            snippet_texts[sid] = sp.content
            snippet_labels[sid] = sp.text
            # Store metadata
            snippet_metadata[sid] = {
                "doc_id": doc_id,
                "segment_id": segment_id,
                "field": sp.field,
            }
    return snippet_texts, snippet_labels, snippet_metadata


def rank_for_segment(
    query_text: str,
    segment_id: str,
    doc: Dict[str, Any],
    config: Step6Config,
    dataset_root: Path,
    cache_base: Optional[Path],
    precomputed_snippets: Optional[Dict[str, List[Any]]] = None,
    image_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    # Get snippets from precomputed store if available, otherwise snippetize
    # snippet_texts maps snippet_id -> content (for indexing/searching)
    # snippet_labels maps snippet_id -> text (field + summary, for display in results)
    snippets_path = dataset_root / 'attributes' / 'atomic_attributes.json' if dataset_root else None
    # Note: snippetizer will check atomic_attributes.json internally, so we still create it
    # but _collect_snippets_from_doc will use precomputed_snippets if available
    snippetizer = Snippetizer(config, snippets_json_path=snippets_path) if not (precomputed_snippets and str(doc.get("doc_id")) in precomputed_snippets) else None
    snippet_texts, snippet_labels, snippet_metadata = _collect_snippets_from_doc(
        doc, segment_id, precomputed_snippets, snippetizer
    )
    doc_id = str(doc.get("doc_id"))
    if not snippet_texts:
        return []
    # Select method
    method_name = config.method.name
    if method_name == "bm25":
        method = BM25SnippetRanker(k1=config.method.bm25.k1, b=config.method.bm25.b)
    elif method_name == "dense":
        method = DenseSnippetRanker(
            model_name=config.method.dense.model_name,
            cache_dir=cache_base,
            normalize_embeddings=config.method.dense.normalize_embeddings,
            dataset_root=dataset_root,
            api_base=getattr(config.method.dense, "api_base", None),
            api_key_env=getattr(config.method.dense, "api_key_env", None),
            request_timeout=getattr(config.method.dense, "request_timeout", 60.0),
        )
    elif method_name == "llm":
        method = LLMJudgeLinkerMethod(
            provider=config.method.llm.provider,
            model=config.method.llm.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(config.method.llm, "api_base", None),
            api_key_env=getattr(config.method.llm, "api_key_env", None),
            image_path=image_path,
            require_reason=getattr(config.method.llm, "require_reason", True),
            use_image=getattr(config.method.llm, "use_image", True),
        )
    elif method_name == "llm_list":
        llm_list_cfg = config.method.llm
        method = LLMListRankerMethod(
            provider=llm_list_cfg.provider,
            model=llm_list_cfg.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(llm_list_cfg, "api_base", None),
            api_key_env=getattr(llm_list_cfg, "api_key_env", None),
            image_path=image_path,
            require_reason=getattr(llm_list_cfg, "require_reason", True),
            use_image=getattr(llm_list_cfg, "use_image", True),
        )
    elif method_name == "llm_expansion":
        llm_expansion_cfg = getattr(config.method, "llm_expansion", None) or config.method.llm
        method = LLMExpansionMethod(
            provider=llm_expansion_cfg.provider,
            model=llm_expansion_cfg.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(llm_expansion_cfg, "api_base", None),
            api_key_env=getattr(llm_expansion_cfg, "api_key_env", None),
            image_path=image_path,
            require_reason=getattr(llm_expansion_cfg, "require_reason", True),
            use_image=getattr(llm_expansion_cfg, "use_image", True),
            expansion_model=getattr(llm_expansion_cfg, "expansion_model", None),
            entailment_model=getattr(llm_expansion_cfg, "entailment_model", None),
        )
    elif method_name == "cross_encode":
        method = CrossEncoderSnippetRanker(
            model_name=config.method.cross_encode.model_name,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            batch_size=config.method.cross_encode.batch_size,
            device=getattr(config.method.cross_encode, "device", None),
        )
    elif method_name == "rerank":
        method = DenseCohereRerankSnippetRanker(
            bm25_k1=getattr(config.method.rerank, "bm25_k1", 1.5),
            bm25_b=getattr(config.method.rerank, "bm25_b", 0.75),
            initial_k=config.method.rerank.initial_k,
            rerank_model=config.method.rerank.rerank_model,
            rerank_top_n=config.method.rerank.rerank_top_n,
            rerank_api_key_env=config.method.rerank.rerank_api_key_env,
            request_timeout=config.method.rerank.request_timeout,
        )
    elif method_name == "random":
        method = RandomSnippetRanker()
    elif method_name == "clip":
        clip_cfg = getattr(config.method, "clip", None)
        if clip_cfg is None:
            from ..config import ClipCfg
            clip_cfg = ClipCfg()
        fusion_method = getattr(clip_cfg, "fusion_method", "linear")
        method = ClipSnippetRanker(
            model_name=getattr(clip_cfg, "model_name", "ViT-B-32"),
            pretrained=getattr(clip_cfg, "pretrained", "openai"),
            fusion_method=fusion_method,
            alpha=getattr(clip_cfg, "alpha", 0.5),
            cache_dir=cache_base,
            dataset_root=dataset_root,
            image_path=image_path,
            device=getattr(clip_cfg, "device", None),
            batch_size=getattr(clip_cfg, "batch_size", 32),
        )
    elif method_name == "llm_rerank":
        llm_rerank_cfg = getattr(config.method, "llm_rerank", None)
        llm_cfg = config.method.llm
        method = DenseLLMRerankSnippetRanker(
            dense_model_name=getattr(llm_rerank_cfg, "dense_model_name", "qwen-2.5-8b") if llm_rerank_cfg else "qwen-2.5-8b",
            dense_normalize_embeddings=getattr(llm_rerank_cfg, "dense_normalize_embeddings", True) if llm_rerank_cfg else True,
            dense_api_base=getattr(llm_rerank_cfg, "dense_api_base", None) if llm_rerank_cfg else None,
            dense_api_key_env=getattr(llm_rerank_cfg, "dense_api_key_env", "OPENROUTER_API_KEY") if llm_rerank_cfg else "OPENROUTER_API_KEY",
            dense_request_timeout=getattr(llm_rerank_cfg, "dense_request_timeout", 60.0) if llm_rerank_cfg else 60.0,
            initial_k=getattr(llm_rerank_cfg, "initial_k", 10) if llm_rerank_cfg else 10,
            provider=llm_cfg.provider,
            model=llm_cfg.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(llm_cfg, "api_base", None),
            api_key_env=getattr(llm_cfg, "api_key_env", "OPENAI_API_KEY"),
            image_path=image_path,
            require_reason=getattr(llm_cfg, "require_reason", True),
            use_image=getattr(llm_cfg, "use_image", True),
            rerank_top_n=getattr(llm_rerank_cfg, "rerank_top_n", 5) if llm_rerank_cfg else 5,
        )
    else:
        raise ValueError("method.name must be one of: bm25, dense, llm, llm_list, llm_expansion, cross_encode, rerank, random, clip, llm_rerank")
    # Score
    method.fit(snippet_texts)
    # Score all snippets (not just top-k)
    # For clip method, pass image_path if available
    if method_name == "clip" and hasattr(method, "score"):
        scored = method.score(query_text, topk=len(snippet_texts), image_path=image_path).scores
    else:
        scored = method.score(query_text, topk=len(snippet_texts)).scores
    # Get rationale from method if available (for llm, llm_list, llm_expansion, and llm_rerank methods)
    rationale_map: Dict[str, str] = {}
    if method_name in {"llm", "llm_list", "llm_expansion", "llm_rerank"} and hasattr(method, "rationale_map"):
        rationale_map = getattr(method, "rationale_map", {})
    # Build ranked rows
    ranked_rows: List[Dict[str, Any]] = []
    for rank, (sid, score) in enumerate(scored, start=1):
        meta = snippet_metadata.get(sid, {})
        field_part = meta.get("field", "")
        row = {
            "rank": rank,
            "snippet_id": sid,
            "doc_id": meta.get("doc_id", doc_id),
            "segment_ids": [meta.get("segment_id", segment_id)],
            "score": float(score),
            "text": snippet_labels.get(sid, ""),
            "content": snippet_texts.get(sid, ""),
            "evidence": {"field": field_part, "char_span": None, "kv_key": None},
        }
        # Add rationale if available
        if sid in rationale_map:
            row["rationale"] = rationale_map[sid]
        ranked_rows.append(row)
    return ranked_rows


def rank_all_snippets_together(
    query_text: str,
    all_snippet_texts: Dict[str, str],
    all_snippet_labels: Dict[str, str],
    all_snippet_metadata: Dict[str, Dict[str, Any]],
    config: Step6Config,
    dataset_root: Path,
    cache_base: Optional[Path],
    image_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Rank all snippets together as one unified list.
    """
    if not all_snippet_texts:
        return []
    
    # Select method
    method_name = config.method.name
    if method_name == "bm25":
        method = BM25SnippetRanker(k1=config.method.bm25.k1, b=config.method.bm25.b)
    elif method_name == "dense":
        method = DenseSnippetRanker(
            model_name=config.method.dense.model_name,
            cache_dir=cache_base,
            normalize_embeddings=config.method.dense.normalize_embeddings,
            dataset_root=dataset_root,
            api_base=getattr(config.method.dense, "api_base", None),
            api_key_env=getattr(config.method.dense, "api_key_env", None),
            request_timeout=getattr(config.method.dense, "request_timeout", 60.0),
        )
    elif method_name == "llm":
        method = LLMJudgeLinkerMethod(
            provider=config.method.llm.provider,
            model=config.method.llm.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(config.method.llm, "api_base", None),
            api_key_env=getattr(config.method.llm, "api_key_env", None),
            image_path=image_path,
            require_reason=getattr(config.method.llm, "require_reason", True),
            use_image=getattr(config.method.llm, "use_image", True),
        )
    elif method_name == "llm_list":
        llm_list_cfg = config.method.llm
        method = LLMListRankerMethod(
            provider=llm_list_cfg.provider,
            model=llm_list_cfg.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(llm_list_cfg, "api_base", None),
            api_key_env=getattr(llm_list_cfg, "api_key_env", None),
            image_path=image_path,
            require_reason=getattr(llm_list_cfg, "require_reason", True),
            use_image=getattr(llm_list_cfg, "use_image", True),
        )
    elif method_name == "llm_expansion":
        llm_expansion_cfg = getattr(config.method, "llm_expansion", None) or config.method.llm
        method = LLMExpansionMethod(
            provider=llm_expansion_cfg.provider,
            model=llm_expansion_cfg.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(llm_expansion_cfg, "api_base", None),
            api_key_env=getattr(llm_expansion_cfg, "api_key_env", None),
            image_path=image_path,
            require_reason=getattr(llm_expansion_cfg, "require_reason", True),
            use_image=getattr(llm_expansion_cfg, "use_image", True),
            expansion_model=getattr(llm_expansion_cfg, "expansion_model", None),
            entailment_model=getattr(llm_expansion_cfg, "entailment_model", None),
        )
    elif method_name == "cross_encode":
        method = CrossEncoderSnippetRanker(
            model_name=config.method.cross_encode.model_name,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            batch_size=config.method.cross_encode.batch_size,
            device=getattr(config.method.cross_encode, "device", None),
        )
    elif method_name == "rerank":
        method = DenseCohereRerankSnippetRanker(
            bm25_k1=getattr(config.method.rerank, "bm25_k1", 1.5),
            bm25_b=getattr(config.method.rerank, "bm25_b", 0.75),
            initial_k=config.method.rerank.initial_k,
            rerank_model=config.method.rerank.rerank_model,
            rerank_top_n=config.method.rerank.rerank_top_n,
            rerank_api_key_env=config.method.rerank.rerank_api_key_env,
            request_timeout=config.method.rerank.request_timeout,
        )
    elif method_name == "clip":
        clip_cfg = getattr(config.method, "clip", None)
        if clip_cfg is None:
            from ..config import ClipCfg
            clip_cfg = ClipCfg()
        fusion_method = getattr(clip_cfg, "fusion_method", "linear")
        method = ClipSnippetRanker(
            model_name=getattr(clip_cfg, "model_name", "ViT-B-32"),
            pretrained=getattr(clip_cfg, "pretrained", "openai"),
            fusion_method=fusion_method,
            alpha=getattr(clip_cfg, "alpha", 0.5),
            cache_dir=cache_base,
            dataset_root=dataset_root,
            image_path=image_path,
            device=getattr(clip_cfg, "device", None),
            batch_size=getattr(clip_cfg, "batch_size", 32),
        )
    elif method_name == "llm_rerank":
        llm_rerank_cfg = getattr(config.method, "llm_rerank", None)
        llm_cfg = config.method.llm
        method = DenseLLMRerankSnippetRanker(
            dense_model_name=getattr(llm_rerank_cfg, "dense_model_name", "qwen-2.5-8b") if llm_rerank_cfg else "qwen-2.5-8b",
            dense_normalize_embeddings=getattr(llm_rerank_cfg, "dense_normalize_embeddings", True) if llm_rerank_cfg else True,
            dense_api_base=getattr(llm_rerank_cfg, "dense_api_base", None) if llm_rerank_cfg else None,
            dense_api_key_env=getattr(llm_rerank_cfg, "dense_api_key_env", "OPENROUTER_API_KEY") if llm_rerank_cfg else "OPENROUTER_API_KEY",
            dense_request_timeout=getattr(llm_rerank_cfg, "dense_request_timeout", 60.0) if llm_rerank_cfg else 60.0,
            initial_k=getattr(llm_rerank_cfg, "initial_k", 10) if llm_rerank_cfg else 10,
            provider=llm_cfg.provider,
            model=llm_cfg.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(llm_cfg, "api_base", None),
            api_key_env=getattr(llm_cfg, "api_key_env", "OPENAI_API_KEY"),
            image_path=image_path,
            require_reason=getattr(llm_cfg, "require_reason", True),
            use_image=getattr(llm_cfg, "use_image", True),
            rerank_top_n=getattr(llm_rerank_cfg, "rerank_top_n", 5) if llm_rerank_cfg else 5,
        )
    elif method_name == "random":
        method = RandomSnippetRanker()
    else:
        raise ValueError("method.name must be one of: bm25, dense, llm, llm_list, llm_expansion, cross_encode, rerank, random, clip, llm_rerank")
    
    # Score all snippets together
    method.fit(all_snippet_texts)
    # For clip method, pass image_path if available
    if method_name == "clip" and hasattr(method, "score"):
        scored = method.score(query_text, topk=len(all_snippet_texts), image_path=image_path).scores
    else:
        scored = method.score(query_text, topk=len(all_snippet_texts)).scores
    
    # Get rationale from method if available
    rationale_map: Dict[str, str] = {}
    if method_name in {"llm", "llm_list", "llm_expansion", "llm_rerank"} and hasattr(method, "rationale_map"):
        rationale_map = getattr(method, "rationale_map", {})
    
    # Build ranked rows
    ranked_rows: List[Dict[str, Any]] = []
    for rank, (sid, score) in enumerate(scored, start=1):
        meta = all_snippet_metadata.get(sid, {})
        field_part = meta.get("field", "")
        row = {
            "rank": rank,
            "snippet_id": sid,
            "doc_id": meta.get("doc_id", ""),
            "segment_ids": [meta.get("segment_id", "")],
            "score": float(score),
            "text": all_snippet_labels.get(sid, ""),
            "content": all_snippet_texts.get(sid, ""),
            "evidence": {"field": field_part, "char_span": None, "kv_key": None},
        }
        # Add rationale if available
        if sid in rationale_map:
            row["rationale"] = rationale_map[sid]
        ranked_rows.append(row)
    
    return ranked_rows


def rank_queries(
    queries: Union[Path, List[Dict[str, Any]]],
    segment_links_path: Path,
    catalog_path: Path,
    artifacts_dir: Path,
    config: Step6Config,
    out_json: Path,
    out_csv: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
):
    set_seed(42)
    # Catalog docs used to generate candidate snippets deterministically per run
    from ..utils.io import read_json
    cat = catalog_path
    docs_list: List[Dict] = []
    if cat.is_dir():
        for fp in sorted(cat.glob("*.json")):
            if fp.name == "ground_truth_mapping.json":
                continue
            try:
                obj = read_json(fp)
            except Exception:
                obj = {}
            if "doc_id" not in obj:
                obj["doc_id"] = fp.name
            docs_list.append(obj)
    else:
        obj = read_json(cat)
        docs_list = obj.get("docs", obj if isinstance(obj, list) else [])

    snippets_path = dataset_root / "attributes" / "atomic_attributes.json"
    precomputed_snippets: Dict[str, List[Any]] = {}
    if snippets_path.exists():
        try:
            precomputed_snippets = read_json(snippets_path)
        except Exception:
            precomputed_snippets = {}
    
    # Check which docs still need processing
    docs_to_process = [doc for doc in docs_list if str(doc.get("doc_id", "")) not in precomputed_snippets]
    
    if docs_to_process:
        print(f'[label_selection] Precomputing snippets for {len(docs_to_process)} docs')
        snippets_path.parent.mkdir(parents=True, exist_ok=True)
        snippetizer = Snippetizer(config, snippets_json_path=snippets_path)
        for doc in tqdm(docs_to_process, desc="Precomputing snippets", unit="doc", leave=False):
            doc_id = str(doc.get("doc_id"))
            if not doc_id:
                continue
            # Process this document
            doc_snippets = []
            for sp in snippetizer.snippetize_doc(doc):
                doc_snippets.append({
                    "id": sp.snippet_id, 
                    "text": sp.text,  # field + summary
                    "content": sp.content  # actual chunk
                })
            
            # Add to precomputed_snippets and save immediately
            precomputed_snippets[doc_id] = doc_snippets
            # Save after each document
            write_json(snippets_path, precomputed_snippets)

    # Build mapping from segment_id -> chosen doc_id and doc_id -> [segment_ids]
    # method selection
    dataset_root = CacheManager.infer_dataset_root(
        dataset_root,
        segment_links_path,
        catalog_path,
        artifacts_dir,
    )
    cache_base = artifacts_dir
    method_name = config.method.name
    if method_name == "bm25":
        method = BM25SnippetRanker(k1=config.method.bm25.k1, b=config.method.bm25.b)
    elif method_name == "dense":
        method = DenseSnippetRanker(
            model_name=config.method.dense.model_name,
            cache_dir=cache_base,
            normalize_embeddings=config.method.dense.normalize_embeddings,
            dataset_root=dataset_root,
            api_base=getattr(config.method.dense, "api_base", None),
            api_key_env=getattr(config.method.dense, "api_key_env", None),
            request_timeout=getattr(config.method.dense, "request_timeout", 60.0),
        )
    elif method_name == "llm" or method_name == "llm_rerank":
        method = LLMJudgeLinkerMethod(
            provider=config.method.llm.provider,
            model=config.method.llm.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(config.method.llm, "api_base", None),
            api_key_env=getattr(config.method.llm, "api_key_env", None),
            require_reason=getattr(config.method.llm, "require_reason", True),
            use_image=getattr(config.method.llm, "use_image", True),
        )
    elif method_name == "llm_list":
        # Fallback to llm config if llm_list doesn't exist
        llm_list_cfg = config.method.llm
        print(f"[label_selection] model: {llm_list_cfg.model}")
        print(f"[label_selection] provider: {llm_list_cfg.provider}")
        print(f"[label_selection] api_base: {getattr(llm_list_cfg, 'api_base', None)}")
        print(f"[label_selection] require_reason: {getattr(llm_list_cfg, 'require_reason', True)}")
        method = LLMListRankerMethod(
            provider=llm_list_cfg.provider,
            model=llm_list_cfg.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(llm_list_cfg, "api_base", None),
            api_key_env=getattr(llm_list_cfg, "api_key_env", None),
            require_reason=getattr(llm_list_cfg, "require_reason", True),
        )
    elif method_name == "llm_expansion":
        llm_expansion_cfg = getattr(config.method, "llm_expansion", None) or config.method.llm
        print(f"[label_selection] model: {llm_expansion_cfg.model}")
        print(f"[label_selection] expansion_model: {getattr(llm_expansion_cfg, 'expansion_model', None) or llm_expansion_cfg.model}")
        print(f"[label_selection] entailment_model: {getattr(llm_expansion_cfg, 'entailment_model', None) or llm_expansion_cfg.model}")
        print(f"[label_selection] provider: {llm_expansion_cfg.provider}")
        print(f"[label_selection] api_base: {getattr(llm_expansion_cfg, 'api_base', None)}")
        print(f"[label_selection] require_reason: {getattr(llm_expansion_cfg, 'require_reason', True)}")
        print(f"[label_selection] use_image: {getattr(llm_expansion_cfg, 'use_image', True)}")
        method = LLMExpansionMethod(
            provider=llm_expansion_cfg.provider,
            model=llm_expansion_cfg.model,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            api_base=getattr(llm_expansion_cfg, "api_base", None),
            api_key_env=getattr(llm_expansion_cfg, "api_key_env", None),
            require_reason=getattr(llm_expansion_cfg, "require_reason", True),
            use_image=getattr(llm_expansion_cfg, "use_image", True),
            expansion_model=getattr(llm_expansion_cfg, "expansion_model", None),
            entailment_model=getattr(llm_expansion_cfg, "entailment_model", None),
        )
    elif method_name == "cross_encode":
        method = CrossEncoderSnippetRanker(
            model_name=config.method.cross_encode.model_name,
            cache_dir=cache_base,
            dataset_root=dataset_root,
            batch_size=config.method.cross_encode.batch_size,
            device=getattr(config.method.cross_encode, "device", None),
        )
        
    elif method_name == "rerank":
        method = DenseCohereRerankSnippetRanker(
            bm25_k1=getattr(config.method.rerank, "bm25_k1", 1.5),
            bm25_b=getattr(config.method.rerank, "bm25_b", 0.75),
            initial_k=config.method.rerank.initial_k,
            rerank_model=config.method.rerank.rerank_model,
            rerank_top_n=config.method.rerank.rerank_top_n,
            rerank_api_key_env=config.method.rerank.rerank_api_key_env,
            request_timeout=config.method.rerank.request_timeout,
        )
    elif method_name == "random":
        method = RandomSnippetRanker()
    elif method_name == "clip":
        clip_cfg = getattr(config.method, "clip", None)
        if clip_cfg is None:
            from ..config import ClipCfg
            clip_cfg = ClipCfg()
        fusion_method = getattr(clip_cfg, "fusion_method", "linear")
        method = ClipSnippetRanker(
            model_name=getattr(clip_cfg, "model_name", "ViT-B-32"),
            pretrained=getattr(clip_cfg, "pretrained", "openai"),
            fusion_method=fusion_method,
            alpha=getattr(clip_cfg, "alpha", 0.5),
            cache_dir=cache_base,
            dataset_root=dataset_root,
            image_path='',
            device=getattr(clip_cfg, "device", None),
            batch_size=getattr(clip_cfg, "batch_size", 32),
        )
    else:
        raise ValueError("method.name must be one of: bm25, dense, llm, llm_list, llm_expansion, cross_encode, rerank, random")
    
    # Determine requirement field value based on method and config
    requirement_value = None
    if method_name in {"llm", "llm_list", "llm_expansion"}:
        if method_name == "llm":
            method_cfg_obj = config.method.llm
        elif method_name == "llm_list":
            method_cfg_obj = config.method.llm
        else:  # llm_expansion
            method_cfg_obj = getattr(config.method, "llm_expansion", None) or config.method.llm
        requirement_value = "reason" if getattr(method_cfg_obj, "require_reason", True) else "information"
    
    # fit on subset of snippets belonging to candidate docs
    # Output structure: {query, query_id, method, model, snippets: {item_id: [{"id": snippet_id, "text": text, "relevance": score, "requirement": "reason"|"information"}, ...]}}
    results_json: Dict[str, List[Dict[str, Any]]] = {}
    csv_rows: List[Dict[str, object]] = []

    if isinstance(queries, Path):
        queries_data = read_json(queries).get("queries", [])
    else:
        queries_data = queries
    if not isinstance(queries_data, list):
        raise ValueError("queries must be a list of query objects")
    queries = queries_data
    
    # Track query info for output (assuming single query per file based on file naming)
    query_text = ""
    query_id = ""

    # Load ground-truth object ranking to select only high-rated items per query
    gt_path = dataset_root / "ground_truth_object_ranking.json"
    gt_map: Dict[str, Dict[str, Any]] = {}
    if gt_path.exists():
        try:
            gt_map = read_json(gt_path)
        except Exception:
            gt_map = {}
    def _normalized_qid(qid: str) -> str:
        return qid

    def _ground_truth_docs_for_query(qid: str) -> List[str]:
        raw = gt_map.get(qid) or gt_map.get(_normalized_qid(qid) + ".json") or gt_map.get(_normalized_qid(qid), {})
        if not isinstance(raw, dict):
            print(f"[label_selection] Ground truth object ranking for query {qid} is not a dictionary")
            return []
        return [doc_id for doc_id, score in raw.items() if isinstance(score, (int, float)) and score >= 4]
        
    links = read_json(Path(segment_links_path)).get("links", [])
    doc_to_segs: Dict[str, List[str]] = {}
    seg_to_doc: Dict[str, str] = {}
    for row in links:
        if row.get("status") == "matched" and row.get("chosen"):
            seg_id = str(row["segment_id"])
            doc_id = str(row["chosen"])
            doc_to_segs.setdefault(doc_id, []).append(seg_id)
            seg_to_doc[seg_id] = doc_id
    # Debug: print segments that do not have metadata (chosen doc_id absent in catalog/index)
    present_docs = {str(d.get("doc_id")) for d in docs_list}
    for row in links:
        if row.get("status") == "matched" and row.get("chosen"):
            seg_id = str(row.get("segment_id"))
            doc_id = str(row.get("chosen"))
            if doc_id not in present_docs:
                print(f"[label_selection] Segment '{seg_id}' has no metadata entry (missing doc_id '{doc_id}' in catalog/index)")
    for q in queries:
        qid = str(q["query_id"])
        # Track first query for metadata (assuming single query per file)
        if not query_id:
            query_text = q.get("text", "")
            query_id = qid
        gt_seg_ids = set(_ground_truth_docs_for_query(qid))
        if not gt_seg_ids:
            # No ground-truth items for this query; skip extraction/ranking
            continue

        seg_ids_for_query = [s for s in seg_to_doc if s in gt_seg_ids]

        # Rank snippets for each segment separately
        for seg_id in tqdm(seg_ids_for_query, desc=f"Selecting Labels for Reccomended Items for Query {qid}", unit="segment", leave=False):
            doc_id = seg_to_doc.get(seg_id)
            if not doc_id:
                continue
            doc = next((d for d in docs_list if str(d.get("doc_id")) == doc_id), None)
            if doc is None:
                print(f"[label_selection] Segment '{seg_id}' has no metadata entry (missing doc_id '{doc_id}' in catalog/index)")
                continue
            
            # Find image path for this item
            image_path = _find_image_path(dataset_root, doc_id)
            
            # Rank snippets for this (segment, doc)
            ranked_rows = rank_for_segment(
                q["text"], 
                seg_id, 
                doc, 
                config, 
                dataset_root, 
                cache_base, 
                precomputed_snippets, 
                image_path=image_path
            )
            
            # Convert ranked_rows to RankedSnippet objects
            ranked: List[RankedSnippet] = [
                RankedSnippet(
                    rank=r["rank"],
                    snippet_id=r["snippet_id"],
                    doc_id=r["doc_id"],
                    segment_ids=r["segment_ids"],
                    score=r["score"],
                    text=r["text"],
                    content=r.get("content", r["text"]),  # backward compat
                    evidence=r["evidence"],
                ) 
                for r in ranked_rows
            ]
            
            # Group snippets by item_id (doc_id)
            item_id = str(doc_id)
            if item_id not in results_json:
                results_json[item_id] = []

            # Track seen snippet IDs to avoid duplicates across segments
            seen_snippet_ids = {s["id"] for s in results_json[item_id]}

            for r in ranked:
                snippet_entry = {
                    "id": r.snippet_id,
                    "text": r.text,
                    "relevance": float(r.score),
                }
                # Add requirement field if method is llm, llm_list, or llm_expansion
                if requirement_value is not None:
                    snippet_entry["requirement"] = requirement_value

                # Add rationale if available (from ranked_rows)
                rationale = None
                for row in ranked_rows:
                    if row.get("snippet_id") == r.snippet_id:
                        rationale = row.get("rationale")
                        break
                if rationale:
                    snippet_entry["rationale"] = str(rationale)

                # Only add if not already present (use highest relevance if duplicate)
                if r.snippet_id not in seen_snippet_ids:
                    results_json[item_id].append(snippet_entry)
                    seen_snippet_ids.add(r.snippet_id)
                else:
                    # Update if this relevance is higher
                    for existing in results_json[item_id]:
                        if existing["id"] == r.snippet_id and r.score > existing["relevance"]:
                            existing["relevance"] = float(r.score)
                            # Also update requirement if it exists
                            if requirement_value is not None:
                                existing["requirement"] = requirement_value
                            # Update rationale if available
                            if rationale:
                                existing["rationale"] = str(rationale)
                            break

            # Add to CSV rows
            for r in ranked:
                csv_rows.append({
                    "query_id": qid,
                    "segment_id": seg_id,
                    "method": method_name,
                    "rank": r.rank,
                    "snippet_id": r.snippet_id,
                    "doc_id": r.doc_id,
                    "segment_ids": ",".join(r.segment_ids),
                    "score": r.score,
                })

    # Get model name from config
    model_name = ""
    if method_name in {"llm", "llm_list", "llm_expansion"}:
        if method_name == "llm":
            model_name = getattr(config.method.llm, "model", "")
        elif method_name == "llm_list":
            llm_list_cfg = config.method.llm
            model_name = getattr(llm_list_cfg, "model", "")
        else:  # llm_expansion
            llm_expansion_cfg = getattr(config.method, "llm_expansion", None) or config.method.llm
            model_name = getattr(llm_expansion_cfg, "model", "")
    elif method_name == "dense":
        model_name = getattr(config.method.dense, "model_name", "")
    
    # Wrap results with query metadata to match object ranker structure
    output_json = {
        "query": query_text,
        "query_id": query_id,
        "method": method_name.upper(),
        "model": model_name,
        "snippets": results_json,
    }
    write_json(out_json, output_json)
    # if out_csv is not None:
    #     write_csv(
    #         out_csv,
    #         csv_rows,
    #         ["query_id", "segment_id", "method", "rank", "snippet_id", "doc_id", "segment_ids", "score"],
    #     )


