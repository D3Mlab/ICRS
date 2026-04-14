from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils.cache_manager import CacheManager
from utils.io_manager import IOManager, OutputSpec
from .helpers import build_segment_entry
from .methods.base import BatchResult, ObjectRankingMethod
from .methods.fusion_dense import FusionDenseObjectRanker
from .methods.dense import DenseObjectRanker
from .methods.umbrella_llm import UmbrellaLLMObjectRanker
from .methods.umbrella_vlm import UmbrellaVLMObjectRanker
from .methods.umbrella_llm_list import UmbrellaLLMListObjectRanker
from .methods.umbrella_vlm_list import UmbrellaVLMListObjectRanker
from .methods.reranker import RerankObjectRanker
from .methods.vison import VisonObjectRanker
from .methods.vison_list import VisonListObjectRanker
from .methods.bm25 import BM25ObjectRanker
from .methods.random import RandomObjectRanker

@dataclass
class BatchRunDetail:
    batch_index: int
    ids: List[str]
    runs_completed: int
    cache_path: Optional[Path] = None


@dataclass
class SegmentRankingResult:
    scores: Dict[str, float]
    rankings: List[Dict[str, Any]]
    histories: Dict[str, List[float]] = field(default_factory=dict)
    field_histories: Dict[str, List[Any]] = field(default_factory=dict)
    batch_details: List[BatchRunDetail] = field(default_factory=list)


def _select_method(
    cfg: Dict[str, Any],
    dataset_root: Optional[Path],
    llm_cache_base: Optional[Path] = None,
) -> ObjectRankingMethod:
    name = str(cfg.get("method", "UMBRELLA_LLM")).upper()
    llm_cfg = dict(cfg.get("llm", {}))
    model = llm_cfg.get("model", "gpt-4o-mini")
    provider = llm_cfg.get("provider", "openai")
    api_key = os.environ.get("OPENROUTER_API_KEY", "") if provider == "openrouter" else os.environ.get("OPENAI_API_KEY", "") if provider == "openai" else os.environ.get("GOOGLE_API_KEY", "") if provider == "google" else ""
    if not api_key:
        raise ValueError("Object ranker method requires llm.api_key")
    temperature = float(llm_cfg.get("temperature", 0.2))
    temp_jitter = float(llm_cfg.get("temperature_jitter", 0.3))
    base_url = llm_cfg.get("base_url") or llm_cfg.get("api_base")
    header_cfg = llm_cfg.get("headers")
    default_headers = header_cfg if isinstance(header_cfg, dict) else None
    
    print(f'CRS method: {name}')
    if 'LM' in name:
        print(f'Model: {model}')
        print(f'Provider: {provider}')

    if name == "UMBRELLA_LLM":
        method: ObjectRankingMethod = UmbrellaLLMObjectRanker(
            api_key=api_key,
            model=model,
            temperature=temperature,
            provider=provider,
            temperature_jitter=temp_jitter,
            api_base=base_url,
            default_headers=default_headers,
        )
    elif name == "UMBRELLA_LLM_LIST":
        method = UmbrellaLLMListObjectRanker(
            api_key=api_key,
            model=model,
            temperature=temperature,
            provider=provider,
            temperature_jitter=temp_jitter,
            api_base=base_url,
            default_headers=default_headers,
        )
    elif name == "UMBRELLA_VLM":
        method = UmbrellaVLMObjectRanker(
            api_key=api_key,
            model=model,
            temperature=temperature,
            provider=provider,
            temperature_jitter=temp_jitter,
            api_base=base_url,
            default_headers=default_headers,
            dataset_root=dataset_root,
        )
    elif name == "UMBRELLA_VLM_LIST":
        method = UmbrellaVLMListObjectRanker(
            api_key=api_key,
            model=model,
            temperature=temperature,
            provider=provider,
            temperature_jitter=temp_jitter,
            api_base=base_url,
            default_headers=default_headers,
            dataset_root=dataset_root,
        )
    elif name == "VISON":
        method = VisonObjectRanker(
            api_key=api_key,
            model=model,
            temperature=temperature,
            provider=provider,
            temperature_jitter=temp_jitter,
            api_base=base_url,
            default_headers=default_headers,
            dataset_root=dataset_root,
        )
    elif name == "VISON_LIST":
        method = VisonListObjectRanker(
            api_key=api_key,
            model=model,
            temperature=temperature,
            provider=provider,
            temperature_jitter=temp_jitter,
            api_base=base_url,
            default_headers=default_headers,
            dataset_root=dataset_root,
        )
    elif name == "RERANK":
        rerank_cfg = dict(cfg.get("rerank", {}))
        method = RerankObjectRanker(
            bm25_k1=float(rerank_cfg.get("bm25_k1", rerank_cfg.get("k1", 1.5))),
            bm25_b=float(rerank_cfg.get("bm25_b", rerank_cfg.get("b", 0.75))),
            initial_k=int(rerank_cfg.get("initial_k", 100)),
            rerank_api_key=rerank_cfg.get("rerank_api_key"),
            rerank_api_key_env=rerank_cfg.get("rerank_api_key_env", "COHERE_API_KEY"),
            rerank_model=rerank_cfg.get("rerank_model", "rerank-3.5"),
            rerank_top_n=int(rerank_cfg.get("rerank_top_n", 50)),
            request_timeout=float(rerank_cfg.get("request_timeout", 30.0)),
        )
    elif name == "FUSION_DENSE":
        dense_cfg = dict(cfg.get("fusion_dense", {}))
        aggregation = dense_cfg.get("aggregation") or cfg.get("fusion", {}).get("aggregation", "mean")
        method = FusionDenseObjectRanker(
            model_name=dense_cfg.get("model_name", "qwen/qwen3-embedding-8b"),
            api_base=dense_cfg.get("api_base"),
            api_key=dense_cfg.get("api_key"),
            api_key_env=dense_cfg.get("api_key_env", "OPENROUTER_API_KEY"),
            aggregation=str(aggregation),
            normalize_embeddings=bool(dense_cfg.get("normalize_embeddings", True)),
            request_timeout=float(dense_cfg.get("request_timeout", 60.0)),
            batch_size=int(dense_cfg.get("batch_size", 64)),
        )
    elif name == "DENSE":
        dense_cfg = dict(cfg.get("dense", {}))
        method = DenseObjectRanker(
            model_name=dense_cfg.get("model_name", "qwen/qwen3-embedding-8b"),
            api_base=dense_cfg.get("api_base"),
            api_key=dense_cfg.get("api_key"),
            api_key_env=dense_cfg.get("api_key_env", "OPENROUTER_API_KEY"),
            normalize_embeddings=bool(dense_cfg.get("normalize_embeddings", True)),
            request_timeout=float(dense_cfg.get("request_timeout", 60.0)),
            batch_size=int(dense_cfg.get("batch_size", 64)),
        )
    elif name == "BM25":
        bm25_cfg = dict(cfg.get("bm25", {}))
        method = BM25ObjectRanker(
            k1=float(bm25_cfg.get("k1", 1.5)),
            b=float(bm25_cfg.get("b", 0.75)),
        )
    elif name == "RANDOM":
        method = RandomObjectRanker()
    else:
        raise ValueError(f"Unsupported object ranking method: {name}")

    try:
        configure = getattr(method, "configure_cache", None)
        if callable(configure):
            configure(
                module="item_recommendation",
                method=name.lower(),
                dataset_root=dataset_root,
                base_dir=llm_cache_base,
                variant=None,
                model_name=getattr(method, "model", None),
            )
    except Exception:
        pass

    return method


def _normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _slugify_value(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = []
    for ch in value.strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or ""


def rank_segments(
    user_query: str,
    segments: List[Dict[str, Any]],
    batch_size: int = 40,
    self_consistency_k: int = 1,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
    method_cfg: Optional[Dict[str, Any]] = None,
    query_id: Optional[str] = None,
) -> SegmentRankingResult:
    if not isinstance(segments, list):
        raise ValueError("segments must be a list of dictionaries")
    if not segments:
        return SegmentRankingResult(scores={}, rankings=[])

    cfg = dict(method_cfg or {})
    llm_cache_base = cfg.get("llm", {}).get("cache_dir") if isinstance(cfg.get("llm"), dict) else None
    dataset_root = CacheManager.infer_dataset_root(cache_dir, cfg.get("cache_dir"), llm_cache_base, cfg.get("dataset_root"))

    if cache_dir is not None:
        step_cache_dir = Path(cache_dir)
    elif cfg.get("cache_dir"):
        step_cache_dir = Path(str(cfg.get("cache_dir")))
    else:
        step_cache_dir = (
            dataset_root
            / CacheManager.CACHE_DIR_NAME  # type: ignore[attr-defined]
            / "item_recommendation"
        )
    step_cache_dir.mkdir(parents=True, exist_ok=True)

    method = _select_method(cfg, dataset_root=dataset_root, llm_cache_base=llm_cache_base)
    method.prepare(segments)


    def _normalize_seg_id(value: str) -> str:
        val = str(value or "").strip()
        if val.endswith(".json") or val.endswith(".png") or val.endswith(".jpg") or val.endswith(".jpeg"):
            return val.rsplit(".", 1)[0]
        return val
    # Filter segments by ground-truth list if available
    allowed_ids: Optional[set[str]] = None
    if query_id and dataset_root:
        gt_path = Path(dataset_root) / "ground_truth_object_ranking.json"
        if gt_path.exists():
            try:
                gt_map = json.loads(gt_path.read_text())
                raw_entry = gt_map.get(query_id) or gt_map.get(_normalize_seg_id(query_id)) or gt_map.get(f"{_normalize_seg_id(query_id)}.json")
                if isinstance(raw_entry, dict):
                    allowed_raw = set(raw_entry.keys())
                    allowed_norm = {_normalize_seg_id(k) for k in allowed_raw}
                    allowed_ids = allowed_raw | allowed_norm
                else:
                    allowed_ids = set()
            except Exception:
                allowed_ids = set()
        else:
            allowed_ids = set()

    entries: List[Dict[str, Any]] = []
    metadata_lookup: Dict[str, Dict[str, Any]] = {}
    for seg in tqdm(segments, desc="Preparing segments", unit="segment", disable=not verbose):
        entry = build_segment_entry(seg)
        sid = entry.get("id")
        if not sid:
            continue
        sid_norm = _normalize_seg_id(sid)
        if allowed_ids is not None:
            if sid not in allowed_ids and sid_norm not in allowed_ids:
                continue
        entries.append(entry)
        metadata_lookup[sid] = {
            "doc_id": seg.get("doc_id"),
            "segment_id": sid,
            "metadata": seg.get("metadata", {}),
        }

    if not entries:
        return SegmentRankingResult(scores={}, rankings=[])

    method_name = str(cfg.get("method", "UMBRELLA_LLM")).upper()
    list_methods = {"UMBRELLA_LLM_LIST", "UMBRELLA_VLM_LIST", "VISON_LIST"}
    if method_name in list_methods:
        effective_batch_size = len(entries)
    else:
        effective_batch_size = max(1, int(batch_size))
    total_batches = (len(entries) + effective_batch_size - 1) // effective_batch_size

    module_name = "item_recommendation"
    # Get model name - check model_name attribute first (for cross-encoder, dense), then model (for LLM)
    model_name = str(getattr(method, "model_name", None) or getattr(method, "model", "unknown"))
    cache_base_dirs = [cache_dir, cfg.get("cache_dir"), llm_cache_base]

    existing_records, _ = CacheManager.load_llm_cache(
        module=module_name,
        method=method_name,
        model_name=model_name,
        dataset=dataset_root,
        base_dirs=cache_base_dirs,
    )

    CACHE_NAMESPACE = "item_recommendation"

    cache_map: Dict[tuple[str, int], Dict[str, Any]] = {}
    for record in existing_records:
        if record.get("namespace") != CACHE_NAMESPACE:
            continue
        key_obj = record.get("key") or {}
        batch_id = key_obj.get("batch_id")
        run_idx = key_obj.get("run")
        if not batch_id or run_idx is None:
            continue
        try:
            content_obj = json.loads(record.get("content", "{}"))
        except Exception:
            continue
        cache_map[(str(batch_id), int(run_idx))] = content_obj

    score_history: Dict[str, List[float]] = {entry["id"]: [] for entry in entries}
    rationale_history: Dict[str, List[Any]] = {entry["id"]: [] for entry in entries}
    field_history: Dict[str, List[Any]] = {entry["id"]: [] for entry in entries}
    batch_details: List[BatchRunDetail] = []

    # set to a single batch witj all entries if the method is 'rerank'
    if method_name == "RERANK":
        effective_batch_size = len(entries)
        total_batches = 1
    
    batch_range = range(0, len(entries), effective_batch_size)
    batch_iter = tqdm(
        batch_range,
        total=total_batches,
        desc="Ranking batches",
        unit="batch",
        disable=not verbose,
    )
    for batch_index, start in enumerate(batch_iter, start=1):
        batch = entries[start : start + effective_batch_size]
        ids_list = [entry["id"] for entry in batch]

        batch_key_raw = _normalize_query(user_query) + "|" + ",".join(ids_list)
        batch_id = hashlib.sha256(batch_key_raw.encode("utf-8")).hexdigest()

        runs_completed = 0
        for run_idx in range(1, 2):
            cached_entry = cache_map.get((batch_id, run_idx))
            if cached_entry is None:
                base_temp = float(getattr(method, "temperature", 0.0))
                temp_used = method.temperature_for_run(base_temp, run_idx)
                batch_result: BatchResult = method.score_batch(
                    user_query=user_query,
                    batch=batch,
                    temperature=temp_used,
                    run_index=run_idx,
                )
                cached_entry = {
                    "scores": batch_result.scores,
                    "parsed_items": batch_result.parsed_items,
                    "field_scores": batch_result.field_scores or {},
                    "temperature": batch_result.temperature if batch_result.temperature is not None else temp_used,
                }
                CacheManager.append_llm_record(
                    module=module_name,
                    method=method_name,
                    model_name=model_name,
                    namespace=CACHE_NAMESPACE,
                    key={"batch_id": batch_id, "run": run_idx},
                    content=json.dumps(cached_entry, ensure_ascii=False),
                    record_meta={"temperature": cached_entry["temperature"]},
                    dataset=dataset_root,
                    base_dirs=cache_base_dirs,
                )
                cache_map[(batch_id, run_idx)] = cached_entry
            runs_completed += 1

            score_map = cached_entry.get("scores", {})
            parsed_items = cached_entry.get("parsed_items", [])
            field_score_map = cached_entry.get("field_scores", {}) or {}
            rationale_map = {
                entry.get("id"): entry for entry in parsed_items if isinstance(entry, dict)
            }

            for sid in ids_list:
                score_value = float(score_map.get(sid, 0.0))
                score_history.setdefault(sid, []).append(score_value)

                rationale_entry = rationale_map.get(sid)
                rationale_history.setdefault(sid, []).append(
                    rationale_entry.get("rationale") if rationale_entry else None
                )

                field_data = field_score_map.get(sid)
                field_history.setdefault(sid, []).append(field_data)

        batch_details.append(
            BatchRunDetail(
                batch_index=batch_index,
                ids=ids_list,
                runs_completed=runs_completed,
                cache_path=None,
            )
        )

    final_scores: Dict[str, float] = {}
    for sid, history in score_history.items():
        if history:
            final_scores[sid] = float(sum(history) / len(history))
        else:
            final_scores[sid] = 0.0

    rankings: List[Dict[str, Any]] = []
    for sid, score in final_scores.items():
        meta = metadata_lookup.get(sid, {})
        rankings.append({
            "segment_id": sid,
            "score": score,
            "doc_id": meta.get("doc_id"),
            "metadata": meta.get("metadata"),
            "history": score_history.get(sid, []),
            "field_history": field_history.get(sid, []),
            "rationales": rationale_history.get(sid, []),
        })

    rankings.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return SegmentRankingResult(
        scores=final_scores,
        rankings=rankings,
        histories=score_history,
        field_histories=field_history,
        batch_details=batch_details,
    )


def rank_from_meta_json(
    user_query: str,
    meta_json_path: Path,
    batch_size: int = 40,
    self_consistency_k: int = 1,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
    method_cfg: Optional[Dict[str, Any]] = None,
    query_id: Optional[str] = None,
) -> SegmentRankingResult:
    data = json.loads(meta_json_path.read_text())
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        segments = data["segments"]
    elif isinstance(data, list):
        segments = data
    else:
        segments = []
        for key in data.keys():
            if key.endswith(".json"):
                #load the meta_data from raw folder
                folder_path = meta_json_path.parent / "raw"
                meta_path = folder_path / key
                meta_data = json.loads(meta_path.read_text())
                segments.append({
                    "segment_id": key.split(".")[0],
                    "doc_id": key,
                    "metadata": meta_data,
                })
    cfg = dict(method_cfg or {})
    cfg.setdefault("dataset_root", str(CacheManager.infer_dataset_root(meta_json_path)))
    return rank_segments(
        user_query=user_query,
        segments=segments,
        batch_size=batch_size,
        self_consistency_k=self_consistency_k,
        cache_dir=cache_dir,
        verbose=verbose,
        method_cfg=cfg,
        query_id=query_id,
    )


class ObjectRanker:
    """High-level wrapper to run object ranking and persist results via IOManager."""

    def __init__(self, io_manager: Optional[IOManager] = None, config_path: Optional[Path] = None) -> None:
        self.io: IOManager = io_manager or IOManager()
        self.config_path: Optional[Path] = config_path
        self._config: Optional[Dict[str, Any]] = None

    def _load_config(self) -> Dict[str, Any]:
        if self._config is not None:
            return self._config
        cfg: Dict[str, Any] = {}
        path = self.config_path or Path("configs/item_recommendation.yaml")
        if path.exists():
            try:
                if path.suffix.lower() in {".yaml", ".yml"}:
                    import yaml  # type: ignore
                    cfg = yaml.safe_load(path.read_text()) or {}
                else:
                    cfg = json.loads(path.read_text() or "{}")
            except Exception:
                cfg = {}
        self._config = cfg if isinstance(cfg, dict) else {}
        return self._config

    def _method_and_model(self, cfg: Dict[str, Any]) -> tuple[str, Optional[str]]:
        method_name = str((cfg or {}).get("method", "UMBRELLA_LLM")).upper()
        llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
        model: Optional[str] = None
        if isinstance(llm_cfg, dict):
            model = llm_cfg.get("model")
        return method_name, model

    def rank(
        self,
        *,
        user_query: str,
        query_id: Optional[str] = None,
        meta_json: Optional[Path],
        out_json: Path,
        batch_size: Optional[int] = None,
        self_consistency_k: int = 1,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        linker_method: Optional[str] = None,
        dataset_root: Optional[Path] = None,
    ) -> Dict[str, Any]:
        cfg = self._load_config()
        method_name, model_name = self._method_and_model(cfg)
        inferred_dataset = CacheManager.infer_dataset_root(meta_json, dataset_root)
        dataset_name = inferred_dataset.name or (inferred_dataset.parts[-1] if inferred_dataset.parts else "default")
        effective_batch_size = batch_size or int(cfg.get("batch_size", 40)) if isinstance(cfg, dict) else (batch_size or 40)
        effective_self_k = int(cfg.get("self_consistency_k", self_consistency_k)) if isinstance(cfg, dict) else self_consistency_k

        def _generate() -> Dict[str, Any]:
            # Prefer pulling linker metadata via IOManager when a method is provided
            segments_list: Optional[List[Dict[str, Any]]] = None
            if linker_method:
                bundle = self.io.load_outputs(module="linker", method=linker_method, dataset=inferred_dataset)
                if bundle is not None:
                    links_payload = bundle.outputs.get("links")
                    # The linker links JSON is an object: {"method": ..., "links": [...]}
                    link_rows: List[Dict[str, Any]] = []
                    if isinstance(links_payload, dict) and isinstance(links_payload.get("links"), list):
                        link_rows = [row for row in links_payload.get("links", []) if isinstance(row, dict)]
                    elif isinstance(links_payload, list):
                        link_rows = [row for row in links_payload if isinstance(row, dict)]

                    segments_list = []
                    meta_dir = inferred_dataset / "attributes" / 'raw'
                    for link in tqdm(link_rows, desc="Loading metadata for segments", unit="segment", disable=not verbose):
                        seg_id = link.get("segment_id")
                        chosen = link.get("chosen")
                        matches = link.get("matches") if isinstance(link.get("matches"), list) else []
                        chosen_doc = chosen if isinstance(chosen, str) and chosen else (matches[0].get("doc_id") if matches else None)
                        if not seg_id or not chosen_doc:
                            continue
                        meta_path = meta_dir / str(chosen_doc)
                        try:
                            meta_obj = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                        except Exception:
                            meta_obj = {}
                        segments_list.append({
                            "segment_id": str(seg_id),
                            "doc_id": str(chosen_doc),
                            "metadata": meta_obj,
                        })
        
            if segments_list is not None:
                cfg_local = dict(cfg or {})
                cfg_local.setdefault("dataset_root", str(inferred_dataset))
                result = rank_segments(
                    user_query=user_query,
                    segments=segments_list,
                    batch_size=effective_batch_size,
                    self_consistency_k=effective_self_k,
                    cache_dir=cache_dir,
                    verbose=verbose,
                    method_cfg=cfg_local,
                    query_id=query_id,
                )
            else:
                meta_json = inferred_dataset / "attributes" / "ground_truth_mapping.json"
                result = rank_from_meta_json(
                    user_query=user_query,
                    meta_json_path=meta_json,
                    batch_size=effective_batch_size,
                    self_consistency_k=effective_self_k,
                    cache_dir=cache_dir,
                    verbose=verbose,
                    method_cfg=cfg,
                    query_id=query_id,
                )
            payload = {
                "query": user_query,
                "query_id": query_id,
                "meta_json": str(meta_json) if meta_json else None,
                "method": method_name,
                "model": model_name,
                "scores": result.scores,
                "rankings": result.rankings,
            }
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            return {"rankings": payload}

        meta_str = str(meta_json) if meta_json else None
        metadata = {
            "query_id": str(query_id) if query_id else "",
            "query": str(user_query or ""),
            "method": method_name,
            "model": model_name or "",
        }
        if meta_str:
            metadata["meta_json"] = meta_str

        bundle = self.io.get_or_generate(
            module="item_recommendation",
            method=method_name,
            dataset=dataset_name,
            specs={
                "rankings": OutputSpec(
                    format="json",
                    mirror_path=out_json,
                    metadata=metadata,
                )
            },
            generator=_generate,
        )
        payload = bundle.outputs.get("rankings") if hasattr(bundle, "outputs") else None
        snapshot_path = self._write_query_snapshot(
            dataset_name=dataset_name,
            method_name=method_name,
            query_id=query_id,
            query_text=user_query,
            payload=payload,
        )
        return {
            "method": method_name,
            "model": model_name,
            "out_json": out_json,
            "payload": payload,
            "snapshot_path": snapshot_path,
        }

    def _write_query_snapshot(
        self,
        *,
        dataset_name: str,
        method_name: str,
        query_id: Optional[str],
        query_text: str,
        payload: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if not payload:
            return None
        method_slug = _slugify_value(method_name) or method_name
        dataset_slug = _slugify_value(dataset_name) or dataset_name or "default"
        if query_id:
            query_slug = _slugify_value(query_id)
        else:
            normalized_query = _normalize_query(query_text or "")
            digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:10]
            query_slug = f"query_{digest}"
        if not query_slug:
            query_slug = "query"
        base_dir = Path(self.io.base_dir)
        snapshot_dir = base_dir / dataset_slug / "item_recommendation" / method_slug
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"{query_slug}_rankings.json"
        try:
            snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception:
            return None
        return str(snapshot_path)

