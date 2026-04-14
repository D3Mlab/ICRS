from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

from utils.cache_manager import CacheManager
from utils.io_manager import IOManager, OutputSpec

from ..config import Step6Config
from ..utils.io import stable_hash
from .runner import build_index, rank_queries


class SnippetRanker:
    def __init__(self, io_manager: Optional[IOManager] = None, config_path: Optional[Path] = None) -> None:
        self.io: IOManager = io_manager or IOManager()
        self.config_path: Optional[Path] = config_path

    def _load_config(self, path: Optional[Path]) -> Step6Config:
        cfg_path = path or Path("configs/label_selection.yaml")
        if str(cfg_path).lower().endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore

                cfg_obj = yaml.safe_load(cfg_path.read_text())
            except Exception:
                cfg_obj = json.loads(cfg_path.read_text())
        else:
            cfg_obj = json.loads(cfg_path.read_text())

        cfg = Step6Config()
        if isinstance(cfg_obj, dict):
            if "index" in cfg_obj:
                idx = cfg_obj["index"]
                cfg.index.granularity = idx.get("granularity", cfg.index.granularity)
                cfg.index.fields_include = idx.get("fields_include", cfg.index.fields_include)
                if "window" in idx:
                    cfg.index.window.size = idx["window"].get("size", cfg.index.window.size)
                    cfg.index.window.stride = idx["window"].get("stride", cfg.index.window.stride)
                if "normalize" in idx:
                    n = idx["normalize"]
                    cfg.index.normalize.lower = n.get("lower", cfg.index.normalize.lower)
                    cfg.index.normalize.deaccent = n.get("deaccent", cfg.index.normalize.deaccent)
                    cfg.index.normalize.strip_punct = n.get("strip_punct", cfg.index.normalize.strip_punct)
            if "method" in cfg_obj:
                m = cfg_obj["method"]
                cfg.method.name = m.get("name", cfg.method.name)
                if "bm25" in m:
                    cfg.method.bm25.k1 = m["bm25"].get("k1", cfg.method.bm25.k1)
                    cfg.method.bm25.b = m["bm25"].get("b", cfg.method.bm25.b)
                if "dense" in m:
                    cfg.method.dense.model_name = m["dense"].get("model_name", cfg.method.dense.model_name)
                    cfg.method.dense.normalize_embeddings = m["dense"].get("normalize_embeddings", cfg.method.dense.normalize_embeddings)
                if "llm" in m:
                    cfg.method.llm.provider = m["llm"].get("provider", cfg.method.llm.provider)
                    cfg.method.llm.model = m["llm"].get("model", cfg.method.llm.model)
                    cfg.method.llm.max_retries = m["llm"].get("max_retries", cfg.method.llm.max_retries)
                    cfg.method.llm.timeout_s = m["llm"].get("timeout_s", cfg.method.llm.timeout_s)
                    if "require_reason" in m["llm"]:
                        cfg.method.llm.require_reason = m["llm"].get("require_reason", True)
                    if "use_image" in m["llm"]:
                        cfg.method.llm.use_image = m["llm"].get("use_image", True)
                if "llm_list" in m:
                    if not hasattr(cfg.method, "llm_list"):
                        from ..config import LLMCfg
                        cfg.method.llm_list = LLMCfg()
                    cfg.method.llm_list.provider = m["llm_list"].get("provider", cfg.method.llm_list.provider)
                    cfg.method.llm_list.model = m["llm_list"].get("model", cfg.method.llm_list.model)
                    cfg.method.llm_list.max_retries = m["llm_list"].get("max_retries", cfg.method.llm_list.max_retries)
                    cfg.method.llm_list.timeout_s = m["llm_list"].get("timeout_s", cfg.method.llm_list.timeout_s)
                    if "require_reason" in m["llm_list"]:
                        cfg.method.llm_list.require_reason = m["llm_list"].get("require_reason", True)
                    if "use_image" in m["llm_list"]:
                        cfg.method.llm_list.use_image = m["llm_list"].get("use_image", True)
                if "llm_expansion" in m:
                    if not hasattr(cfg.method, "llm_expansion"):
                        from ..config import LLMCfg
                        cfg.method.llm_expansion = LLMCfg()
                    cfg.method.llm_expansion.provider = m["llm_expansion"].get("provider", cfg.method.llm_expansion.provider)
                    cfg.method.llm_expansion.model = m["llm_expansion"].get("model", cfg.method.llm_expansion.model)
                    cfg.method.llm_expansion.max_retries = m["llm_expansion"].get("max_retries", cfg.method.llm_expansion.max_retries)
                    cfg.method.llm_expansion.timeout_s = m["llm_expansion"].get("timeout_s", cfg.method.llm_expansion.timeout_s)
                    if "require_reason" in m["llm_expansion"]:
                        cfg.method.llm_expansion.require_reason = m["llm_expansion"].get("require_reason", True)
                    if "use_image" in m["llm_expansion"]:
                        cfg.method.llm_expansion.use_image = m["llm_expansion"].get("use_image", True)
                    if "expansion_model" in m["llm_expansion"]:
                        cfg.method.llm_expansion.expansion_model = m["llm_expansion"].get("expansion_model", None)
                    if "entailment_model" in m["llm_expansion"]:
                        cfg.method.llm_expansion.entailment_model = m["llm_expansion"].get("entailment_model", None)
                if "clip" in m:
                    if not hasattr(cfg.method, "clip"):
                        from ..config import ClipCfg
                        cfg.method.clip = ClipCfg()
                    cfg.method.clip.model_name = m["clip"].get("model_name", cfg.method.clip.model_name)
                    cfg.method.clip.pretrained = m["clip"].get("pretrained", cfg.method.clip.pretrained)
                    cfg.method.clip.fusion_method = m["clip"].get("fusion_method", cfg.method.clip.fusion_method)
                    cfg.method.clip.alpha = m["clip"].get("alpha", cfg.method.clip.alpha)
                    cfg.method.clip.device = m["clip"].get("device", cfg.method.clip.device)
                    cfg.method.clip.batch_size = m["clip"].get("batch_size", cfg.method.clip.batch_size)
                    if "require_reason" in m["clip"]:
                        cfg.method.clip.require_reason = m["clip"].get("require_reason", True)
                    if "summarize_conversation" in m["clip"]:
                        cfg.method.clip.summarize_conversation = m["clip"].get("summarize_conversation", True)
                    if "gemini_model" in m["clip"]:
                        cfg.method.clip.gemini_model = m["clip"].get("gemini_model", "gemini-2.0-flash-exp")
                    if "gemini_api_key" in m["clip"]:
                        cfg.method.clip.gemini_api_key = m["clip"].get("gemini_api_key", None)
                if "llm_rerank" in m:
                    if not hasattr(cfg.method, "llm_rerank"):
                        from ..config import LLMRerankCfg
                        cfg.method.llm_rerank = LLMRerankCfg()
                    cfg.method.llm_rerank.dense_model_name = m["llm_rerank"].get("dense_model_name", cfg.method.llm_rerank.dense_model_name)
                    cfg.method.llm_rerank.dense_normalize_embeddings = m["llm_rerank"].get("dense_normalize_embeddings", cfg.method.llm_rerank.dense_normalize_embeddings)
                    cfg.method.llm_rerank.dense_api_base = m["llm_rerank"].get("dense_api_base", cfg.method.llm_rerank.dense_api_base)
                    cfg.method.llm_rerank.dense_api_key_env = m["llm_rerank"].get("dense_api_key_env", cfg.method.llm_rerank.dense_api_key_env)
                    cfg.method.llm_rerank.dense_request_timeout = m["llm_rerank"].get("dense_request_timeout", cfg.method.llm_rerank.dense_request_timeout)
                    cfg.method.llm_rerank.initial_k = m["llm_rerank"].get("initial_k", cfg.method.llm_rerank.initial_k)
                    cfg.method.llm_rerank.provider = m["llm_rerank"].get("provider", cfg.method.llm_rerank.provider)
                    cfg.method.llm_rerank.model = m["llm_rerank"].get("model", cfg.method.llm_rerank.model)
                    cfg.method.llm_rerank.api_base = m["llm_rerank"].get("api_base", cfg.method.llm_rerank.api_base)
                    cfg.method.llm_rerank.api_key_env = m["llm_rerank"].get("api_key_env", cfg.method.llm_rerank.api_key_env)
                    if "require_reason" in m["llm_rerank"]:
                        cfg.method.llm_rerank.require_reason = m["llm_rerank"].get("require_reason", True)
                    if "use_image" in m["llm_rerank"]:
                        cfg.method.llm_rerank.use_image = m["llm_rerank"].get("use_image", True)
                    cfg.method.llm_rerank.rerank_top_n = m["llm_rerank"].get("rerank_top_n", cfg.method.llm_rerank.rerank_top_n)
            if "limits" in cfg_obj:
                l = cfg_obj["limits"]
                cfg.limits.max_snippets_per_doc = l.get("max_snippets_per_doc", cfg.limits.max_snippets_per_doc)
                cfg.limits.topk = l.get("topk", cfg.limits.topk)
            if "logging" in cfg_obj:
                lg = cfg_obj["logging"]
                cfg.logging.save_details = lg.get("save_details", cfg.logging.save_details)
                cfg.logging.verbose = lg.get("verbose", cfg.logging.verbose)
        return cfg

    def rank(
        self,
        *,
        segment_links_path: Path,
        catalog_path: Path,
        artifacts_dir: Path,
        out_json: Path,
        out_csv: Optional[Path] = None,
        config_path: Optional[Path] = None,
        queries: Optional[List[Dict[str, Any]]] = None,
        query_text: Optional[str] = None,
        query_id: Optional[str] = None,
        queries_path: Optional[Path] = None,
        dataset: Optional[Union[str, Path]] = None,
        dataset_root: Optional[Path] = None,
    ) -> Dict[str, Any]:
        out_json = Path(out_json)
        out_csv_path = Path(out_csv) if out_csv is not None else None
        cfg = self._load_config(config_path or self.config_path)
        method_name = str(cfg.method.name).upper()
        # Append _linear or _late suffix for clip method based on fusion_method
        # Then append _REASON if require_reason is True
        if method_name == "CLIP":
            clip_cfg = getattr(cfg.method, "clip", None)
            if clip_cfg:
                fusion_method = getattr(clip_cfg, "fusion_method", "linear")
                method_name = f"CLIP_{fusion_method.upper()}"
                # Append _REASON suffix if require_reason is True
                require_reason = getattr(clip_cfg, "require_reason", True)
                if require_reason:
                    method_name = f"{method_name}_REASON"
            else:
                method_name = "CLIP_LINEAR"
        
        # Append _REASON or _INFO suffix for llm, llm_list, and llm_expansion methods based on require_reason
        # Also append _VISUAL or _TEXT based on use_image
        if method_name in {"LLM", "LLM_LIST", "LLM_EXPANSION"}:
            # Get the specific config for the method to check require_reason and use_image
            if method_name == "LLM":
                method_cfg_obj = cfg.method.llm
            elif method_name == "LLM_LIST":
                method_cfg_obj = cfg.method.llm
            else:  # LLM_EXPANSION
                method_cfg_obj = getattr(cfg.method, "llm_expansion", None) or cfg.method.llm
            
            require_reason = getattr(method_cfg_obj, "require_reason", True)
            use_image = getattr(method_cfg_obj, "use_image", True)
            suffix = "_REASON" if require_reason else "_INFO"
            image_suffix = "_VISUAL" if use_image else "_TEXT"
            method_name = f"{method_name}{suffix}{image_suffix}"
        
        model_name: Optional[str] = None
        # Extract base method name (without suffix) for model lookup
        base_method_name = method_name.replace("_REASON", "").replace("_INFO", "").replace("_LINEAR", "").replace("_LATE", "").replace("_VISUAL", "").replace("_TEXT", "")
        if base_method_name == "DENSE":
            model_name = cfg.method.dense.model_name
        elif base_method_name in {"LLM", "LLM_LIST", "LLM_EXPANSION"}:
            # Use the specific config object for model name
            if base_method_name == "LLM":
                model_name = getattr(cfg.method.llm, "model", None)
            elif base_method_name == "LLM_LIST":
                llm_list_cfg = cfg.method.llm
                model_name = getattr(llm_list_cfg, "model", None)
            else:  # LLM_EXPANSION
                llm_expansion_cfg = getattr(cfg.method, "llm_expansion", None) or cfg.method.llm
                model_name = getattr(llm_expansion_cfg, "model", None)
        elif base_method_name == "CLIP":
            clip_cfg = getattr(cfg.method, "clip", None)
            if clip_cfg:
                model_name = f"{getattr(clip_cfg, 'model_name', 'ViT-B-32')}_{getattr(clip_cfg, 'pretrained', 'openai')}"
        
        # Use "na" if model_name is not available
        effective_model_name = model_name if model_name else "na"

        inferred_root = dataset_root or CacheManager.infer_dataset_root(
            queries_path,
            segment_links_path,
            catalog_path,
            artifacts_dir,
        )
        dataset_path = Path(inferred_root)
        if dataset is not None:
            dataset_name = dataset.name if isinstance(dataset, Path) and dataset.name else str(dataset)
        else:
            dataset_name = dataset_path.name or (dataset_path.parts[-1] if dataset_path.parts else "default")
        queries_payload = self._prepare_queries_input(
            queries=queries,
            query_text=query_text,
            query_id=query_id,
            queries_path=queries_path,
        )
        effective_query_id = query_id or (queries_payload[0].get("query_id") if queries_payload else None)

        def generate() -> Dict[str, Any]:
            manifest = artifacts_dir / "snippet_index_manifest.json"
            if not manifest.exists():
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                build_index(catalog_path, artifacts_dir, cfg)
            rank_queries(
                queries=queries_payload,
                segment_links_path=segment_links_path,
                catalog_path=catalog_path,
                artifacts_dir=artifacts_dir,
                config=cfg,
                out_json=out_json,
                out_csv=out_csv_path,
                dataset_root=dataset_path,
            )
            json_payload = json.loads(out_json.read_text()) if out_json.exists() else {}
            outputs = {"snippets_json": json_payload}
            if out_csv_path is not None and out_csv_path.exists():
                outputs["snippets_csv"] = out_csv_path.read_text()
            return outputs

        query_meta: Dict[str, Any] = {}
        if effective_query_id:
            query_meta["query_id"] = str(effective_query_id)

        specs = {
            "snippets_json": OutputSpec(format="json", mirror_path=out_json, metadata=query_meta),
        }

        # Include model name in method path: model_name/method_name
        method_with_model = f"{effective_model_name}/{method_name}"

        bundle = self.io.get_or_generate(
            module="label_selection",
            method=method_with_model,
            dataset=dataset_name,
            specs=specs,
            generator=generate,
        )

        snapshot_json, snapshot_csv = self._write_query_snapshots(
            dataset_name=dataset_name,
            method_name=method_name,
            model_name=effective_model_name,
            query_id=effective_query_id,
            queries_payload=queries_payload,
            source_json=out_json,
            source_csv=out_csv_path,
        )

        return {
            "method": method_name,
            "model": model_name,
            "json_path": str(out_json),
            "csv_path": str(out_csv_path) if out_csv_path is not None else None,
            "payload": bundle.outputs,
            "query_id": effective_query_id,
            "snapshot_json": snapshot_json,
            "snapshot_csv": snapshot_csv,
        }

    def _prepare_queries_input(
        self,
        *,
        queries: Optional[List[Dict[str, Any]]],
        query_text: Optional[str],
        query_id: Optional[str],
        queries_path: Optional[Path],
    ) -> List[Dict[str, Any]]:
        if queries:
            return queries
        if query_text:
            return [{"query_id": query_id or "q_001", "text": query_text}]
        if queries_path is not None:
            data = json.loads(queries_path.read_text())
            qlist = data.get("queries") if isinstance(data, dict) else None
            if isinstance(qlist, list) and qlist:
                return qlist
        raise ValueError("Provide either queries, query_text, or queries_path for snippet ranking")

    def _write_query_snapshots(
        self,
        *,
        dataset_name: str,
        method_name: str,
        model_name: str,
        query_id: Optional[str],
        queries_payload: List[Dict[str, Any]],
        source_json: Path,
        source_csv: Optional[Path],
    ) -> tuple[Optional[str], Optional[str]]:
        if not source_json.exists():
            return None, None

        method_slug = _slugify_value(method_name) or method_name
        model_slug = _slugify_value(model_name) or model_name
        dataset_slug = _slugify_value(dataset_name) or dataset_name or "default"
        if query_id:
            query_slug = _slugify_value(query_id)
        else:
            query_text = ""
            if queries_payload and isinstance(queries_payload[0], dict):
                query_text = str(queries_payload[0].get("text") or "")
            query_slug = f"query_{stable_hash(query_text)[:10]}"
        if not query_slug:
            query_slug = "query"

        base_dir = Path(self.io.base_dir)
        # method_slug already includes _REASON or _INFO suffix if applicable
        # Include model_name in the path: label_selection/model_name/method_name
        snapshot_dir = base_dir / dataset_slug / "label_selection" / model_slug / method_slug
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        json_target = snapshot_dir / f"{query_slug.replace('_json', '')}.json"
        shutil.copy2(source_json, json_target)
        #remove the source json from file system
        
        if os.path.isfile(source_json):
            os.remove(source_json)

        csv_target_path: Optional[Path] = None
        if source_csv is not None and source_csv.exists():
            csv_target_path = snapshot_dir / f"{query_slug}_snippets_csv.csv"
            # shutil.copy2(source_csv, csv_target_path)

        return str(json_target), (str(csv_target_path) if csv_target_path else None)


def _slugify_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    cleaned = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or ""


__all__ = ["SnippetRanker", "build_index", "rank_queries"]

