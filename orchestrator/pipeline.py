from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from item_segmentation.sam_segmenter import SamSegmenter
from item_recommendation.pipeline import ObjectRanker
from linker.pipeline import MetaDataLinker
from label_selection.pipeline import SnippetRanker  # type: ignore


from utils.cache_manager import CacheManager
from utils.io_manager import IOManager

class ICRSPipeline:
    def __init__(self):
        self.io = IOManager()
        self._step_metadata: Dict[str, Optional[str]] = {
            "segmenter_method": None,
            "segmenter_model": None,
            "describer_method": None,
            "describer_model": None,
            "linker_method": None,
            "linker_model": None,
            "item_recommendation_method": None,
            "item_recommendation_model": None,
            "label_selection_method": None,
            "label_selection_model": None,
        }

    # Step 1: Segmentation
    def segment(self, image_path: Path, out_dir: Path, exhaustive: bool = False, device: str = "cpu", model: str = "vit_b") -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        seg = SamSegmenter(model_type=model, device_str=device)
        saved, masks = seg.segment_image(
            image_path=image_path,
            output_dir=out_dir,
            exhaustive=exhaustive,
        )
        self._step_metadata["segmenter_method"] = seg.__class__.__name__
        self._step_metadata["segmenter_model"] = model
        return {"saved_files": saved, "num_masks": len(masks), "out_dir": str(out_dir)}


    # Step 2: Attribute Lookup
    def link(self, segments_json: Path, docs_json: Path, out_links: Path, 
    out_meta: Path, linker_cfg: Optional[Path] = None, **kwargs) -> tuple[str, Optional[str]]:
        linker = MetaDataLinker(io_manager=self.io, config_path=linker_cfg)
        res = linker.link(
            segments_path=segments_json,
            docs_path=docs_json,
            out_links=out_links,
            out_meta=out_meta,
            **kwargs,
        )
        method_used = str(res.get("method", "")).upper() if res.get("method") else None
        model_used = res.get("model")
        if method_used:
            self._step_metadata["linker_method"] = method_used
        if model_used:
            self._step_metadata["linker_model"] = model_used
        return method_used or "", model_used

    # Step 3: Item Reccomendation (post-link metadata)
    def reccomend_items(
        self,
        query: str,
        query_id: Optional[str],
        meta_json: Path,
        out_json: Path,
        api_key: str,
        batch_size: Optional[int] = None,
        self_consistency_k: int = 1,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
        config_path: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        ranker = ObjectRanker(io_manager=self.io, config_path=config_path)
        res = ranker.rank(
            user_query=query,
            query_id=query_id,
            meta_json=meta_json,
            out_json=out_json,
            batch_size=batch_size,
            self_consistency_k=self_consistency_k,
            cache_dir=cache_dir,
            verbose=verbose,
            linker_method=self._step_metadata.get("linker_method"),
            dataset_root=CacheManager.infer_dataset_root(meta_json),
        )
        method_used = res.get("method")
        model_used = res.get("model")
        if method_used:
            self._step_metadata["item_recommendation_method"] = str(method_used)
        if model_used:
            self._step_metadata["item_recommendation_model"] = str(model_used)
        payload = res.get("payload") if isinstance(res, dict) else None
        snapshot_path = res.get("snapshot_path") if isinstance(res, dict) else None
        return (payload if isinstance(payload, dict) else {}), (str(snapshot_path) if snapshot_path else None)

    # Step 4: label selection
    def label_selection(
        self,
        base_dir: Path,
        query_text: str,
        query_id: Optional[str],
        data_root: Path,
        artifacts_dir: Path,
        out_json: Path,
        config_path: Optional[Path] = None,
        seg_links_path: Optional[Path] = None,
        out_csv: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if not query_text:
            raise ValueError("Provide --query id/text for snippet ranking")
        q_identifier = query_id or "q_001"
        query_payload = [{"query_id": q_identifier, "text": query_text}]
        dataset_name = base_dir.name or (base_dir.parts[-1] if base_dir.parts else "default")

        seg_links_dst = self._resolve_segment_links_path(
            base_dir=base_dir,
            dataset_name=dataset_name,
            provided_path=seg_links_path,
        )

        catalog_path = base_dir / "attributes" / 'raw'
        label_selection_step_art_dir = artifacts_dir / "label_selection_step"
        label_selection_step_art_dir.mkdir(parents=True, exist_ok=True)
        ranker = SnippetRanker(io_manager=self.io, config_path=config_path)
        csv_path = out_csv if out_csv is not None else out_json.with_suffix(".csv")
        res = ranker.rank(
            segment_links_path=seg_links_dst,
            catalog_path=catalog_path,
            artifacts_dir=label_selection_step_art_dir,
            out_json=out_json,
            out_csv=csv_path,
            config_path=config_path,
            queries=query_payload,
            query_id=q_identifier,
            dataset=dataset_name,
            dataset_root=base_dir,
        )
        method_name = res.get("method")
        model_used = res.get("model")
        if method_name:
            self._step_metadata["label_selection_method"] = method_name
        if model_used:
            self._step_metadata["label_selection_model"] = model_used
        return {
            "label_selection_out_json": str(out_json),
            "label_selection_out_csv": str(csv_path) if csv_path else None,
            "label_selection_method": method_name,
            "label_selection_model": model_used,
            "snippet_rankings_json": res.get("snapshot_json"),
            "snippet_rankings_csv": res.get("snapshot_csv"),
        }

    def _resolve_segment_links_path(
        self,
        *,
        base_dir: Path,
        dataset_name: str,
        provided_path: Optional[Path],
    ) -> Path:
        candidates: List[Path] = []
        if provided_path and provided_path.exists():
            return provided_path

        linker_method = self._step_metadata.get("linker_method")
        if linker_method:
            try:
                bundle = self.io.load_outputs(module="linker", method=linker_method, dataset=dataset_name)
            except Exception:
                bundle = None
            if bundle is not None:
                descriptor = bundle.descriptors.get("links")
                if descriptor and descriptor.file.exists():
                    return descriptor.file

        fallback = base_dir / "attributes" / "segment_links.json"
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            "Could not locate segment links file for snippet ranking. "
            f"Tried provided path and linker outputs for dataset '{dataset_name}'."
        )

    # Auto orchestrate: detect given artifacts and fill missing steps
    def run(
        self,
        image: Optional[Path] = None,
        crops_dir: Optional[Path] = None,
        item_recommendation_json: Optional[Path] = None,
        docs_json: Optional[Path] = None,
        out_links: Optional[Path] = None,
        out_meta: Optional[Path] = None,
        linker_cfg: Optional[Path] = None,
        item_recommendation_cfg: Optional[Path] = None,
        api_key: str = "",
        query: str = "",
        device: str = "cpu",
        model: str = "vit_b",
        exhaustive: bool = False,
        w_text: float = 0.7,
        w_image: float = 0.3,
        k: int = 20,
        m: int = 5,
        min_match_threshold: float = 0.65,
        min_margin: float = 0.05,
        tfidf_max_features: int = 20000,
        image_hist_bins: int = 64,
        llm_enable: bool = False,
        seed: int = 42,
        self_consistency_k: int = 5,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        self._step_metadata = {
        "segmenter_method": None,
        "segmenter_model": None,
        "describer_method": None,
        "describer_model": None,
        "linker_method": None,
        "linker_model": None,
        "item_recommendation_method": None,
        "item_recommendation_model": None,
        "label_selection_method": None,
        "label_selection_model": None,
        }

        def derive_base_dir(image_path: Optional[Path], crops_path: Optional[Path]) -> Path:
            if crops_path is not None:
                try:
                    return crops_path.parent.parent
                except Exception:
                    return crops_path.parent
            if image_path is not None:
                parts = image_path.parts
                if "data" in parts:
                    i = parts.index("data")
                    if i + 1 < len(parts):
                        return Path(*parts[: i + 2])
                return image_path.parent.parent if image_path.parent.name.lower().startswith("env") else image_path.parent
            return Path("data")

        base_dir = derive_base_dir(image, crops_dir)
        rank_dir = base_dir / "rankings"
        seg_dir = base_dir / "segments" / "crops"
        default_cache = base_dir / "cache"
        dataset_root_for_queries = CacheManager.infer_dataset_root(base_dir)

        query_text = query or ""
        query_id: Optional[str] = None
        
        if query_text:
            print(f"[orchestrator] Query text: {query_text}")
            query_text, query_id = self._resolve_query_input(query_text, dataset_root_for_queries)

        # Step 1: crops_dir
        if crops_dir is None:
            if image is None:
                raise ValueError("Either provide crops_dir or image to generate segments.")
            crops_dir = seg_dir
            print('Generating segments...')
            self.segment(image, crops_dir, exhaustive=exhaustive, device=device, model=model)

        # Step 2: Item Reccomendation
        ranking_payload: Optional[Dict[str, Any]] = None
        object_snapshot: Optional[str] = None
        rank_cfg_path = item_recommendation_cfg or Path("configs/item_recommendation.yaml")
        if query_text:
            if not api_key:
                raise ValueError("LLM API key required to rank segments. Provide --api_key.")
            if item_recommendation_json is None:
                item_recommendation_json = rank_dir / "item_recommendation.json"
            print('Item reccomendation...')
            ranking_payload, object_snapshot = self.reccomend_items(
                query=query_text,
                query_id=query_id,
                meta_json=out_meta,
                out_json=item_recommendation_json,
                api_key=api_key,
                batch_size=None,
                self_consistency_k=self_consistency_k,
                cache_dir=(cache_dir or default_cache),
                verbose=verbose,
                config_path=rank_cfg_path,
            )
        else:
            if item_recommendation_json is None:
                item_recommendation_json = rank_dir / "item_recommendation.json"
            ranking_payload = {}
            object_snapshot = None

        # Step 3: label selection
        data_root = Path(".")
        artifacts_root = Path("artifacts")
        label_selection_cfg = Path("configs/label_selection.yaml")
        label_selection_out_json = rank_dir / "snippet_rankings.json"
        label_selection_out_csv = rank_dir / "snippet_rankings.csv"
        print('Label selection...')
        label_selection_out = self.label_selection(
            base_dir=base_dir,
            query_text=query_text,
            query_id=query_id,
            data_root=data_root,
            artifacts_dir=artifacts_root,
            seg_links_path=out_links,
            config_path=label_selection_cfg,
            out_json=label_selection_out_json,
            out_csv=label_selection_out_csv,
        )
        return {
            "crops_dir": str(crops_dir),
            "out_links": str(out_links),
            "out_meta": str(out_meta),
            "item_recommendation_json": str(item_recommendation_json) if item_recommendation_json else None,
            "item_recommendation": ranking_payload,
            "object_rankings_json": object_snapshot,
            "base_dir": str(base_dir),
            "query_id": query_id,
            "query_text": query_text,
            **label_selection_out,
            **{k: v for k, v in self._step_metadata.items() if v},
        }

    def _resolve_query_input(self, query_arg: str, dataset_root: Path) -> Tuple[str, Optional[str]]:
        query_arg = (query_arg or "").strip()
        if not query_arg:
            return "", None

        candidates: List[Path] = []
        dataset_root = dataset_root.resolve()

        candidates.append(dataset_root / "conversation" / "agg" / "pre_gt.json")


        for candidate in candidates:
            try:
                if not candidate.exists():
                    continue
                data = json.loads(candidate.read_text())
            except Exception:
                continue
            if isinstance(data, dict) and query_arg in data:
                text = str(data[query_arg]).strip()
                if text:
                    return text, query_arg

        try:
            q_path = Path(query_arg)
            if q_path.suffix.lower() == ".json" and q_path.exists():
                from label_selection.utils.query import queryReformatter  # type: ignore

                text = queryReformatter.reformat_file_to_text(q_path)
                if text:
                    return text, q_path.stem
        except Exception:
            pass

        return query_arg, query_arg or None




