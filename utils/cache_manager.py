from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union


def _as_path(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


class CacheManager:
    """Centralized manager for embedding and LLM caches."""

    DEFAULT_DATASET = "fashion"
    DATA_DIR_NAME = "data"
    CACHE_DIR_NAME = "cache"
    EMBEDDINGS_DIR_NAME = "embeddings"
    LLM_RESPONSES_DIR_NAME = "llm_response"

    @staticmethod
    def sanitize(value: str) -> str:
        return (
            str(value)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
            .replace("|", "_")
            .replace(",", "_")
        )

    @classmethod
    def infer_dataset_root(cls, *candidates: Optional[Union[str, Path]]) -> Path:
        """Infer dataset root (e.g., data/fashion) from a list of candidate paths."""

        for cand in candidates:
            if not cand:
                continue
            p = _as_path(cand)
            if p is None:
                continue
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p.absolute()
            parts = resolved.parts
            if cls.DATA_DIR_NAME in parts:
                idx = parts.index(cls.DATA_DIR_NAME)
                if idx + 1 < len(parts):
                    return Path(*parts[: idx + 2])
        return Path(cls.DATA_DIR_NAME) / cls.DEFAULT_DATASET

    @classmethod
    def _resolve_dataset_root(
        cls,
        dataset: Optional[Union[str, Path]] = None,
        *fallback_candidates: Optional[Union[str, Path]],
    ) -> Path:
        if dataset is not None:
            p = _as_path(dataset)
            if p is not None:
                if not p.is_absolute():
                    if cls.DATA_DIR_NAME not in p.parts:
                        p = Path(cls.DATA_DIR_NAME) / p
                    p = p.resolve()
                if p.is_file():
                    p = p.parent
                return p
        return cls.infer_dataset_root(*fallback_candidates)

    @classmethod
    def _embedding_dir(cls, dataset_root: Path, module: str, method: str) -> Path:
        path = (
            dataset_root
            / cls.CACHE_DIR_NAME
            / cls.EMBEDDINGS_DIR_NAME
            / cls.sanitize(module)
            / cls.sanitize(method)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def _llm_dir(cls, dataset_root: Path, module: str, method: str) -> Path:
        path = (
            dataset_root
            / cls.CACHE_DIR_NAME
            / cls.LLM_RESPONSES_DIR_NAME
            / cls.sanitize(module)
            / cls.sanitize(method)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def _embedding_file(
        cls,
        dataset_root: Path,
        module: str,
        method: str,
        model_name: str,
        variant: Optional[str],
    ) -> Path:
        directory = cls._embedding_dir(dataset_root, module, method)
        filename = cls.sanitize(model_name)
        if variant:
            filename = f"{filename}__{cls.sanitize(variant)}"
        return directory / f"{filename}.pkl"

    @classmethod
    def _llm_file(
        cls,
        dataset_root: Path,
        module: str,
        method: str,
        model_name: str,
        variant: Optional[str],
    ) -> Path:
        directory = cls._llm_dir(dataset_root, module, method)
        filename = cls.sanitize(model_name)
        if variant:
            filename = f"{filename}__{cls.sanitize(variant)}"
        return directory / f"{filename}.json"

    @staticmethod
    def _safe_write_pickle(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
            pickle.dump(payload, tmp)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    @staticmethod
    def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    @classmethod
    def load_embedding_cache(
        cls,
        *,
        module: str,
        method: str,
        model_name: str,
        dataset: Optional[Union[str, Path]] = None,
        base_dirs: Iterable[Optional[Union[str, Path]]] = (),
        variant: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dataset_root = cls._resolve_dataset_root(dataset, *base_dirs)
        path = cls._embedding_file(dataset_root, module, method, model_name, variant)
        if not path.exists():
            return {}, {}
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
        except Exception:
            return {}, {}
        if not isinstance(payload, dict):
            return {}, {}
        meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        records = payload.get("records", {}) if isinstance(payload.get("records"), dict) else {}
        if meta.get("module") != module or meta.get("method") != method or meta.get("model") != model_name:
            return {}, {}
        if variant and meta.get("variant") != variant:
            return {}, {}
        stored_dataset = meta.get("dataset_root")
        try:
            if stored_dataset and Path(stored_dataset) != dataset_root:
                return {}, {}
        except Exception:
            return {}, {}
        return records, meta

    @classmethod
    def save_embedding_cache(
        cls,
        *,
        module: str,
        method: str,
        model_name: str,
        records: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        dataset: Optional[Union[str, Path]] = None,
        base_dirs: Iterable[Optional[Union[str, Path]]] = (),
        variant: Optional[str] = None,
    ) -> Path:
        dataset_root = cls._resolve_dataset_root(dataset, *base_dirs)
        path = cls._embedding_file(dataset_root, module, method, model_name, variant)
        payload_meta = {
            "module": module,
            "method": method,
            "model": model_name,
            "variant": variant,
            "dataset_root": str(dataset_root),
        }
        if meta:
            payload_meta.update(meta)
        payload = {"meta": payload_meta, "records": records}
        cls._safe_write_pickle(path, payload)
        return path

    @classmethod
    def load_llm_cache(
        cls,
        *,
        module: str,
        method: str,
        model_name: str,
        require_reason: bool = False,
        dataset: Optional[Union[str, Path]] = None,
        base_dirs: Iterable[Optional[Union[str, Path]]] = (),
        variant: Optional[str] = None,
    ) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
        dataset_root = cls._resolve_dataset_root(dataset, *base_dirs)
        path = cls._llm_file(dataset_root, module, method, model_name, variant)
        if not path.exists():
            return [], {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return [], {}
        if not isinstance(payload, dict):
            return [], {}
        meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
        if meta.get("module") != module or meta.get("method") != method or meta.get("model") != model_name:
            if 'require_reason' in meta and meta['require_reason'] != require_reason:
                return [], {}
            else:
                return [], {}
        if variant and meta.get("variant") != variant:
            if 'require_reason' in meta and meta['require_reason'] != require_reason:
                return [], {}
            else:
                return [], {}
        stored_dataset = meta.get("dataset_root")
        try:
            if stored_dataset and Path(stored_dataset) != dataset_root:
                if 'require_reason' in meta and meta['require_reason'] != require_reason:
                    return [], {}
                else:
                    return [], {}
        except Exception:
            return [], {}
        return records, meta

    @classmethod
    def save_llm_cache(
        cls,
        *,
        module: str,
        method: str,
        model_name: str,
        require_reason: bool = False,
        records: list[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
        dataset: Optional[Union[str, Path]] = None,
        base_dirs: Iterable[Optional[Union[str, Path]]] = (),
        variant: Optional[str] = None,
    ) -> Path:
        dataset_root = cls._resolve_dataset_root(dataset, *base_dirs)
        path = cls._llm_file(dataset_root, module, method, model_name, variant)
        payload_meta = {
            "module": module,
            "method": method,
            "model": model_name,
            "variant": variant,
            "dataset_root": str(dataset_root),
            "require_reason": require_reason,
        }
        if meta:
            payload_meta.update(meta)
        payload = {"meta": payload_meta, "records": records}
        cls._safe_write_json(path, payload)
        return path

    @classmethod
    def get_llm_record(
        cls,
        *,
        module: str,
        method: str,
        model_name: str,
        namespace: str,
        require_reason: bool = False,
        key: Dict[str, Any],
        dataset: Optional[Union[str, Path]] = None,
        base_dirs: Iterable[Optional[Union[str, Path]]] = (),
        variant: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        records, meta = cls.load_llm_cache(
            module=module,
            method=method,
            model_name=model_name,
            dataset=dataset,
            base_dirs=base_dirs,
            variant=variant,
            require_reason=require_reason,
        )
        if not records:
            return None
        for rec in records:
            if rec.get("namespace") == namespace and rec.get("key") == key:
                return {"record": rec, "meta": meta}
        return None

    @classmethod
    def append_llm_record(
        cls,
        *,
        module: str,
        method: str,
        model_name: str,
        namespace: str,
        require_reason: bool = False,
        key: Dict[str, Any],
        content: str,
        record_meta: Optional[Dict[str, Any]] = None,
        dataset: Optional[Union[str, Path]] = None,
        base_dirs: Iterable[Optional[Union[str, Path]]] = (),
        variant: Optional[str] = None,
    ) -> Path:
        records, meta = cls.load_llm_cache(
            module=module,
            method=method,
            model_name=model_name,
            dataset=dataset,
            base_dirs=base_dirs,
            variant=variant,
            require_reason=require_reason,
        )
        filtered = [r for r in records if not (r.get("namespace") == namespace and r.get("key") == key)]
        filtered.append({"namespace": namespace, "key": key, "content": content, "meta": record_meta or {}})
        return cls.save_llm_cache(
            module=module,
            method=method,
            model_name=model_name,
            records=filtered,
            meta=meta,
            dataset=dataset,
            base_dirs=base_dirs,
            variant=variant,
        )

    @classmethod
    def embedding_records_path(
        cls,
        *,
        module: str,
        method: str,
        model_name: str,
        dataset: Optional[Union[str, Path]] = None,
        base_dirs: Iterable[Optional[Union[str, Path]]] = (),
        variant: Optional[str] = None,
    ) -> Path:
        dataset_root = cls._resolve_dataset_root(dataset, *base_dirs)
        return cls._embedding_file(dataset_root, module, method, model_name, variant)

    @classmethod
    def llm_records_path(
        cls,
        *,
        module: str,
        method: str,
        model_name: str,
        dataset: Optional[Union[str, Path]] = None,
        base_dirs: Iterable[Optional[Union[str, Path]]] = (),
        variant: Optional[str] = None,
    ) -> Path:
        dataset_root = cls._resolve_dataset_root(dataset, *base_dirs)
        return cls._llm_file(dataset_root, module, method, model_name, variant)

