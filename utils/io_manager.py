from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union


def _sanitize(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace(",", "_")
    )


def _to_path(value: Union[str, Path]) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _last_segment(value: Union[str, Path]) -> str:
    if isinstance(value, Path):
        parts = [p for p in value.parts if p]
    else:
        parts = [p for p in Path(value).parts if p]
    return parts[-1] if parts else "default"


@dataclass
class OutputSpec:
    format: str = "json"
    mirror_path: Optional[Union[str, Path]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    storage_path: Optional[Union[str, Path]] = None


@dataclass
class OutputDescriptor:
    name: str
    format: str
    file: Path
    metadata: Dict[str, Any]


@dataclass
class OutputBundle:
    dataset: str
    module: str
    method: str
    outputs: Dict[str, Any]
    descriptors: Dict[str, OutputDescriptor]


class IOManager:
    def __init__(self, base_dir: Union[str, Path] = "results"):
        self.base_dir = _to_path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_or_generate(
        self,
        *,
        module: str,
        method: str,
        dataset: Union[str, Path],
        specs: Dict[str, OutputSpec],
        generator: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> OutputBundle:
        dataset_name = self._normalize_dataset(dataset)
        module_key = _sanitize(module)
        # Handle method keys with slashes (e.g., "model_name/method_name")
        # Split on "/" before sanitizing, then join with a special separator
        if "/" in method:
            method_parts = method.split("/", 1)
            # Order: model_name/method_name -> model_key__MODEL__method_key
            method_key = f"{_sanitize(method_parts[0])}__MODEL__{_sanitize(method_parts[1])}"
        else:
            method_key = _sanitize(method)

        # existing = self._load_bundle(module_key, method_key, dataset_name, specs)
        # if existing:
        #     self._mirror_outputs(existing, specs)
        #     return existing

        if generator is None:
            raise FileNotFoundError(
                f"No stored outputs for module '{module}' method '{method}' (dataset={dataset_name})"
            )

        produced = generator()
        if not isinstance(produced, dict):
            raise ValueError("Generator callable must return a dict mapping output name to payload")

        missing = [name for name in specs.keys() if name not in produced]
        if missing:
            raise ValueError(f"Generator did not produce outputs for: {', '.join(missing)}")

        bundle = self._save_bundle(
            module_key=module_key,
            module_name=module,
            method_key=method_key,
            method_name=method,
            dataset=dataset_name,
            outputs=produced,
            specs=specs,
        )
        self._mirror_outputs(bundle, specs)
        return bundle

    def load_outputs(
        self,
        *,
        module: str,
        method: str,
        dataset: Union[str, Path],
    ) -> Optional[OutputBundle]:
        dataset_name = self._normalize_dataset(dataset)
        module_key = _sanitize(module)
        # Handle method keys with slashes (e.g., "model_name/method_name")
        if "/" in method:
            method_parts = method.split("/", 1)
            # Order: model_name/method_name -> model_key__MODEL__method_key
            method_key = f"{_sanitize(method_parts[0])}__MODEL__{_sanitize(method_parts[1])}"
        else:
            method_key = _sanitize(method)
        return self._load_bundle(module_key, method_key, dataset_name, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_dataset(self, dataset: Union[str, Path]) -> str:
        if isinstance(dataset, Path):
            if dataset.name:
                return dataset.name
            return _last_segment(dataset)
        ds = str(dataset).strip()
        if not ds:
            return "default"
        parts = [p for p in Path(ds).parts if p]
        return parts[-1] if parts else ds

    def _dataset_dir(self, dataset: str) -> Path:
        path = self.base_dir / _sanitize(dataset)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _module_dir(self, dataset: str, module_key: str) -> Path:
        path = self._dataset_dir(dataset) / module_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_storage_path(self, storage_path: Optional[Union[str, Path]], default_name: str) -> Path:
        if storage_path is None:
            return Path(default_name)
        candidate = _to_path(storage_path)
        parts = [_sanitize(str(part)) for part in candidate.parts if str(part)]
        if not parts:
            return Path(default_name)
        return Path(*parts)

    def _registry_path(self, dataset: str, module_key: str) -> Path:
        return self._module_dir(dataset, module_key) / "registry.json"

    def _load_registry(self, dataset: str, module_key: str) -> Dict[str, Any]:
        path = self._registry_path(dataset, module_key)
        if not path.exists():
            return {"methods": {}}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {"methods": {}}

    def _save_registry(self, dataset: str, module_key: str, registry: Dict[str, Any]) -> None:
        path = self._registry_path(dataset, module_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
            json.dump(registry, tmp, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    def _load_bundle(
        self,
        module_key: str,
        method_key: str,
        dataset: str,
        specs: Optional[Dict[str, OutputSpec]],
    ) -> Optional[OutputBundle]:
        registry = self._load_registry(dataset, module_key)
        methods = registry.get("methods", {})
        entry = methods.get(method_key)
        if not entry:
            return None

        outputs_def = entry.get("outputs", {})
        if not outputs_def:
            return None

        target_names = list(specs.keys()) if specs else list(outputs_def.keys())
        outputs: Dict[str, Any] = {}
        descriptors: Dict[str, OutputDescriptor] = {}
        module_dir = self._module_dir(dataset, module_key)

        for name in target_names:
            spec_meta = specs[name].metadata if specs and name in specs else {}
            stored = outputs_def.get(name)
            if not stored:
                return None
            stored_meta = stored.get("metadata", {})
            if spec_meta and any(stored_meta.get(k) != v for k, v in spec_meta.items()):
                return None
            fmt = stored.get("format", "json")
            if fmt == "json_dir":
                dir_relative = stored.get("dir")
                if not dir_relative:
                    return None
                file_path = module_dir / dir_relative
                if not file_path.exists() or not file_path.is_dir():
                    return None
            else:
                file_relative = stored.get("file", "")
                if not file_relative:
                    return None
                file_path = module_dir / file_relative
                if not file_path.exists():
                    return None
            try:
                payload = self._read_payload(file_path, fmt)
            except Exception:
                return None
            outputs[name] = payload
            descriptors[name] = OutputDescriptor(
                name=name,
                format=fmt,
                file=file_path,
                metadata=stored_meta,
            )

        return OutputBundle(
            dataset=dataset,
            module=entry.get("module", module_key),
            method=entry.get("method", method_key),
            outputs=outputs,
            descriptors=descriptors,
        )

    def _save_bundle(
        self,
        *,
        module_key: str,
        module_name: str,
        method_key: str,
        method_name: str,
        dataset: str,
        outputs: Dict[str, Any],
        specs: Dict[str, OutputSpec],
    ) -> OutputBundle:
        registry = self._load_registry(dataset, module_key)
        module_dir = self._module_dir(dataset, module_key)
        
        # Parse method_key to extract model_name and method_name if it contains a separator
        # For label_selection, method_key format is "model_name/method_name" (before sanitization)
        # After sanitization, slashes become underscores, so we check for the pattern
        # We'll use a special separator "__MODEL__" that won't be confused with normal underscores
        method_parts = method_key.split("__MODEL__")
        if len(method_parts) == 2:
            model_key, base_method_key = method_parts
            # Create subdirectory structure: model_name/method_name
            method_dir = module_dir / model_key / base_method_key
        else:
            # No model separator, use method_key directly
            method_dir = module_dir
            base_method_key = method_key
        
        outputs_def: Dict[str, Any] = {}
        descriptors: Dict[str, OutputDescriptor] = {}

        for name, spec in specs.items():
            fmt = spec.format.lower()
            payload = outputs[name]
            if fmt == "json_dir":
                default_dir = f"{base_method_key}_{_sanitize(name)}"
                relative_path = self._resolve_storage_path(spec.storage_path, default_dir)
                # If we have a model subdirectory, create path under model_dir
                if len(method_parts) == 2:
                    dir_path = method_dir / relative_path.name
                    # Store relative path from module_dir: model_name/method_name/dir_name
                    stored_relative_path = Path(model_key) / base_method_key / relative_path.name
                else:
                    dir_path = module_dir / relative_path
                    stored_relative_path = relative_path
                self._write_payload(dir_path, payload, fmt)
                outputs_def[name] = {
                    "dir": str(stored_relative_path),
                    "format": fmt,
                    "metadata": spec.metadata,
                }
                descriptors[name] = OutputDescriptor(
                    name=name,
                    format=fmt,
                    file=dir_path,
                    metadata=spec.metadata,
                )
                continue

            fname = f"{base_method_key}_{_sanitize(name)}.{self._extension_for(fmt)}"
            # If we have a model subdirectory, place file there
            if len(method_parts) == 2:
                file_path = method_dir / fname
                # Store relative path from module_dir: model_name/method_name/filename
                stored_file_path = Path(model_key) / base_method_key / fname
            else:
                file_path = module_dir / fname
                stored_file_path = Path(fname)
            self._write_payload(file_path, payload, fmt)
            outputs_def[name] = {
                "file": str(stored_file_path),
                "format": fmt,
                "metadata": spec.metadata,
            }
            descriptors[name] = OutputDescriptor(
                name=name,
                format=fmt,
                file=file_path,
                metadata=spec.metadata,
            )

        registry.setdefault("methods", {})[method_key] = {
            "module": module_name,
            "method": method_name,
            "outputs": outputs_def,
        }
        self._save_registry(dataset, module_key, registry)

        return OutputBundle(
            dataset=dataset,
            module=module_name,
            method=method_name,
            outputs={name: outputs[name] for name in specs.keys()},
            descriptors=descriptors,
        )

    def _mirror_outputs(self, bundle: OutputBundle, specs: Dict[str, OutputSpec]) -> None:
        for name, spec in specs.items():
            if spec.mirror_path is None:
                continue
            payload = bundle.outputs.get(name)
            descriptor = bundle.descriptors.get(name)
            if payload is None or descriptor is None:
                continue
            mirror_path = _to_path(spec.mirror_path)
            self._write_payload(mirror_path, payload, descriptor.format)

    # ------------------------------------------------------------------
    # Payload read/write utilities
    # ------------------------------------------------------------------
    def _extension_for(self, fmt: str) -> str:
        if fmt == "json":
            return "json"
        if fmt == "csv":
            return "csv"
        if fmt == "text":
            return "txt"
        if fmt == "json_dir":
            return "json"
        return fmt

    def _write_payload(self, path: Path, payload: Any, fmt: str) -> None:
        fmt = fmt.lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
                json.dump(payload, tmp, indent=2, ensure_ascii=False)
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        elif fmt == "csv":
            data = payload if isinstance(payload, str) else str(payload)
            with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        elif fmt == "text":
            data = payload if isinstance(payload, str) else str(payload)
            with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        elif fmt == "json_dir":
            if not isinstance(payload, dict):
                raise ValueError("json_dir payload must be a dict of name -> json payload")
            path.mkdir(parents=True, exist_ok=True)
            normalized: Dict[str, Any] = {}
            for key, value in payload.items():
                file_key = str(key)
                file_name = file_key if file_key.endswith(".json") else f"{file_key}.json"
                normalized[file_name] = value

            existing_files = {p.name for p in path.glob("*.json")}
            target_files = set(normalized.keys())
            for stale in existing_files - target_files:
                try:
                    (path / stale).unlink()
                except Exception:
                    pass

            for file_name, value in normalized.items():
                target_file = path / file_name
                self._write_payload(target_file, value, "json")
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    def _read_payload(self, path: Path, fmt: str) -> Any:
        fmt = fmt.lower()
        if fmt == "json":
            return json.loads(path.read_text())
        if fmt in {"csv", "text"}:
            return path.read_text()
        if fmt == "json_dir":
            if not path.exists() or not path.is_dir():
                return {}
            result: Dict[str, Any] = {}
            for file_path in sorted(path.glob("*.json")):
                try:
                    content = json.loads(file_path.read_text())
                except Exception:
                    continue
                seg_id = None
                if isinstance(content, dict):
                    seg_val = content.get("id")
                    if isinstance(seg_val, str) and seg_val:
                        seg_id = seg_val
                if seg_id is None:
                    name = file_path.name
                    seg_id = name[:-5] if name.endswith(".json") else name
                result[seg_id] = content
            return result
        raise ValueError(f"Unsupported output format: {fmt}")