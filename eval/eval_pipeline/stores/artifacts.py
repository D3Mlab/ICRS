"""
ArtifactStore enforcing a manifest key→path map.

Responsibilities:
- load/save manifest at artifacts/manifest.json
- require(keys): assert presence; give remediation hints on failure
- put(key, path): register new artifact path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


class ArtifactStore:
    def __init__(self, artifacts_dir: Path, manifest_path: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.manifest_path = manifest_path
        self._manifest: Dict[str, str] = {}
        self._loaded = False

    def load(self) -> None:
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text())
        else:
            self._manifest = {}
        self._loaded = True

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2, ensure_ascii=False))

    def require(self, keys: Iterable[str]) -> None:
        if not self._loaded:
            self.load()
        missing: List[str] = [k for k in keys if k not in self._manifest]
        if missing:
            hints = []
            # Simple hints per key prefix
            for k in missing:
                if k.startswith("step1."):
                    hints.append("python -m eval_pipeline.cli step1 …")
                elif k.startswith("step2."):
                    hints.append("python -m eval_pipeline.cli step2 …")
                elif k.startswith("step3."):
                    hints.append("python -m eval_pipeline.cli step3 …")
                elif k.startswith("step4."):
                    hints.append("python -m eval_pipeline.cli step4 …")
            hint_str = " or ".join(sorted(set(hints))) if hints else "run the appropriate step"
            raise RuntimeError(f"Missing artifacts: {missing}. Generate by: {hint_str}.")

    def get_path(self, key: str) -> Path:
        if not self._loaded:
            self.load()
        p = self._manifest.get(key)
        if p is None:
            raise KeyError(f"Artifact not found in manifest: {key}")
        return Path(p)

    def put(self, key: str, path: Path) -> None:
        if not self._loaded:
            self.load()
        rel = str(path)
        self._manifest[key] = rel



