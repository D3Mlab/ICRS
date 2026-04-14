from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from utils.cache_manager import CacheManager
from .base import LinkMethod
from utils.llm_base import LLMBase


PROMPT_VERSION = "v1"

class LLMJudgeLinker(LLMBase):
    """Specialized LLM client for judgment prompts with persistent caching and usage metadata."""

    def judge(self, prompt: str, *, version: str = "v1", step: str = "linking", method: str = "LLM_JUDGE", temperature: Optional[float] = None, extra_components: Optional[Dict[str, str]] = None) -> Any:
    
        components = {
            "variant": "judge",
            "version": version,
            "used_by": "judge",
            "step": step,
            "method": method,
        }
        if extra_components:
            components.update({k: v for k, v in extra_components.items() if isinstance(v, str)})

        content = self.chat_completion_with_cache(namespace="judge", prompt=prompt, temperature=temperature, components=components)
        obj = self.try_parse_json(str(content))
        if obj is None:
            # Fallback: attempt naive parse by stripping code fences already handled in try_parse_json
            raise ValueError("LLMJudgeLinker: failed to parse JSON from completion")
        return obj

    @staticmethod
    def build_judge_prompt(segment: Dict[str, Any], doc: Dict[str, Any]) -> str:
        seg_text = str(segment.get("description", ""))
        seg_img = str(segment.get("image_path", ""))
        title = str(doc.get("title") or doc.get("product_name") or "")
        desc = str(doc.get("description") or doc.get("product_description") or "")
        attrs = doc.get("attributes") or doc.get("product_detail") or {}
        attrs_txt = ", ".join([f"{k}: {v}" for k, v in attrs.items()]) if isinstance(attrs, dict) else str(attrs)
        return (
            "You are a product matching judge. Return ONLY JSON.\n"
            f"SEGMENT_DESC: {seg_text}\n"
            f"DOC_TITLE: {title}\nDOC_DESC: {desc}\nDOC_ATTRS: {attrs_txt}\n"
            "Schema: {\"relevance\": float in [0,1], \"reason\": string<=30}\n"
            "JSON ONLY: {\"relevance\":0.0,\"reason\":\"...\"}"
        )

    def judge_pair(self, segment: Dict[str, Any], doc: Dict[str, Any], *, version: str = "v1", step: str = "linking", method: str = "LLM_JUDGE", temperature: Optional[float] = None) -> Any:
        prompt = self.build_judge_prompt(segment, doc)
        seg_id = str(segment.get("id"))
        doc_id = str(doc.get("doc_id"))
        components = {"segment_id": seg_id, "doc_id": doc_id}
        return self.judge(prompt, version=version, step=step, method=method, temperature=temperature, extra_components=components)



class LLMLinkJudge(LinkMethod):
    def __init__(self, provider: str = "hf-local", model: str = "stub", cache_dir: Path | None = None, api_key: str = "", dataset_root: Optional[Path] = None):
        self.provider = provider
        self.model = model
        self.docs: List[Dict[str, Any]] = []
        self.dataset_root = CacheManager.infer_dataset_root(dataset_root, cache_dir)
        self.cache_base = cache_dir
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client: LLMJudgeLinker | None = None

    def _ensure_client(self) -> None:
        if self.provider != "openai":
            return
        if self._client is None:
            if not self.api_key:
                raise ValueError("LLMLinkJudge: OPENAI_API_KEY is required for provider 'openai'")
            self._client = LLMJudgeLinker(provider="openai", api_key=self.api_key, model=self.model, temperature=0.0)
            try:
                self._client.configure_cache(
                    module="linker",
                    method="llm_judge",
                    dataset_root=self.dataset_root,
                    base_dir=self.cache_base,
                    variant=None,
                )
            except Exception:
                pass

    def prepare(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        if self.provider == "openai":
            self._ensure_client()

    def _call_llm(self, segment: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
        if self.provider == "openai":
            assert self._client is not None
            data = self._client.judge_pair(segment, doc, version=PROMPT_VERSION, step="linking", method="LLM_JUDGE", temperature=0.0)
            if isinstance(data, dict):
                return data
            try:
                return json.loads(str(data))
            except Exception:
                return {"relevance": 0.0, "reason": "parse_fail"}
        else:
            raise ValueError("LLMLinkJudge: only openai provider is supported")

    def score_segment(self, segment: Dict[str, Any], top_k: int) -> List[Tuple[str, float, str]]:
        seg_id = str(segment.get("id"))
        pairs: List[Tuple[str, float, str]] = []
        for d in self.docs:
            doc_id = str(d.get("doc_id"))
            data = self._call_llm(segment, d)
            score = float(data.get("relevance", 0.0))
            reason = str(data.get("reason", ""))
            pairs.append((doc_id, score, reason))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return pairs[:top_k]


