from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import sys

from .text_dense import DenseLinker
from utils.llm_base import LLMBase
from label_selection.utils.text import normalize


class LLMExpansion(LLMBase):
    """Specialized LLM client for query expansion with persistent caching and usage metadata."""

    def expand(self, field_name: str, text: str, *, variant: str = "qe", version: str = "v1", step: str = "linking", method: str = "TEXT_DENSE_EXPANSION", extra_guidance: Optional[str] = None, temperature: Optional[float] = None) -> str:
        guidance = (
            extra_guidance
            or "You are enhancing a short query field for dense retrieval.\n"
               "Task: Expand the text by adding semantically related terms, synonyms, abbreviations, brand/model variants, materials, and usage contexts that may appear in product catalogs.\n"
               "Constraints: Keep it faithful (no contradictions), avoid hallucinations, prefer concise phrases, and keep length under 80 tokens per field.\n"
               "Guidance: Consider modern query expansion approaches from recent literature (e.g., synonym/context expansion)."
        )
        prompt = (
            f"{guidance}\n"
            f"Field: {field_name}\n"
            f"Original: {text}\n"
            "Output: A single line of expanded text that augments the original without repeating it verbatim."
        )
        components = {
            "field": field_name,
            "variant": variant,
            "version": version,
            "used_by": "expansion",
            "step": step,
            "method": method,
        }
        file_tag = f"qe_linker_{method}"
        content = self.chat_completion_with_cache(namespace="qe", prompt=prompt, temperature=temperature, components=components, file_tag=file_tag)
        return str(content or "").strip()

class QueryExpansionLinker(DenseLinker):
    def __init__(
        self,
        fields: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
        embed_provider: str = "local",
        embed_api_key: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        llm_api_key: str = "",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.2,
        variant_suffix: str = "qe",
        llm_cache_dir: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
    ):
        super().__init__(
            fields=fields,
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            cache_dir=cache_dir,
            embed_provider=embed_provider,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            dataset_root=dataset_root,
        )
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.variant_suffix = variant_suffix  # used to separate segment caches from non-QE
        self._llm: LLMExpansion = LLMExpansion(provider=llm_provider, api_key=llm_api_key, model=llm_model, temperature=llm_temperature)
        self.method_name = "TEXT_DENSE_EXPANSION"

        try:
            self._llm.configure_cache(
                module="linker",
                method=self.method_name.lower(),
                dataset_root=self.dataset_root,
                base_dir=llm_cache_dir or cache_dir,
                variant=self.variant_suffix,
            )
        except Exception as e:
            print(f"[QueryExpansionLinker] Failed to configure LLM cache: {e}", file=sys.stderr)


    def _expand_text(self, field_name: str, text: str) -> str:
        try:
            # Cached expansion via dedicated LLMExpansion helper (persists to disk)
            expanded = self._llm.expand(field_name=field_name, text=text, variant=str(self.variant_suffix or "qe"), version="v1", step="linking", method="TEXT_DENSE_EXPANSION", temperature=self.llm_temperature)
            if not expanded:
                return text
            # Merge original + expansion for stronger signal
            return f"{text} \n {expanded}"
        except Exception as e:
            raise ValueError(f"Failed to expand text for field {field_name}")

    def _expanded_query_text(self, segment: Dict[str, Any]) -> str:
        skip_keys = {"id", "confidence"}
        expanded_segment = json.loads(json.dumps(segment))
        expanded_parts: List[str] = []

        def path_label(path: List[Any]) -> str:
            if not path:
                return "description"
            parts: List[str] = []
            for p in path:
                if isinstance(p, int):
                    parts.append(f"[{p}]")
                else:
                    if parts:
                        parts.append(f".{p}")
                    else:
                        parts.append(str(p))
            return "".join(parts)

        def assign(path: List[Any], value: Any) -> None:
            cur = expanded_segment
            for idx, key in enumerate(path):
                last = idx == len(path) - 1
                if isinstance(key, int):
                    if last:
                        if isinstance(cur, list) and 0 <= key < len(cur):
                            cur[key] = value
                    else:
                        if isinstance(cur, list) and 0 <= key < len(cur):
                            cur = cur[key]
                        else:
                            return
                else:
                    if last:
                        if isinstance(cur, dict):
                            cur[key] = value
                    else:
                        if isinstance(cur, dict) and key in cur:
                            cur = cur[key]
                        else:
                            return

        def traverse(value: Any, path: List[Any]) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(k, str) and k.lower() in skip_keys:
                        continue
                    traverse(v, path + [k])
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    traverse(item, path + [i])
            elif isinstance(value, str):
                text = value.strip()
                if not text:
                    return
                if path and isinstance(path[-1], str) and path[-1].lower() in skip_keys:
                    return
                base_text = normalize(text)
                label = path_label(path)
                expanded_text = self._expand_text(label, base_text)
                expanded_parts.append(expanded_text)
                assign(path, expanded_text)

        traverse(segment, [])

        combined = " \n ".join(expanded_parts) if expanded_parts else normalize(str(segment.get("description", "")))
        self.save_expanded_query(expanded_segment)
        return combined

    def save_expanded_query(self, expanded_segment: Dict[str, Any]) -> None:
        if not hasattr(self, "dataset_root") or self.dataset_root is None:
            return
        out_dir = Path(self.dataset_root) / "expanded_descriptions"
        out_dir.mkdir(parents=True, exist_ok=True)
        seg_id = str(expanded_segment.get("id") or "segment")
        fname = f"{seg_id}.json"
        fpath = out_dir / fname
        fpath.write_text(json.dumps(expanded_segment, ensure_ascii=False, indent=2))

    def _build_query_text(self, segment: Dict[str, Any]) -> str:  # type: ignore[override]
        # Produce expanded-and-normalized query text; parent handles caching/scoring
        return normalize(self._expanded_query_text(segment))

    def _doc_cache_variant(self) -> str:  # type: ignore[override]
        base_variant = super()._doc_cache_variant()
        suffix = self._sanitize_label(self.variant_suffix) if self.variant_suffix else "qe"
        return f"{base_variant}_{suffix}"

    def _segment_cache_variant(self) -> str:  # type: ignore[override]
        base_variant = super()._segment_cache_variant()
        suffix = self._sanitize_label(self.variant_suffix) if self.variant_suffix else "qe"
        return f"{base_variant}_{suffix}"


