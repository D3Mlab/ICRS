from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.cache_manager import CacheManager

from .base import MethodResult, RelevanceMethod
from .dense import DenseSnippetRanker
from ..utils.io import stable_hash

from google import genai  # type: ignore
from openai import OpenAI


PROMPT_VERSION = "v2"  # Updated to include rationale


class DenseLLMRerankSnippetRanker(RelevanceMethod):
    """Dense retrieval followed by LLM rerank on the top-K snippets."""

    def __init__(
        self,
        *,
        dense_model_name: str = "qwen-2.5-8b",
        dense_normalize_embeddings: bool = True,
        dense_api_base: Optional[str] = None,
        dense_api_key_env: Optional[str] = None,
        dense_request_timeout: float = 60.0,
        initial_k: int = 10,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        cache_dir: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        image_path: Optional[Path] = None,
        require_reason: bool = True,
        use_image: bool = True,
        rerank_top_n: int = 5,
    ) -> None:
        # Dense retriever setup
        self.dense = DenseSnippetRanker(
            model_name='qwen/qwen3-embedding-8b',
            normalize_embeddings=dense_normalize_embeddings,
            cache_dir=cache_dir,
            dataset_root=dataset_root,
            api_base=dense_api_base,
            api_key_env=dense_api_key_env or "OPENROUTER_API_KEY",
            request_timeout=dense_request_timeout,
        )
        self.initial_k = max(1, int(initial_k))
        self.rerank_top_n = max(1, int(rerank_top_n))
        
        # LLM setup
        self.provider = (provider or "openai").lower()
        self.model = model
        self.dataset_root = CacheManager.infer_dataset_root(dataset_root, cache_dir)
        self.cache_base = cache_dir
        self.cache_module = "label_selection"
        self.cache_method = "llm_rerank"
        self.cache_variant = None
        self.api_base = api_base
        self.api_key_env = "OPENROUTER_API_KEY" if self.provider == "openrouter" else "OPENAI_API_KEY"
        self.api_key = os.getenv(self.api_key_env)
        self.google_model = None
        self._google_client = None
        self.image_path = image_path
        self.require_reason = require_reason
        self.use_image = use_image
        self.rationale_map: Dict[str, str] = {}  # Store rationale for each snippet
        
        if self.provider == "google":
            self._google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        self.snippet_texts: Dict[str, str] = {}
        print(f'[label_selection] provider: {self.provider}')
        print(f'[label_selection] model: {self.model}')

    def fit(self, snippet_texts: Dict[str, str]) -> None:
        self.snippet_texts = dict(snippet_texts)
        self.dense.fit(snippet_texts)

    def _read_image_b64(self, path: Path) -> str:
        """Read image file and return as base64 string."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_llm_single(self, prompt: str) -> str:
        # Build content with image if available and use_image is True
        if self.use_image and self.image_path and self.image_path.exists():
            image_b64 = self._read_image_b64(self.image_path)
            image_ext = self.image_path.suffix.lower()
            mime_type = "image/png" if image_ext == ".png" else "image/jpeg"
        else:
            print(f'[label_selection] image_path: {self.image_path}')
            image_b64 = None
            mime_type = None

        if self.provider in {"openai", "openrouter"}:
            try:
                client_kwargs: Dict[str, Any] = {}
                client_kwargs["api_key"] = self.api_key
                if self.provider == "openrouter":
                    base_url = self.api_base or "https://openrouter.ai/api/v1"
                    client_kwargs["base_url"] = base_url
                    headers: Dict[str, str] = {}
                    referer = os.getenv("OPENROUTER_SITE_URL")
                    if referer:
                        headers["HTTP-Referer"] = referer
                    title = os.getenv("OPENROUTER_APP_NAME")
                    if title:
                        headers["X-Title"] = title
                    if headers:
                        client_kwargs["default_headers"] = headers  # type: ignore
                client = OpenAI(**client_kwargs)
            except Exception as exc:
                raise RuntimeError(f"{self.provider} provider not available: {exc}") from exc
            for attempt in range(3):
                try:
                    if image_b64:
                        content = [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}", "detail": "high"}},
                        ]
                    else:
                        content = prompt
                    print(f"Content: {self.model}")
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": content}],
                        temperature=0,
                    )
                    return resp.choices[0].message.content or ""
                except Exception:
                    time.sleep(0.5 * (attempt + 1))
            raise RuntimeError("LLM call failed after retries")
        if self.provider == "google":
            if not self._google_client:
                raise RuntimeError("Google provider is not initialized properly")
            for attempt in range(3):
                try:
                    if image_b64:
                        parts = [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                        ]
                    else:
                        parts = [{"text": prompt}]
                    contents = [{"role": "user", "parts": parts}]
                    response = self._google_client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config={"temperature": 0}
                    )
                    return self._extract_google_text(response)
                except Exception:
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        raise
        raise RuntimeError("LLM provider not configured. Set provider to openai, openrouter, or google.")

    def _build_rerank_prompt(self, query_text: str, candidate_ids: List[str]) -> str:
        """Build a prompt that ranks the candidate snippets."""
        if not candidate_ids:
            raise RuntimeError("No candidate snippets to rank.")

        # Build criteria based on require_reason flag
        reason = 'answer or address on why this item can be recommended to the that user require you to inform user about based on the given conversation'
        additional_info = 'directly provide additional information for this item that you think user need to know to support his/her decision making based on current given conversation but currently not explicitly requested'
        
        if self.require_reason:
            criteria_text = reason
        else:
            criteria_text = additional_info
        
        instruction = f"""You are a helpful shopping assistant for a user that wearing a AR glass.

Task:
You will have the following information:
1. a conversation between you and the user, 
2. multiple key from the catalog of the reccomended items.
{'3. the segment of the item visual appearance as an image. (you should use both coversation and image to make decision)' if self.use_image else ''}

For EACH key, determine you if want do display the key and its content as a label on the AR glass based on the fact that if you think the content in this key can {criteria_text}

RANK all keys from most satisfy to least satisfy the above requirement.

Conversation: {query_text}

You will be given {len(candidate_ids)} keys below. Rank them and turn the top {min(len(candidate_ids), self.rerank_top_n)} keys from 1 (most satisfy) to {min(len(candidate_ids), self.rerank_top_n)} ({min(len(candidate_ids), self.rerank_top_n)}th most satisfy).

Return ONLY a JSON array with this exact format:
[
  {{"id": "1", "rank": 1, "rationale": "brief explanation why this key is ranked first"}},
  {{"id": "2", "rank": 2, "rationale": "brief explanation why this key is ranked second"}},
  ...
]

For each key, provide a brief rationale explaining why you assigned that rank based on the requirment.

The ranking should be:
- Rank 1 = most relevant key
- Rank 2 = second most relevant key
- ...
- Rank {min(len(candidate_ids), self.rerank_top_n)} = the {min(len(candidate_ids), self.rerank_top_n)}th most relevant key

KEYS:
"""
        for idx, sid in enumerate(candidate_ids, 1):
            text = self.snippet_texts.get(sid, "")
            instruction += f"\n[{idx}] KEY_ID: {sid}\nKEY_TEXT: {text}\n---\n"
        
        instruction += "\nReturn ONLY the JSON array, no other text."
        return instruction

    def _parse_batch_response(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse the batch response, handling various formats."""
        if not raw_text:
            return []
        
        # Strip markdown code fences if present
        text = raw_text.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        
        # Try to parse as JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                # Sometimes LLM returns {"ranking": [...]} or {"results": [...]}
                for key in ["ranking", "ranks", "results", "items", "snippets"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                # If it's a dict with id/rank, wrap it
                if "id" in parsed or "rank" in parsed:
                    return [parsed]
        except json.JSONDecodeError:
            # Try to extract JSON array from text
            m = re.search(r'\[[\s\S]*?\]', text)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
        
        return []

    def _extract_google_text(self, response: Any) -> str:
        """
        Extract plain text from a Google GenAI response.

        Handles both:
        - New `genai.Client` HttpResponse with `candidates[0].content.parts[*].text`
        - Any direct `.text` attribute on the response
        Also strips markdown code fences like ```json ... ``` so that json.loads works.
        """
        # Prefer a top-level `text` attribute if present
        text = getattr(response, "text", None)

        # Fallback: look into `candidates[0].content.parts[*].text`
        if not text:
            candidates = getattr(response, "candidates", None)
            if candidates:
                parts: List[str] = []
                first = candidates[0]
                content = getattr(first, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        value = getattr(part, "text", None)
                        if value:
                            parts.append(value)
                elif getattr(first, "parts", None):
                    for part in first.parts:
                        value = getattr(part, "text", None)
                        if value:
                            parts.append(value)
                if parts:
                    text = "\n".join(parts)

        if not text:
            return ""

        raw = str(text).strip()

        # If the model wrapped JSON in fenced code blocks, unwrap it.
        # Example:
        # ```json
        # { "relevance": 3 }
        # ```
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
        if m:
            return m.group(1).strip()

        return raw

    def _llm_rerank(self, query_text: str, candidate_ids: List[str]) -> Optional[Dict[str, float]]:
        """Rerank candidates using LLM."""
        if not candidate_ids:
            return {}
        
        prompt = self._build_rerank_prompt(query_text, candidate_ids)
        raw_text = self._call_llm_single(prompt)
        
        # Parse the response to get rankings
        parsed = self._parse_batch_response(raw_text)
        
        # Build rank map and rationale map: snippet_id -> rank
        rank_map: Dict[str, int] = {}
        rerank_scores: Dict[str, float] = {}
        self.rationale_map = {}  # Reset rationale map
        
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    sid = str(item.get("id") or "")
                    # clean sid to int only, remove any non-numeric characters
                    sid = ''.join(filter(str.isdigit, sid))
                    try:
                        _ = int(sid)
                    except Exception:
                        print(f'[label_selection] Error: Invalid snippet id: {sid}')
                        continue
                    rank = item.get("rank")
                    rationale = item.get("rationale", "")
                    if sid and rank is not None:
                        try:
                            rank_int = int(rank)
                            rank_map[sid] = rank_int
                            # Convert rank to score: rank 1 -> highest score, rank N -> lowest score
                            # Score = rerank_top_n - rank + 1, so rank 1 gets rerank_top_n, rank N gets 1
                            rerank_scores[sid] = float(self.rerank_top_n - rank_int + 1)
                            if rationale:
                                self.rationale_map[sid] = str(rationale).strip()
                        except Exception:
                            continue
        
        return rerank_scores if rerank_scores else None

    def score(self, query_text: str, topk: int) -> MethodResult:
        """Score snippets using dense retrieval followed by LLM reranking."""
        if not self.snippet_texts:
            return MethodResult(scores=[])

        # Dense retrieval to get initial ranking
        dense_topk = max(topk, self.initial_k)
        dense_result = self.dense.score(query_text, topk=dense_topk).scores
        if not dense_result:
            return MethodResult(scores=[])

        candidate_ids = [sid for sid, _ in dense_result[: self.initial_k]]
        if not candidate_ids:
            return MethodResult(scores=dense_result[:topk])
        
        # LLM rerank the candidates
        rerank_scores = self._llm_rerank(query_text, candidate_ids)
        if not rerank_scores:
            return MethodResult(scores=dense_result[:topk])

        # Combine dense and rerank scores
        # Offset rerank scores so top-k remain above others while preserving rerank ordering
        dense_top_max = max(score for _, score in dense_result[: self.initial_k])
        score_map: Dict[str, float] = {sid: float(score) for sid, score in dense_result}
        for sid, rerank_score in rerank_scores.items():
            # Add rerank score on top of dense max score
            score_map[sid] = dense_top_max + float(rerank_score)

        pairs = list(score_map.items())
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return MethodResult(scores=pairs[: min(topk, len(pairs))])

