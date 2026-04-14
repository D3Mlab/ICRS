from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.cache_manager import CacheManager

from .base import MethodResult, RelevanceMethod
from ..utils.io import stable_hash

from google import genai  # type: ignore
from openai import OpenAI


PROMPT_VERSION = "v2"  # Updated to include rationale


class LLMJudgeLinkerMethod(RelevanceMethod):
    def __init__(
        self,
        provider: str,
        model: str,
        cache_dir: Optional[Path],
        dataset_root: Optional[Path] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        image_path: Optional[Path] = None,
        require_reason: bool = True,
        use_image: bool = True,
    ):
        self.provider = (provider or "openai").lower()
        self.model = model
        self.dataset_root = CacheManager.infer_dataset_root(dataset_root, cache_dir)
        self.cache_base = cache_dir
        self.cache_module = "label_selection"
        self.cache_method = "llm_judge"
        self.cache_variant = None
        self.snippet_ids: List[str] = []
        self.snippet_texts: Dict[str, str] = {}
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

    def fit(self, snippet_texts: Dict[str, str]) -> None:
        self.snippet_ids = sorted(snippet_texts.keys())
        self.snippet_texts = snippet_texts

    def _read_image_b64(self, path: Path) -> str:
        """Read image file and return as base64 string."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_llm_single(self, prompt: str) -> str:
        # Build content with image if available and use_image is True
        print(f'[label_selection] use_image: {self.use_image} with image_path: {self.image_path}')
        if self.use_image and self.image_path and self.image_path.exists():
            image_b64 = self._read_image_b64(self.image_path)
            image_ext = self.image_path.suffix.lower()
            mime_type = "image/png" if image_ext == ".png" else "image/jpeg"
        else:
            print(f'[label_selection] image_path: {self.image_path}')
            image_b64 = None
            mime_type = None
        print(f'[label_selection] provider: {self.provider}')
        print(f'[label_selection] model: {self.model}')
        if self.provider in {"openai", "openrouter"}:
            try:
                client_kwargs: Dict[str, Any] = {}
                key = os.getenv(self.api_key_env) or os.getenv("OPENAI_API_KEY")
                if key:
                    client_kwargs["api_key"] = key
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
                    if not image_b64 and self.use_image:
                        raise RuntimeError("Image is required but not provided")
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
                    print(f"Error: {exc}")
                    print(f"Attempt: {attempt}")
                    time.sleep(0.5 * (attempt + 1))
            raise RuntimeError("LLM call failed after retries")
        raise RuntimeError("LLM provider not configured. Set provider to openai, openrouter, or google.")

    def _build_batch_prompt(self, query_text: str) -> str:
        """Build a single prompt that scores all snippets in one batch."""
        if not self.snippet_ids:
            raise RuntimeError("fit() must be called before score(). No snippets available.")

        # Criteria text (EIS vs IN-style objective)
        reason = (
            "Determine whether the snippet directly answers or addresses information that the user has "
            "explicitly requested, stated, or confirmed in the conversation (i.e., explicit requests/preferences)."
        )
        additional_info = (
            "Determine whether the snippet provides additional, non-obvious information that would proactively "
            "help the user make a better decision, even though it was not explicitly requested in the conversation."
        )

        criteria_text = reason if self.require_reason else additional_info

        # Optional visual input line
        visual_line = "3. (Optional) a visual segment of the item's appearance as an image.\n" if self.use_image else ""

        instruction = f"""You are an assistant for an immersive conversational recommendation system.
    The user is wearing AR glasses and is physically present in a store.
    The recommended item is already visible in the user's field of view.

    Context:
    You are given:
    1. A conversation between the system and the user.
    2. A set of attribute snippets describing the recommended item.
    {visual_line}Your task:
    For EACH attribute snippet, decide whether it should be shown as an immersive textual label
    to support the user's decision-making in this physical setting.

    Evaluation Objective:
    {criteria_text}
    
    Conversation:
    {query_text}

    Scoring Scale (0–3):
    0 = Clearly irrelevant or does not satisfy the evaluation objective
    1 = Weak relevance or marginally related information
    2 = Moderately relevant but not among the most helpful attributes
    3 = Highly relevant and clearly satisfies the evaluation objective

    Instructions:
    You will be given {len(self.snippet_ids)} attribute snippets.
    For EACH snippet, assign a relevance score from 0 to 3.

    Output Format:
    Return ONLY a JSON array in the following exact format:
    [
    {{"id": "1", "relevance": 3, "rationale": "brief justification"}},
    {{"id": "2", "relevance": 1, "rationale": "brief justification"}},
    ...
    ]

    Output Requirements:
    - Include ALL snippet IDs exactly once.
    - Use the exact IDs provided below.
    - Provide a concise rationale grounded in the evaluation objective.
    - Do NOT include any text outside the JSON array.

    SNIPPETS:
    """

        for idx, sid in enumerate(self.snippet_ids, 1):
            text = self.snippet_texts.get(sid, "")
            instruction += f"\n[{idx}] SNIPPET_ID: {sid}\nSNIPPET_TEXT: {text}\n---\n"

        instruction += "\nReturn ONLY the JSON array, no other text."
        return instruction

    def score(self, query_text: str, topk: int) -> MethodResult:
        """Score all keys in a single batch call."""
        # Ensure fit() has been called
        if not self.snippet_ids or not self.snippet_texts:
            raise RuntimeError("fit() must be called before score(). No snippets available.")
        
        # Handle empty snippet list
        if len(self.snippet_ids) == 0:
            return MethodResult(scores=[])
        
        prompt = self._build_batch_prompt(query_text)

        # key_obj = {
        #     "query_hash": stable_hash(query_text),
        #     "snippet_ids_hash": stable_hash(",".join(self.snippet_ids)),
        #     "model": self.model,
        #     "prompt_version": PROMPT_VERSION,
        # }
        # namespace = "judge_batch"
        # cached = CacheManager.get_llm_record(
        #     module=self.cache_module,
        #     method=self.cache_method,
        #     model_name=self.model,
        #     namespace=namespace,
        #     key=key_obj,
        #     dataset=self.dataset_root,
        #     base_dirs=[self.cache_base],
        #     variant=self.cache_variant,
        #     require_reason=self.require_reason,
        # )
        # if cached is not None:
        #     record = cached.get("record", {})
        #     raw_text = record.get("content", "")
        # else:
        raw_text = self._call_llm_single(prompt)
        #     CacheManager.append_llm_record(
        #         module=self.cache_module,
        #         method=self.cache_method,
        #         model_name=self.model,
        #         namespace=namespace,
        #         key=key_obj,
        #         content=str(raw_text),
        #         record_meta={"provider": self.provider},
        #         dataset=self.dataset_root,
        #         base_dirs=[self.cache_base],
        #         variant=self.cache_variant,
        #     )

        # Parse the response
        
        parsed = self._parse_batch_response(raw_text)
        # Build score map and rationale map
        score_map: Dict[str, float] = {}
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
                        print('[label_selection] Error: Invalid snippet id: {sid}')
                        continue
                    rel = item.get("relevance")
                    rationale = item.get("rationale", "")
                    if sid and rel is not None:
                        try:
                            score_map[sid] = float(rel)
                            if rationale:
                                self.rationale_map[sid] = str(rationale).strip()
                        except Exception:
                            continue

        # Ensure all snippets have scores (default to 0.0 if missing)
        pairs: List[Tuple[str, float]] = []
        for sid in self.snippet_ids:
            score = score_map.get(sid, 0.0)
            pairs.append((sid, float(score)))

        pairs.sort(key=lambda x: (-x[1], x[0]))
        return MethodResult(scores=pairs[:topk])

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
                # Sometimes LLM returns {"scores": [...]} or {"results": [...]}
                for key in ["scores", "results", "items", "snippets"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                # If it's a dict with id/relevance, wrap it
                if "id" in parsed or "relevance" in parsed:
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


