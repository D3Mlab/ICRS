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
from ..utils.io import stable_hash

from google import genai  # type: ignore
from openai import OpenAI


PROMPT_VERSION = "v2"  # Updated to include rationale


class LLMListRankerMethod(RelevanceMethod):
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
        self.cache_method = "llm_list"
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
        print(f'[label_selection] provider: {self.provider}')
        print(f'[label_selection] model: {self.model}')
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
                    if not image_b64 and self.use_image:
                        raise RuntimeError("Image is required but not provided")
                    print(f"Content: {self.model}")
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": content}],
                        temperature=0,
                    )
                    return resp.choices[0].message.content or ""
                except Exception as exc:
                    print(f"Error: {exc}")
                    print(f"Attempt: {attempt}")
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

    def _build_batch_prompt(self, query_text: str) -> str:
        """Build a single robust prompt that ranks all snippets in one batch."""
        if not self.snippet_ids:
            raise RuntimeError("fit() must be called before score(). No snippets available.")

        # Map require_reason to evaluation objective / criterion text
        # require_reason=True  -> EIS-style (explicitly address stated requests/preferences)
        # require_reason=False -> IN-style (proactively provide non-obvious helpful info)
        eis_criterion = (
            "Prefer snippets that directly answer or address the user's explicitly stated "
            "requests, questions, or preferences in the conversation."
        )
        in_criterion = (
            "Prefer snippets that provide additional, non-obvious information that would "
            "proactively help the user make a better decision, even if it was not explicitly "
            "requested in the conversation."
        )

        objective_name = "EIS (Explicit Information Satisfaction)" if self.require_reason else "IN (Implicit Information Need)"
        criteria_text = eis_criterion if self.require_reason else in_criterion

        instruction = f"""You are an assistant for an immersive conversational recommendation system.
    The user is wearing AR glasses and is physically present in a store.
    The recommended item is already visible in the user's field of view.

    Context:
    You are given:
    1) A conversation between the system and the user.
    2) A set of attribute snippets describing the recommended item.
    {'3) A visual segment of the item appearance.' if self.use_image else ''}

    Your task:
    Rank the attribute snippets by how suitable they are to display as immersive textual labels
    to support the user's decision-making in this physical setting.

    Evaluation Objective:
    {objective_name}
    {criteria_text}

    Conversation:
    {query_text}

    Ranking Instruction:
    You will be given {len(self.snippet_ids)} attribute snippets below.
    Rank them from most suitable to least suitable under the evaluation objective and constraints above.
    Return ONLY the TOP 5 snippets, ranked 1 (most suitable) to 5 (fifth).

    Output Format:
    Return ONLY a JSON array with this exact format:
    [
    {{"id": "<INFORMATION_ID>", "rank": 1, "rationale": "brief justification"}},
    {{"id": "<INFORMATION_ID>", "rank": 2, "rationale": "brief justification"}},
    {{"id": "<INFORMATION_ID>", "rank": 3, "rationale": "brief justification"}},
    {{"id": "<INFORMATION_ID>", "rank": 4, "rationale": "brief justification"}},
    {{"id": "<INFORMATION_ID>", "rank": 5, "rationale": "brief justification"}}
    ]

    Output Requirements:
    - Include exactly 5 entries.
    - Use the exact INFORMATION_ID values provided below.
    - Rationale must be grounded in the evaluation objective and constraints.
    - Do NOT include any text outside the JSON array.

    SNIPPETS:
    """

        for idx, sid in enumerate(self.snippet_ids, 1):
            text = self.snippet_texts.get(sid, "")
            instruction += f"\n[{idx}] INFORMATION_ID: {sid}\nINFORMATION_TEXT: {text}\n---\n"

        instruction += "\nReturn ONLY the JSON array, no other text."
        return instruction

    def score(self, query_text: str, topk: int) -> MethodResult:
        """Rank all snippets in a single batch call, then convert ranks to scores."""
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
        #     "require_reason": self.require_reason,
        # }
        # namespace = "list_rank"
        # cached = CacheManager.get_llm_record(
        #     module=self.cache_module,
        #     method=self.cache_method,
        #     model_name=self.model,
        #     namespace=namespace,
        #     key=key_obj,
        #     dataset=self.dataset_root,
        #     base_dirs=[self.cache_base],
        #     variant=self.cache_variant,
        # )
        # if cached is not None:
        #     record = cached.get("record", {})
        #     raw_text = record.get("content", "")
        # else:
        raw_text = self._call_llm_single(prompt)
            # CacheManager.append_llm_record(
            #     module=self.cache_module,
            #     method=self.cache_method,
            #     model_name=self.model,
            #     namespace=namespace,
            #     key=key_obj,
            #     content=str(raw_text),
            #     record_meta={"provider": self.provider},
            #     dataset=self.dataset_root,
            #     base_dirs=[self.cache_base],
            #     variant=self.cache_variant,
            # )

        # Parse the response to get rankings
        parsed = self._parse_batch_response(raw_text)

        # Build rank map and rationale map: snippet_id -> rank
        rank_map: Dict[str, int] = {}
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
                    rank = item.get("rank")
                    rationale = item.get("rationale", "")
                    if sid and rank is not None:
                        try:
                            rank_map[sid] = int(rank)
                            if rationale:
                                self.rationale_map[sid] = str(rationale).strip()
                        except Exception:
                            continue

        # Convert ranks to scores: rank 1 gets highest score, rank 2 gets second highest, etc.
        # Use descending scores so higher rank (smaller number) = higher score
        num_snippets = len(self.snippet_ids)
        if num_snippets == 0:
            return MethodResult(scores=[])
        
        # Assign scores: rank 1 -> score = num_snippets, rank 2 -> score = num_snippets - 1, etc.
        # This ensures rank 1 has the highest score
        pairs: List[Tuple[str, float]] = []
        for sid in self.snippet_ids:
            rank = rank_map.get(sid)
            if rank is not None:
                # Convert rank to score: rank 1 -> highest score, rank N -> lowest score
                # Score = num_snippets - rank + 1, so rank 1 gets num_snippets, rank N gets 1
                score = float(5 - rank + 1)
            else:
                # If snippet not in ranking, assign lowest score
                score = 0.0
            pairs.append((sid, score))

        # Sort by score descending, then by snippet_id for determinism
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

