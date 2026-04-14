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

from tqdm import tqdm

PROMPT_VERSION = "v3"  # Updated for expansion pipeline


class LLMExpansionMethod(RelevanceMethod):
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
        expansion_model: Optional[str] = None,  # Smaller model for expansion/extraction
        entailment_model: Optional[str] = None,  # Larger model for entailment
    ):
        self.provider = (provider or "openai").lower()
        # Use separate models: smaller for expansion/extraction, larger for entailment
        # Default: use provided model for all tasks if not specified
        self.model = model  # Keep for backward compatibility
        if self.provider == 'openai':
            self.expansion_model = 'gpt-4o-mini' # Smaller model for state extraction, visibility, question expansion
            self.entailment_model = 'gpt-5.1'  # Larger model for entailment scoring
        if self.provider == 'google':
            self.expansion_model = 'gemini-2.5-flash' # Smaller model for state extraction, visibility, question expansion
            self.entailment_model = 'gemini-2.5-pro'  # Larger model for entailment scoring
        if self.provider == 'openrouter':
            self.expansion_model = 'qwen/qwen3-vl-8b-a3b-instruct' # Smaller model for state extraction, visibility, question expansion
            self.entailment_model = 'qwen/qwen3-vl-30b-a3b-instruct'  # Larger model for entailment scoring
        self.dataset_root = CacheManager.infer_dataset_root(dataset_root, cache_dir)
        self.cache_base = cache_dir
        self.cache_module = "label_selection"
        self.cache_method = "llm_expansion"
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

        self.meta_data_fields: Dict[str, List[str]] = {}
        self.enviorment_attributes: List[str] = []
        self.query_attributes: List[str] = []
        
        # Pipeline state
        self.dialogue_state: Optional[Dict[str, Any]] = None
        self.visibility_map: Optional[Dict[str, float]] = None
        self.expanded_questions: List[Dict[str, Any]] = []  # List of {question, probability, attribute_type}
        self.lambda_penalty: float = 0.3  # Weight for visibility penalty
        self.attribute_weights: Dict[str, float] = {
            "fabric": 1.0,
            "care": 0.8,
            "fit": 1.2,
            "warmth": 1.0,
            "opacity": 0.7,
            "lining": 0.9,
            "stretch": 0.8,
            "itch": 1.1,
            "durability": 0.9,
            "waterproof": 1.0,
            "breathability": 0.8,
        }

    def fit(self, snippet_texts: Dict[str, str]) -> None:
        self.snippet_ids = sorted(snippet_texts.keys())
        self.snippet_texts = snippet_texts

    def _read_image_b64(self, path: Path) -> str:
        """Read image file and return as base64 string."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_llm_single(self, prompt: str, model: Optional[str] = None, system: Optional[str] = None) -> str:
        """Call LLM with optional model override. Defaults to self.model."""
        use_model = model or self.model
        # Build content with image if available and use_image is True
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
                    # Build messages list for OpenAI/OpenRouter
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    if image_b64:
                        user_content = [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}", "detail": "high"}},
                        ]
                    else:
                        user_content = prompt
                    messages.append({"role": "user", "content": user_content})
                    
                    resp = client.chat.completions.create(
                        model=use_model,
                        messages=messages,
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
                    # Google API uses different content format
                    # Prepend system message to prompt if provided
                    full_prompt = f"{system}\n\n{prompt}" if system else prompt
                    
                    if image_b64:
                        # For images, use parts format
                        parts = [
                            {"text": full_prompt},
                            {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                        ]
                    else:
                        # For text-only, wrap in parts format
                        parts = [{"text": full_prompt}]
                    contents = [{"role": "user", "parts": parts}]

                    response = self._google_client.models.generate_content(
                        model=use_model,
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

    def _extract_dialogue_state(self, conversation: str) -> Dict[str, Any]:
        """Extract dialogue state S from conversation C."""
        system = 'you are an expert in shopping assistant that helps users find the best products for their needs.'
        prompt = f"""Analyze the following shopping conversation and extract the dialogue state.

        Conversation:
        {conversation}

        Extract the following information as a JSON object. 

        For example,
        {{
        'context': "string (e.g., 'shopping', 'movie', 'music', 'food', 'travel', 'other')",
        "occasion": "string (e.g., 'farm visit', 'office', 'casual outing')",
        "formality_band": "string (e.g., 'casual', 'business casual', 'formal')",
        "comfort_constraints": ["list of comfort requirements"],
        "indoor_outdoor": "string ('indoor', 'outdoor', 'both')",
        "temperature_assumptions": "string (e.g., 'cool', 'warm', 'variable')",
        "style_preference": "string ('look nice', 'flashy', 'understated', etc.)",
        "activity_type": "string (e.g., 'walking', 'sitting', 'active')",
        "time_of_day": "string (e.g., 'afternoon', 'morning', 'evening')",
        "weather_conditions": ["list of weather factors mentioned"]
        }}


        DO NOT LIMIT TO THE EXAMPLE ABOVE.  DIALOGUE STATE SHOULD BE DEPEND ON CONVERSATION.

        Return ONLY the JSON object, no other text."""
        
        raw_text = self._call_llm_single(prompt, model=self.expansion_model, system=system)
        
        # Parse JSON response
        try:
            text = raw_text.strip()
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
            if m:
                text = m.group(1).strip()
            state = json.loads(text)
            return state if isinstance(state, dict) else {}
        except Exception:
            return {}

    def _extract_visibility_map(self) -> Dict[str, float]:
        """Extract visibility map V = VisibleAttr(I, J) using VLM."""

        system = 'you are a user who is shopping for an item. You are looking at the item image and you are trying to understand the item without any metadata.'
        prompt = """Analyze the item image and determine which attributes can be visually inferred without metadata.

        return a list of attributes that can be visually inferred without metadata.

            Return ONLY a JSON object with this exact format:
            {{
            "attributes": ["list of attributes"]
            }}

            For example,
            {{
            "attributes": ["mesh finishing", "sneaker design", "white color", "synthetic leather material"]
            }}"""

        raw_text = self._call_llm_single(prompt, model=self.expansion_model, system=system)
        # Parse JSON response
        try:
            text = raw_text.strip()
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
            if m:
                text = m.group(1).strip()
            visibility = json.loads(text)
            return visibility["attributes"] if isinstance(visibility, dict) and "attributes" in visibility else []
        except Exception as e:
            print(f"[llm_expansion] Error extracting visibility map: {e}")
            return {}

    def _expand_questions(
        self, 
        dialogue_state: Dict[str, Any], 
        visibility_map: List[str],
        category: Optional[str] = None,
        query_text: str = ""
    ) -> List[Dict[str, Any]]:
        """Generate anticipatory questions Q = {(q_n, p_n, a_n)}."""
        state_str = json.dumps(dialogue_state, indent=2)
        visibility_str = ", ".join(visibility_map)
        system = f"""You are a user who is shopping for an item. """

        prompt = f"""
        Based on 
        1. the provided conversations between you and an shopping assistant, 
        2. the dialogue state extracted from the conversations 
        3. the visibility map of an item that reccomend to you by the shopping assistant
        4. the image segment of the item that the shopping assistant reccomended to you
        
        generate plausible follow-up questions, you, as the user, for the item that the shopping assistant reccomended to you, would ask before purchasing.

        Focus on NON-VISIBLE uncertainties - questions about attributes that cannot be determined from the image.

        Conversation:
        {query_text}

        Dialogue State:
        {state_str}

        Visibility Map (attributes that can be visually inferred without metadata):
        {visibility_str}

        Category: {category or "general"}

        Generate 8-12 questions. For each question, provide:
        - question: The actual question text (e.g., "Is it machine washable?", "How warm is it?", etc.)
        - probability: Likelihood you would ask this based on the dialogue state and visibility (0.0 to 1.0)
        - attribute_type: one word summary of what the question is ask for (e.g., care, warmth, fit, etc.)

        Return ONLY a JSON array:
        [
        {{"question": "Is it machine washable?", "probability": 0.15, "attribute_type": "care"}},
        {{"question": "How warm is it?", "probability": 0.20, "attribute_type": "warmth"}},
        ...
]
"""
        raw_text = self._call_llm_single(prompt, model=self.expansion_model, system=system)
        # Parse JSON response
        try:
            text = raw_text.strip()
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
            if m:
                text = m.group(1).strip()
            m = re.search(r'\[[\s\S]*?\]', text)
            if m:
                text = m.group(0)
            questions = json.loads(text)
            if isinstance(questions, list):
                # Normalize probabilities
                total_prob = sum(q.get("probability", 0.0) for q in questions if isinstance(q, dict))
                if total_prob > 0:
                    for q in questions:
                        if isinstance(q, dict):
                            q["probability"] = q.get("probability", 0.0) / total_prob
                return questions
        except Exception:
            pass
        return []

    def _build_augmented_conversation(self, original_conversation: str, item_id: Optional[str] = None, topk: int = 1) -> str:
        """Build augmented conversation C_i^{+} = C ⊕ (Assistant recommends item i) ⊕ Q_i."""
        # Format expanded questions as hypothetical seeker questions
        questions_list = []
        # for each attribute type, add the question to the questions_list
        # get unique attribute types
        unique_attribute_types = list(set(q.get("attribute_type", "") for q in self.expanded_questions))
        for attribute_type in unique_attribute_types:
            # sort the self.expanded_questions by probability descending for the given attribute type
            # filter the attribute type questions by the given attribute type
            attribute_type_questions = [q for q in self.expanded_questions if q.get("attribute_type", "") == attribute_type]
            # sort the attribute type questions by probability descending
            attribute_type_questions.sort(key=lambda x: x.get("probability", 0.0), reverse=True)
            # add the topk questions to the questions_list
            questions_list.extend([q.get("question", "") for q in attribute_type_questions[:topk]])
        
        # Build augmented conversation
        item_ref = f"Item #{item_id}" if item_id else "this item"
        questions_text = "\n".join(questions_list)
        
        augmented = f"""{original_conversation}\nAssistant: I'd recommend {item_ref}. Before you decide, do you want to confirm anything about this item?\nSeeker:
        {questions_text}"""
                
        return augmented.strip(), len(questions_list)
    
    def _build_batch_prompt(self, query_text: str) -> str:
        """Build a single prompt that ranks all snippets in one batch."""
        if not self.snippet_ids:
            raise RuntimeError("fit() must be called before score(). No snippets available.")
        
        instruction = f"""You are a helpful shopping assistant.

    Task:
    We will provide you with 
    1. a conversation between you and the user, 
    2. multiple snippets from an item's metadata.
    3. the segment of the item visual appearance as an image.

Your task is to RANK all snippets from most relevant to least relevant based on the conversation.

For each snippet, consider if it can answer the questions that the user would ask to understand this items based on the conversation.
Conversation: {query_text}

You will be given {len(self.snippet_ids)} snippets below. Rank them and turn the top 5 items from 1 (most relevant) to 5 (fifth most relevant).

Return ONLY a JSON array with this exact format:
[
  {{"id": "1", "rank": 1, "rationale": "brief explanation why this snippet is ranked first"}},
  {{"id": "2", "rank": 2, "rationale": "brief explanation why this snippet is ranked second"}},
  ...
]

For each snippet, provide a brief rationale explaining why you assigned that rank based on the requirment and if it can be inferred directly from the visual or physical appearance of the item provided.

The ranking should be:
- Rank 1 = most relevant snippet
- Rank 2 = second most relevant snippet
- ...
- Rank 5 = the fifth most relevant snippet

SNIPPETS:
"""
        for idx, sid in enumerate(self.snippet_ids, 1):
            text = self.snippet_texts.get(sid, "")
            instruction += f"\n[{idx}] SNIPPET_ID: {sid}\nSNIPPET_TEXT: {text}\n---\n"
        
        instruction += "\nReturn ONLY the JSON array, no other text."
        return instruction

    def score(self, query_text: str, topk: int) -> MethodResult:
        """Score snippets using the expansion pipeline: State + Visibility + Questions + Entailment."""
        # Ensure fit() has been called
        if not self.snippet_ids or not self.snippet_texts:
            raise RuntimeError("fit() must be called before score(). No snippets available.")
        
        # Handle empty snippet list
        if len(self.snippet_ids) == 0:
            return MethodResult(scores=[])
        
        # Step 1: Extract dialogue state S
        print(f"[llm_expansion] Extracting dialogue state with model: {self.expansion_model}")
        self.dialogue_state = self._extract_dialogue_state(query_text)

        print(f"[llm_expansion] Dialogue state: {self.dialogue_state}")
        
        # Step 2: Extract visibility map V
        print(f"[llm_expansion] Extracting visibility map with model: {self.expansion_model}")
        self.visibility_map = self._extract_visibility_map()

        print(f"[llm_expansion] Visibility map: {self.visibility_map}")
        
        # Step 3: Expand questions Q = {(q_n, p_n, a_n)}
        print(f"[llm_expansion] Expanding questions with model: {self.expansion_model}")
        category = self.dialogue_state.get("style_category") or self.dialogue_state.get("occasion", "")
        self.expanded_questions = self._expand_questions(
            self.dialogue_state, 
            self.visibility_map,
            category,
            query_text
        )

        print(f"[llm_expansion] Expanded questions: {self.expanded_questions}")
        
        if not self.expanded_questions:
            print("[llm_expansion] Warning: No questions generated, falling back to simple ranking")
            # Fallback to original method
            return self._fallback_score(query_text, topk)
        
        # Step 4: Build augmented conversation C_i^{+} = C ⊕ (Assistant recommends item i) ⊕ Q_i
        # Extract item ID from snippet_ids if available (e.g., "item_2.json#field#0_10" -> "item_2")
        item_id = None
        if self.snippet_ids:
            first_sid = self.snippet_ids[0]
            # Try to extract item ID from snippet ID format
            if "#" in first_sid:
                item_id = first_sid.split("#")[0]
            elif "." in first_sid:
                item_id = first_sid.split(".")[0]
        
        augmented_conversation, num_questions = self._build_augmented_conversation(query_text, item_id)
        print(f"[llm_expansion] Building augmented conversation with {num_questions} questions...")
        # Step 5: Use larger model to rank snippets based on augmented conversation
        print(f"[llm_expansion] Ranking snippets with model: {self.entailment_model}")
        prompt = self._build_batch_prompt(augmented_conversation)
        
        key_obj = {
            "query_hash": stable_hash(augmented_conversation),
            "snippet_ids_hash": stable_hash(",".join(self.snippet_ids)),
            "model": self.entailment_model,
            "prompt_version": PROMPT_VERSION,
            "require_reason": self.require_reason,
        }
        namespace = "expansion_rank"
        cached = CacheManager.get_llm_record(
            module=self.cache_module,
            method=self.cache_method,
            model_name=self.entailment_model,
            namespace=namespace,
            key=key_obj,
            dataset=self.dataset_root,
            base_dirs=[self.cache_base],
            variant=self.cache_variant,
        )
        
        if cached is not None:
            record = cached.get("record", {})
            raw_text = record.get("content", "")
        else:
            raw_text = self._call_llm_single(prompt, model=self.entailment_model)
            CacheManager.append_llm_record(
                module=self.cache_module,
                method=self.cache_method,
                model_name=self.entailment_model,
                namespace=namespace,
                key=key_obj,
                content=str(raw_text),
                record_meta={"provider": self.provider},
                dataset=self.dataset_root,
                base_dirs=[self.cache_base],
                variant=self.cache_variant,
            )
        
        # Parse the response to get rankings (similar to llm_list.py)
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
                        print(f'[label_selection] Error: Invalid snippet id: {sid}')
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
        
        # Convert ranks to scores (similar to llm_list.py)
        num_snippets = len(self.snippet_ids)
        if num_snippets == 0:
            return MethodResult(scores=[])
        
        pairs: List[Tuple[str, float]] = []
        for sid in self.snippet_ids:
            rank = rank_map.get(sid)
            if rank is not None:
                # Convert rank to score: rank 1 -> highest score, rank N -> lowest score
                # Score = num_snippets - rank + 1, so rank 1 gets num_snippets, rank N gets 1
                # But limit to top 5 like llm_list
                score = float(5 - rank + 1) if rank <= 5 else 0.0
            else:
                # If snippet not in ranking, assign lowest score
                score = 0.0
            pairs.append((sid, score))
        
        # Sort by score descending, then by snippet_id for determinism
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return MethodResult(scores=pairs[:topk])
    
    def _fallback_score(self, query_text: str, topk: int) -> MethodResult:
        """Fallback to original ranking method if expansion fails."""
        prompt = self._build_batch_prompt(query_text)

        key_obj = {
            "query_hash": stable_hash(query_text),
            "snippet_ids_hash": stable_hash(",".join(self.snippet_ids)),
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "require_reason": self.require_reason,
        }
        namespace = "list_rank"
        cached = CacheManager.get_llm_record(
            module=self.cache_module,
            method=self.cache_method,
            model_name=self.model,
            namespace=namespace,
            key=key_obj,
            dataset=self.dataset_root,
            base_dirs=[self.cache_base],
            variant=self.cache_variant,
        )
        if cached is not None:
            record = cached.get("record", {})
            raw_text = record.get("content", "")
        else:
            raw_text = self._call_llm_single(prompt)
            CacheManager.append_llm_record(
                module=self.cache_module,
                method=self.cache_method,
                model_name=self.model,
                namespace=namespace,
                key=key_obj,
                content=str(raw_text),
                record_meta={"provider": self.provider},
                dataset=self.dataset_root,
                base_dirs=[self.cache_base],
                variant=self.cache_variant,
            )

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
                        print(f'[label_selection] Error: Invalid snippet id: {sid}')
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

        # Convert ranks to scores
        pairs: List[Tuple[str, float]] = []
        for sid in self.snippet_ids:
            rank = rank_map.get(sid)
            if rank is not None:
                score = float(5 - rank + 1)
            else:
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

