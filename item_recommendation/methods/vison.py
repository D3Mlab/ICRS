from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm_base import LLMBase

from .base import BatchResult, ObjectRankingMethod


class VisonObjectRanker(LLMBase, ObjectRankingMethod):
    """VLM-based relevance scoring using only the product image (no metadata)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        provider: str = "openai",
        temperature_jitter: float = 0.3,
        api_base: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        dataset_root: Optional[Path] = None,
    ) -> None:
        super().__init__(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            base_url=api_base,
            default_headers=default_headers,
        )
        self.temperature_jitter = max(0.0, float(temperature_jitter))
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def temperature_for_run(self, base_temperature: float, run_index: int) -> float:
        if self.temperature_jitter <= 0.0:
            return max(0.0, min(1.0, base_temperature))
        jitter = random.uniform(-self.temperature_jitter, self.temperature_jitter)
        temp = base_temperature + jitter
        return max(0.0, min(1.0, temp))

    @staticmethod
    def _read_image_b64(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_path(self, segment_id: str) -> Optional[Path]:
        """Locate the image for the given segment_id under dataset_root/segments/crops."""
        if not self.dataset_root:
            return None
        item_id = str(segment_id).strip()
        if item_id.endswith(".json"):
            item_id = item_id[:-5]
        elif item_id.endswith(".png"):
            item_id = item_id[:-4]
        base_dir = self.dataset_root / "segments" / "crops"
        candidates = [base_dir / f"{item_id}.png", base_dir / f"{item_id}.jpg", base_dir / f"{item_id}.jpeg"]
        for path in candidates:
            if path.exists():
                return path
        return None

    def score_batch(
        self,
        user_query: str,
        batch: List[Dict[str, Any]],
        temperature: float,
        run_index: int,
    ) -> BatchResult:
        prompt = self._build_prompt(user_query, batch[0].get("id"))

        content_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for item in batch:
            segment_id = str(item.get("id", ""))
            image_path = self._get_image_path(segment_id)
            if not image_path or not image_path.exists():
                # Skip items without images
                continue

            try:
                image_b64 = self._read_image_b64(image_path)
            except Exception as exc:
                print(f"[Vison] Failed to load image for {segment_id}: {exc}")
                continue

            ext = image_path.suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            else:
                mime_type = "image/png"

            content_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}", "detail": "high"},
                }
            )
            # Minimal text so the model can reference the item id
            content_items.append({"type": "text", "text": f"Item ID: {segment_id}"})

        if len(content_items) == 1:
            # No images added
            return BatchResult(scores={}, parsed_items=[], raw_response=None, temperature=temperature, prompt=prompt)

        int_item_id = None
        try:
            int_item_id = int(batch[0].get("id"))
        except Exception:
            pass
        if 'item_' in batch[0].get("id") or int_item_id is None or int_item_id > 1000000:
            content = "You are a helpful shopping assistant that scores products based only on images." 
        else:
            content = "You are a helpful movie assistant that scores movies based only on posters."
        print(f"[item_recommendation] content: {content}")
        response = self.generate_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": content,
                },
                {"role": "user", "content": content_items},
            ],
            temperature=temperature,
        )

        raw_text = response.choices[0].message.content  # type: ignore[assignment]
        data = self.try_parse_json(raw_text)

        scores: Dict[str, float] = {}
        parsed_items: List[Dict[str, Any]] = []

        def _record(entry: Dict[str, Any]) -> None:
            sid = str(entry.get("id"))
            if not sid:
                return
            try:
                score = float(entry.get("score", 1))
            except Exception:
                score = 1.0
            score = max(0.0, min(3.0, score))
            scores[sid] = score
            parsed_items.append({"id": sid, "score": score, "rationale": entry.get("rationale")})

        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    _record(row)
        elif isinstance(data, dict):
            if isinstance(data.get("results"), list):
                for row in data["results"]:
                    if isinstance(row, dict):
                        _record(row)
            elif isinstance(data.get("items"), list):
                for row in data["items"]:
                    if isinstance(row, dict):
                        _record(row)

        # Ensure all items in batch have scores (None if missing)
        for item in batch:
            sid = item.get("id")
            if sid not in scores:
                parsed_items.append({"id": sid, "score": None, "rationale": "not_returned_by_model"})

        return BatchResult(
            scores=scores,
            parsed_items=parsed_items,
            raw_response=raw_text,
            temperature=temperature,
            prompt=prompt,
        )

    @staticmethod
    def _build_prompt(user_query: str, item_id: str) -> str:
        
        int_item_id = None
        try:
            int_item_id = int(item_id)
        except Exception:
            pass
        if 'item_' in item_id or int_item_id is None or int_item_id > 1000000:
            return f"""
                    You are a helpful shopping assistant that helps users find the best products for their needs.

                    Task: For each catalog item, you will see:
                    1. A product image (showing the visual appearance of the item)
                    2. A short line with the Item ID

                    Based on the given conversation:

                    Use ONLY the image to make your decision. Do not rely on any metadata.

                    For each item, assign a score using the UMBRELLA 0-3 scale:
                    - 0 = definitely not relevant to the user requirement / user will definitely not like the item
                    - 1 = weak/minimal evidence
                    - 2 = user will likely like the item
                    - 3 = user will definitely like the item

                    Guidance:
                    - You must base your judgment purely on the image (color, style, material appearance, design, overall aesthetic).
                    - In your rationale, explicitly reference what you see in the image.

                    Return ONLY a JSON array with entries:
                    {{"id": "segment_id", "score": 0-3, "rationale": "..."}} for example:
                    [
                        {{"id": "123", "score": 2, "rationale": "The product is about a boy who is a superhero."}},
                        {{"id": "456", "score": 3, "rationale": "The product is about a boy who is a superhero."}},
                    ]

                    Conversation: {user_query}

                    You will receive items in the following format:
                    - Each item shows an image
                    - Each item will have a short text line with the Item ID

                    For each item, analyze ONLY the image to determine relevance to the user query.
                    """
        else:
            print('[Vison] using movie prompt')
            return f"""
                    You are helpfull movie assistant that helps users find the best movies for their needs.

                    Task: For each catalog item, you will see:
                    1. A movie poster (showing the visual appearance of the movie)
                    2. A short line with the Item ID

                    Based on the given conversation:

                    Use ONLY the image to make your decision. Do not rely on any metadata.

                    For each item, assign a score that if you think the user will like the movie using the UMBRELLA 0-3 scale:
                    - 0 = definitely not relevant to the user requirement / user will definitely not like the movie
                    - 1 = weak/minimal evidence
                    - 2 = user will likely like the movie
                    - 3 = user will definitely like the movie

                    Guidance:
                    - You must base your judgment purely on the image (color, style, material appearance, design, overall aesthetic).
                    - In your rationale, explicitly reference what you see in the image.

                    Return ONLY a JSON array with entries for each movie:
                    {{"id": "segment_id", "score": 0-3, "rationale": "..."}}
                    for example:
                    [
                        {{"id": "123", "score": 2, "rationale": "The movie is about a boy who is a superhero."}},
                        {{"id": "456", "score": 3, "rationale": "The movie is about a boy who is a superhero."}},
                    ]

                    Conversation: {user_query}

                    You will receive items in the following format:
                    - Each item shows an image
                    - Each item will have a short text line with the Item ID

                    For each item, analyze ONLY the image to determine relevance to the user query.
                    """