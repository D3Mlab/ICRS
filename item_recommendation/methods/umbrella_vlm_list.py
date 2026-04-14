from __future__ import annotations

import base64
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm_base import LLMBase
from .base import BatchResult, ObjectRankingMethod


class UmbrellaVLMListObjectRanker(LLMBase, ObjectRankingMethod):
    """VLM-based item ranking (list-style) using images + metadata."""

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
        if not self.dataset_root:
            return None
        item_id = str(segment_id).strip()
        if item_id.endswith(".json"):
            item_id = item_id[:-5]
        elif item_id.endswith(".png"):
            item_id = item_id[:-4]
        base_dir = self.dataset_root / "segments" / "crops"
        candidates = [
            base_dir / f"{item_id}.png",
            base_dir / f"{item_id}.jpg",
            base_dir / f"{item_id}.jpeg",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _get_metadata_path(self, segment_id: str) -> Optional[Path]:
        if not self.dataset_root:
            return None
        item_id = str(segment_id).strip()
        if item_id.endswith(".json"):
            item_id = item_id[:-5]
        elif item_id.endswith(".png"):
            item_id = item_id[:-4]
        meta_path = self.dataset_root / "attributes" / 'raw' / f"{item_id}.json"
        if meta_path.exists():
            return meta_path
        return None

    def _load_metadata(self, segment: Dict[str, Any], segment_id: str) -> Dict[str, Any]:
        metadata = segment.get("metadata", {})
        if metadata and isinstance(metadata, dict):
            return metadata
        meta_path = self._get_metadata_path(segment_id)
        if meta_path and meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception:
                pass
        return {}

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
            doc_id = item.get("doc_id")
            metadata = self._load_metadata(item, segment_id)

            image_path = self._get_image_path(segment_id)
            image_b64: Optional[str] = None
            if image_path and image_path.exists():
                try:
                    image_b64 = self._read_image_b64(image_path)
                    ext = image_path.suffix.lower()
                    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
                    content_items.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                                "detail": "high",
                            },
                        }
                    )
                except Exception as e:
                    print(f"[UmbrellaVLMList] Failed to load image for {segment_id}: {e}")
                    image_b64 = None

            item_text_parts = [f"Item ID: {segment_id}"]
            if doc_id:
                item_text_parts.append(f"Document ID: {doc_id}")
            if metadata:
                item_text_parts.append("Metadata:")
                item_text_parts.append(json.dumps(metadata, indent=2))
            summary = item.get("text", "")
            if summary:
                item_text_parts.append(f"Summary: {summary}")
            if image_b64:
                item_text_parts.append("(Image provided above)")
            content_items.append({"type": "text", "text": "\n".join(item_text_parts)})

        response = self.generate_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant that ranks products for the user." if 'item_' in batch[0].get("id") else "You are a helpful movie assistant that ranks movies for the user."},
                {"role": "user", "content": content_items},
            ],
            temperature=temperature,
        )

        raw_text = response.choices[0].message.content  # type: ignore[assignment]
        data = self.try_parse_json(raw_text)
        if data is None:
            trimmed = raw_text.strip()
            if trimmed.startswith("{") and not trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed + "}")
                except Exception:
                    data = None
        scores: Dict[str, float] = {}
        parsed_items: List[Dict[str, Any]] = []
        order: List[str] = []
        rationale: Dict[str, Any] = {}

        def _ingest_list(items: List[Any]) -> None:
            for row in items:
                if isinstance(row, dict):
                    sid = str(row.get("id") or row.get("segment_id") or "").strip()
                    if sid:
                        order.append(sid)
                        rationale[sid] = row.get("reason") or row.get("rationale")
                elif isinstance(row, str):
                    sid = row.strip()
                    if sid:
                        order.append(sid)

        if isinstance(data, list):
            _ingest_list(data)
        elif isinstance(data, dict):
            ranking = data.get("ranking") or data.get("results") or data.get("items")
            if isinstance(ranking, list):
                _ingest_list(ranking)
        print(order)
        #add item_ prefix for all order that does not have item_ prefix when there is such prefix in batch[0].get("id")
        if 'item_' in batch[0].get("id"):
            order = [f'item_{sid}' if 'item_' not in sid else sid for sid in order]
        print(order)
        if order:
            max_score = float(len(order))
            for idx, sid in enumerate(order):
                score = max_score - float(idx)
                scores[sid] = score
                parsed_items.append(
                    {
                        "id": sid,
                        "score": score,
                        "rationale": rationale.get(sid),
                        "doc_id": next((it.get("doc_id") for it in batch if str(it.get("id")) == sid), None),
                    }
                )

        for item in batch:
            sid = str(item.get("id"))
            if sid not in scores:
                parsed_items.append(
                    {
                        "id": sid,
                        "score": None,
                        "rationale": "not_returned_by_model",
                        "doc_id": item.get("doc_id"),
                    }
                )

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
You are helpfull shopping assistant that helps users find the best products for their needs.

Task: For each catalog item, you will see:
1. A product image - showing the visual appearance of the item
2. Metadata information - including product details, specifications, and attributes

You will also see a conversation between you, the assistant, and the user. Based on these information, rank ALL items from you think that most fit to the user's request to the least fit for the user.

Return JSON ONLY as:
{{"ranking":[{{"id":"<item_id>","reason":"..."}}, ...]}} for example:
{{"ranking": [
    {{"id": "123", "reason": "The product is about a boy who is a superhero."}},
    {{"id": "456", "reason": "The product is about a boy who is a superhero."}},
]}}

Please strictly follow the format and do not add any other text. Make sure the item_id is the EXACT same as the one in the metadata.

- Include every item exactly once
- Order is best to worst (top of array = #1)
- Reference both the image and metadata in your reasoning

Guidance: 
- IMPORTANT: You must consider BOTH the image and metadata together when assigning the score
- Use the image to assess: visual attributes (color, style, material appearance, overall aesthetic, design elements)
- Use the metadata to assess: functional attributes, specifications, product details, categories, brand information
- Combine insights from both sources
- In your rationale, explicitly reference both visual elements (from the image) and metadata fields that influenced your decision

Conversation:
{user_query}

Items follow as alternating [image] and [text] blocks."""
        else:
            print('[item_recommendation] using movie prompt')
            return f"""
You are a movie assistant that helps users find the best movies for their needs.

Task: For each movie, you will see:
1. A movie poster - showing the movie poster
2. Metadata information - including movie details, specifications, and attributes

You will also see a conversation between you, the assistant, and the user. Based on these information, rank ALL movies from you think that most fit to the user's request to the least fit for the user.

Return JSON ONLY as:
{{"ranking":[{{"id":"<movie_id>","reason":"..."}}, ...]}} for example:
{{"ranking": [
    {{"id": "123", "reason": "The movie is about a boy who is a superhero."}},
    {{"id": "456", "reason": "The movie is about a boy who is a superhero."}},
]}}

Please strictly follow the format and do not add any other text.

- Include every item exactly once
- Order is best to worst (top of array = #1)
- Reference both the image and metadata in your reasoning

Guidance: 
- IMPORTANT: You must consider BOTH the image and metadata together when assigning the score
- Use the image to assess: visual attributes (color, style, material appearance, overall aesthetic, design elements)
- Use the metadata to assess: functional attributes, specifications, product details, categories, brand information
- Combine insights from both sources
- In your rationale, explicitly reference both visual elements (from the image) and metadata fields that influenced your decision

Conversation:
{user_query}

Items follow as alternating [image] and [text] blocks."""


