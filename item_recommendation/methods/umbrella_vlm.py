from __future__ import annotations

import base64
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm_base import LLMBase

from .base import BatchResult, ObjectRankingMethod


class UmbrellaVLMObjectRanker(LLMBase, ObjectRankingMethod):
    """VLM-based relevance scoring using segment images and metadata with the UMBRELLA rubric."""

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
        """Read image file and encode as base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_path(self, segment_id: str) -> Optional[Path]:
        """Get the image path for a segment ID.
        
        Extracts item ID from segment_id (e.g., "item_1" from "item_1" or "item_1.json")
        and looks for the image in {dataset_root}/segments/crops/{item_id}.png
        
        The dataset_root should point to the dataset directory (e.g., data/fashion)
        """
        if not self.dataset_root:
            return None
        
        # Extract item ID from segment_id
        # Handle formats like "item_1", "item_1.json", "item_1.png", etc.
        item_id = str(segment_id).strip()
        if item_id.endswith(".json"):
            item_id = item_id[:-5]
        elif item_id.endswith(".png"):
            item_id = item_id[:-4]
        
        # Look for image in segments/crops folder relative to dataset_root
        # dataset_root should be like data/fashion, so segments/crops is directly under it
        image_path = self.dataset_root / "segments" / "crops" / f"{item_id}.png"
        if image_path.exists():
            return image_path
        
        # Try alternative locations or extensions
        for ext in [".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            alt_path = self.dataset_root / "segments" / "crops" / f"{item_id}{ext}"
            if alt_path.exists():
                return alt_path
        
        return None

    def _get_metadata_path(self, segment_id: str) -> Optional[Path]:
        """Get the metadata path for a segment ID.
        
        Extracts item ID from segment_id and looks for metadata in data/{dataset}/attributes/{item_id}.json
        """
        if not self.dataset_root:
            return None
        
        # Extract item ID from segment_id
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
        """Load metadata for a segment, preferring the metadata already in the segment dict."""
        # First, try to use metadata already in the segment
        metadata = segment.get("metadata", {})
        if metadata and isinstance(metadata, dict):
            return metadata
        
        # Fallback: try to load from file
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
        """Score a batch of segments using VLM with images and metadata."""
        prompt = self._build_prompt(user_query, batch[0].get("id"))
        
        # Build content with images and metadata for each item
        # Structure: prompt, then for each item: [image (if available), item metadata text]
        content_items: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]
        
        # Process each item and add its image + metadata together
        for item in batch:
            segment_id = str(item.get("id", ""))
            doc_id = item.get("doc_id")
            metadata = self._load_metadata(item, segment_id)
            
            # Get image path and encode if available
            image_path = self._get_image_path(segment_id)
            image_b64: Optional[str] = None
            if image_path and image_path.exists():
                try:
                    image_b64 = self._read_image_b64(image_path)
                    # Determine image format from extension
                    ext = image_path.suffix.lower()
                    if ext in [".jpg", ".jpeg"]:
                        mime_type = "image/jpeg"
                    elif ext == ".png":
                        mime_type = "image/png"
                    else:
                        mime_type = "image/png"  # default
                    
                    # Add image to content - this image corresponds to the item below
                    content_items.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                            "detail": "high"
                        }
                    })
                except Exception as e:
                    print(f"[UmbrellaVLM] Failed to load image for {segment_id}: {e}")
                    image_b64 = None
            
            # Build metadata text for this item
            fields = item.get("fields", [])
            if not fields and metadata:
                # Flatten metadata if fields not already provided
                from ..helpers import flatten_metadata
                fields = [{"name": name, "value": value} for name, value in flatten_metadata(metadata)]
            
            # Create item description text that includes ID and metadata
            item_text_parts = [f"Item ID: {segment_id}"]
            if doc_id:
                item_text_parts.append(f"Document ID: {doc_id}")
            
            if fields:
                item_text_parts.append("Metadata:")
                for field in fields:
                    name = field.get("name", "")
                    value = field.get("value", "")
                    if value:
                        item_text_parts.append(f"  - {name}: {value}")
            
            summary = item.get("text", "")
            if summary:
                item_text_parts.append(f"Summary: {summary}")
            
            if image_b64:
                item_text_parts.append("(Image provided above)")
            else:
                item_text_parts.append("(No image available)")
            
            # Add item metadata text right after its image (if any)
            content_items.append({
                "type": "text",
                "text": "\n".join(item_text_parts)
            })
        
        int_item_id = None
        try:
            int_item_id = int(batch[0].get("id"))
        except Exception:
            pass
        if 'item_' in batch[0].get("id") or int_item_id is None or int_item_id > 1000000:
            content = "You are a helpful shopping assistant that helps users find the best products for their needs. You can see product images and metadata."
        else:
            content = "You are a helpful movie assistant that helps users find the best movies for their needs. You can see movie posters and metadata."
        print(f"[item_recommendation] content: {content}")
        # Make API call
        response = self.generate_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": content,
                },
                {
                    "role": "user",
                    "content": content_items,
                },
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
            parsed_items.append({
                "id": sid,
                "score": score,
                "rationale": entry.get("rationale"),
                "doc_id": entry.get("doc_id"),
            })

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

        # Ensure all items in batch have scores
        for item in batch:
            sid = item.get("id")
            if sid not in scores:
                parsed_items.append({
                    "id": sid,
                    "score": None,
                    "rationale": "not_returned_by_model",
                    "doc_id": item.get("doc_id"),
                })

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
1. A product image (if available) - showing the visual appearance of the item
2. Metadata information - including product details, specifications, and attributes
3. a conversation between you, the assistant, and the user

Base on the given conversation,

You must use BOTH the image AND the metadata to make your decision. Do not rely on only one source of information.

For each item, assign a score that if you think the user will like the item using the UMBRELLA 0-3 scale.

Scoring:
- 0 = definitely not relevant to the user requirement / user will definitely not like the item
- 1 = weak/minimal evidence
- 2 = user will likely like the item
- 3 = user will definitely like the item

Guidance: 
- IMPORTANT: You must consider BOTH the image and metadata together when assigning the score
- Use the image to assess: visual attributes (color, style, material appearance, overall aesthetic, design elements)
- Use the metadata to assess: functional attributes, specifications, product details, categories, brand information
- Combine insights from both sources
- In your rationale, explicitly reference both visual elements (from the image) and metadata fields that influenced your decision

Return ONLY a JSON array with entries:
{{"id": "segment_id", "score": 0-3, "rationale": "..."}} 

for example:
[
    {{"id": "123", "score": 2, "rationale": "The product is about a boy who is a superhero."}},
    {{"id": "456", "score": 3, "rationale": "The product is about a boy who is a superhero."}},
]

The rationale should mention both image observations and metadata details that led to the score.

Conversation: {user_query}

You will receive items in the following format:
- Each item may have an image (shown above the item description)
- Each item will have a text description with: Item ID, Document ID (if available), Metadata fields, and Summary
- Items are presented one after another: [Image (if available)] followed by [Item description text]

For each item, analyze BOTH the image (if provided) AND the metadata to determine relevance to the user query.
"""
        else:
            print('[object ranker] using movie prompt')
            return f"""
You are helpfull movie assistant that helps users find the best movies for their needs.

Task: For each catalog item, you will see:
1. A product image (if available) - showing the poster of the movie
2. Metadata information - including movie details, specifications, and attributes
3. a conversation between you, the assistant, and the user

Base on the given conversation,

You must use BOTH the image AND the metadata to make your decision. Do not rely on only one source of information.

For each item, assign a score that if you think the user will like the movie using the UMBRELLA 0-3 scale.

Scoring:
- 0 = definitely not relevant to the user requirement / user will definitely not like the movie
- 1 = weak/minimal evidence
- 2 = user will likely like the movie
- 3 = user will definitely like the movie

Guidance: 
- IMPORTANT: You must consider BOTH the image and metadata together when assigning the score
- Use the image to assess: visual attributes (color, style, material appearance, overall aesthetic, design elements)
- Use the metadata to assess: functional attributes, specifications, product details, categories, brand information
- Combine insights from both sources
- In your rationale, explicitly reference both visual elements (from the image) and metadata fields that influenced your decision

Return ONLY a JSON array with entries:
{{"id": "segment_id", "score": 0-3, "rationale": "..."}} for example:
[
    {{"id": "123", "score": 2, "rationale": "The movie is about a boy who is a superhero."}},
    {{"id": "456", "score": 3, "rationale": "The movie is about a boy who is a superhero."}},
]

The rationale should mention both image observations and metadata details that led to the score.

Conversation: {user_query}

You will receive items in the following format:
- Each item may have an poster (shown above the item description)
- Each item will have a text description with: Item ID, Document ID (if available), Metadata fields, and Summary
- Items are presented one after another: [Image (if available)] followed by [Item description text]

For each item, analyze BOTH the image (if provided) AND the metadata to determine relevance to the user query.
"""

