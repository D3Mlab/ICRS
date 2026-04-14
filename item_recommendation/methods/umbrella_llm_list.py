from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from utils.llm_base import LLMBase

from .base import BatchResult, ObjectRankingMethod


class UmbrellaLLMListObjectRanker(LLMBase, ObjectRankingMethod):
    """LLM-based item ranking (list-style) using metadata + conversation context."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        provider: str = "openai",
        temperature_jitter: float = 0.3,
        api_base: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
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

    def temperature_for_run(self, base_temperature: float, run_index: int) -> float:
        if self.temperature_jitter <= 0.0:
            return max(0.0, min(1.0, base_temperature))
        jitter = random.uniform(-self.temperature_jitter, self.temperature_jitter)
        temp = base_temperature + jitter
        return max(0.0, min(1.0, temp))

    def score_batch(
        self,
        user_query: str,
        batch: List[Dict[str, Any]],
        temperature: float,
        run_index: int,
    ) -> BatchResult:
        prompt = self._build_prompt(user_query, batch[0].get("id"))
        payload = [
            {
                "id": item.get("id"),
                "doc_id": item.get("doc_id"),
                "metadata_fields": item.get("fields", []),
                "summary": item.get("text", ""),
            }
            for item in batch
        ]
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
        ]
        response = self.generate_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant that ranks products for the user." if 'item_' in batch[0].get("id") else "You are a helpful movie assistant that ranks movies for the user."},
                {"role": "user", "content": content},
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
        # Assign descending fake scores so higher rank -> higher score
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

        # Ensure all items included
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
1. Metadata information - including product details, specifications, and attributes

You will also see a conversation between you, the assistant, and the user. Based on these information, rank ALL items from you think that most fit to the user's request to the least fit for the user.

Return JSON ONLY as:
{{"ranking":[{{"id":"<item_id>","reason":"..."}}, ...]}} for example:
{{"ranking": [
    {{"id": "123", "reason": "The product is about a boy who is a superhero."}},
    {{"id": "456", "reason": "The product is about a boy who is a superhero."}},
]}}

You must strictly follow the format and do not add any other text.

- Include every item exactly once
- Order is best to worst (top of array = #1)
- Use the provided metadata wording in reasons; be concise

Conversation:
{user_query}

Items JSON (id, doc_id, metadata_fields, summary) follows."""
        else:
            print('[item_recommendation] using movie prompt')
            return f"""
You are a movie assistant that helps users find the best movies for their needs.

Task: For each movie, you will see:
1. Metadata information - including movie details, specifications, and attributes

You will also see a conversation between you, the assistant, and the user. Based on these information, rank ALL movies from you think that most fit to the user's request to the least fit for the user.

Return JSON ONLY as:
{{"ranking":[{{"id":"<movie_id>","reason":"..."}}, ...]}} for example:
"ranking": [
    {{"id": "123", "reason": "The movie is about a boy who is a superhero."}},
    {{"id": "456", "reason": "The movie is about a boy who is a superhero."}},
]

- Include every item exactly once
- Order is best to worst (top of array = #1)
- Use the provided metadata wording in reasons; be concise

Conversation:
{user_query}

Items JSON (id, doc_id, metadata_fields, summary) follows."""


