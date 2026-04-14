from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from utils.llm_base import LLMBase

from .base import BatchResult, ObjectRankingMethod


class UmbrellaLLMObjectRanker(LLMBase, ObjectRankingMethod):
    """LLM-based relevance scoring using linked metadata and the UMBRELLA rubric."""

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
                {"role": "system", "content": "You are helpfull shopping assistant that helps users find the best products for their needs." if 'item_' in batch[0].get("id") else "You are helpfull movie assistant that helps users find the best movies for their needs."},
                {"role": "user", "content": content},
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

                    Task: Base on the given conversation between you, the assistant, and the user, and also the catalog item described by metadata fields, 
                    assign a score that if you think the user will like the item, to the catalog item using the UMBRELLA 0-3 scale.

                    Scoring:
                    - 0 = definitely not relevant to the user requirement / user will definitely not like the item
                    - 1 = weak/minimal evidence
                    - 2 = user will likely like the item
                    - 3 = user will definitely like the item

                    Guidance: Consider how strongly the metadata suggests the item satisfies the user query.
                    Be decisive; avoid using score "1" as a default. Reference specific fields in the rationale.

                    Return ONLY a JSON array with entries:
                    {{"id": "segment_id", "score": 0-3, "rationale": "..."}} for example:
                    [
                        {{"id": "123", "score": 2, "rationale": "The product is about a boy who is a superhero."}},
                        {{"id": "456", "score": 3, "rationale": "The product is about a boy who is a superhero."}},
                    ]

                    Conversation: {user_query}
                    You will receive a JSON array of objects, each with:
                    - id: segment identifier
                    - doc_id: linked document id (optional)
                    - metadata_fields: list of {{"name", "value"}}
                    """
        else:
            print('[object ranker] using movie prompt')
            return f"""
                You are helpfull movie assistant that helps users find the best movies for their needs.

                Task: Base on the given conversation between you, the assistant, and the user, and also the catalog item described by metadata fields, 
                assign a score that if you think the user will like the movie, to the movie using the UMBRELLA 0-3 scale.

                Scoring:
                - 0 = definitely not relevant to the user requirement / user will definitely not like the movie
                - 1 = weak/minimal evidence
                - 2 = user will likely like the item
                - 3 = user will definitely like the item

                Guidance: Consider how strongly the metadata suggests the item satisfies the user query.
                Be decisive; avoid using score "1" as a default. Reference specific fields in the rationale.

                Return ONLY a JSON array with entries for each movie:
                {{"id": "movie_id", "score": 0-3, "rationale": "..."}}, for example:
                [
                    {{"id": "123", "score": 2, "rationale": "The movie is about a boy who is a superhero."}},
                    {{"id": "456", "score": 3, "rationale": "The movie is about a boy who is a superhero."}},
                ]

                Conversation: {user_query}
                You will receive a JSON array of objects, each with:
                - id: segment identifier
                - doc_id: linked document id (optional)
                - metadata_fields: list of {{"name", "value"}}
                    """
            

                

