from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Optional, Any
from ..config import Step6Config


class Merger:
    """
    Merges semantically identical snippets by grouping them together.
    Uses LLM to identify snippets that have the same meaning but different wording.
    """

    def __init__(self, cfg: Step6Config):
        self.cfg = cfg
        self._openai_client = None
        # Use OpenAI GPT-4o-mini
        self._model = "gpt-4o-mini"

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY env var.")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def merge_semantically_identical(
        self, snippets: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Merge semantically identical snippets.
        
        Args:
            snippets: List of dicts with keys 'id', 'text', 'content'
            
        Returns:
            List of dicts where:
            - Semantically identical snippets have been merged (same text assigned to all IDs in a semantic group)
            - Each snippet has an 'old_text' field with the original text before merging
            - Duplicates (same 'text' and 'content') are removed, keeping only the first occurrence
        """
        if not snippets or len(snippets) < 2:
            return snippets

        # Build mapping from id to snippet for quick lookup
        id_to_snippet = {s["id"]: s for s in snippets}

        # Call LLM to find semantically identical groups
        semantic_groups = self._find_semantic_groups(snippets)
        print(f'[label_selection] semantic_groups: {semantic_groups}')
        # Create a mapping: snippet_id -> representative_text
        # The LLM returns groups where the key is the representative text (one of the snippet texts) for that group
        id_to_representative_text: Dict[str, str] = {}
        for representative_text, group_ids in semantic_groups.items():
            # The LLM should have used one of the actual snippet texts as the key
            # Use the representative text from the LLM as the merged text for all IDs in this group
            for sid in group_ids:
                if sid in id_to_snippet:
                    id_to_representative_text[sid] = representative_text

        # Update snippets with merged texts, saving old text
        merged_snippets = []
        for snippet in snippets:
            sid = snippet["id"]
            merged_snippet = snippet.copy()
            # Save the old text before replacing
            # If this snippet is in a semantic group, use the representative text
            if sid in id_to_representative_text:
                merged_snippet["old_text"] = snippet["text"]
                merged_snippet["text"] = id_to_representative_text[sid]
            else:
                merged_snippet["old_text"] = 'NA'
            merged_snippets.append(merged_snippet)

        # Deduplicate: remove snippets that have exactly the same 'text' and 'content'
        seen = {}  # (text, content) -> first snippet with these values
        deduplicated = []
        for snippet in merged_snippets:
            text = snippet["text"]
            content = snippet["content"]
            key = (text, content)
            if key not in seen:
                seen[key] = snippet
                deduplicated.append(snippet)
            # If duplicate, skip it (keep only the first occurrence)

        return deduplicated

    def _find_semantic_groups(
        self, snippets: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """
        Use LLM to find groups of semantically identical snippets.
        
        Returns:
            Dict mapping representative text to list of snippet IDs that have the same meaning
        """
        if len(snippets) < 2:
            return {}

        client = self._get_openai_client()

        # Build input format for LLM: list of {id, text} pairs
        snippet_list = [
            {"id": s["id"], "text": s["text"]} for s in snippets
        ]

        system_instruction = """You are analyzing snippets to find groups that are semantically identical (mean the same thing but use different words).

        For each group of snippets that have the same meaning:
        - a summarized keywords that cover the general meaning of the group, should be in the similiar length as group
        - List all snippet IDs that belong to that group as values (including the ID of the snippet whose text you used as the key)

        Ignore snippets that are unique or don't have semantic duplicates.

        You should ensure that the entire snippet text between two snippets has the same meaning, not just part of it.
        For example 'review.cusioning' is not the same as 'product.cusioning' since the first is a review and the second is a product. and review and product does not mean the same things.

        You do not need return if text that are eaxctly the same.

        You MUST return valid JSON in this exact format:
        {
        "groups": {
            "representative_text_here": ["id1", "id2", "id3"],
            "another_representative_text": ["id4", "id5"]
        }
        }

        Requirements:
        - Keys is the summarized keywords you generated for the group
        - IDs must match EXACTLY one of the input snippet IDs, do not add or remove any part of the ID
        - Values are lists of all snippet IDs that have the same meaning
        - Only return groups with 2+ IDs. Do not include single snippets.
        - Return only valid JSON, no other text.

        Example:
        {
        "groups": {
            "The jacket is warm and comfortable": ["id1", "id5", "id12"],
            "Lightweight design for easy wear": ["id3", "id8"]
        }
        }"""

        user_message = f"""Find all groups of snippets that mean the same thing but use different wording.

Snippets to analyze:
{json.dumps(snippet_list, ensure_ascii=False, indent=2)}

Return your response as valid JSON only."""

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )

                text = (resp.choices[0].message.content or "").strip()
                data = json.loads(text)
                groups = data.get("groups", {})

                # Validate that all IDs in groups exist in our snippets
                valid_groups: Dict[str, List[str]] = {}
                snippet_ids = {s["id"] for s in snippets}
                for rep_text, group_ids in groups.items():
                    if not isinstance(group_ids, list):
                        continue
                    # Filter to only include valid IDs and groups with 2+ IDs
                    valid_ids = [sid for sid in group_ids if isinstance(sid, str) and sid in snippet_ids]
                    if len(valid_ids) >= 2:
                        valid_groups[rep_text] = valid_ids

                return valid_groups

            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))

        # If all attempts fail, return empty dict (no merging)
        print(f"Warning: Failed to find semantic groups: {last_err}")
        return {}

