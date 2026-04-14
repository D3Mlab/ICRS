from __future__ import annotations

import os
import json
import sys
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

from nltk.corpus import stopwords

from ..config import Step6Config
from ..utils.text import normalize
from ..utils.idgen import snippet_id_for_text
from ..utils.io import read_json


@dataclass
class SnippetPiece:
    snippet_id: str
    doc_id: str
    field: str
    text: str
    content: str


class Snippetizer:
    """
    Pipeline:
      1) Flatten doc -> collect text fields
      2) Stanza sentence split
      3) If sentence is short enough => keep as-is (skip LLM)
      4) Else Gemini Flash micro-chunk into *few, larger* semantic pieces
      5) Validation-only filter: drop chunks that become empty after removing:
         - punctuation/symbols (by tokenization)
         - NLTK stopwords
         (Original chunk text is preserved when saved.)
      6) normalize() and emit SnippetPiece
    """

    _WORD_RE = re.compile(r"[a-zA-Z0-9]+")

    def __init__(self, cfg: Step6Config, snippets_json_path: Optional[Path] = None):
        self.cfg = cfg
        self._stanza = None
        self._genai = None
        self._stopwords: Optional[Set[str]] = None
        self._snippets_json_path = snippets_json_path
        self._cached_snippets: Optional[Dict[str, Any]] = None

        # Use OpenAI GPT-4o-mini
        self._model = "gpt-4o-mini"

        # Heuristics to reduce LLM calls + prevent tiny chunks
        self._short_sentence_max_words = int(getattr(cfg, "short_sentence_max_words", 10))
        self._min_chunk_words = int(getattr(cfg, "min_chunk_words", 4))
        self._max_chunks_per_sentence = int(getattr(cfg, "max_chunks_per_sentence", 3))

    # -------------------------
    # NLTK stopwords + filter
    # -------------------------
    def _get_stopwords(self) -> Set[str]:
        """
        Requires one-time: nltk.download("stopwords")
        """
        if self._stopwords is None:
            self._stopwords = set(stopwords.words("english"))
        return self._stopwords

    def _has_meaningful_content(self, text: str) -> bool:
        """
        Validation-only cleaning:
          - Extract alphanumeric tokens (drops punctuation/symbols)
          - Remove NLTK stopwords
          - Keep chunk only if >=1 meaningful token remains

        IMPORTANT:
          - Does NOT modify the stored chunk.
        """
        if not text or not text.strip():
            return False

        tokens = self._WORD_RE.findall(text.lower())
        if not tokens:
            return False

        sw = self._get_stopwords()
        meaningful = [t for t in tokens if t not in sw]
        return len(meaningful) > 0

    def _word_count(self, text: str) -> int:
        return len(self._WORD_RE.findall(text))

    # -------------------------
    # Stanza sentence split
    # -------------------------
    def _get_stanza(self):
        if self._stanza is None:
            import stanza  # type: ignore

            processors = "tokenize"
            try:
                stanza.Pipeline(lang="en", processors=processors, use_gpu=False, verbose=False)
            except Exception:
                stanza.download("en", processors=processors, verbose=False)

            self._stanza = stanza.Pipeline(lang="en", processors=processors, use_gpu=False, verbose=False)
        return self._stanza

    def _sentence_split(self, text: str) -> List[str]:
        nlp = self._get_stanza()
        doc = nlp(text)
        return [s.text.strip() for s in doc.sentences if s.text and s.text.strip()]

    # -------------------------
    # OpenAI micro-chunking
    # -------------------------
    def _get_openai_client(self):
        if self._genai is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY env var.")
            self._genai = OpenAI(api_key=api_key)
        return self._genai

    def _gemini_micro_chunks(self, sentences: List[str]) -> List[List[Tuple[str, str]]]:
        """
        Input: list of sentences (ONLY long sentences; short ones should be bypassed before calling)
        Output: parallel list of (chunk, summary) tuples

        Strong constraints to avoid over-splitting:
          - prefer 1-3 chunks per sentence
          - each chunk should be >= min_chunk_words when possible
          - do not split enumerations/spec lists unless a clause boundary exists
        """
        if not sentences:
            return []

        client = self._get_openai_client()

        response_schema = {
            "type": "object",
            "properties": {
                "chunks_by_sentence": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chunk": {"type": "string"},
                            },
                            "required": ["chunk"],
                        },
                    },
                }
            },
            "required": ["chunks_by_sentence"],
        }

        system_instruction = f"""
Goal:
Split each sentence into 1 - MAX 3 semantic chunks.

You MUST return JSON that matches this exact schema:
{json.dumps(response_schema, indent=2)}

The response must be a JSON object with a "chunks_by_sentence" array. Each element in "chunks_by_sentence" is an array of objects for one sentence, where each object has "chunk" (string) field.

IMPORTANT: 
- ONLY split if there is a clear different meaning statement in one stenece. You can just output the original sentence if you think it is not worth splitting.
- Each chunk should be a standalone meaning unit (claim/reason/contrast/condition/statement).
- You should ensure the chunk maintained its original meaning and context with the original sentence, the meaning should not be maniuplated by the chunk. For example, if the sentence is 'The user do not think the jacket is warm and comfortable', then the chunk should be 'The user do not think the jacket is warm' and The user do not think the jacket is warm comfortable'.
- Each chunk should be as a complete sentence to is self-contained, contained the full expression of the sentence it comes from.
    For example,
    sentence: Lightweight cotton for breathability and comfort, it should have two chunks:
    ('Lightweight cotton for breathability' and 'Lightweight cotton for comfort')
"""

        user_message = f"""Split each sentence into 1 - MAX 3 semantic chunks..

Sentences:
{json.dumps(sentences, ensure_ascii=False, indent=2)}

Constraints:
- Max chunks per sentence: {self._max_chunks_per_sentence}
- Min chunk words: {self._min_chunk_words}

Return your response as valid JSON matching the schema provided in the system instructions."""

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                )

                text = (resp.choices[0].message.content or "").strip()
                data = json.loads(text)
                out = data.get("chunks_by_sentence", [])

                cleaned: List[List[Tuple[str, str]]] = []
                for chunks in out:
                    if not isinstance(chunks, list):
                        chunks = [chunks]
                    # Extract chunk, with placeholder for keywords
                    chunk_summary_pairs: List[Tuple[str, str]] = []
                    for item in chunks:
                        if isinstance(item, dict):
                            chunk_text = str(item.get("chunk", "")).strip()
                            # Placeholder for keywords (not used, but kept for compatibility)
                            summary_text = 'placeholder'
                            if chunk_text:
                                chunk_summary_pairs.append((chunk_text, summary_text))
                        elif isinstance(item, str):
                            # Fallback: if it's just a string, use it as chunk with empty summary
                            chunk_text = item.strip()
                            if chunk_text:
                                chunk_summary_pairs.append((chunk_text, ""))
                    cleaned.append(chunk_summary_pairs)
                return cleaned

            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))

        raise RuntimeError(f"OpenAI micro-chunking failed: {last_err}")

    def _summarize_reviews(self, reviews: List[str], count = 0) -> List[str]:
        """
        Summarize a list of reviews into exactly 3 semantic chunks.
        Returns a list of 3 summary chunks.
        """
        if not reviews:
            return []
        
        # Combine all reviews into a single text
        combined_reviews = "\n\n".join(reviews)
        
        client = self._get_openai_client()
        
        response_schema = {
            "type": "object",
            "properties": {
                "summary_chunks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                }
            },
            "required": ["summary_chunks"],
        }
        
        system_instruction = f"""
Goal:
Summarize the provided description into 1-3 sentences that capture the main themes, opinions, and key points across the description.

You MUST return JSON that matches this exact schema:
{json.dumps(response_schema, indent=2)}

- Capture distinct themes or aspects from the description
- Be comprehensive and self-contained
- Represent the collective sentiment and key points from the description
- Be written as complete, coherent sentences
"""
        
        user_message = f"""Summarize the following description into exactly 3 sentence:

Description:
{combined_reviews}

Return your response as valid JSON with 3 sentences."""
        
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                )
                
                text = (resp.choices[0].message.content or "").strip()
                data = json.loads(text)
                summary_chunks = data.get("summary_chunks", [])
                
                # Filter out empty chunks
                chunks = [chunk.strip() for chunk in summary_chunks if chunk.strip()]
                if len(chunks) == 0:
                    if count < 3:
                        print(f'summarizing reviews again {count} times')
                        return self._summarize_reviews(reviews, count + 1)
                    else:
                        return [reviews[0]]
                else:
                    return chunks
                
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        
        raise RuntimeError(f"OpenAI review summarization failed: {last_err}")

    def _postfilter_chunks(self, chunk_summaries: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        After LLM returns chunks with summaries, enforce:
          - remove junk (stopword/punct-only)
          - avoid too-small chunks by merging forward when needed
          - cap number of chunks
          - preserve/combine summaries when chunks are merged
        """
        # 1) remove junk
        kept = [(chunk, summary) for chunk, summary in chunk_summaries if self._has_meaningful_content(chunk)]

        if not kept:
            return []

        # 2) merge small chunks forward
        merged: List[Tuple[str, str]] = []
        i = 0
        while i < len(kept):
            cur_chunk, cur_summary = kept[i]
            # if self._word_count(cur_chunk) < self._min_chunk_words and i + 1 < len(kept):
            #     # merge with next
            #     nxt_chunk, nxt_summary = kept[i + 1]
            #     glue = "" if cur_chunk.endswith((",", ";", ":", "—", "-", "–")) else " "
            #     merged_chunk = (cur_chunk + glue + nxt_chunk).strip()
            #     # Combine summaries: use both if they're different, otherwise just one
            #     if cur_summary and nxt_summary and cur_summary != nxt_summary:
            #         # Combine unique words from both summaries
            #         all_words = (cur_summary + " " + nxt_summary).split()
            #         unique_words = []
            #         seen = set()
            #         for word in all_words:
            #             if word.lower() not in seen:
            #                 unique_words.append(word)
            #                 seen.add(word.lower())
            #             if len(unique_words) >= 5:
            #                 break
            #         combined_summary = " ".join(unique_words[:5])
            #     else:
            #         combined_summary = cur_summary or nxt_summary
            #     merged.append((merged_chunk, combined_summary))
            #     i += 2
            # else:
            merged.append((cur_chunk, cur_summary))
            i += 1

        # final pass: drop junk again (in case merge created weird stuff)
        merged = [(chunk, summary) for chunk, summary in merged if self._has_meaningful_content(chunk)]
        return merged

    # def _get_term_summary(self, chunk: str) -> str:
    #     """
    #     Generate a concise term summary for a chunk using OpenAI.
    #     Returns a short phrase (3-5 words) that summarizes the key terms/concepts in the chunk.
    #     """
    #     if not chunk or not chunk.strip():
    #         return ""
        
    #     client = self._get_openai_client()
        
    #     system_instruction = (
    #         "Extract the key terms and concepts from the given text chunk.\n"
    #         "Output a concise phrase (3-5 words maximum) that summarizes the main topics/terms.\n"
    #         "Use nouns, key adjectives, and important descriptors. Avoid stopwords.\n"
    #         "Return ONLY the summary phrase, no explanation or punctuation."
    #     )
        
    #     last_err: Optional[Exception] = None
    #     for attempt in range(3):
    #         try:
    #             resp = client.chat.completions.create(
    #                 model=self._model,
    #                 messages=[
    #                     {"role": "system", "content": system_instruction},
    #                     {"role": "user", "content": chunk},
    #                 ],
    #                 temperature=0.1,
    #             )
                
    #             summary = (resp.choices[0].message.content or "").strip()
    #             # Clean up any extra formatting
    #             summary = re.sub(r'[^\w\s-]', '', summary).strip()
    #             # Limit to reasonable length
    #             words = summary.split()[:5]
    #             return " ".join(words)
                
    #         except Exception as e:
    #             last_err = e
    #             time.sleep(0.6 * (attempt + 1))
        
    #     # Fallback: return empty if all attempts fail
    #     return ""

    # -------------------------
    # Flatten doc -> text fields
    # -------------------------
    def _extract_text_fields(self, doc: Dict) -> List[Tuple[str, str]]:
        """
        Returns list of (field, chunk_text) where chunk_text is:
          - a whole short sentence (no LLM), OR
          - one of few larger semantic chunks (LLM + postfilter)
        Excludes 'doc_id' and any key containing 'id' (case-insensitive).
        """

        def flatten(obj: Any, prefix: str = "", paraent_structure: str = "") -> List[Tuple[str, str]]:
            rows: List[Tuple[str, str]] = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else str(k)
                    
                    # Special handling for review_snippets field
                    if k == "description" or k == 'features' and isinstance(v, list) and len(v) > 3:
                        # Check if all items in the list are strings (reviews)
                        if all(isinstance(item, str) for item in v):
                            try:
                                # Summarize reviews into 3 semantic chunks
                                summary_chunks = self._summarize_reviews(v)
                                print(f'{k}: {summary_chunks}')
                                # Process each summary chunk through normal sentence splitting
                                    # 1) sentence split
                                sents = summary_chunks
                                
                                # 2) bypass LLM for short sentences; batch only long ones
                                long_sents: List[str] = []
                                set_of_non_split_fields: Set[str] = ['name', 'brand', 'rating', 'price', 'title']
                                for s in sents:
                                    if self._word_count(s) <= 10 or any(non in key for non in set_of_non_split_fields):
                                        rows.append((key, s))
                                    else:
                                        long_sents.append(s)
                                
                                # 3) LLM micro-chunk long sentences only
                                if long_sents:
                                    chunks_by_sent = self._gemini_micro_chunks(long_sents)
                                    for s, chunk_summaries in zip(long_sents, chunks_by_sent):
                                        post = self._postfilter_chunks(chunk_summaries)
                                        if not post:
                                            # fallback: keep original sentence if LLM output is unusable
                                            rows.append((key, s))
                                        else:
                                            for ch, summary in post:
                                                # Append summary to field key
                                                summary = re.sub(r'[^\w\s-]', '', summary).replace(' ', '_').strip().lower()
                                                field_with_summary = f"{key}.{summary}" if summary else key
                                                rows.append((field_with_summary, ch))
                            except Exception as e:
                                # Fallback: if summarization fails, process reviews normally
                                print(f"Warning: Review summarization failed for {key}: {e}", file=sys.stderr)
                                rows.extend(flatten(v, key))
                        else:
                            # Not all items are strings, process normally
                            rows.extend(flatten(v, key))
                    else:
                        rows.extend(flatten(v, key))

            elif isinstance(obj, list):
                for idx, v in enumerate(obj):
                    key = f"{prefix}" if prefix else f"[{idx}]"
                    rows.extend(flatten(v, key, paraent_structure = 'LIST'))

            else:
                if obj is None:
                    return rows

                field = prefix
                if field == "doc_id" or "id" in field.lower():
                    return rows

                val = str(obj).strip()
                if not val:
                    return rows

                # Check if the value starts with 'Keywords' or 'key word:' and handle as comma-separated list
                val_lower = val.lower()
                if val_lower.startswith('keywords') or val_lower.startswith('key word'):
                    # Extract the substring after 'Keywords' or 'key word:'
                    if val_lower.startswith('keywords'):
                        # Find where 'keywords' ends (could be 'keywords:', 'keywords ', etc.)
                        prefix_end = len('keywords')
                        if len(val) > prefix_end and val[prefix_end] in [':', ' ']:
                            prefix_end += 1
                        keywords_str = val[prefix_end:].strip()
                    else:  # 'key word:'
                        keywords_str = val[len('key word:'):].strip()
                    
                    # Split by comma and create separate entries for each keyword
                    keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                    for keyword in keywords:
                        #remove the punctuation from the keyword
                        keyword = re.sub(r'[^\w\s-]', '', keyword).strip().replace(' ', '_').lower()
                        fields = f"{field}.keywords.{keyword}"
                        rows.append((fields, keyword))
                    return rows
                    
                set_of_non_split_fields: Set[str] = ['name', 'brand', 'rating', 'price','title']
                if  any(non in field for non in set_of_non_split_fields):
                    rows.append((field, val))

                elif self._word_count(val) <= 10:
                    rows.append((field, val))
                else:
                    # 3) LLM micro-chunk long sentences only
                    chunks_by_sent = self._gemini_micro_chunks([val])
                    for chunk_summaries in chunks_by_sent:
                        post = self._postfilter_chunks(chunk_summaries)
                        tem_sentence = ''
                        for ch, summary in post:
                            # Append summary to field key
                            summary = re.sub(r'[^\w\s-]', '', summary).replace(' ', '_').strip().lower()
                            field_with_summary = f"{field}.{summary}" if summary else field
                            if len(ch.split(' ')) < 5:
                                tem_sentence += ch + ' '
                                if len(tem_sentence.split(' ')) > 10:
                                    rows.append((field_with_summary, tem_sentence))
                                    tem_sentence = ''
                            rows.append((field_with_summary, ch))

            return rows

        return flatten(doc)

    # -------------------------
    # Main API
    # -------------------------
    def _load_snippets_json(self) -> Dict[str, Any]:
        """Lazy load atomic_attributes.json if not already loaded."""
        if self._cached_snippets is not None:
            return self._cached_snippets
        
        if self._snippets_json_path and self._snippets_json_path.exists():
            try:
                self._cached_snippets = read_json(self._snippets_json_path)
                return self._cached_snippets
            except Exception:
                self._cached_snippets = {}
                return {}
        else:
            self._cached_snippets = {}
            return {}

    def snippetize_doc(self, doc: Dict) -> List[SnippetPiece]:
        doc_id = str(doc.get("doc_id"))
        
        # Check if doc is already in atomic_attributes.json
        snippets_data = self._load_snippets_json()
        if doc_id in snippets_data:
            # Skip processing if already exists
            print(f"Doc {doc_id} already exists in atomic_attributes.json")
            return []
        
        pieces: List[SnippetPiece] = []

        for field, chunk in self._extract_text_fields(doc):
            # validation-only junk filter (preserve original chunk if kept)
            if not self._has_meaningful_content(chunk):
                continue

            norm = normalize(chunk, **self.cfg.index.normalize.__dict__)
            if not norm:
                continue

            # text = field key (may include summary like "description [Warmth]")
            # content = the actual chunk/sentence text
            text = field.strip()
            content = norm.strip()
            sid = snippet_id_for_text(doc_id, field, (0, len(content)))

            pieces.append(
                SnippetPiece(
                    snippet_id=sid,
                    doc_id=doc_id,
                    field=field,
                    text=text,
                    content=content,
                )
            )

        return pieces
