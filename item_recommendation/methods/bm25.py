from __future__ import annotations

from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from item_recommendation.helpers import top_level_metadata_fields

from .base import BatchResult, ObjectRankingMethod


class BM25ObjectRanker(ObjectRankingMethod):
    """BM25-based relevance scoring using metadata text."""

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.item_ids: List[str] = []
        self.item_texts: List[str] = []

    @staticmethod
    def _build_item_text(item: Dict[str, Any]) -> str:
        """Build text representation of an item from its metadata."""
        metadata = item.get("metadata") or {}
        parts = [f"{name}: {value}" for name, value in top_level_metadata_fields(metadata)]
        summary = "; ".join(parts)
        if summary:
            return summary
        return str(item.get("text") or "")

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for tokenization."""
        # Simple normalization: lowercase and split on whitespace
        return text.lower().strip()
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        normalized = BM25ObjectRanker._normalize(text)
        tokens = [tok for tok in normalized.split() if tok]
        return tokens

    @staticmethod
    def _bm25_score_to_umbrella(bm25_score: float, min_score: float = 0.0, max_score: float = 10.0) -> float:
        """Convert BM25 score to UMBRELLA 0-3 scale.
        
        BM25 scores can vary widely. We normalize them to [0, 3] range.
        If min/max are provided, we use them for normalization.
        Otherwise, we use a sigmoid-like transformation.
        """
        # # Normalize to [0, 1] range if we have min/max bounds
        # if max_score > min_score:
        #     normalized = (bm25_score - min_score) / (max_score - min_score)
        #     normalized = max(0.0, min(1.0, normalized))
        # else:
        #     # Use a sigmoid-like transformation for unbounded scores
        #     # Scale so that typical BM25 scores map reasonably
        #     normalized = 1.0 / (1.0 + 2.0 ** (-bm25_score / 2.0))
        
        # Map [0, 1] to [0, 3]
        return bm25_score

    def prepare(self, objects: List[Dict[str, Any]]) -> None:
        """Prepare BM25 index from the batch of objects."""
        self.item_ids = []
        self.item_texts = []
        tokenized_corpus: List[List[str]] = []
        
        for item in objects:
            sid = str(item.get("id") or "")
            if not sid:
                continue
            
            item_text = self._build_item_text(item)
            if not item_text:
                continue
            
            tokens = self._tokenize(item_text)
            if not tokens:
                continue
            
            self.item_ids.append(sid)
            self.item_texts.append(item_text)
            tokenized_corpus.append(tokens)
        
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self.bm25 = None

    def score_batch(
        self,
        user_query: str,
        batch: List[Dict[str, Any]],
        temperature: float,
        run_index: int,
    ) -> BatchResult:
        """Score a batch of items using BM25."""
        if not batch:
            return BatchResult(scores={})
        
        # Prepare BM25 index from this batch
        self.prepare(batch)
        
        if self.bm25 is None or not self.item_ids:
            # If no valid items, return empty scores
            scores: Dict[str, float] = {}
            parsed_items: List[Dict[str, Any]] = []
            for item in batch:
                sid = str(item.get("id") or "")
                if sid:
                    parsed_items.append({
                        "id": sid,
                        "score": None,
                        "rationale": "no_text_available",
                        "doc_id": item.get("doc_id"),
                    })
            return BatchResult(
                scores=scores,
                parsed_items=parsed_items,
                field_scores={},
                raw_response=None,
                temperature=None,
                prompt=None,
            )
        
        # Tokenize query
        query_tokens = self._tokenize(user_query)
        if not query_tokens:
            # Empty query - return zero scores
            scores: Dict[str, float] = {sid: 0.0 for sid in self.item_ids}
            parsed_items: List[Dict[str, Any]] = []
            for item in batch:
                sid = str(item.get("id") or "")
                if sid:
                    scores[sid] = scores.get(sid, 0.0)
                    parsed_items.append({
                        "id": sid,
                        "score": 0.0,
                        "rationale": "empty_query",
                        "doc_id": item.get("doc_id"),
                    })
            return BatchResult(
                scores=scores,
                parsed_items=parsed_items,
                field_scores={},
                raw_response=None,
                temperature=None,
                prompt=None,
            )
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Find min/max for normalization
        if len(bm25_scores) > 0:
            min_score = float(min(bm25_scores))
            max_score = float(max(bm25_scores))
        else:
            min_score = 0.0
            max_score = 0.0
        
        # Convert to UMBRELLA scores and build results
        scores: Dict[str, float] = {}
        parsed_items: List[Dict[str, Any]] = []
        
        for idx, sid in enumerate(self.item_ids):
            if idx < len(bm25_scores):
                bm25_score = float(bm25_scores[idx])
                umbrella_score = self._bm25_score_to_umbrella(bm25_score, min_score, max_score)
                scores[sid] = umbrella_score
                parsed_items.append({
                    "id": sid,
                    "score": umbrella_score,
                    "rationale": f"bm25_score={bm25_score:.4f}",
                    "doc_id": batch[idx].get("doc_id") if idx < len(batch) else None,
                })
        
        # Ensure all items in batch have scores
        for item in batch:
            sid = str(item.get("id") or "")
            if sid and sid not in scores:
                parsed_items.append({
                    "id": sid,
                    "score": None,
                    "rationale": "not_in_bm25_index",
                    "doc_id": item.get("doc_id"),
                })
        
        parsed_items.sort(key=lambda entry: float(entry.get("score", 0.0)) if entry.get("score") is not None else -1, reverse=True)
        return BatchResult(
            scores=scores,
            parsed_items=parsed_items,
            field_scores={},
            raw_response=None,
            temperature=None,
            prompt=None,
        )

