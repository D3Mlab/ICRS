from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from openai import OpenAI

import numpy as np
from PIL import Image

from utils.cache_manager import CacheManager
from .base import MethodResult, RelevanceMethod
from ..utils.io import stable_hash

try:
    import open_clip
    import torch
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False
    open_clip = None  # type: ignore
    torch = None  # type: ignore

try:
    from google import genai  # type: ignore
    _HAS_GOOGLE_GENAI = True
except ImportError:
    _HAS_GOOGLE_GENAI = False
    genai = None  # type: ignore


DEFAULT_DENSE_EMBED_DIM = 1536
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ClipEmbeddingCache:
    """Cache for CLIP embeddings (text and image)."""

    def __init__(
        self,
        cache_dir: Optional[Path],
        model_name: str,
        dataset_root: Optional[Path] = None,
    ) -> None:
        self.cache_base = cache_dir
        self.dataset_root = CacheManager.infer_dataset_root(dataset_root, cache_dir)
        self.model_name = model_name
        self.enabled = cache_dir is not None
        self.module = "label_selection"
        self.method = "clip"

    def _load_records(self, variant: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not self.enabled:
            return {}, {}
        records, meta = CacheManager.load_embedding_cache(
            module=self.module,
            method=self.method,
            model_name=self.model_name,
            dataset=self.dataset_root,
            base_dirs=[self.cache_base],
            variant=variant,
        )
        if not isinstance(records, dict):
            records = {}
        return records, meta or {}

    def _save_records(self, records: Dict[str, Any], variant: str, config: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        CacheManager.save_embedding_cache(
            module=self.module,
            method=self.method,
            model_name=self.model_name,
            records=records,
            meta={"config": config},
            dataset=self.dataset_root,
            base_dirs=[self.cache_base],
            variant=variant,
        )

    def get_text_embeddings(
        self,
        *,
        keys: List[str],
        texts: List[str],
        variant: str,
        config: Dict[str, Any],
        encode_fn: Callable[[List[str]], np.ndarray],
    ) -> np.ndarray:
        """Get text embeddings with caching."""
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)

        if not self.enabled:
            return encode_fn(texts)

        records, meta = self._load_records(variant)
        if meta.get("config") != config:
            records = {}

        ordered: List[Optional[np.ndarray]] = [None] * len(keys)
        missing_indices: List[int] = []

        for idx, key in enumerate(keys):
            rec = records.get(key)
            if rec and rec.get("text") == texts[idx] and rec.get("embedding") is not None:
                ordered[idx] = np.asarray(rec["embedding"], dtype=np.float32)
            else:
                missing_indices.append(idx)

        if missing_indices:
            need_texts = [texts[i] for i in missing_indices]
            embeddings = encode_fn(need_texts)
            for local_idx, emb in zip(missing_indices, embeddings):
                vec = np.asarray(emb, dtype=np.float32)
                ordered[local_idx] = vec
                records[keys[local_idx]] = {"text": texts[local_idx], "embedding": vec}

        for idx, vec in enumerate(ordered):
            if vec is None:
                emb = encode_fn([texts[idx]])[0]
                vec = np.asarray(emb, dtype=np.float32)
                ordered[idx] = vec
                records[keys[idx]] = {"text": texts[idx], "embedding": vec}

        stacked = np.stack([np.asarray(vec, dtype=np.float32) for vec in ordered], axis=0)
        self._save_records(records, variant, config)
        return stacked.astype(np.float32)

    def get_image_embeddings(
        self,
        *,
        keys: List[str],
        images: List[Path],
        variant: str,
        config: Dict[str, Any],
        encode_fn: Callable[[List[Path]], np.ndarray],
    ) -> np.ndarray:
        """Get image embeddings with caching."""
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)

        if not self.enabled:
            return encode_fn(images)

        records, meta = self._load_records(variant)
        if meta.get("config") != config:
            records = {}

        ordered: List[Optional[np.ndarray]] = [None] * len(keys)
        missing_indices: List[int] = []

        for idx, key in enumerate(keys):
            rec = records.get(key)
            image_key = images[idx].name if images[idx] else "none"
            if rec and rec.get("image_key") == image_key and rec.get("embedding") is not None:
                ordered[idx] = np.asarray(rec["embedding"], dtype=np.float32)
            else:
                missing_indices.append(idx)

        if missing_indices:
            need_images = [images[i] for i in missing_indices]
            embeddings = encode_fn(need_images)
            for local_idx, emb in zip(missing_indices, embeddings):
                vec = np.asarray(emb, dtype=np.float32)
                ordered[local_idx] = vec
                image_key = images[local_idx].name if images[local_idx] else "none"
                records[keys[local_idx]] = {"image_key": image_key, "embedding": vec}

        for idx, vec in enumerate(ordered):
            if vec is None:
                emb = encode_fn([images[idx]])[0]
                vec = np.asarray(emb, dtype=np.float32)
                ordered[idx] = vec
                image_key = images[idx].name if images[idx] else "none"
                records[keys[idx]] = {"image_key": image_key, "embedding": vec}

        stacked = np.stack([np.asarray(vec, dtype=np.float32) for vec in ordered], axis=0)
        self._save_records(records, variant, config)
        return stacked.astype(np.float32)


class ClipSnippetRanker(RelevanceMethod):
    """
    CLIP-style cross-modal embedding matching for snippet ranking.
    Supports both linear and late fusion strategies.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        fusion_method: str = "linear",  # "linear" or "late"
        alpha: float = 0.5,
        require_reason: bool = True,
        cache_dir: Optional[Path] = None,
        dataset_root: Optional[Path] = None,
        image_path: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        summarize_conversation: bool = True,
        gemini_model: str = "gemini-2.0-flash-exp",  # or "gemini-2.5-flash-lite" when available
        gemini_api_key: Optional[str] = None,
    ):
        if not _HAS_OPEN_CLIP:
            raise RuntimeError(
                "open_clip and torch packages are required for ClipSnippetRanker. "
                "Install with: pip install open-clip-torch torch pillow"
            )
        self.model_name = 'ViT-L-14'
        self.pretrained = pretrained
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.require_reason = require_reason
        self.cache_dir = cache_dir
        self.dataset_root = dataset_root
        self.image_path = image_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.cache = ClipEmbeddingCache(self.cache_dir, f"{model_name}_{pretrained}", dataset_root=self.dataset_root)
        
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.preprocess: Optional[Any] = None
        self.snippet_ids: List[str] = []
        self.snippet_texts: Dict[str, str] = {}
        self.snippet_embs: Optional[np.ndarray] = None
        self.embedding_dim: Optional[int] = None

        # Dense text embedding (OpenRouter Qwen3-8B)
        self.dense_model_name = "qwen/qwen3-embedding-8b"
        self.dense_api_base = "https://openrouter.ai/api/v1"
        self.dense_api_key = os.getenv("OPENROUTER_API_KEY")
        self.dense_request_timeout = 60.0
        self.dense_batch_size = 16
        if not self.dense_api_key:
            raise ValueError("ClipSnippetRanker dense embeddings require OPENROUTER_API_KEY or OPENAI_API_KEY.")
        dense_headers = self._openrouter_headers(allow_blank=True)
        self._dense_client = OpenAI(
            api_key=self.dense_api_key,
            base_url=self.dense_api_base,
            default_headers=dense_headers or None,
            timeout=self.dense_request_timeout,
        )
        
        # Gemini summarization settings
        self.summarize_conversation = summarize_conversation
        self.gemini_model = gemini_model
        self._gemini_client: Optional[Any] = None
        if self.summarize_conversation:
            if not _HAS_GOOGLE_GENAI:
                raise RuntimeError(
                    "google-genai package is required for conversation summarization. "
                    "Install with: pip install google-genai"
                )
            api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable or gemini_api_key parameter required "
                    "for conversation summarization"
                )
            self._gemini_client = genai.Client(api_key=api_key)

    def _get_model(self):
        """Lazy load the CLIP model."""
        if self.model is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()
        return self.model, self.tokenizer, self.preprocess

    def _summarize_conversation(self, conversation: str) -> str:
        """Summarize conversation to 1 sentence using Gemini 2.5 Flash Lite with caching."""
        if not self.summarize_conversation or not self._gemini_client:
            return conversation
        
        # Build cache key (used for both cache check and save)
        cache_key = {
            "conversation_hash": stable_hash(conversation),
            "model": self.gemini_model,
        }
        
        # Check cache first
        if self.cache_dir:
            cached = CacheManager.get_llm_record(
                module="label_selection",
                method="clip",
                model_name=self.gemini_model,
                namespace="conversation_summary",
                key=cache_key,
                dataset=self.dataset_root,
                base_dirs=[self.cache_dir],
                variant=None,
            )
            if cached is not None:
                record = cached.get("record", {})
                cached_summary = record.get("content", "")
                if cached_summary:
                    return str(cached_summary).strip()
        
        # Not in cache, call Gemini API
        prompt = f"""Summarize the following shopping conversation into exactly one concise sentence that captures the user's main request, preferences, and needs. Focus on what the user is looking for.

Conversation:
{conversation}

Summary (one sentence only):"""
        
        try:
            response = self._gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"temperature": 0}
            )
            # Extract text from response
            if hasattr(response, "text"):
                summary = response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                parts = response.candidates[0].content.parts
                summary = " ".join(part.text for part in parts if hasattr(part, "text")).strip()
            else:
                summary = conversation  # Fallback to original if extraction fails
            
            summary = summary if summary else conversation
            
            # Save to cache
            if self.cache_dir:
                CacheManager.append_llm_record(
                    module="label_selection",
                    method="clip",
                    model_name=self.gemini_model,
                    namespace="conversation_summary",
                    key=cache_key,
                    content=summary,
                    record_meta={"provider": "google"},
                    dataset=self.dataset_root,
                    base_dirs=[self.cache_dir],
                    variant=None,
                )
            
            return summary
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to summarize conversation with Gemini: {e}. Using original conversation.")
            return conversation

    def _cache_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "fusion_method": self.fusion_method,
            "alpha": self.alpha,
            "require_reason": self.require_reason,
        }

    def _openrouter_headers(self, allow_blank: bool = False) -> Dict[str, str]:
        """Build headers for OpenRouter embedding requests."""
        api_key = self.dense_api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key and not allow_blank:
            raise ValueError(
                "ClipSnippetRanker dense embedding path requires OPENROUTER_API_KEY or OPENAI_API_KEY."
            )
        headers: Dict[str, str] = {}
        referer = os.getenv("OPENROUTER_SITE_URL")
        if referer:
            headers["HTTP-Referer"] = referer
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if app_name:
            headers["X-Title"] = app_name
        return headers

    def _encode_texts_dense(self, texts: Optional[List[str]]) -> np.ndarray:
        """Encode texts using OpenRouter dense embedder (Qwen3-8B)."""
        if not texts:
            dim = self.embedding_dim or DEFAULT_DENSE_EMBED_DIM
            return np.zeros((0, dim), dtype=np.float32)

        vectors: List[np.ndarray] = []

        for start in range(0, len(texts), self.dense_batch_size):
            chunk = texts[start : start + self.dense_batch_size]
            resp = self._dense_client.embeddings.create(
                model=self.dense_model_name,
                input=chunk,
            )
            data = getattr(resp, "data", None)
            if not isinstance(data, list):
                raise ValueError("OpenRouter embeddings response missing 'data' entries")
            chunk_vecs: List[np.ndarray] = []
            for item in data:
                embedding = getattr(item, "embedding", None)
                if embedding is None:
                    raise ValueError("OpenRouter embeddings response missing 'embedding'")
                chunk_vecs.append(np.asarray(embedding, dtype=np.float32))
            if not chunk_vecs:
                raise ValueError("OpenRouter returned empty embeddings list")
            vectors.append(np.stack(chunk_vecs, axis=0))

        embs = np.vstack(vectors)
        self.embedding_dim = embs.shape[1]
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms
        return embs.astype(np.float32)

    def _encode_texts(self, texts: Optional[List[str]]) -> np.ndarray:
        """Encode texts using CLIP text encoder. Encodes each text separately."""
        if not texts:
            dim = self.embedding_dim or 512
            return np.zeros((0, dim), dtype=np.float32)

        model, tokenizer, _ = self._get_model()
        
        # Tokenize all texts at once to ensure tokenizer sees all differences
        # This ensures each text gets unique tokens
        with torch.no_grad():
            # Debug: Check if texts are actually different
            unique_texts = len(set(texts))
            if unique_texts < len(texts):
                import warnings
                # Print more detail to see what's different
                sample_texts = [t[:300] for t in texts[:3]]

            
            # Tokenize all texts together - this ensures each gets unique tokenization
            all_text_tokens = tokenizer(texts).to(self.device)
            
            # Debug: Check if tokens are different
            if len(texts) > 1:
                first_tokens = all_text_tokens[0]
                tokens_different = any(not torch.equal(first_tokens, all_text_tokens[i]) for i in range(1, min(3, len(texts))))



            # Encode all texts together (batch encoding)
            all_text_features = model.encode_text(all_text_tokens)
            
            # Debug: Check features before normalization
            if len(texts) > 1:
                first_features = all_text_features[0]
                features_different_before_norm = any(
                    not torch.allclose(first_features, all_text_features[i], atol=1e-5) 
                    for i in range(1, min(3, len(texts)))
                )
            
            # L2 normalize (standard for CLIP embeddings)
            all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
            
            # Extract each embedding separately
            embeddings_list = [
                all_text_features[i].cpu().numpy().astype(np.float32)
                for i in range(len(texts))
            ]
            
            # Verify embeddings are different (debug check)
            if len(embeddings_list) > 1:
                first_emb = embeddings_list[0]
                all_same = all(np.allclose(first_emb, emb, atol=1e-6) for emb in embeddings_list[1:])
        
        embs = np.vstack(embeddings_list)
        if self.embedding_dim is None:
            self.embedding_dim = embs.shape[1]
        return embs

    def _encode_images(self, image_paths: Optional[List[Path]]) -> np.ndarray:
        """Encode images using CLIP image encoder."""
        if not image_paths:
            dim = self.embedding_dim or 512
            return np.zeros((0, dim), dtype=np.float32)

        model, _, preprocess = self._get_model()
        
        embeddings_list = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = []
            for img_path in batch_paths:
                if img_path and img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    images.append(preprocess(img))
                else:
                    # Use a blank image if path doesn't exist
                    images.append(preprocess(Image.new("RGB", (224, 224), color="black")))
            
            if images:
                with torch.no_grad():
                    image_tensor = torch.stack(images).to(self.device)
                    image_features = model.encode_image(image_tensor)
                    # L2 normalize
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    embeddings_list.append(image_features.cpu().numpy().astype(np.float32))
        
        if not embeddings_list:
            dim = self.embedding_dim or 512
            return np.zeros((len(image_paths), dim), dtype=np.float32)
        
        embs = np.vstack(embeddings_list)
        if self.embedding_dim is None:
            self.embedding_dim = embs.shape[1]
        return embs

    def _encode_texts_cached(self, texts: List[str], variant: str) -> np.ndarray:
        """Encode texts with caching."""
        keys = [f"text::{stable_hash(t)}" for t in texts]
        return self.cache.get_text_embeddings(
            keys=keys,
            texts=texts,
            variant=variant,
            config=self._cache_config(),
            encode_fn=self._encode_texts,
        )

    def _encode_images_cached(self, image_paths: List[Path], variant: str) -> np.ndarray:
        """Encode images with caching."""
        keys = [f"image::{img_path.name if img_path else 'none'}" for img_path in image_paths]
        return self.cache.get_image_embeddings(
            keys=keys,
            images=image_paths,
            variant=variant,
            config=self._cache_config(),
            encode_fn=self._encode_images,
        )

    def fit(self, snippet_texts: Dict[str, str]) -> None:
        """Store snippet texts (encoding happens per-snippet in score() with conversation)."""
        self.snippet_ids = sorted(snippet_texts.keys())
        self.snippet_texts = snippet_texts

    def score(self, query_text: str, topk: int, image_path: Optional[Path] = None) -> MethodResult:
        """
        Score snippets against query and image.
        Each snippet is embedded separately with the conversation appended using a template.
        
        Args:
            query_text: Natural language query (conversation)
            topk: Number of top results to return
            image_path: Optional path to image segment (uses self.image_path if not provided)
        """
        if not self.snippet_ids or not self.snippet_texts:
            return MethodResult(scores=[])

        # Build criteria text based on require_reason flag (mirroring llm_list.py)
        reason = (
            "answer or address on why this item can be recommended that requires you to "
            "specificly inform user that user’s explicitly stated requests and preferences "
            "in the conversation"
        )
        additional_info = (
            "directly provide additional information for this item that you think satisfy "
            "user proactive information needs of this item after reccomending this item "
            "based on current given conversation but has not been explicitly requested by "
            "user. And this information requires you, as assistant, to specificly inform users"
        )
        criteria_text = reason if self.require_reason else additional_info

        # Use the original conversation for dense embeddings (no summarization)
        query_for_embedding = query_text

        # Encode original query once with dense embedder (shared across all snippets) - NO CACHING
        query_emb = self._encode_texts_dense([query_for_embedding])[0]
        
        # Encode image if available (shared across all snippets) - NO CACHING
        img_emb: Optional[np.ndarray] = None
        effective_image_path = image_path or self.image_path
        if effective_image_path and effective_image_path.exists():
            img_emb = self._encode_images([effective_image_path])[0]
        # Build combined texts for all snippets using template
        # IMPORTANT: Put snippet FIRST to avoid truncation (CLIP max 77 tokens)
        # If conversation is long, it will be truncated, but snippet will be preserved
        template = (
            "Snippet: {snippet}\n\n"
        )
        # Debug: Check if snippet_texts actually has different values
        unique_snippets = len(set(self.snippet_texts.values()))
        if unique_snippets < len(self.snippet_texts):
            import warnings
            # Show which snippet IDs have which texts
            snippet_id_to_text = {sid: self.snippet_texts[sid] for sid in self.snippet_ids[:5]}
        
        # Debug: Verify snippet_texts is not empty or None
        if not self.snippet_texts:
            raise ValueError("snippet_texts is empty! Cannot encode snippets.")
        
        combined_texts = [
            template.format(
                snippet=self.snippet_texts[snippet_id],
            )
            for snippet_id in self.snippet_ids
        ]

        
        # Encode all snippets in batch with dense embedder - NO CACHING, always compute fresh embeddings
        snippet_emb_dense_all = self._encode_texts_dense(combined_texts)
        snippet_emb_clip_all = self._encode_texts(combined_texts)
        
        # Ensure we have the right number of embeddings
        if len(snippet_emb_dense_all) != len(self.snippet_ids):
            raise ValueError(f"Mismatch: {len(snippet_emb_dense_all)} embeddings for {len(self.snippet_ids)} snippets")
        
        # Verify embeddings are actually different (debug check)
        if len(snippet_emb_dense_all) > 1:
            # Check if all embeddings are identical (which would indicate a bug)
            first_emb = snippet_emb_dense_all[0]
            all_same = all(np.allclose(first_emb, emb, atol=1e-6) for emb in snippet_emb_dense_all[1:])
        
        # Compute fusion scores for all snippets using explicit indexing
        scores: List[float] = []
        for idx, snippet_id in enumerate(self.snippet_ids):
            snippet_emb_dense = snippet_emb_dense_all[idx]
            snippet_emb_clip = snippet_emb_clip_all[idx]
            if self.fusion_method == "linear":
                # Linear fusion: e_joint = normalize(alpha * e_query + (1-alpha) * e_img)
                if img_emb is not None:
                    joint_emb = self.alpha * query_emb - (1 - self.alpha) * img_emb
                    # L2 normalize
                    joint_emb = joint_emb / (np.linalg.norm(joint_emb) + 1e-8)
                    score = float(np.dot(snippet_emb_dense, joint_emb))
                else:
                    # No image, just use query
                    import warnings
                    warnings.warn("No image found, using query only")
                    score = float(np.dot(snippet_emb_dense, query_emb))
            elif self.fusion_method == "late":
                # Late fusion: s_j = alpha * cos(e_query, e_aj) + (1-alpha) * cos(e_img, e_aj)
                query_score = float(np.dot(snippet_emb_dense, query_emb))
                if img_emb is not None:
                    img_score = float(np.dot(snippet_emb_clip, img_emb))
                    score = self.alpha * query_score - (1 - self.alpha) * img_score
                else:
                    # No image, just use query
                    import warnings
                    warnings.warn("No image found, using query only")
                    score = query_score
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
            scores.append(score)

        pairs: List[Tuple[str, float]] = list(zip(self.snippet_ids, scores))
        pairs.sort(key=lambda x: (-x[1], x[0]))
        if topk <= 0:
            return MethodResult(scores=[])
        return MethodResult(scores=pairs[: min(topk, len(pairs))])

