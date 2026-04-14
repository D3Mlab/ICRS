from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NormalizeCfg:
    lower: bool = True
    deaccent: bool = True
    strip_punct: bool = True


@dataclass
class WindowCfg:
    size: int = 2
    stride: int = 1


@dataclass
class IndexCfg:
    granularity: str = "mixed"
    fields_include: List[str] = field(default_factory=lambda: [
        "title", "description", "attributes", "specs", "reviews.text"
    ])
    fields_exclude: List[str] = field(default_factory=lambda: [
        "doc_id", "id", "method", "segment_file"
    ])
    window: WindowCfg = field(default_factory=WindowCfg)
    normalize: NormalizeCfg = field(default_factory=NormalizeCfg)


@dataclass
class BM25Cfg:
    k1: float = 1.5
    b: float = 0.75


@dataclass
class DenseCfg:
    model_name: str = "qwen-2.5-8b"
    normalize_embeddings: bool = True
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    request_timeout: float = 60.0


@dataclass
class LLMCfg:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    max_retries: int = 2
    timeout_s: int = 30
    api_base: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    require_reason: bool = True
    use_image: bool = True
    expansion_model: Optional[str] = None  # Smaller model for expansion/extraction (defaults to model if not set)
    entailment_model: Optional[str] = None  # Larger model for entailment (defaults to model if not set)


@dataclass
class CrossEncoderCfg:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    device: Optional[str] = None


@dataclass
class RerankCfg:
    dense_model_name: str = "qwen-2.5-8b"
    dense_normalize_embeddings: bool = True
    dense_api_base: str = "https://openrouter.ai/api/v1"
    dense_api_key_env: str = "OPENROUTER_API_KEY"
    dense_request_timeout: float = 60.0
    initial_k: int = 10
    rerank_model: str = "cohere/rerank-3.5"
    rerank_top_n: int = 10
    rerank_api_key_env: str = "COHERE_API_KEY"
    request_timeout: float = 30.0


@dataclass
class ClipCfg:
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    fusion_method: str = "linear"  # "linear" or "late"
    alpha: float = 0.5
    device: Optional[str] = None
    batch_size: int = 32
    require_reason: bool = True
    summarize_conversation: bool = True
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_api_key: Optional[str] = None


@dataclass
class LLMRerankCfg:
    dense_model_name: str = "qwen-2.5-8b"
    dense_normalize_embeddings: bool = True
    dense_api_base: str = "https://openrouter.ai/api/v1"
    dense_api_key_env: str = "OPENROUTER_API_KEY"
    dense_request_timeout: float = 60.0
    initial_k: int = 10
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_base: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    require_reason: bool = True
    use_image: bool = True
    rerank_top_n: int = 5


@dataclass
class MethodCfg:
    name: str = "bm25"  # bm25 | dense | llm | llm_list | llm_expansion | cross_encode | rerank | clip | llm_rerank
    bm25: BM25Cfg = field(default_factory=BM25Cfg)
    dense: DenseCfg = field(default_factory=DenseCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    llm_list: LLMCfg = field(default_factory=LLMCfg)
    llm_expansion: LLMCfg = field(default_factory=LLMCfg)
    cross_encode: CrossEncoderCfg = field(default_factory=CrossEncoderCfg)
    rerank: RerankCfg = field(default_factory=RerankCfg)
    clip: ClipCfg = field(default_factory=ClipCfg)
    llm_rerank: LLMRerankCfg = field(default_factory=LLMRerankCfg)


@dataclass
class LimitsCfg:
    max_snippets_per_doc: int = 2000
    topk: int = 100


@dataclass
class LoggingCfg:
    save_details: bool = True
    verbose: bool = False


@dataclass
class Step6Config:
    index: IndexCfg = field(default_factory=IndexCfg)
    method: MethodCfg = field(default_factory=MethodCfg)
    limits: LimitsCfg = field(default_factory=LimitsCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)


