from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    jina_api_key: str

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    lock_ttl_seconds: int = 120
    lock_wait_timeout_seconds: float = 30.0
    lock_poll_interval_seconds: float = 0.10

    langfuse_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    collection_name: str = "multihop_rag"
    dense_embedding_model: str = "gemini-embedding-001"
    decomposition_model: str = "gemini-2.5-flash"
    generation_model: str = "gemini-2.5-flash"
    sparse_model: str = "Qdrant/bm25"
    jina_reranker_model: str = "jina-reranker-v3.5"
    jina_rerank_url: str = "https://api.jina.ai/v1/rerank"
    dense_embedding_dim: int = 1536
    dense_retrieval_limit: int = 50
    sparse_retrieval_limit: int = 50
    hybrid_limit: int = 100
    jina_top_n: int = 80
    jina_max_doc_length: int = 3000
    per_subquery_final_limit: int = 10
    decomposition_temperature: float = 0.0
    final_answer_temperature: float = 0.0
    max_subquery_concurrency: int = 4
    sparse_thread_concurrency: int = 4
    jina_timeout_seconds: float = 60.0
    rag_cache_version: str = "v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
