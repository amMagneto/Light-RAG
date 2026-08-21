from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Light-RAG"
    log_level: str = "INFO"

    database_url: str = "sqlite+aiosqlite:///./rag_app.db"

    admin_api_key: str | None = None

    groq_api_key: str | None = None
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    llm_rpm_limit: int = 25

    embedding_model: str = "intfloat/e5-base-v2"
    chroma_dir: str = "./chroma_store"
    collection_name: str = "light_rag_chunks"
    memory_collection_name: str = "light_rag_memories"

    storage_dir: str = "./storage"

    chunk_size: int = 900
    chunk_overlap: int = 150

    dense_top_k: int = 25
    sparse_top_k: int = 25
    rrf_k: int = 60
    candidate_pool: int = 40
    rerank_top_k: int = 8

    min_relevance: float = 0.35
    strong_relevance: float = 0.55
    min_lexical_for_top_rank: float = 0.22
    max_refinement_attempts: int = 2

    parent_expansion: bool = True
    parent_max_chars: int = 2400

    memory_recall_k: int = 3
    enable_memory_extraction: bool = True

    @property
    def llm_available(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def uploads_dir(self) -> Path:
        return Path(self.storage_dir) / "uploads"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
