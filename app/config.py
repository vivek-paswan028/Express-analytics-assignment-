from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    google_api_key: str = ""
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.0

    embedding_model: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    retrieval_top_k: int = 5
    max_retries: int = 2

    chunk_size: int = 800
    chunk_overlap: int = 100
    max_upload_size_mb: int = 5
    auto_ingest_corpus: bool = False

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    feedback_log_path: str = "./data/feedback.jsonl"

    tavily_api_key: str = ""

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()

    @property
    def feedback_log_file(self) -> Path:
        p = Path(self.feedback_log_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def effective_api_key(self) -> str:
        return self.gemini_api_key.strip() or self.google_api_key.strip()

    @property
    def has_llm_key(self) -> bool:
        return bool(self.effective_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

