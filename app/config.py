"""
app/config.py
─────────────
Centralised settings loaded from environment variables / .env file.
All modules import `get_settings()` rather than reading os.environ directly.
This makes the settings testable (override via env vars in pytest) and
documents every configurable parameter in one place.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unrecognised env vars
    )

    # ── LLM ──────────────────────────────────────────────────────────
    google_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.0

    # ── Embeddings ───────────────────────────────────────────────────
    # Local sentence-transformers model; downloaded once to HF cache.
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Vector store ─────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    # ── Retrieval ────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    # Max query-rewrite + re-retrieval cycles before returning fallback.
    # 2 is a deliberate choice: one retry handles vocabulary mismatch,
    # a second handles broader reformulation. A third rarely helps on a
    # small local corpus and adds noticeable latency.
    max_retries: int = 2

    # ── Ingestion ────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_upload_size_mb: int = 5

    # ── FastAPI ───────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    # ── Feedback ─────────────────────────────────────────────────────
    feedback_log_path: str = "./data/feedback.jsonl"

    # ── Optional bonus ───────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── Derived helpers ───────────────────────────────────────────────
    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()

    @property
    def feedback_log_file(self) -> Path:
        p = Path(self.feedback_log_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def has_llm_key(self) -> bool:
        return bool(self.google_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (constructed once per process)."""
    return Settings()

