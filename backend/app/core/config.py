"""
core/config.py
All application settings loaded from environment variables / .env
"""

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

    # ── App ──────────────────────────────────────────────────
    app_name: str = "Zupee Drama AI Companion"
    app_version: str = "2.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Cohere LLM ───────────────────────────────────────────
    cohere_api_key: str = ""
    cohere_model: str = "command-a-03-2025"   # ← updated from command-r-plus
    cohere_timeout: int = 60
    max_tokens: int = 512
    temperature: float = 0.85

    # ── Embeddings ───────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k_retrieval: int = 5

    # ── Paths ─────────────────────────────────────────────────
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    vector_store_dir: str = "data/processed/vector_stores"
    memory_store_dir: str = "data/memory_store"

    # ── Features ─────────────────────────────────────────────
    spoiler_protection_enabled: bool = True
    memory_enabled: bool = True
    memory_max_turns: int = 20
    memory_summary_threshold: int = 10
    intent_enabled: bool = True

    # ── Computed helpers ─────────────────────────────────────
    @property
    def vector_store_path(self) -> Path:
        return Path(self.vector_store_dir)

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_dir)

    @property
    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    @property
    def memory_store_path(self) -> Path:
        return Path(self.memory_store_dir)


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of Settings()."""
    return Settings()