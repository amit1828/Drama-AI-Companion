"""
rag/embedder.py
Singleton embedding model wrapper using sentence-transformers.

Provides:
  - embed_texts()  — batch embed a list of strings
  - embed_query()  — embed a single query string
"""

from __future__ import annotations
from functools import lru_cache
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Model singleton
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model():
    """Load the embedding model once and cache it for the process lifetime."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    settings = get_settings()
    logger.info("loading_embedding_model", model=settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("embedding_model_loaded", model=settings.embedding_model)
    return model


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of text strings.

    Args:
        texts: Non-empty list of strings to embed.
        batch_size: Sentences per forward pass.
        show_progress: Print tqdm progress bar (useful for large ingests).

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32.
    """
    if not texts:
        raise ValueError("embed_texts() called with empty list")

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # Unit vectors → cosine sim = dot product
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        np.ndarray of shape (embedding_dim,), dtype float32.
    """
    return embed_texts([query])[0]


def get_embedding_dim() -> int:
    """Return the embedding dimension of the loaded model."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()