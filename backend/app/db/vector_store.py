"""
db/vector_store.py
FAISS-based vector store manager.

Each (show_id, character_id) pair gets its own FAISS index file so
that character retrieval is always scoped and never bleeds across shows.

Index files are stored at:
  data/processed/vector_stores/{show_id}__{character_id}.index
  data/processed/vector_stores/{show_id}__{character_id}.meta.json
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed — vector store will use brute-force numpy fallback")


_locks: dict[str, Lock] = {}


def _key(show_id: str, character_id: str) -> str:
    return f"{show_id}__{character_id}"


def _base_path(show_id: str, character_id: str) -> Path:
    settings = get_settings()
    p = Path(settings.vector_store_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / _key(show_id, character_id)


def _get_lock(show_id: str, character_id: str) -> Lock:
    k = _key(show_id, character_id)
    if k not in _locks:
        _locks[k] = Lock()
    return _locks[k]


# ─────────────────────────────────────────────────────────────
# Chunk metadata schema stored alongside the index
# ─────────────────────────────────────────────────────────────

class ChunkMeta:
    __slots__ = ["text", "show_id", "character_id", "episode_number", "scene_id", "chunk_index"]

    def __init__(
        self,
        text: str,
        show_id: str,
        character_id: str,
        episode_number: int,
        scene_id: str,
        chunk_index: int,
    ):
        self.text = text
        self.show_id = show_id
        self.character_id = character_id
        self.episode_number = episode_number
        self.scene_id = scene_id
        self.chunk_index = chunk_index

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "show_id": self.show_id,
            "character_id": self.character_id,
            "episode_number": self.episode_number,
            "scene_id": self.scene_id,
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkMeta":
        return cls(**d)


# ─────────────────────────────────────────────────────────────
# Core operations
# ─────────────────────────────────────────────────────────────

def upsert_chunks(
    show_id: str,
    character_id: str,
    episode_number: int,
    texts: list[str],
    embeddings: np.ndarray,
    scene_id: str = "default",
) -> int:
    """
    Add new chunks to the FAISS index for a character.
    Removes any existing chunks for the same episode first (idempotent re-ingest).

    Returns number of vectors added.
    """
    lock = _get_lock(show_id, character_id)
    base = _base_path(show_id, character_id)
    meta_path = Path(str(base) + ".meta.pkl")
    index_path = Path(str(base) + ".index")

    with lock:
        # Load existing metadata
        existing_metas: list[ChunkMeta] = []
        existing_embeddings: Optional[np.ndarray] = None

        if meta_path.exists():
            with open(meta_path, "rb") as f:
                existing_metas = pickle.load(f)

        # Remove stale chunks for this episode
        if existing_metas:
            keep_indices = [
                i for i, m in enumerate(existing_metas)
                if not (m.episode_number == episode_number and m.character_id == character_id)
            ]
            existing_metas = [existing_metas[i] for i in keep_indices]

            if index_path.exists() and FAISS_AVAILABLE and keep_indices:
                old_index = faiss.read_index(str(index_path))
                dim = old_index.d
                # Reconstruct kept embeddings
                all_vecs = np.zeros((old_index.ntotal, dim), dtype=np.float32)
                for i in range(old_index.ntotal):
                    all_vecs[i] = old_index.reconstruct(i)
                existing_embeddings = all_vecs[keep_indices] if keep_indices else None

        # Build new metas
        new_metas = [
            ChunkMeta(
                text=text,
                show_id=show_id,
                character_id=character_id,
                episode_number=episode_number,
                scene_id=scene_id,
                chunk_index=i,
            )
            for i, text in enumerate(texts)
        ]

        all_metas = existing_metas + new_metas

        # Build combined embeddings
        new_emb = embeddings.astype(np.float32)
        if existing_embeddings is not None and len(existing_embeddings) > 0:
            all_embeddings = np.vstack([existing_embeddings, new_emb])
        else:
            all_embeddings = new_emb

        # Build / rebuild FAISS index
        if FAISS_AVAILABLE:
            dim = all_embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner Product (cosine after normalisation)
            faiss.normalize_L2(all_embeddings)
            index.add(all_embeddings)
            faiss.write_index(index, str(index_path))
        else:
            # Numpy fallback — save raw embeddings
            np.save(str(index_path) + ".npy", all_embeddings)

        with open(meta_path, "wb") as f:
            pickle.dump(all_metas, f)

        logger.info(
            "chunks_upserted",
            show_id=show_id,
            character_id=character_id,
            episode=episode_number,
            new_chunks=len(new_metas),
            total_chunks=len(all_metas),
        )
        return len(new_metas)


def search(
    show_id: str,
    character_id: str,
    query_embedding: np.ndarray,
    max_episode: int,
    top_k: int = 5,
) -> list[tuple[ChunkMeta, float]]:
    """
    Search the vector index for a character, filtered to episodes ≤ max_episode.

    Returns list of (ChunkMeta, score) sorted by relevance descending.
    """
    base = _base_path(show_id, character_id)
    meta_path = Path(str(base) + ".meta.pkl")
    index_path = Path(str(base) + ".index")

    if not meta_path.exists():
        logger.warning("no_index_found", show_id=show_id, character_id=character_id)
        return []

    with open(meta_path, "rb") as f:
        all_metas: list[ChunkMeta] = pickle.load(f)

    # Episode filter — this is the spoiler fence
    allowed_indices = [
        i for i, m in enumerate(all_metas)
        if m.episode_number <= max_episode
    ]

    if not allowed_indices:
        return []

    query = query_embedding.astype(np.float32).reshape(1, -1)

    if FAISS_AVAILABLE and index_path.exists():
        index = faiss.read_index(str(index_path))
        faiss.normalize_L2(query)
        # Search all, then filter by allowed
        search_k = min(index.ntotal, top_k * 10)
        scores, indices = index.search(query, search_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx in allowed_indices:
                results.append((all_metas[idx], float(score)))
            if len(results) >= top_k:
                break
    else:
        # Numpy cosine similarity fallback
        emb_path = str(index_path) + ".npy"
        if not Path(emb_path).exists():
            return []
        all_embs = np.load(emb_path)
        allowed_embs = all_embs[allowed_indices]
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(allowed_embs, axis=1, keepdims=True) + 1e-10
        sims = (allowed_embs / norms) @ q_norm.T
        top_local = np.argsort(sims[:, 0])[::-1][:top_k]
        results = [
            (all_metas[allowed_indices[i]], float(sims[i, 0]))
            for i in top_local
        ]

    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]