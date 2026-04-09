"""
rag/ingest.py
═══════════════════════════════════════════════════════════════
Full ingestion pipeline — episode-first design.

NEW FLOW (v3):
  ONE episode file uploaded by admin
       │
       ▼
  scene_parser.parse_episode_for_characters()
       │  splits into blocks, detects who was present in each
       │
       ├─► Character A script (only blocks A witnessed)
       │         → chunked → embedded → stored in A's FAISS index
       │
       ├─► Character B script (only blocks B witnessed)
       │         → chunked → embedded → stored in B's FAISS index
       │
       └─► Character C was absent → no knowledge → skipped

This guarantees:
  - Shylock cannot narrate Bassanio choosing the casket (wasn't there)
  - Portia cannot describe "Hath not a Jew eyes?" (wasn't there)
  - Each character speaks only from their own witnessed experience

OLD FLOW (deprecated but kept for compatibility):
  ingest_script(show_id, character_id, episode_number, text)
  Still available for manual / script-level ingestion.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import hashlib
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.utils.text_utils import clean_script_text, split_into_chunks
from app.rag.embedder import embed_texts
from app.rag.scene_parser import parse_episode_for_characters
from app.db.vector_store import upsert_chunks
from app.db.metadata_store import (
    record_ingested_episode,
    list_characters,
    get_show,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _scene_id(show_id: str, character_id: str, episode_number: int) -> str:
    raw = f"{show_id}:{character_id}:{episode_number}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _ingest_character_text(
    show_id: str,
    character_id: str,
    episode_number: int,
    text: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """
    Chunk, embed, and store text for ONE character.
    Returns number of chunks stored.
    """
    if not text.strip():
        return 0

    chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        return 0

    chunk_texts = [c.text for c in chunks]
    embeddings = embed_texts(chunk_texts)
    scene_id = _scene_id(show_id, character_id, episode_number)

    stored = upsert_chunks(
        show_id=show_id,
        character_id=character_id,
        episode_number=episode_number,
        texts=chunk_texts,
        embeddings=embeddings,
        scene_id=scene_id,
    )

    record_ingested_episode(
        show_id=show_id,
        character_id=character_id,
        episode_number=episode_number,
        chunk_count=stored,
        filename=filename,
    )

    return stored


# ─────────────────────────────────────────────────────────────
# NEW: Episode-level ingestion (one file → all characters)
# ─────────────────────────────────────────────────────────────

def ingest_episode(
    show_id: str,
    episode_number: int,
    raw_text: str,
    filename: str = "unknown",
) -> dict:
    """
    Ingest a complete episode script and automatically fan out to
    all registered characters — each receiving ONLY the blocks
    where they were physically present.
    """
    settings = get_settings()

    logger.info(
        "episode_ingest_start",
        show_id=show_id,
        episode=episode_number,
        raw_chars=len(raw_text),
        filename=filename,
    )

    # Validate show
    show = get_show(show_id)
    if not show:
        raise ValueError(f"Show '{show_id}' is not registered.")

    # Load characters
    chars = list_characters(show_id)
    if not chars:
        raise ValueError(f"No characters registered for show '{show_id}'.")

    registered = {c["character_id"]: c["name"] for c in chars}

    # Clean text
    cleaned = clean_script_text(raw_text)
    if len(cleaned) < 100:
        raise ValueError("Episode script too short after cleaning.")

    # Parse scenes
    character_scripts = parse_episode_for_characters(
        raw_text=cleaned,
        show_id=show_id,
        episode_number=episode_number,
        registered_characters=registered,
    )

    # Ingest per character
    per_character_results: dict[str, int] = {}
    absent_characters: list[str] = []

    for char_id in registered:
        if char_id in character_scripts:
            cs = character_scripts[char_id]
            chunks_stored = _ingest_character_text(
                show_id,
                char_id,
                episode_number,
                cs.full_text,
                filename,
                settings.chunk_size,
                settings.chunk_overlap,
            )
            per_character_results[char_id] = chunks_stored
        else:
            absent_characters.append(char_id)

    total_chunks = sum(per_character_results.values())

    return {
        "show_id": show_id,
        "episode_number": episode_number,
        "filename": filename,
        "total_chunks_stored": total_chunks,
        "characters_ingested": per_character_results,
        "characters_absent": absent_characters,
    }


# ─────────────────────────────────────────────────────────────
# LEGACY
# ─────────────────────────────────────────────────────────────

def ingest_script(
    show_id: str,
    character_id: str,
    episode_number: int,
    raw_text: str,
    filename: str = "unknown",
) -> dict:
    settings = get_settings()

    cleaned = clean_script_text(raw_text)
    stored = _ingest_character_text(
        show_id,
        character_id,
        episode_number,
        cleaned,
        filename,
        settings.chunk_size,
        settings.chunk_overlap,
    )

    return {
        "show_id": show_id,
        "character_id": character_id,
        "episode_number": episode_number,
        "chunks_created": stored,
    }


def ingest_from_file(
    filepath: str | Path,
    show_id: str,
    episode_number: int,
    character_id: str | None = None,
) -> dict:
    path = Path(filepath)
    raw_text = path.read_text(encoding="utf-8")

    if character_id:
        return ingest_script(show_id, character_id, episode_number, raw_text, path.name)
    else:
        return ingest_episode(show_id, episode_number, raw_text, path.name)