"""
rag/filters.py
Spoiler protection logic.

The core rule:
  A user watching episode N must NEVER receive context from episodes > N.
  This is enforced at two levels:
    1. vector_store.search() filters by max_episode at retrieval time.
    2. filters.apply_spoiler_fence() double-checks retrieved chunks here.

Also provides character-scoping utilities.
"""

from __future__ import annotations
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.vector_store import ChunkMeta

logger = get_logger(__name__)


@dataclass
class FilteredChunks:
    allowed: list[ChunkMeta]
    blocked: list[ChunkMeta]
    spoiler_protection_triggered: bool


def apply_spoiler_fence(
    chunks: list[tuple[ChunkMeta, float]],
    max_episode: int,
) -> FilteredChunks:
    """
    Secondary spoiler filter applied after retrieval.

    Removes any chunks whose episode_number exceeds the user's current episode.
    Logs a warning if any slipped through the primary filter (indicates a bug).

    Args:
        chunks: Retrieved (ChunkMeta, score) pairs.
        max_episode: The episode the user is currently watching.

    Returns:
        FilteredChunks with allowed and blocked lists.
    """
    settings = get_settings()

    if not settings.spoiler_protection_enabled:
        return FilteredChunks(
            allowed=[m for m, _ in chunks],
            blocked=[],
            spoiler_protection_triggered=False,
        )

    allowed = []
    blocked = []

    for meta, score in chunks:
        if meta.episode_number <= max_episode:
            allowed.append(meta)
        else:
            blocked.append(meta)
            logger.warning(
                "spoiler_chunk_blocked",
                episode=meta.episode_number,
                max_allowed=max_episode,
                show_id=meta.show_id,
                character_id=meta.character_id,
            )

    triggered = len(blocked) > 0

    return FilteredChunks(
        allowed=allowed,
        blocked=blocked,
        spoiler_protection_triggered=triggered,
    )


def scope_to_character(
    chunks: list[ChunkMeta],
    character_id: str,
    include_shared: bool = True,
) -> list[ChunkMeta]:
    """
    Filter chunks to only those belonging to a specific character.
    Optionally include chunks tagged as 'shared' (scene descriptions
    without a specific character owner).

    Args:
        chunks: List of ChunkMeta to filter.
        character_id: Target character.
        include_shared: Whether to include 'shared'/'narrator' tagged chunks.
    """
    result = []
    for chunk in chunks:
        if chunk.character_id == character_id:
            result.append(chunk)
        elif include_shared and chunk.character_id in ("shared", "narrator", None, ""):
            result.append(chunk)
    return result


def deduplicate_chunks(chunks: list[ChunkMeta], similarity_threshold: int = 80) -> list[ChunkMeta]:
    """
    Remove near-duplicate chunks based on text overlap.
    Uses simple word-overlap ratio as a cheap deduplication step.
    """
    seen_words: list[set] = []
    unique: list[ChunkMeta] = []

    for chunk in chunks:
        words = set(chunk.text.lower().split())
        is_dup = False
        for seen in seen_words:
            if not seen or not words:
                continue
            overlap = len(words & seen) / max(len(words | seen), 1) * 100
            if overlap >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(chunk)
            seen_words.append(words)

    return unique