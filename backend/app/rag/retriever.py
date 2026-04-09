"""
rag/retriever.py
Orchestrates the retrieval phase of RAG:
  user query → embedding → FAISS search → spoiler filter → dedup → return chunks
"""

from __future__ import annotations
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embedder import embed_query
from app.rag.filters import apply_spoiler_fence, deduplicate_chunks
from app.db.vector_store import search, ChunkMeta

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[ChunkMeta]
    scores: list[float]
    spoiler_protection_triggered: bool
    query_used: str


def retrieve(
    query: str,
    show_id: str,
    character_id: str,
    episode_number: int,
    top_k: int | None = None,
) -> RetrievalResult:
    """
    Full retrieval pipeline for a user query.

    1. Embed the query.
    2. Search FAISS index (already episode-filtered).
    3. Apply secondary spoiler fence.
    4. Deduplicate.
    5. Return ordered chunks with scores.

    Args:
        query: The user's raw message.
        show_id: Show context.
        character_id: Character being chatted with.
        episode_number: User's current episode (spoiler ceiling).
        top_k: Override default retrieval count.

    Returns:
        RetrievalResult with chunks ordered by relevance.
    """
    settings = get_settings()
    k = top_k or settings.top_k_retrieval

    logger.info(
        "retrieval_start",
        show_id=show_id,
        character_id=character_id,
        episode=episode_number,
        query_len=len(query),
    )

    # ── Embed query ──────────────────────────────────────────
    query_embedding = embed_query(query)

    # ── Vector search (primary episode filter inside) ────────
    raw_results = search(
        show_id=show_id,
        character_id=character_id,
        query_embedding=query_embedding,
        max_episode=episode_number,
        top_k=k * 3,   # Over-fetch so dedup still leaves us with k
    )

    if not raw_results:
        logger.info("retrieval_empty", show_id=show_id, character_id=character_id)
        return RetrievalResult(
            chunks=[],
            scores=[],
            spoiler_protection_triggered=False,
            query_used=query,
        )

    # ── Secondary spoiler fence ──────────────────────────────
    filtered = apply_spoiler_fence(raw_results, max_episode=episode_number)

    # ── Deduplicate ──────────────────────────────────────────
    unique_chunks = deduplicate_chunks(filtered.allowed)

    # ── Trim to top_k ────────────────────────────────────────
    # Scores map: chunk → score (filtered.allowed is in order)
    score_map = {id(m): s for m, s in raw_results}
    final_chunks = unique_chunks[:k]
    final_scores = [score_map.get(id(c), 0.0) for c in final_chunks]

    logger.info(
        "retrieval_complete",
        retrieved=len(raw_results),
        after_filter=len(filtered.allowed),
        after_dedup=len(unique_chunks),
        final=len(final_chunks),
        spoiler_triggered=filtered.spoiler_protection_triggered,
    )

    return RetrievalResult(
        chunks=final_chunks,
        scores=final_scores,
        spoiler_protection_triggered=filtered.spoiler_protection_triggered,
        query_used=query,
    )