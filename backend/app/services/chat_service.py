"""
services/chat_service.py
═══════════════════════════════════════════════════════════════
Business logic layer — orchestrates the full RAG + Memory + Intent pipeline:

  request
    │
    ├── 1. Classify intent (fast rules → LLM fallback)
    ├── 2. Load/update conversation memory
    ├── 3. Extract entities from user message
    ├── 4. RAG retrieval (spoiler-fenced)
    ├── 5. Build system prompt (persona + memory + RAG + intent tone)
    ├── 6. Generate reply via Cohere
    ├── 7. Save updated memory (with optional auto-summarisation)
    └── 8. Return ChatResponse
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import asyncio

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.metadata_store import get_character, get_show
from app.models.chat import ChatRequest, ChatResponse, RetrievedChunk
from app.rag.retriever import retrieve
from app.rag.prompt_builder import build_system_prompt, build_messages_payload
from app.rag.generator import generate_reply
from app.memory.memory_store import (
    load_memory,
    save_memory,
    build_memory_context,
    extract_entities_from_text,
)
from app.memory.summariser import summarise_and_compress
from app.intent.classifier import classify_intent, get_tone_hint

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Uncertainty guard (ANTI-SPOILER CONTROL)  new
# ─────────────────────────────────────────────
def should_force_uncertainty(retrieved_chunks, scores, threshold=0.75):
    if not retrieved_chunks:
        return True
    if max(scores) < threshold:
        return True
    return False


class ChatService:
    """Stateless service — orchestrates every chat request end-to-end."""

    async def chat(self, request: ChatRequest) -> ChatResponse:
        settings = get_settings()
        show_id = request.show_id
        character_id = request.character_id
        episode_number = request.episode_number
        session_id = request.session_id

        # ── Validate show / character ─────────────────────────
        char_meta = get_character(show_id, character_id)
        character_name = char_meta["name"] if char_meta else character_id.title()

        logger.info(
            "chat_request",
            show_id=show_id,
            character_id=character_id,
            episode=episode_number,
            session_id=session_id,
            msg_preview=request.user_message[:60],
        )

        # ── Step 1: Intent classification ─────────────────────
        intent_result = classify_intent(
            text=request.user_message,
            use_llm_fallback=settings.intent_enabled,
        )
        tone_hint = get_tone_hint(intent_result.intent)
        logger.info(
            "intent_classified",
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            method=intent_result.method,
        )

        # ── Step 2: Load conversation memory ──────────────────
        mem = load_memory(
            session_id=session_id,
            show_id=show_id,
            character_id=character_id,
        )

        # ── Step 3: Entity extraction from user message ───────
        new_entities = extract_entities_from_text(request.user_message)
        for key, val in new_entities.items():
            mem.update_entity(key, val)

        # ── Step 4: RAG retrieval ─────────────────────────────
        retrieval = retrieve(
            query=request.user_message,
            show_id=show_id,
            character_id=character_id,
            episode_number=episode_number,
        )
        # ── Step 4.5: Uncertainty / hallucination guard ───────
        force_uncertain = should_force_uncertainty(
            retrieval.chunks,
            retrieval.scores,
            threshold=0.75
        )

        # ── Step 5: Build memory context string ───────────────
        memory_context = build_memory_context(mem, max_recent_turns=8)

        # ── Step 6: Build system prompt ───────────────────────
        system_prompt = build_system_prompt(
            show_id=show_id,
            character_id=character_id,
            episode_number=episode_number,
            retrieved_chunks=retrieval.chunks,
            memory_context=memory_context,
            intent_label=intent_result.intent.value,
            tone_hint=tone_hint,
            force_uncertain=force_uncertain,   # for hallucination control in low-confidence retrievals
        )

        # ── Step 7: Build messages (recent history only) ──────
        messages = build_messages_payload(
            history=request.history,
            user_message=request.user_message,
            max_history_turns=4,
        )

        # ── Step 8: Generate reply ────────────────────────────
        # generate_reply is a sync/blocking function (tenacity retry).
        # Run it in a thread so we don't block the async event loop.
        reply, is_fallback = await asyncio.to_thread(
            generate_reply,
            system_prompt,
            messages,
        )
        if is_fallback:
            logger.warning("chat_used_fallback", character_id=character_id)

        # ── Step 9: Update memory with this exchange ──────────
        mem.add_turn(role="user", content=request.user_message, intent=intent_result.intent.value)
        mem.add_turn(role="assistant", content=reply, intent=intent_result.intent.value)

        # Auto-summarise if short_term is getting long.
        # summarise_and_compress is a plain sync function returning bool — do NOT await it.
        # Run it in a thread to avoid blocking the event loop during the Cohere API call inside.
        if len(mem.short_term) >= settings.memory_summary_threshold:
            await asyncio.to_thread(summarise_and_compress, mem)

        save_memory(mem)

        # ── Step 10: Package response ─────────────────────────
        retrieved_chunks_out = [
            RetrievedChunk(
                text=chunk.text[:300],
                show_id=chunk.show_id,
                character_id=chunk.character_id,
                episode_number=chunk.episode_number,
                scene_id=chunk.scene_id,
                score=round(score, 4),
            )
            for chunk, score in zip(retrieval.chunks, retrieval.scores)
        ]

        logger.info(
            "chat_response_ready",
            character_id=character_id,
            session_id=session_id,
            reply_len=len(reply),
            chunks_used=len(retrieved_chunks_out),
            memory_turns=mem.total_turns,
        )

        return ChatResponse(
            reply=reply,
            character_id=character_id,
            character_name=character_name,
            show_id=show_id,
            session_id=session_id,
            intent=intent_result.intent.value,
            intent_confidence=round(intent_result.confidence, 2),
            retrieved_chunks=retrieved_chunks_out,
            spoiler_protected=retrieval.spoiler_protection_triggered,
            memory_turn_count=mem.total_turns,
        )


# ── Singleton ─────────────────────────────────────────────────
_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service