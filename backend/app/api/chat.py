"""
api/chat.py
FastAPI router:
  POST /chat               — send a message to a character
  GET  /shows              — list all registered shows
  GET  /shows/{id}         — single show with characters
  POST /memory/reset       — clear a session's conversation memory
  GET  /memory/sessions    — list all stored sessions (admin)
  GET  /memory/view        — view raw memory JSON for a session
  GET  /intent/classify    — debug: classify an intent without chatting
  GET  /health             — liveness + Cohere API key check
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.models.chat import (
    ChatRequest, ChatResponse,
    MemoryResetRequest, MemoryResetResponse,
)
from app.services.chat_service import get_chat_service
from app.db.metadata_store import list_shows, get_show, get_character
from app.memory.memory_store import delete_memory, list_sessions
from app.intent.classifier import classify_intent
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Chat"])


# ─────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with a drama character",
    description=(
        "Send a message to a specific character. "
        "Pass a stable `session_id` to maintain conversation memory across requests. "
        "The `episode_number` enforces spoiler protection."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    show = get_show(request.show_id)
    if not show:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Show '{request.show_id}' not found. Register it first via /admin/register.",
        )
    char = get_character(request.show_id, request.character_id)
    if not char:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{request.character_id}' not found in show '{request.show_id}'.",
        )

    service = get_chat_service()
    try:
        return await service.chat(request)
    except Exception as e:
        logger.error("chat_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ─────────────────────────────────────────────────────────────
# Show / Character discovery
# ─────────────────────────────────────────────────────────────

@router.get("/shows", summary="List all registered shows")
async def list_all_shows() -> JSONResponse:
    shows = list_shows()
    sanitised = []
    for show in shows:
        s = dict(show)
        s["characters"] = {
            cid: {k: v for k, v in cdata.items() if k != "persona_prompt"}
            for cid, cdata in show.get("characters", {}).items()
        }
        sanitised.append(s)
    return JSONResponse({"shows": sanitised})


@router.get("/shows/{show_id}", summary="Get a single show with its characters")
async def get_show_detail(show_id: str) -> JSONResponse:
    show = get_show(show_id)
    if not show:
        raise HTTPException(status_code=404, detail=f"Show '{show_id}' not found.")
    s = dict(show)
    s["characters"] = {
        cid: {k: v for k, v in cdata.items() if k != "persona_prompt"}
        for cid, cdata in show.get("characters", {}).items()
    }
    return JSONResponse(s)


# ─────────────────────────────────────────────────────────────
# Memory endpoints
# ─────────────────────────────────────────────────────────────

@router.post(
    "/memory/reset",
    response_model=MemoryResetResponse,
    summary="Clear a session's conversation memory",
)
async def reset_memory(req: MemoryResetRequest) -> MemoryResetResponse:
    deleted = delete_memory(req.session_id)
    return MemoryResetResponse(
        success=deleted,
        session_id=req.session_id,
        message="Memory cleared." if deleted else "No memory found for this session.",
    )


@router.get("/memory/sessions", summary="List all stored conversation sessions")
async def list_memory_sessions(
    show_id: str | None = Query(default=None),
    character_id: str | None = Query(default=None),
) -> JSONResponse:
    sessions = list_sessions(show_id=show_id, character_id=character_id)
    return JSONResponse({"sessions": sessions, "total": len(sessions)})


@router.get("/memory/view", summary="View raw memory JSON for a session")
async def view_memory(session_id: str = Query(...)) -> JSONResponse:
    import json
    from app.memory.memory_store import _memory_path
    path = _memory_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No memory found for this session_id.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


# ─────────────────────────────────────────────────────────────
# Intent debug
# ─────────────────────────────────────────────────────────────

@router.get("/intent/classify", summary="Classify intent of a message (debug)")
async def debug_intent(
    message: str = Query(..., description="Message text to classify"),
) -> JSONResponse:
    result = classify_intent(message, use_llm_fallback=True)
    return JSONResponse({
        "message": message,
        "intent": result.intent.value,
        "confidence": result.confidence,
        "method": result.method,
    })


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

@router.get("/health", summary="Liveness + Cohere API key check")
async def health() -> JSONResponse:
    """
    Returns service status and confirms the Cohere API key is set.
    Does NOT make a live Cohere API call — just checks key presence
    to avoid unnecessary spend on health probes.
    """
    from app.core.config import get_settings
    settings = get_settings()

    cohere_key_set = bool(settings.cohere_api_key)

    return JSONResponse({
        "status": "ok",
        "cohere_key_configured": cohere_key_set,
        "cohere_model": settings.cohere_model,
        "memory_enabled": settings.memory_enabled,
        "intent_enabled": settings.intent_enabled,
        "spoiler_protection": settings.spoiler_protection_enabled,
    })
