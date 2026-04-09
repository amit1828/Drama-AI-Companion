"""
api/upload.py
═══════════════════════════════════════════════════════════════
Upload and admin routes:

  POST /upload/episode           ← NEW primary endpoint
       Upload one script per episode. System auto-distributes
       scene knowledge to all registered characters.

  POST /upload                   ← Legacy single-character override
       Still available for manual corrections.

  POST /admin/register/show
  POST /admin/register/character
  GET  /admin/episodes
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.models.chat import EpisodeUploadResponse, UploadResponse
from app.services.ingest_service import get_ingest_service
from app.db.metadata_store import (
    register_show,
    register_character,
    get_show,
    _load as _load_meta,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Upload & Admin"])


# ─────────────────────────────────────────────────────────────
# NEW: Episode-level upload (primary workflow)
# ─────────────────────────────────────────────────────────────

@router.post(
    "/upload/episode",
    response_model=EpisodeUploadResponse,
    summary="Upload a full episode script (recommended)",
    description=(
        "Upload ONE script file for an entire episode. "
        "The system automatically parses the script, detects which registered "
        "characters are present in each scene, and indexes only the relevant "
        "portions into each character's knowledge base. "
        "Characters absent from a scene will have no knowledge of it — "
        "they cannot narrate events they didn't witness."
    ),
)
async def upload_episode(
    file: UploadFile = File(..., description="Episode script (.txt or .md)"),
    show_id: str = Form(..., description="Show identifier"),
    episode_number: int = Form(..., ge=1, description="Episode number"),
) -> EpisodeUploadResponse:

    if not get_show(show_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Show '{show_id}' not found. Register it first.",
        )

    file_content = await file.read()
    filename = file.filename or "episode.txt"
    service = get_ingest_service()

    try:
        return await service.ingest_episode_upload(
            show_id=show_id,
            episode_number=episode_number,
            filename=filename,
            file_content=file_content,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error("episode_upload_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ─────────────────────────────────────────────────────────────
# LEGACY: Single-character upload (override / corrections)
# ─────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="[Override] Upload a script for a single character",
    description=(
        "Legacy endpoint. Uploads a script tagged to ONE specific character. "
        "Use this ONLY for manual corrections or to add character-specific background "
        "lore that isn't in the main episode script. "
        "For normal episode ingestion, use POST /upload/episode instead."
    ),
)
async def upload_script_legacy(
    file: UploadFile = File(...),
    show_id: str = Form(...),
    character_id: str = Form(...),
    episode_number: int = Form(..., ge=1),
) -> UploadResponse:

    if not get_show(show_id):
        raise HTTPException(status_code=404, detail=f"Show '{show_id}' not found.")

    file_content = await file.read()
    filename = file.filename or "script.txt"
    service = get_ingest_service()

    try:
        return await service.ingest_upload(
            show_id=show_id,
            character_id=character_id,
            episode_number=episode_number,
            filename=filename,
            file_content=file_content,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("legacy_upload_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# Admin — Register Show
# ─────────────────────────────────────────────────────────────

@router.post("/admin/register/show", summary="Register a new show")
async def register_show_endpoint(
    show_id: str = Form(...),
    name: str = Form(...),
    genre: str = Form(...),
    description: str = Form(default=""),
) -> JSONResponse:
    try:
        register_show(show_id=show_id, name=name, genre=genre, description=description)
        return JSONResponse({"success": True, "message": f"Show '{show_id}' registered."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# Admin — Register Character
# ─────────────────────────────────────────────────────────────

@router.post("/admin/register/character", summary="Register a character with persona")
async def register_character_endpoint(
    show_id: str = Form(...),
    character_id: str = Form(...),
    name: str = Form(...),
    role: str = Form(...),
    persona_prompt: str = Form(...),
    emoji: str = Form(default="🎭"),
    lore_json: str = Form(default="{}"),
) -> JSONResponse:
    import json
    if not get_show(show_id):
        raise HTTPException(status_code=404, detail=f"Show '{show_id}' not found.")
    try:
        lore = json.loads(lore_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="lore_json must be valid JSON.")
    try:
        register_character(
            show_id=show_id,
            character_id=character_id,
            name=name,
            role=role,
            persona_prompt=persona_prompt,
            emoji=emoji,
            lore=lore,
        )
        return JSONResponse({"success": True, "message": f"Character '{character_id}' registered."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# Admin — List Episodes
# ─────────────────────────────────────────────────────────────

@router.get("/admin/episodes", summary="List all ingested episodes")
async def list_episodes(
    show_id: str | None = None,
    character_id: str | None = None,
) -> JSONResponse:
    data = _load_meta()
    episodes = data.get("ingested_episodes", [])
    if show_id:
        episodes = [e for e in episodes if e["show_id"] == show_id]
    if character_id:
        episodes = [e for e in episodes if e["character_id"] == character_id]
    return JSONResponse({"episodes": episodes, "total": len(episodes)})