"""
services/ingest_service.py
═══════════════════════════════════════════════════════════════
Business logic for the new episode-first upload flow.

NEW endpoint  POST /upload/episode
  - Admin uploads ONE file per episode
  - System auto-parses and fans out to all registered characters
  - Each character gets only what they witnessed

LEGACY endpoint  POST /upload  (kept, now labelled clearly)
  - Single character, single episode — for corrections/overrides
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from pathlib import Path

from app.core.logging import get_logger
from app.models.chat import EpisodeUploadResponse, UploadResponse
from app.rag.ingest import ingest_episode, ingest_script

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}
MAX_BYTES = 10 * 1024 * 1024   # 10 MB


def _decode(file_content: bytes, filename: str) -> str:
    try:
        return file_content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_content.decode("latin-1")
            logger.warning("file_decoded_latin1", filename=filename)
            return text
        except Exception:
            raise ValueError("Could not decode file. Please ensure it is UTF-8 encoded.")


def _validate(filename: str, file_content: bytes, min_chars: int = 100) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if len(file_content) > MAX_BYTES:
        raise ValueError(f"File too large ({len(file_content)//1024} KB). Maximum is 10 MB.")
    text = _decode(file_content, filename)
    if len(text.strip()) < min_chars:
        raise ValueError(f"Script is too short (< {min_chars} chars). Upload a full episode script.")
    return text


class IngestService:

    # ── NEW: Episode-level ────────────────────────────────────

    async def ingest_episode_upload(
        self,
        show_id: str,
        episode_number: int,
        filename: str,
        file_content: bytes,
    ) -> EpisodeUploadResponse:
        """
        Ingest a complete episode script.
        Automatically distributes scene knowledge to the right characters.
        """
        raw_text = _validate(filename, file_content, min_chars=100)

        logger.info(
            "episode_upload_received",
            show_id=show_id,
            episode=episode_number,
            filename=filename,
            size_bytes=len(file_content),
        )

        summary = ingest_episode(
            show_id=show_id,
            episode_number=episode_number,
            raw_text=raw_text,
            filename=filename,
        )

        ingested = summary["characters_ingested"]
        absent = summary["characters_absent"]
        total = summary["total_chunks_stored"]

        ingested_summary = ", ".join(
            f"{cid}({n} chunks)" for cid, n in ingested.items()
        ) or "none"
        absent_summary = ", ".join(absent) or "none"

        message = (
            f"Episode {episode_number} ingested from '{filename}'. "
            f"{total} total chunks stored. "
            f"Characters with knowledge: [{ingested_summary}]. "
            f"Characters absent from this episode: [{absent_summary}]."
        )

        return EpisodeUploadResponse(
            success=True,
            show_id=show_id,
            episode_number=episode_number,
            filename=filename,
            total_chunks_stored=total,
            characters_ingested=ingested,
            characters_absent=absent,
            message=message,
        )

    # ── LEGACY: Single-character ──────────────────────────────

    async def ingest_upload(
        self,
        show_id: str,
        character_id: str,
        episode_number: int,
        filename: str,
        file_content: bytes,
    ) -> UploadResponse:
        """
        Legacy: Ingest a script for a single character.
        Use for manual overrides or character-specific lore additions.
        """
        raw_text = _validate(filename, file_content, min_chars=50)

        logger.info(
            "legacy_upload_received",
            show_id=show_id,
            character_id=character_id,
            episode=episode_number,
            filename=filename,
        )

        summary = ingest_script(
            show_id=show_id,
            character_id=character_id,
            episode_number=episode_number,
            raw_text=raw_text,
            filename=filename,
        )

        return UploadResponse(
            success=True,
            show_id=show_id,
            character_id=character_id,
            episode_number=episode_number,
            chunks_created=summary["chunks_created"],
            message=(
                f"[Override] '{filename}' ingested for {character_id} — "
                f"{summary['chunks_created']} chunks created."
            ),
        )


# ── Singleton ─────────────────────────────────────────────────
_ingest_service: IngestService | None = None


def get_ingest_service() -> IngestService:
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = IngestService()
    return _ingest_service