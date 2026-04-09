"""
models/chat.py
All Pydantic schemas for request / response validation.
"""

from __future__ import annotations
import uuid
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


# ─────────────────────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    show_id: str
    character_id: str
    episode_number: int = Field(..., ge=1)
    user_message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    history: list[Message] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def normalise(self) -> "ChatRequest":
        self.user_message = self.user_message.strip()
        if not self.session_id:
            self.session_id = f"{self.show_id}__{self.character_id}__{uuid.uuid4().hex[:12]}"
        return self


class RetrievedChunk(BaseModel):
    text: str
    show_id: str
    character_id: Optional[str]
    episode_number: int
    scene_id: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    character_id: str
    character_name: str
    show_id: str
    session_id: str
    intent: str = "general"
    intent_confidence: float = 0.0
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    spoiler_protected: bool = False
    memory_turn_count: int = 0


# ─────────────────────────────────────────────────────────────
# Upload endpoint — NEW episode-level
# ─────────────────────────────────────────────────────────────

class EpisodeUploadResponse(BaseModel):
    """Response for the new episode-level upload."""
    success: bool
    show_id: str
    episode_number: int
    filename: str
    total_chunks_stored: int
    characters_ingested: dict[str, int]   # {character_id: chunk_count}
    characters_absent: list[str]          # characters not found in this episode
    message: str


class UploadResponse(BaseModel):
    """Legacy single-character upload response (kept for compatibility)."""
    success: bool
    show_id: str
    character_id: str
    episode_number: int
    chunks_created: int
    message: str


# ─────────────────────────────────────────────────────────────
# Memory endpoints
# ─────────────────────────────────────────────────────────────

class MemoryResetRequest(BaseModel):
    session_id: str


class MemoryResetResponse(BaseModel):
    success: bool
    session_id: str
    message: str


# ─────────────────────────────────────────────────────────────
# Character / Show metadata
# ─────────────────────────────────────────────────────────────

class CharacterMeta(BaseModel):
    character_id: str
    name: str
    role: str
    persona_prompt: str
    emoji: str


class ShowMeta(BaseModel):
    show_id: str
    name: str
    genre: str
    characters: list[CharacterMeta]