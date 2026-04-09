"""
db/metadata_store.py
JSON-file-based metadata store for shows, characters, and ingested episodes.
In production this would be replaced by PostgreSQL / Firestore.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from threading import Lock
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_lock = Lock()


def _store_path() -> Path:
    settings = get_settings()
    p = Path(settings.processed_dir) / "metadata.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load() -> dict:
    path = _store_path()
    if not path.exists():
        return {"shows": {}, "ingested_episodes": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    path = _store_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# Show / Character registration
# ─────────────────────────────────────────────────────────────

def register_show(
    show_id: str,
    name: str,
    genre: str,
    description: str = "",
) -> None:
    with _lock:
        data = _load()
        if show_id not in data["shows"]:
            data["shows"][show_id] = {
                "show_id": show_id,
                "name": name,
                "genre": genre,
                "description": description,
                "characters": {},
            }
            _save(data)
            logger.info("show_registered", show_id=show_id)


def register_character(
    show_id: str,
    character_id: str,
    name: str,
    role: str,
    persona_prompt: str,
    emoji: str = "🎭",
    lore: Optional[dict] = None,
) -> None:
    with _lock:
        data = _load()
        if show_id not in data["shows"]:
            raise ValueError(f"Show '{show_id}' not registered. Call register_show first.")
        data["shows"][show_id]["characters"][character_id] = {
            "character_id": character_id,
            "name": name,
            "role": role,
            "persona_prompt": persona_prompt,
            "emoji": emoji,
            "lore": lore or {},
        }
        _save(data)
        logger.info("character_registered", show_id=show_id, character_id=character_id)


# ─────────────────────────────────────────────────────────────
# Episode tracking (for spoiler protection)
# ─────────────────────────────────────────────────────────────

def record_ingested_episode(
    show_id: str,
    character_id: str,
    episode_number: int,
    chunk_count: int,
    filename: str,
) -> None:
    with _lock:
        data = _load()
        entry = {
            "show_id": show_id,
            "character_id": character_id,
            "episode_number": episode_number,
            "chunk_count": chunk_count,
            "filename": filename,
        }
        # Deduplicate
        data["ingested_episodes"] = [
            e for e in data["ingested_episodes"]
            if not (
                e["show_id"] == show_id
                and e["character_id"] == character_id
                and e["episode_number"] == episode_number
            )
        ]
        data["ingested_episodes"].append(entry)
        _save(data)


def get_max_available_episode(show_id: str, character_id: str) -> int:
    data = _load()
    episodes = [
        e["episode_number"]
        for e in data["ingested_episodes"]
        if e["show_id"] == show_id and e["character_id"] == character_id
    ]
    return max(episodes, default=0)


# ─────────────────────────────────────────────────────────────
# Lookups
# ─────────────────────────────────────────────────────────────

def get_show(show_id: str) -> Optional[dict]:
    return _load()["shows"].get(show_id)


def get_character(show_id: str, character_id: str) -> Optional[dict]:
    show = get_show(show_id)
    if not show:
        return None
    return show["characters"].get(character_id)


def list_shows() -> list[dict]:
    return list(_load()["shows"].values())


def list_characters(show_id: str) -> list[dict]:
    show = get_show(show_id)
    if not show:
        return []
    return list(show["characters"].values())