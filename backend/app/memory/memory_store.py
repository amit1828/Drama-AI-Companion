"""
memory/memory_store.py
═══════════════════════════════════════════════════════════════
Per-session conversation memory with three layers:

  1. SHORT-TERM  — full recent turns (last N turns, in-RAM + persisted)
  2. LONG-TERM   — auto-summarised older turns (LLM-compressed)
  3. ENTITY MEM  — extracted facts (names, relationships, key events)

Storage layout (JSON files):
  data/memory_store/{show_id}__{character_id}__{session_id}.json

Session ID = show_id + character_id + user fingerprint (or UUID).
The frontend sends a stable session_id so memory persists across page reloads.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_locks: dict[str, Lock] = {}


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

class Turn:
    """A single conversation exchange."""
    __slots__ = ["role", "content", "timestamp", "intent", "turn_id"]

    def __init__(
        self,
        role: str,
        content: str,
        intent: str = "general",
        turn_id: Optional[str] = None,
    ):
        self.role = role                          # "user" | "assistant"
        self.content = content
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.intent = intent
        self.turn_id = turn_id or str(uuid.uuid4())[:8]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        t = cls(role=d["role"], content=d["content"], intent=d.get("intent", "general"))
        t.timestamp = d.get("timestamp", t.timestamp)
        t.turn_id = d.get("turn_id", t.turn_id)
        return t


class ConversationMemory:
    """
    Full memory object for one (session, character) pair.

    Fields:
      session_id       Stable ID linking reloads to the same conversation
      show_id          Show context
      character_id     Character being chatted with
      short_term       List[Turn] — recent full turns (max memory_max_turns)
      long_term_summary  Compressed text of older turns
      entities         Dict of extracted facts: {"Priya": "journalist", ...}
      total_turns      Running count including summarised turns
      created_at       ISO timestamp
      last_active      ISO timestamp
    """

    def __init__(
        self,
        session_id: str,
        show_id: str,
        character_id: str,
    ):
        self.session_id = session_id
        self.show_id = show_id
        self.character_id = character_id
        self.short_term: list[Turn] = []
        self.long_term_summary: str = ""
        self.entities: dict[str, str] = {}       # entity_name → description
        self.total_turns: int = 0
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_active = self.created_at

    def add_turn(self, role: str, content: str, intent: str = "general") -> Turn:
        t = Turn(role=role, content=content, intent=intent)
        self.short_term.append(t)
        self.total_turns += 1
        self.last_active = datetime.now(timezone.utc).isoformat()
        return t

    def update_entity(self, name: str, fact: str) -> None:
        self.entities[name] = fact

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "show_id": self.show_id,
            "character_id": self.character_id,
            "short_term": [t.to_dict() for t in self.short_term],
            "long_term_summary": self.long_term_summary,
            "entities": self.entities,
            "total_turns": self.total_turns,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationMemory":
        m = cls(
            session_id=d["session_id"],
            show_id=d["show_id"],
            character_id=d["character_id"],
        )
        m.short_term = [Turn.from_dict(t) for t in d.get("short_term", [])]
        m.long_term_summary = d.get("long_term_summary", "")
        m.entities = d.get("entities", {})
        m.total_turns = d.get("total_turns", len(m.short_term))
        m.created_at = d.get("created_at", m.created_at)
        m.last_active = d.get("last_active", m.created_at)
        return m


# ─────────────────────────────────────────────────────────────
# Storage helpers
# ─────────────────────────────────────────────────────────────

def _memory_path(session_id: str) -> Path:
    settings = get_settings()
    p = Path(settings.memory_store_dir)
    p.mkdir(parents=True, exist_ok=True)
    # Sanitise session_id for use as filename
    safe = session_id.replace("/", "_").replace("\\", "_")[:80]
    return p / f"{safe}.json"


def _get_lock(session_id: str) -> Lock:
    if session_id not in _locks:
        _locks[session_id] = Lock()
    return _locks[session_id]


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def load_memory(
    session_id: str,
    show_id: str,
    character_id: str,
) -> ConversationMemory:
    """
    Load existing memory for a session or create a fresh one.
    Always returns a ConversationMemory object.
    """
    path = _memory_path(session_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mem = ConversationMemory.from_dict(data)
            logger.info(
                "memory_loaded",
                session_id=session_id,
                turns=len(mem.short_term),
                total=mem.total_turns,
            )
            return mem
        except Exception as e:
            logger.warning("memory_load_failed", session_id=session_id, error=str(e))

    # Fresh memory
    mem = ConversationMemory(
        session_id=session_id,
        show_id=show_id,
        character_id=character_id,
    )
    logger.info("memory_created_fresh", session_id=session_id)
    return mem


def save_memory(mem: ConversationMemory) -> None:
    """Persist memory to disk (thread-safe)."""
    lock = _get_lock(mem.session_id)
    path = _memory_path(mem.session_id)
    with lock:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(mem.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("memory_save_failed", session_id=mem.session_id, error=str(e))


def trim_short_term(mem: ConversationMemory, max_turns: Optional[int] = None) -> list[Turn]:
    """
    If short_term exceeds max_turns, pop the oldest half and return them
    as candidates for summarisation. Caller is responsible for summarising.
    """
    settings = get_settings()
    limit = max_turns or settings.memory_max_turns
    if len(mem.short_term) <= limit:
        return []

    cutoff = len(mem.short_term) - (limit // 2)
    to_summarise = mem.short_term[:cutoff]
    mem.short_term = mem.short_term[cutoff:]
    logger.info(
        "short_term_trimmed",
        session_id=mem.session_id,
        trimmed=len(to_summarise),
        remaining=len(mem.short_term),
    )
    return to_summarise


def build_memory_context(mem: ConversationMemory, max_recent_turns: int = 8) -> str:
    """
    Build a string block summarising the conversation memory.
    Injected into the LLM prompt so the character "remembers" the user.

    Returns multi-section text:
      [Long-term summary] (if exists)
      [Key facts about user] (entities)
      [Recent conversation] (last N turns)
    """
    parts: list[str] = []

    if mem.long_term_summary:
        parts.append(
            f"EARLIER IN THIS CONVERSATION (summary):\n{mem.long_term_summary}"
        )

    if mem.entities:
        facts = "\n".join(f"  • {k}: {v}" for k, v in mem.entities.items())
        parts.append(f"WHAT I KNOW ABOUT YOU:\n{facts}")

    recent = mem.short_term[-max_recent_turns:]
    if recent:
        lines = []
        for turn in recent:
            label = "You" if turn.role == "user" else "Me"
            lines.append(f"  {label}: {turn.content}")
        parts.append(f"OUR RECENT EXCHANGE:\n" + "\n".join(lines))

    return "\n\n".join(parts)


def extract_entities_from_text(text: str) -> dict[str, str]:
    """
    Simple heuristic entity extractor for user messages.
    Looks for self-introduction patterns and relationship statements.

    Examples detected:
      "my name is Rahul"       → {"user_name": "Rahul"}
      "I'm from Mumbai"        → {"user_location": "Mumbai"}
      "I love thrillers"       → {"user_preference": "thrillers"}
      "I've watched episode 3" → {"watched_episode": "3"}
    """
    import re
    entities: dict[str, str] = {}
    text_lower = text.lower()

    # Name
    name_match = re.search(r"(?:my name is|i(?:'m| am) called|call me)\s+([a-z][a-z\s]{1,20})", text_lower)
    if name_match:
        entities["user_name"] = name_match.group(1).strip().title()

    # Location
    loc_match = re.search(r"(?:i(?:'m| am) from|i live in)\s+([a-z][a-z\s]{1,25})", text_lower)
    if loc_match:
        entities["user_location"] = loc_match.group(1).strip().title()

    # Episode watched
    ep_match = re.search(r"(?:watched|seen|finished|on)\s+episode\s+(\d+)", text_lower)
    if ep_match:
        entities["watched_episode"] = ep_match.group(1)

    # Favourite genre / preference
    pref_match = re.search(r"i (?:love|like|enjoy|prefer)\s+([a-z][a-z\s]{1,30})", text_lower)
    if pref_match:
        entities["user_preference"] = pref_match.group(1).strip()

    return entities


def delete_memory(session_id: str) -> bool:
    """Delete a session's memory file. Returns True if deleted."""
    path = _memory_path(session_id)
    if path.exists():
        path.unlink()
        logger.info("memory_deleted", session_id=session_id)
        return True
    return False


def list_sessions(show_id: Optional[str] = None, character_id: Optional[str] = None) -> list[dict]:
    """List all stored sessions (for admin panel)."""
    settings = get_settings()
    p = Path(settings.memory_store_dir)
    if not p.exists():
        return []

    results = []
    for f in p.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            if show_id and d.get("show_id") != show_id:
                continue
            if character_id and d.get("character_id") != character_id:
                continue
            results.append({
                "session_id": d.get("session_id"),
                "show_id": d.get("show_id"),
                "character_id": d.get("character_id"),
                "total_turns": d.get("total_turns", 0),
                "last_active": d.get("last_active"),
            })
        except Exception:
            continue

    return sorted(results, key=lambda x: x.get("last_active", ""), reverse=True)