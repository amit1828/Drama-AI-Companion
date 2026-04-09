"""
rag/scene_parser.py
═══════════════════════════════════════════════════════════════
Episode-level script parser.
...
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class SceneBlock:
    """A coherent block of script text (scene or sub-scene)."""
    block_index: int
    raw_text: str
    present_characters: set[str]
    location: str = ""


@dataclass
class CharacterScript:
    character_id: str
    episode_number: int
    show_id: str
    blocks: list[SceneBlock] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n---\n\n".join(b.raw_text for b in self.blocks)

    @property
    def total_chars(self) -> int:
        return sum(len(b.raw_text) for b in self.blocks)

    @property
    def block_count(self) -> int:
        return len(self.blocks)


# ─────────────────────────────────────────────────────────────
# Script splitter
# ─────────────────────────────────────────────────────────────

_SCENE_BREAK_PATTERNS = [
    re.compile(r"^(SCENE|ACT|EPISODE|INT\.|EXT\.)\s", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^={3,}", re.MULTILINE),
    re.compile(r"^-{3,}", re.MULTILINE),
    re.compile(r"^\[(?:Enter|Exit|Exeunt)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\*{3,}", re.MULTILINE),
]

_ENTER_PATTERN = re.compile(r"\[(?:Enter|Re-enter)\s+([^\]]+)\]", re.IGNORECASE)
_EXIT_PATTERN = re.compile(r"\[(?:Exit|Exeunt)\s*([^\]]*)\]", re.IGNORECASE)

_DIALOGUE_PATTERN = re.compile(
    r"^([A-Z][A-Z\s\-\'\.]{1,40})(?:\s*\([^)]*\))?:\s",
    re.MULTILINE
)


def split_into_scene_blocks(raw_text: str, min_block_chars: int = 100) -> list[str]:
    text = raw_text.strip()

    splits: list[int] = [0]
    for pattern in _SCENE_BREAK_PATTERNS[:3]:
        for m in pattern.finditer(text):
            if m.start() > 0:
                splits.append(m.start())

    splits = sorted(set(splits))

    if len(splits) > 1:
        blocks = []
        for i, start in enumerate(splits):
            end = splits[i + 1] if i + 1 < len(splits) else len(text)
            block = text[start:end].strip()
            if len(block) >= min_block_chars:
                blocks.append(block)
        if blocks:
            return blocks

    enter_positions = [m.start() for m in _ENTER_PATTERN.finditer(text)]
    if len(enter_positions) > 1:
        blocks = []
        for i, start in enumerate(enter_positions):
            end = enter_positions[i + 1] if i + 1 < len(enter_positions) else len(text)
            block = text[start:end].strip()
            if len(block) >= min_block_chars:
                blocks.append(block)
        if blocks:
            return blocks

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunk_size = max(3, len(paragraphs) // 8)
    blocks = []
    for i in range(0, len(paragraphs), chunk_size):
        block = "\n\n".join(paragraphs[i:i + chunk_size])
        if len(block) >= min_block_chars:
            blocks.append(block)
    return blocks if blocks else [text]


# ─────────────────────────────────────────────────────────────
# Character presence detector
# ─────────────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    return re.sub(r"[^\w\s]", "", name.lower()).strip()


def detect_present_characters(
    block_text: str,
    registered_characters: dict[str, str],
) -> set[str]:

    present: set[str] = set()
    block_lower = block_text.lower()

    char_aliases: dict[str, list[str]] = {}
    for char_id, display_name in registered_characters.items():
        aliases = [
            _normalise(char_id),
            _normalise(display_name),
        ]
        first_name = display_name.split()[0].lower()
        if len(first_name) > 2:
            aliases.append(first_name)

        parts = display_name.split()
        if len(parts) > 1:
            aliases.append(parts[-1].lower())

        char_aliases[char_id] = list(set(a for a in aliases if len(a) > 1))

    for m in _ENTER_PATTERN.finditer(block_text):
        names_in_entry = m.group(1).lower()
        for char_id, aliases in char_aliases.items():
            for alias in aliases:
                if alias in names_in_entry:
                    present.add(char_id)
                    break

    for m in _DIALOGUE_PATTERN.finditer(block_text):
        speaker_raw = m.group(1).lower().strip()
        for char_id, aliases in char_aliases.items():
            for alias in aliases:
                if alias in speaker_raw or speaker_raw in alias:
                    present.add(char_id)
                    break

    for m in _EXIT_PATTERN.finditer(block_text):
        names_in_exit = m.group(1).lower() if m.group(1) else ""
        if names_in_exit and "exeunt" not in names_in_exit:
            for char_id, aliases in char_aliases.items():
                for alias in aliases:
                    if alias in names_in_exit:
                        present.add(char_id)
                        break

    for char_id, aliases in char_aliases.items():
        if char_id not in present:
            mention_count = sum(
                block_lower.count(alias)
                for alias in aliases
                if len(alias) > 2
            )
            if mention_count >= 2:
                present.add(char_id)

    return present


# ─────────────────────────────────────────────────────────────
# Main parser entry point
# ─────────────────────────────────────────────────────────────

def parse_episode_for_characters(
    raw_text: str,
    show_id: str,
    episode_number: int,
    registered_characters: dict[str, str],
) -> dict[str, CharacterScript]:

    logger.info(
        "parse_episode_start",
        show_id=show_id,
        episode=episode_number,
        chars=list(registered_characters.keys()),
        raw_chars=len(raw_text),
    )

    raw_blocks = split_into_scene_blocks(raw_text)
    logger.info("blocks_split", count=len(raw_blocks))

    scene_blocks: list[SceneBlock] = []
    for i, block_text in enumerate(raw_blocks):
        present = detect_present_characters(block_text, registered_characters)
        scene_blocks.append(SceneBlock(
            block_index=i,
            raw_text=block_text,
            present_characters=present,
        ))

    character_scripts: dict[str, CharacterScript] = {}
    for char_id in registered_characters:
        char_blocks = [b for b in scene_blocks if char_id in b.present_characters]
        if char_blocks:
            character_scripts[char_id] = CharacterScript(
                character_id=char_id,
                episode_number=episode_number,
                show_id=show_id,
                blocks=char_blocks,
            )

    return character_scripts