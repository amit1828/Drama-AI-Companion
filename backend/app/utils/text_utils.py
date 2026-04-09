"""
utils/text_utils.py
Text cleaning, normalisation, and chunking utilities shared across the RAG pipeline.
"""

from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    word_count: int


# ─────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────

def clean_script_text(raw: str) -> str:
    """
    Normalise raw script text:
    - Unicode NFC normalisation
    - Remove zero-width chars and BOM
    - Collapse multiple blank lines to single blank line
    - Strip trailing whitespace per line
    """
    text = unicodedata.normalize("NFC", raw)
    # Remove BOM and zero-width characters
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "")
    # Normalise Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip per-line trailing whitespace
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ blank lines → 2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_dialogue(text: str) -> str:
    """
    Extract only dialogue lines from a screenplay-style script.
    Handles common formats:
      CHARACTER NAME
      Dialogue text here.

      CHARACTER NAME (V.O.)
      Dialogue text.
    """
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Detect all-caps character cue (optionally followed by parenthetical)
        if re.match(r"^[A-Z][A-Z\s\-'\.]+(\s*\(.*\))?$", line) and len(line) < 60:
            character = line
            # Collect dialogue lines that follow
            dialogue_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                r"^[A-Z][A-Z\s\-'\.]+(\s*\(.*\))?$", lines[i].strip()
            ):
                dialogue_lines.append(lines[i].strip())
                i += 1
            if dialogue_lines:
                result.append(f"{character}: {' '.join(dialogue_lines)}")
            continue
        i += 1
    return "\n".join(result)


# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[TextChunk]:
    """
    Split text into overlapping word-based chunks.

    Strategy:
    1. Prefer splitting at paragraph boundaries.
    2. Fall back to sentence boundaries.
    3. Fall back to word boundaries.

    Args:
        text: Cleaned script text.
        chunk_size: Target words per chunk.
        overlap: Number of words to overlap between adjacent chunks.

    Returns:
        List of TextChunk objects with positional metadata.
    """
    # First try paragraph-aware splitting
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[TextChunk] = []
    current_words: list[str] = []
    current_start_char = 0
    char_pos = 0
    chunk_index = 0

    def flush_chunk(words: list[str], start: int, end: int) -> TextChunk:
        nonlocal chunk_index
        c = TextChunk(
            text=" ".join(words),
            chunk_index=chunk_index,
            char_start=start,
            char_end=end,
            word_count=len(words),
        )
        chunk_index += 1
        return c

    for para in paragraphs:
        para_words = para.split()
        para_char_start = text.find(para, char_pos)
        if para_char_start == -1:
            para_char_start = char_pos
        para_char_end = para_char_start + len(para)
        char_pos = para_char_end

        current_words.extend(para_words)

        if len(current_words) >= chunk_size:
            # Emit chunk
            chunk_char_end = para_char_end
            chunks.append(flush_chunk(current_words[:chunk_size], current_start_char, chunk_char_end))
            # Keep overlap
            current_words = current_words[chunk_size - overlap:]
            current_start_char = chunk_char_end - (overlap * 5)  # approximate

    # Flush remainder
    if current_words:
        chunks.append(flush_chunk(current_words, current_start_char, len(text)))

    return chunks


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def truncate_to_tokens(text: str, max_words: int = 300) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def format_history_for_prompt(history: list[dict], max_turns: int = 6) -> str:
    """Convert message history list to a readable string for prompt injection."""
    recent = history[-max_turns * 2:]
    lines = []
    for msg in recent:
        role_label = "Fan" if msg["role"] == "user" else "Character"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)