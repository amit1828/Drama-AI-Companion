"""
memory/summariser.py
═══════════════════════════════════════════════════════════════
Compresses older conversation turns into a concise long-term
memory summary using the Cohere API.

This keeps the context window manageable while ensuring the
character "remembers" earlier conversation topics.

Called automatically by ChatService when short_term exceeds
memory_summary_threshold.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.memory.memory_store import ConversationMemory, Turn, trim_short_term, save_memory

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────

_SUMMARISE_PROMPT = """\
You are a memory assistant for a drama character AI companion.
Below is a portion of a conversation between a user and a drama character.
Write a concise 3-5 sentence summary capturing:
- Key topics the user asked about
- Any personal details the user revealed about themselves
- Emotional tone and any important revelations
- What the character disclosed or hinted at

Keep it in third-person, factual, and under 150 words.
Do not include any preamble — output only the summary text itself.

CONVERSATION TO SUMMARISE:
{conversation_text}

SUMMARY:"""


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _turns_to_text(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        label = "User" if t.role == "user" else "Character"
        lines.append(f"{label}: {t.content}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def summarise_and_compress(mem: ConversationMemory) -> bool:
    """
    Check if short_term needs compression. If so:
      1. Pop the oldest turns from short_term.
      2. Summarise them via Cohere.
      3. Merge with existing long_term_summary.
      4. Save updated memory.

    Returns True if summarisation occurred, False if not needed.
    """
    from app.core.config import get_settings
    settings = get_settings()

    if not settings.memory_enabled:
        return False

    to_summarise = trim_short_term(mem, max_turns=settings.memory_max_turns)
    if not to_summarise:
        return False

    logger.info(
        "summarising_turns",
        session_id=mem.session_id,
        turns_to_summarise=len(to_summarise),
    )

    conversation_text = _turns_to_text(to_summarise)

    # Prepend existing summary so we don't lose prior history
    if mem.long_term_summary:
        conversation_text = (
            f"[Previous summary: {mem.long_term_summary}]\n\n"
            + conversation_text
        )

    try:
        new_summary = _call_cohere_summariser(conversation_text)
        mem.long_term_summary = new_summary
        save_memory(mem)
        logger.info(
            "summarisation_complete",
            session_id=mem.session_id,
            summary_len=len(new_summary),
        )
        return True

    except Exception as e:
        logger.error("summarisation_failed", session_id=mem.session_id, error=str(e))
        # Graceful fallback — keep a plain excerpt rather than losing history
        mem.long_term_summary = (
            f"[Earlier topics discussed: {conversation_text[:400]}...]"
        )
        save_memory(mem)
        return False


def _call_cohere_summariser(conversation_text: str) -> str:
    """
    Call the Cohere Chat API to produce a factual memory summary.

    Uses a low temperature (0.2) for deterministic, factual output.
    Sends a single user turn — no chat history needed for summarisation.
    """
    import cohere
    from app.core.config import get_settings
    settings = get_settings()

    client = cohere.Client(
        api_key=settings.cohere_api_key,
        timeout=settings.cohere_timeout,
    )

    prompt = _SUMMARISE_PROMPT.format(conversation_text=conversation_text)

    response = client.chat(
        model=settings.cohere_model,
        message=prompt,
        temperature=0.2,      # Low temperature — factual summary, not creative
        max_tokens=250,       # Summaries should be short
    )

    return response.text.strip()
