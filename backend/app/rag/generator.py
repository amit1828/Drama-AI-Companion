"""
rag/generator.py
═══════════════════════════════════════════════════════════════
LLM generation layer using the Cohere API (command-r-plus).

Sends the assembled system prompt + message history to Cohere's
Chat endpoint and returns the character's in-character reply.

Cohere Chat API concepts used here:
  - preamble      : the system-level persona / instruction block
  - chat_history  : prior turns as List[{"role", "message"}]
  - message       : the latest user turn

Features:
  - Singleton Cohere client (one SDK instance for the process)
  - Retry with exponential back-off via tenacity
  - Structured logging of token usage and latency
  - Graceful in-character fallbacks on any hard error
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import time
from functools import lru_cache

import cohere
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Cohere client — one instance for the whole process lifetime
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_client() -> cohere.Client:
    settings = get_settings()
    if not settings.cohere_api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Add it to your .env file."
        )
    return cohere.Client(
        api_key=settings.cohere_api_key,
        timeout=settings.cohere_timeout,
    )


# ─────────────────────────────────────────────────────────────
# In-character fallback messages (returned on hard error)
# ─────────────────────────────────────────────────────────────

_FALLBACK_RESPONSES = [
    "Something… pulled me away for a moment. What were you saying?",
    "I feel strange all of a sudden. Ask me again — I lost the thread.",
    "The world flickered. Say that once more.",
    "Forgive me — I was somewhere else. Repeat that for me?",
]
_fallback_idx = 0


def _next_fallback() -> str:
    global _fallback_idx
    msg = _FALLBACK_RESPONSES[_fallback_idx % len(_FALLBACK_RESPONSES)]
    _fallback_idx += 1
    return msg


# ─────────────────────────────────────────────────────────────
# Message format converter
# ─────────────────────────────────────────────────────────────

def _to_cohere_history(messages: list[dict]) -> tuple[list[dict], str]:
    """
    Split the messages list into:
      - chat_history : all turns except the final user message
      - last_message : the latest user message string

    Cohere Chat expects:
      chat_history = [{"role": "USER"|"CHATBOT", "message": "..."}]
      message      = "current user input"

    Role mapping:
      "user"      -> "USER"
      "assistant" -> "CHATBOT"
    """
    role_map = {"user": "USER", "assistant": "CHATBOT"}

    if not messages:
        return [], ""

    # The last element must be the user's current message
    last = messages[-1]
    history_raw = messages[:-1]

    chat_history = [
        {
            "role": role_map.get(m["role"], "USER"),
            "message": m["content"],
        }
        for m in history_raw
    ]

    return chat_history, last.get("content", "")


# ─────────────────────────────────────────────────────────────
# Core Cohere call (with retry)
# ─────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((cohere.errors.TooManyRequestsError,)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
def _call_cohere(system_prompt: str, messages: list[dict]) -> str:
    """
    Call the Cohere Chat API.

    Args:
        system_prompt : Full character persona + RAG + memory block.
                        Maps to Cohere's `preamble` parameter.
        messages      : Full message list ending with the user's latest turn.

    Returns:
        The character's reply as a plain string.
    """
    settings = get_settings()
    client = _get_client()

    chat_history, user_message = _to_cohere_history(messages)

    t0 = time.monotonic()

    response = client.chat(
        model=settings.cohere_model,
        preamble=system_prompt,          # System-level instructions
        chat_history=chat_history,       # Prior conversation turns
        message=user_message,            # Current user input
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        p=0.92,                          # top-p nucleus sampling
    )

    elapsed = round(time.monotonic() - t0, 2)

    # Log token usage if Cohere returns it
    usage = getattr(response, "meta", None)
    if usage and hasattr(usage, "tokens"):
        tokens = usage.tokens
        logger.info(
            "cohere_call_complete",
            model=settings.cohere_model,
            input_tokens=getattr(tokens, "input_tokens", "?"),
            output_tokens=getattr(tokens, "output_tokens", "?"),
            elapsed_s=elapsed,
        )
    else:
        logger.info(
            "cohere_call_complete",
            model=settings.cohere_model,
            elapsed_s=elapsed,
        )

    return response.text


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def generate_reply(
    system_prompt: str,
    messages: list[dict],
) -> tuple[str, bool]:
    """
    Generate a character reply via Cohere command-r-plus.

    Args:
        system_prompt : Assembled system block (persona + memory + RAG + intent).
        messages      : Full conversation list ending with the user's latest turn.

    Returns:
        (reply_text, is_fallback)
        is_fallback=True means an error occurred and a safe
        in-character fallback was returned instead.
    """
    try:
        reply = _call_cohere(system_prompt=system_prompt, messages=messages)
        reply = _clean_reply(reply)
        return reply.strip(), False

    except cohere.errors.UnauthorizedError:
        logger.error("cohere_auth_error", hint="Check COHERE_API_KEY in .env")
        return "I cannot speak right now. The connection to my world is broken.", True

    except cohere.errors.TooManyRequestsError:
        logger.error("cohere_rate_limit")
        return _next_fallback(), True

    except Exception as e:
        logger.error("cohere_generation_failed", error=str(e))
        return _next_fallback(), True


def _clean_reply(text: str) -> str:
    """
    Remove any accidental role-prefix artefacts that Cohere
    occasionally prepends, and ensure the reply ends cleanly.
    """
    import re
    # Strip leading role prefixes like "Assistant: " or "Character: "
    text = re.sub(
        r"^(assistant|character|chatbot|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.lstrip("\n ")

    # If the reply ends mid-sentence (no terminal punctuation),
    # drop the trailing fragment to avoid abrupt cut-offs.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences and not text.rstrip().endswith((".", "!", "?", "…")):
        if len(sentences) > 1:
            text = " ".join(sentences[:-1])

    return text.strip()
