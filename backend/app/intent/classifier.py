"""
intent/classifier.py
═══════════════════════════════════════════════════════════════
Intent classification for user messages.

Two-stage pipeline:
  Stage 1 — Rule-based fast path (keyword + regex matching).
             Handles ~80% of messages with zero API cost.
  Stage 2 — Cohere LLM fallback for ambiguous messages.

Intent taxonomy (10 intents):
  ┌──────────────────────┬────────────────────────────────────┐
  │ Intent               │ Description                        │
  ├──────────────────────┼────────────────────────────────────┤
  │ plot_question        │ "What happened in ep 3?"           │
  │ character_backstory  │ "Tell me about your past"          │
  │ relationship_query   │ "Do you trust Vikram?"             │
  │ emotional_support    │ "Are you scared?"                  │
  │ lore_request         │ "Tell me a secret"                 │
  │ scene_continuation   │ "What happened next?" / roleplay   │
  │ out_of_character     │ "Are you an AI?" / real-world Q    │
  │ greeting             │ "Hello" / "Hey"                    │
  │ farewell             │ "Bye" / "See you"                  │
  │ general              │ Everything else                    │
  └──────────────────────┴────────────────────────────────────┘
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Intent enum
# ─────────────────────────────────────────────────────────────

class Intent(str, Enum):
    PLOT_QUESTION       = "plot_question"
    CHARACTER_BACKSTORY = "character_backstory"
    RELATIONSHIP_QUERY  = "relationship_query"
    EMOTIONAL_SUPPORT   = "emotional_support"
    LORE_REQUEST        = "lore_request"
    SCENE_CONTINUATION  = "scene_continuation"
    OUT_OF_CHARACTER    = "out_of_character"
    GREETING            = "greeting"
    FAREWELL            = "farewell"
    GENERAL             = "general"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float     # 0.0 – 1.0
    method: str           # "rules" | "llm" | "fallback" | "disabled"
    raw_label: str        # original string before enum cast


# ─────────────────────────────────────────────────────────────
# Rule patterns (Stage 1 — zero API cost)
# ─────────────────────────────────────────────────────────────

_RULES: list[tuple[Intent, list[str]]] = [
    (Intent.GREETING, [
        r"^(hi|hey|hello|namaste|hii+|good (morning|evening|afternoon))[!.,\s]*$",
        r"^(what'?s up|yo|sup)[!.,\s]*$",
    ]),
    (Intent.FAREWELL, [
        r"^(bye|goodbye|see you|see ya|cya|take care|good ?night)[!.,\s]*$",
        r"(have to go|gotta go|talk later|catch you later)",
    ]),
    (Intent.OUT_OF_CHARACTER, [
        r"\b(are you an? (ai|bot|robot|language model|llm|gpt|claude|cohere|llama))\b",
        r"\b(who (made|created|built|trained) you)\b",
        r"\b(real world|in real life|actually|outside the (show|drama|series))\b",
        r"\b(what is your model|anthropic|openai|meta|ollama|cohere)\b",
    ]),
    (Intent.LORE_REQUEST, [
        r"\b(tell me (a )?secret|behind the scenes|what (nobody|no one) knows)\b",
        r"\b(lore|backstory detail|hidden truth|what really happened)\b",
        r"\b(exclusive|insider|confidential)\b",
    ]),
    (Intent.CHARACTER_BACKSTORY, [
        r"\b(tell me about your(self| past| history| childhood| life))\b",
        r"\b(where (are you|did you come) from|how did you (become|end up|start))\b",
        r"\b(your (family|parents|mother|father|childhood|memories|past))\b",
        r"\b(who (were|are) you before)\b",
    ]),
    (Intent.RELATIONSHIP_QUERY, [
        r"\b(do you (trust|like|love|hate|fear|know))\b",
        r"\b(what do you think (of|about))\b",
        r"\b(your (relationship|feelings|opinion) (with|about|towards|on))\b",
        r"\b(are you and .+ (friends|enemies|allies|close))\b",
    ]),
    (Intent.EMOTIONAL_SUPPORT, [
        r"\b(are you (okay|alright|scared|afraid|sad|happy|angry|upset|lonely))\b",
        r"\b(how (do you feel|are you feeling|are you doing))\b",
        r"\b(what (are you feeling|do you (feel|fear|want|need)))\b",
        r"\b(does it (hurt|scare|bother|worry) you)\b",
    ]),
    (Intent.PLOT_QUESTION, [
        r"\b(what happened (in|during|after|before)|what (happened|occurs?) (next|then))\b",
        r"\b(episode \d+|ep\.?\s*\d+)\b",
        r"\b(why did .+ happen|how did .+ (happen|end|start))\b",
        r"\b(what (is|was) the (plot|story|storyline|scene))\b",
    ]),
    (Intent.SCENE_CONTINUATION, [
        r"\b(what (happens|happened) next|continue (the scene|the story))\b",
        r"\b(keep going|go on|and then|what (do|did) you do next)\b",
        r"\b(let'?s (roleplay|continue|keep|pretend))\b",
        r"\b(act as if|imagine (that|you are|if))\b",
    ]),
]

_compiled_rules: list[tuple[Intent, list[re.Pattern]]] = [
    (intent, [re.compile(pat, re.IGNORECASE) for pat in patterns])
    for intent, patterns in _RULES
]


# ─────────────────────────────────────────────────────────────
# Stage 1: Rule-based classifier
# ─────────────────────────────────────────────────────────────

def _classify_by_rules(text: str) -> IntentResult | None:
    """Return an IntentResult if any rule matches, else None."""
    text_stripped = text.strip()
    for intent, patterns in _compiled_rules:
        for pattern in patterns:
            if pattern.search(text_stripped):
                confidence = 1.0 if intent in (Intent.GREETING, Intent.FAREWELL) else 0.85
                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    method="rules",
                    raw_label=intent.value,
                )
    return None


# ─────────────────────────────────────────────────────────────
# Stage 2: Cohere LLM classifier
# ─────────────────────────────────────────────────────────────

_LLM_CLASSIFY_PROMPT = """\
Classify the following user message into exactly ONE of these intent categories:
  plot_question, character_backstory, relationship_query, emotional_support,
  lore_request, scene_continuation, out_of_character, greeting, farewell, general

User message: "{message}"

Rules:
- Respond with ONLY the category name, nothing else.
- No explanation, no punctuation, no quotes.
- If unsure, respond: general

Category:"""


def _classify_by_cohere(text: str) -> IntentResult:
    """
    Call the Cohere API to classify intent.

    Uses temperature=0.0 and max_tokens=10 — we only need
    one word back, so this call is extremely cheap and fast.
    """
    try:
        import cohere
        from app.core.config import get_settings
        settings = get_settings()

        client = cohere.Client(
            api_key=settings.cohere_api_key,
            timeout=settings.cohere_timeout,
        )

        response = client.chat(
            model=settings.cohere_model,
            message=_LLM_CLASSIFY_PROMPT.format(message=text[:500]),
            temperature=0.0,    # Deterministic — classification, not generation
            max_tokens=10,      # We only need one label word
        )

        raw = (
            response.text
            .strip()
            .lower()
            .replace(".", "")
            .replace('"', "")
            .replace("'", "")
            .split()[0]         # Take only the first word — guard against verbose output
        )

        try:
            intent = Intent(raw)
            return IntentResult(
                intent=intent,
                confidence=0.75,
                method="llm",
                raw_label=raw,
            )
        except ValueError:
            logger.warning("cohere_intent_unknown_label", raw=raw)
            return IntentResult(
                intent=Intent.GENERAL,
                confidence=0.5,
                method="llm",
                raw_label=raw,
            )

    except Exception as e:
        logger.warning("cohere_intent_failed", error=str(e))
        return IntentResult(
            intent=Intent.GENERAL,
            confidence=0.3,
            method="fallback",
            raw_label="general",
        )


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def classify_intent(text: str, use_llm_fallback: bool = True) -> IntentResult:
    """
    Classify the intent of a user message.

    Stage 1: Fast regex rules (zero API cost).
    Stage 2: Cohere LLM fallback for ambiguous messages.

    Args:
        text: Raw user message.
        use_llm_fallback: Call Cohere if rules don't match.

    Returns:
        IntentResult with intent, confidence, and method used.
    """
    from app.core.config import get_settings
    settings = get_settings()

    if not settings.intent_enabled:
        return IntentResult(
            intent=Intent.GENERAL,
            confidence=1.0,
            method="disabled",
            raw_label="general",
        )

    # Stage 1: Rules — handles ~80% of messages, zero cost
    rule_result = _classify_by_rules(text)
    if rule_result:
        logger.debug(
            "intent_by_rules",
            intent=rule_result.intent,
            preview=text[:60],
        )
        return rule_result

    # Stage 2: Cohere LLM fallback
    if use_llm_fallback:
        llm_result = _classify_by_cohere(text)
        logger.debug(
            "intent_by_cohere",
            intent=llm_result.intent,
            preview=text[:60],
        )
        return llm_result

    return IntentResult(
        intent=Intent.GENERAL,
        confidence=0.5,
        method="fallback",
        raw_label="general",
    )


# ─────────────────────────────────────────────────────────────
# Intent → response tone hints (unchanged — pure text)
# ─────────────────────────────────────────────────────────────

INTENT_TONE_HINTS: dict[Intent, str] = {
    Intent.PLOT_QUESTION: (
        "The user is asking about story events. Ground your answer in specific "
        "scene details from the script context. Be precise but stay in character."
    ),
    Intent.CHARACTER_BACKSTORY: (
        "The user wants to know your personal history. Share selectively — "
        "reveal enough to be compelling, hold back enough to stay mysterious."
    ),
    Intent.RELATIONSHIP_QUERY: (
        "The user is asking about your feelings or relationships with other characters. "
        "Be emotionally honest but guarded — these are complicated topics for you."
    ),
    Intent.EMOTIONAL_SUPPORT: (
        "The user is checking on your emotional state. Be vulnerable but in character — "
        "show the emotion your character would show in this moment of the story."
    ),
    Intent.LORE_REQUEST: (
        "The user wants exclusive behind-the-scenes insight. You may hint at hidden "
        "truths that a viewer wouldn't know — but frame them as personal confessions, "
        "not as an omniscient narrator."
    ),
    Intent.SCENE_CONTINUATION: (
        "The user wants to continue or re-enact a scene. Play along dramatically — "
        "describe action, emotion, and dialogue as if the scene is happening right now."
    ),
    Intent.OUT_OF_CHARACTER: (
        "The user is trying to break the fourth wall. Stay firmly in character. "
        "If asked if you are an AI, deflect with confusion or in-world logic. "
        "Never acknowledge being a language model or the Cohere API."
    ),
    Intent.GREETING: (
        "The user is greeting you. Respond in character with an appropriate greeting "
        "that reflects the current emotional state of your character and the story."
    ),
    Intent.FAREWELL: (
        "The user is saying goodbye. Give a warm, in-character farewell that leaves "
        "them wanting to return. Hint at unfinished business or unresolved feelings."
    ),
    Intent.GENERAL: (
        "Respond naturally and in character. Match the emotional register of the user's message."
    ),
}


def get_tone_hint(intent: Intent) -> str:
    return INTENT_TONE_HINTS.get(intent, INTENT_TONE_HINTS[Intent.GENERAL])
