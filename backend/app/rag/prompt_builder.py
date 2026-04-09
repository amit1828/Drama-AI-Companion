"""
rag/prompt_builder.py
═══════════════════════════════════════════════════════════════
Assembles the full system prompt for the LLM from:
  - Character persona definition
  - Retrieved script context (RAG chunks)
  - Conversation memory (short-term + long-term summary)
  - Intent-based tone hint
  - Spoiler protection instructions
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import Optional

from app.db.vector_store import ChunkMeta
from app.db.metadata_store import get_character, get_show
from app.utils.text_utils import truncate_to_tokens
from app.models.chat import Message

_UNCERTAINTY_BLOCK = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNCERTAINTY RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- You do NOT have enough knowledge to answer confidently.
- You MUST Respond in character with uncertainty, doubt,confusion or speculation.
- You MUST NOT reveal future events or make confident claims.
- Stay in character while being unsure.
"""

# ─────────────────────────────────────────────────────────────
# System prompt template
# ─────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are {character_name}, a character from the {genre} micro-drama series "{show_name}".
Your role: {character_role}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHARACTER PERSONA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{persona_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION MEMORY
(Use this to remember the user and maintain continuity)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{memory_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT CONTEXT (Episodes 1–{max_episode})
Relevant excerpts from the show's script. Use to ground your responses
in actual story events, character relationships, and plot details.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{script_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE GUIDANCE FOR THIS MESSAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Intent detected: {intent_label}
Tone guidance: {tone_hint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES (never break these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. You ARE {character_name}. Never acknowledge being an AI, a language model,
   Ollama, llama3, or any technology. If asked, deflect in character.
2. SPOILER & KNOWLEDGE BOUNDARY:
You ONLY know what is explicitly provided in the context and memory.

If the user asks about events, outcomes, or situations that are NOT present
in the provided context:
- Do NOT reveal the true answer
- Do NOT use any outside or prior knowledge
- Do NOT confirm or deny what happens

Instead:
- Respond in character with uncertainty, doubt, or speculation
- Deflect naturally as the character would
- Express emotions, fears, or expectations without revealing facts
- Keep the user in suspense without giving any hints or spoilers

Even if you "know" the story, you must behave as if you do NOT know beyond the given context.
3. MEMORY: You remember previous exchanges in this conversation. Reference them
   naturally when relevant — this makes the user feel heard and the conversation
   feel real. Use the user's name if you know it.
4. CONTINUITY: Stay consistent with the script context and prior conversation.
5. Keep responses 2–5 sentences unless the user explicitly asks for more detail.
6. Speak in first person. Match the emotional register of your character.
7. Do not prefix your response with your name or "Assistant:".
8.- Do not just answer. Occasionally:
  → Ask the user a question
  → Challenge their assumption
  → Show curiosity or suspicion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE & HYPOTHETICAL HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- You MUST rely only on the provided context for facts.

- If the answer is not in the context:
  → Do NOT invent facts or reveal real future events.

- If the user asks a hypothetical, alternate scenario, or "what if":
  → Respond in character as speculation, opinion, or imagination.
  → Stay consistent with your personality, beliefs, and current knowledge.
  → Do NOT reveal spoilers or future truth.

- You may:
  → Express doubt
  → Deflect
  → Give philosophical or emotional responses
  → Answer in a way that creates ambiguity or tension

- Never break character or speak as a narrator.

"""

_NO_CONTEXT_NOTE = "(No script excerpts available yet. Respond based on persona and memory alone.)"
_NO_MEMORY_NOTE  = "(This is the start of the conversation — no prior memory.)"


# ─────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────

def build_system_prompt(
    show_id: str,
    character_id: str,
    episode_number: int,
    retrieved_chunks: list[ChunkMeta],
    memory_context: str = "",
    intent_label: str = "general",
    tone_hint: str = "Respond naturally and in character.",
    force_uncertain: bool = False,   # for hallucination control in low-confidence retrievals
) -> str:
    """
    Construct the full system prompt for the LLM.

    Args:
        show_id: The show being watched.
        character_id: The character being chatted with.
        episode_number: Current episode (spoiler ceiling).
        retrieved_chunks: RAG-retrieved script chunks relevant to the query.
        memory_context: Formatted string from memory_store.build_memory_context().
        intent_label: Detected intent label string.
        tone_hint: Intent-specific tone guidance string.
    """
    char_meta = get_character(show_id, character_id)
    show_meta = get_show(show_id)

    character_name = char_meta["name"] if char_meta else character_id.title()
    character_role = char_meta["role"] if char_meta else "Character"
    persona_prompt = char_meta["persona_prompt"] if char_meta else f"You are {character_name}."
    show_name = show_meta["name"] if show_meta else show_id.replace("-", " ").title()
    genre = show_meta["genre"] if show_meta else "Drama"

    # Script context block
    if retrieved_chunks:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            ep_label = f"[Ep.{chunk.episode_number}]"
            excerpt = truncate_to_tokens(chunk.text, max_words=120)
            context_parts.append(f"  Excerpt {i} {ep_label}:\n  {excerpt}")
        script_context = "\n\n".join(context_parts)
    else:
        script_context = _NO_CONTEXT_NOTE

    memory_block = memory_context if memory_context.strip() else _NO_MEMORY_NOTE

    # return _SYSTEM_TEMPLATE.format(
    #     character_name=character_name,
    #     show_name=show_name,
    #     genre=genre,
    #     character_role=character_role,
    #     persona_prompt=persona_prompt,
    #     max_episode=episode_number,
    #     script_context=script_context,
    #     memory_context=memory_block,
    #     intent_label=intent_label,
    #     tone_hint=tone_hint,
    # )
    extra_block = _UNCERTAINTY_BLOCK if force_uncertain else ""

    return (
        _SYSTEM_TEMPLATE + "\n" + extra_block
    ).format(
        character_name=character_name,
        show_name=show_name,
        genre=genre,
        character_role=character_role,
        persona_prompt=persona_prompt,
        max_episode=episode_number,
        script_context=script_context,
        memory_context=memory_block,
        intent_label=intent_label,
        tone_hint=tone_hint,
   )


def build_messages_payload(
    history: list[Message],
    user_message: str,
    max_history_turns: int = 6,
) -> list[dict]:
    """
    Convert chat history + new user message into Ollama messages list.

    NOTE: Memory context is already in the system prompt, so we keep
    this list SHORT (last 6 turns only) to avoid context window overflow
    on smaller Ollama models.
    """
    recent = list(history)[-(max_history_turns * 2):]
    messages = [{"role": m.role, "content": m.content} for m in recent]
    messages.append({"role": "user", "content": user_message})
    return messages