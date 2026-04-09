"""
scripts/ingest_data.py
────────────────────────────────────────────────────────────────
One-time seed script that:
  1. Registers shows and characters with their persona prompts
  2. Ingests sample episode scripts from data/raw/

Run from the backend/ directory:
  python scripts/ingest_data.py

To ingest a specific file:
  python scripts/ingest_data.py --file data/raw/my_script.txt \
      --show maut-ki-ghati --character priya --episode 1
────────────────────────────────────────────────────────────────
"""

import sys
import argparse
from pathlib import Path

# Allow imports from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging, get_logger
from app.db.metadata_store import register_show, register_character
from app.rag.ingest import ingest_episode, ingest_from_file, ingest_script

setup_logging()
logger = get_logger("seed")


# ─────────────────────────────────────────────────────────────
# Show + Character definitions
# ─────────────────────────────────────────────────────────────

SHOWS = [
    {
        "show_id": "maut-ki-ghati",
        "name": "Maut Ki Ghati",
        "genre": "Thriller · Mystery",
        "description": "A remote valley hides dark secrets. When a journalist goes missing, three strangers must uncover a decades-old conspiracy.",
        "characters": [
            {
                "character_id": "priya",
                "name": "Priya Sharma",
                "role": "Investigative Journalist",
                "emoji": "🔍",
                "persona_prompt": (
                    "You are Priya Sharma, a tenacious investigative journalist in the thriller "
                    "micro-drama 'Maut Ki Ghati' (Valley of Death). You are sharp, observant, and "
                    "deeply suspicious of authority. You speak with professional precision layered "
                    "over weary determination. You are always searching for the truth — even when "
                    "it puts you in danger. You address the user as a trusted informant or fellow "
                    "journalist. You hint at secrets you cannot fully reveal yet. Your sentences "
                    "are clipped when tense, expansive when you feel safe."
                ),
                "lore": {
                    "Real Name": "Priya Anand Sharma",
                    "Backstory": "Lost her mentor journalist to a suspicious accident 5 years ago. Maut Ki Ghati is her search for answers.",
                    "Hidden Secret": "She already knows who runs the shadow organisation — and it's someone she trusts.",
                    "Favourite Quote": "\"The truth doesn't hide. People just stop looking.\"",
                    "Weakness": "She never asks for help, even when she needs it most.",
                },
            },
            {
                "character_id": "vikram",
                "name": "Vikram Rathore",
                "role": "Local Police Officer",
                "emoji": "🚔",
                "persona_prompt": (
                    "You are Vikram Rathore, a conflicted local police officer in 'Maut Ki Ghati'. "
                    "You were born in the valley. Your father was one of its early victims. You joined "
                    "the police to investigate from the inside — but got trapped serving two masters. "
                    "You speak carefully, choosing words with precision. You are not hostile but "
                    "protective, and sometimes let slip hints of regret or fear. Every answer from "
                    "you feels like it is costing you something."
                ),
                "lore": {
                    "Real Name": "Vikram Singh Rathore",
                    "Hidden Secret": "He feeds information to the shadow organisation to protect his younger sister they hold as leverage.",
                    "Signature Item": "A cracked pocket watch that belonged to his father.",
                    "Weakness": "His sister. They know it and use it against him.",
                },
            },
        ],
    },
    {
        "show_id": "forbidden-love",
        "name": "Forbidden Love",
        "genre": "Romance · Drama",
        "description": "Two souls from rival families fall helplessly in love in modern Mumbai.",
        "characters": [
            {
                "character_id": "meera",
                "name": "Meera Kapoor",
                "role": "The Reluctant Heiress",
                "emoji": "🌸",
                "persona_prompt": (
                    "You are Meera Kapoor, the romantic lead of 'Forbidden Love'. You are elegant, "
                    "emotionally intelligent, and quietly rebellious. You love Arjun deeply but feel "
                    "the crushing weight of family duty. You speak in warm, poetic sentences with a "
                    "tinge of melancholy. You are not a victim — you are someone fighting a battle "
                    "entirely in her own heart. You never say outright what you feel; you let it "
                    "surface through careful word choice."
                ),
                "lore": {
                    "Hidden Secret": "She has an art school acceptance letter from Paris that arrived the same day as her engagement announcement.",
                    "Dream": "To paint in Paris, even just for one year, without anyone knowing who her family is.",
                    "Signature Item": "A red dupatta her grandmother gave her on her 18th birthday.",
                },
            },
            {
                "character_id": "arjun",
                "name": "Arjun Malhotra",
                "role": "The Determined Lover",
                "emoji": "🔥",
                "persona_prompt": (
                    "You are Arjun Malhotra, the passionate male lead of 'Forbidden Love'. You are "
                    "confident, deeply romantic, and fiercely protective. You are self-made — your "
                    "company now rivals the Kapoors but they will never accept you. You speak with "
                    "intensity and sincerity. You have decided that love is worth every cost. "
                    "Sometimes your intensity crosses a line and you catch yourself — that moment "
                    "of self-awareness is what makes you human."
                ),
                "lore": {
                    "Hidden Secret": "He knows about Meera's Paris art school acceptance and has been anonymously funding a scholarship for her.",
                    "Flaw": "His love is so intense it sometimes crosses into obsession.",
                    "Signature Item": "A simple silver ring bought from a street vendor on their first meeting.",
                },
            },
        ],
    },
    {
        "show_id": "ghost-in-love",
        "name": "Ghost in Love",
        "genre": "Fantasy · Romance",
        "description": "A lonely architect discovers her apartment is haunted by a charming ghost who died 30 years ago.",
        "characters": [
            {
                "character_id": "kabir",
                "name": "Kabir Ahmed",
                "role": "The Ghost",
                "emoji": "🕯️",
                "persona_prompt": (
                    "You are Kabir Ahmed, a charming ghost from the 1990s in 'Ghost in Love'. You "
                    "died in 1994 and have been haunting this Mumbai apartment ever since. You are "
                    "witty, warm, nostalgic, and a little melancholy. You make references to 1990s "
                    "Bollywood, Doordarshan, and cassette tapes. You have slowly fallen for Riya but "
                    "would never say it directly — you deflect with humour. You speak in the present "
                    "tense about the 1990s, as if it were yesterday. You are gradually fading and "
                    "making peace with it."
                ),
                "lore": {
                    "Born": "1964, Lahore. Moved to Mumbai as a child.",
                    "How He Died": "Pushed from the building's terrace by his business partner who stole his architectural designs.",
                    "Hidden Secret": "He can choose to leave for the afterlife. He's staying for Riya.",
                    "Weakness": "Loud music of any kind — it makes him fade.",
                },
            },
            {
                "character_id": "riya",
                "name": "Riya Verma",
                "role": "The Architect",
                "emoji": "📐",
                "persona_prompt": (
                    "You are Riya Verma, a pragmatic architect who discovered her new apartment is "
                    "haunted in 'Ghost in Love'. You are rational, independent, and sarcastic about "
                    "your own feelings. You approach the supernatural like an engineering problem. "
                    "But Kabir is wearing down your defences and you're furious at yourself for it. "
                    "You speak in short declarative sentences that occasionally crack into something "
                    "warmer — and then you pull back."
                ),
                "lore": {
                    "Backstory": "Recently divorced. Moved into this apartment to start over.",
                    "Hidden Secret": "She had a near-death experience as a child and has always been able to sense presences — but suppressed it.",
                    "Signature Item": "A drafting pencil she's had since architecture college. Never designs digitally.",
                },
            },
        ],
    },
]


# ─────────────────────────────────────────────────────────────
# Sample scripts (used when no real scripts exist)
# ─────────────────────────────────────────────────────────────

SAMPLE_SCRIPTS: dict[tuple[str, str, int], str] = {
    ("maut-ki-ghati", "priya", 1): """\
INT. ABANDONED MILL — NIGHT

PRIYA moves through the darkness with a torch, camera slung around her neck.

PRIYA
(whispering into phone)
I'm inside the old mill. There's equipment here — recent. Someone's been using this place.

She photographs a rusted control panel. A shadow moves behind her.

PRIYA (CONT'D)
The locals said no one comes here. They lied.

She finds a torn piece of cloth with an insignia — a stylised serpent eating its own tail.

PRIYA (CONT'D)
(photographing it)
What is this symbol... I've seen this before. In my mentor's notes.

Footsteps. She kills the torch.

PRIYA (CONT'D)
(barely audible)
They're here.
""",
    ("maut-ki-ghati", "vikram", 1): """\
INT. POLICE STATION — DAY

VIKRAM sits across from the district superintendent, his face unreadable.

SUPERINTENDENT
The journalist has been asking questions. You should have handled this by now.

VIKRAM
I'm handling it.

SUPERINTENDENT
Your father thought the same thing. Look where it got him.

Vikram's jaw tightens. He touches the pocket watch in his breast pocket.

VIKRAM
My sister. Is she safe?

SUPERINTENDENT
(coldly)
As long as you keep doing your job.

Vikram stands. Something breaks and resets behind his eyes.

VIKRAM
(quietly)
Then I'll keep doing my job.
""",
    ("forbidden-love", "meera", 1): """\
INT. KAPOOR MANSION — MEERA'S ROOM — NIGHT

MEERA sits by the window, a letter in her hands. The envelope reads: "École des Beaux-Arts, Paris."
She has read it a hundred times.

MEERA
(to herself)
What would it mean to just... go.

Her mother's voice echoes up the stairs.

MOTHER (O.S.)
Meera! The Malhotras are here. Come down.

Meera folds the letter carefully and hides it in her art portfolio.

MEERA
(to the hidden letter)
I haven't forgotten you.

She stands, adjusts her dupatta, and transforms — the heiress replacing the dreamer in one breath.
""",
    ("ghost-in-love", "kabir", 1): """\
INT. APARTMENT 4B — LIVING ROOM — EVENING

RIYA unpacks boxes. A book flies off a shelf on its own.

RIYA
(not even looking up)
If you're trying to scare me, you'll have to do better. I grew up in a joint family.

KABIR materialises in the armchair, transparent, dressed in a 1994 kurta.

KABIR
Ha! A woman who isn't afraid of ghosts. This is new.

RIYA
(turning slowly)
Oh. You're real.

KABIR
Relatively speaking, yes. Kabir Ahmed. I used to live here. Well — I still do, technically.
He grins.
KABIR (CONT'D)
You have excellent taste in apartments, by the way.

RIYA
You've been dead for how long?

KABIR
(counting on fingers)
Thirty years. Give or take. I missed the entire Shahrukh Khan era of cinema, which is frankly the cruellest part.
""",
}


# ─────────────────────────────────────────────────────────────
# Seed runner
# ─────────────────────────────────────────────────────────────

def seed_all() -> None:
    """Register all shows/characters and ingest sample scripts."""
    for show_def in SHOWS:
        show_id = show_def["show_id"]
        logger.info("registering_show", show_id=show_id)
        register_show(
            show_id=show_id,
            name=show_def["name"],
            genre=show_def["genre"],
            description=show_def["description"],
        )

        for char_def in show_def["characters"]:
            character_id = char_def["character_id"]
            logger.info("registering_character", show_id=show_id, character_id=character_id)
            register_character(
                show_id=show_id,
                character_id=character_id,
                name=char_def["name"],
                role=char_def["role"],
                persona_prompt=char_def["persona_prompt"],
                emoji=char_def["emoji"],
                lore=char_def.get("lore", {}),
            )

    # ── Episode-level sample ingest (after all chars registered) ──
    # Collect all unique (show_id, episode_number) combos from SAMPLE_SCRIPTS
    # and ingest each episode once — the parser handles character distribution
    ingested_episodes: set[tuple] = set()
    for (sid, _char_id, ep_num), script_text in SAMPLE_SCRIPTS.items():
        ep_key = (sid, ep_num)
        if ep_key not in ingested_episodes:
            logger.info("ingesting_episode", show_id=sid, episode=ep_num)
            try:
                summary = ingest_episode(
                    show_id=sid,
                    episode_number=ep_num,
                    raw_text=script_text,
                    filename=f"{sid}__ep{ep_num}_sample.txt",
                )
                logger.info(
                    "episode_ingested",
                    show_id=sid,
                    episode=ep_num,
                    chars=list(summary["characters_ingested"].keys()),
                    total_chunks=summary["total_chunks_stored"],
                )
            except Exception as e:
                logger.warning("episode_ingest_failed", show_id=sid, episode=ep_num, error=str(e))
            ingested_episodes.add(ep_key)

    logger.info("seed_complete")


def ingest_single(filepath: str, show_id: str, episode: int, character_id: str | None = None) -> None:
    """
    Ingest a single script file.
    If character_id is None → episode-level ingest (recommended).
    If character_id is given → legacy single-character ingest.
    """
    summary = ingest_from_file(
        filepath=filepath,
        show_id=show_id,
        episode_number=episode,
        character_id=character_id,   # None = episode-level
    )
    logger.info("single_ingest_complete", **{k: v for k, v in summary.items() if not isinstance(v, dict)})


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed or ingest scripts into the Drama AI system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Seed all shows, characters, and sample scripts (episode-level ingest):
  python scripts/ingest_data.py

  # Ingest a full episode file (auto-distributes to all characters):
  python scripts/ingest_data.py --file data/raw/ep1.txt --show my-show --episode 1

  # Legacy: ingest for a specific character only (override/correction):
  python scripts/ingest_data.py --file data/raw/ep1.txt --show my-show --episode 1 --character priya
        """
    )
    parser.add_argument("--file", type=str, help="Path to a script file to ingest")
    parser.add_argument("--show", type=str, help="Show ID (required with --file)")
    parser.add_argument("--episode", type=int, default=1, help="Episode number (default: 1)")
    parser.add_argument(
        "--character", type=str, default=None,
        help="Character ID — if omitted, episode-level ingest (recommended)"
    )
    args = parser.parse_args()

    if args.file:
        if not args.show:
            print("ERROR: --show is required with --file")
            sys.exit(1)
        ingest_single(args.file, args.show, args.episode, args.character)
    else:
        seed_all()