"""
frontend/app.py
Streamlit UI for Zupee Drama AI Companion v2 (Cohere + Memory + Intent)

Pages:
  🎬  Watch & Chat   — show → character → full memory chat
  📤  Upload Script  — upload episode scripts
  🗂️  Admin Panel    — register shows/characters, manage memory sessions
"""

import json
import uuid
import requests
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Zupee Drama AI Companion",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; font-weight: 400 !important; }
.stApp { background: #0d0d14; color: #f0ede8; }
section[data-testid="stSidebar"] { background: #111118 !important; border-right: 1px solid rgba(255,255,255,0.07); }

.user-msg {
    background: #1e1e2e; border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px 14px 4px 14px; padding: 12px 16px;
    margin: 4px 0 4px 20%; color: #f0ede8; font-size: 14px; line-height: 1.6;
}
.char-msg {
    background: #1a1512; border: 1px solid rgba(232,160,69,0.15);
    border-radius: 14px 14px 14px 4px; padding: 12px 16px;
    margin: 4px 20% 4px 0; color: #f0ede8; font-size: 14px; line-height: 1.6;
}
.char-name-label { font-size: 11px; color: #e8a045; font-weight: 500; letter-spacing: 0.3px; margin-bottom: 4px; }

.intent-badge {
    display: inline-block; font-size: 10px; padding: 2px 8px;
    border-radius: 10px; margin-left: 8px; font-weight: 500; letter-spacing: 0.4px;
    text-transform: uppercase;
}
.intent-plot        { background: rgba(61,156,138,0.2); color: #3d9c8a; }
.intent-backstory   { background: rgba(122,92,156,0.2); color: #a07ad4; }
.intent-emotion     { background: rgba(201,75,122,0.2); color: #e0508a; }
.intent-lore        { background: rgba(232,160,69,0.2); color: #e8a045; }
.intent-ooc         { background: rgba(201,75,61,0.2);  color: #e07060; }
.intent-greeting    { background: rgba(76,175,110,0.2); color: #4caf6e; }
.intent-scene       { background: rgba(53,130,220,0.2); color: #4a90e2; }
.intent-general     { background: rgba(90,86,80,0.2);   color: #9b9690; }

.memory-indicator {
    font-size: 11px; color: #5a5650; padding: 4px 12px;
    border-top: 1px solid rgba(255,255,255,0.05); text-align: center;
}
.ep-context {
    background: rgba(232,160,69,0.06); border-left: 3px solid #e8a045;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-size: 13px; color: #9b9690; margin: 8px 0;
}
.session-pill {
    font-size: 10px; color: #3d6e5a; background: rgba(61,110,90,0.15);
    border: 1px solid rgba(61,110,90,0.3); border-radius: 10px;
    padding: 2px 8px; font-family: monospace;
}
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Intent badge helper
# ─────────────────────────────────────────────────────────────
INTENT_CLASS = {
    "plot_question":       "intent-plot",
    "character_backstory": "intent-backstory",
    "relationship_query":  "intent-backstory",
    "emotional_support":   "intent-emotion",
    "lore_request":        "intent-lore",
    "scene_continuation":  "intent-scene",
    "out_of_character":    "intent-ooc",
    "greeting":            "intent-greeting",
    "farewell":            "intent-greeting",
    "general":             "intent-general",
}
INTENT_LABEL = {
    "plot_question":       "Plot",
    "character_backstory": "Backstory",
    "relationship_query":  "Relationship",
    "emotional_support":   "Emotion",
    "lore_request":        "Lore",
    "scene_continuation":  "Scene",
    "out_of_character":    "OOC",
    "greeting":            "Greeting",
    "farewell":            "Farewell",
    "general":             "General",
}


def intent_badge(intent: str) -> str:
    cls = INTENT_CLASS.get(intent, "intent-general")
    label = INTENT_LABEL.get(intent, intent)
    return f'<span class="intent-badge {cls}">{label}</span>'


# ─────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_shows() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/shows", timeout=5)
        r.raise_for_status()
        return r.json().get("shows", [])
    except Exception as e:
        st.error(f"Cannot reach backend: {e}")
        return []


def send_chat(show_id, character_id, episode_number, user_message, session_id) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/chat", json={
            "show_id": show_id,
            "character_id": character_id,
            "episode_number": episode_number,
            "user_message": user_message,
            "session_id": session_id,
            "history": [],   # Memory handles history now
        }, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def reset_memory(session_id: str) -> bool:
    try:
        r = requests.post(f"{API_BASE}/memory/reset", json={"session_id": session_id}, timeout=10)
        r.raise_for_status()
        return r.json().get("success", False)
    except Exception:
        return False


def fetch_sessions(show_id=None, character_id=None) -> list[dict]:
    try:
        params = {}
        if show_id: params["show_id"] = show_id
        if character_id: params["character_id"] = character_id
        r = requests.get(f"{API_BASE}/memory/sessions", params=params, timeout=5)
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception:
        return []


def check_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.json()
    except Exception:
        return {"status": "error", "cohere_key_configured": False}


def upload_episode(file_bytes, filename, show_id, episode_number) -> dict | None:
    """NEW: Upload one file for the entire episode — auto-distributes to all characters."""
    try:
        r = requests.post(
            f"{API_BASE}/upload/episode",
            files={"file": (filename, file_bytes, "text/plain")},
            data={"show_id": show_id, "episode_number": str(episode_number)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        st.error(f"Upload error {r.status_code}: {r.text}")
        return None
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def upload_script_override(file_bytes, filename, show_id, character_id, episode_number) -> dict | None:
    """LEGACY: Upload a script for a single character (override/correction use only)."""
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes, "text/plain")},
            data={"show_id": show_id, "character_id": character_id, "episode_number": str(episode_number)},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        st.error(f"Upload error {r.status_code}: {r.text}")
        return None
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def register_show_api(show_id, name, genre, description) -> bool:
    try:
        r = requests.post(f"{API_BASE}/admin/register/show",
            data={"show_id": show_id, "name": name, "genre": genre, "description": description},
            timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(str(e)); return False


def register_character_api(show_id, character_id, name, role, persona_prompt, emoji, lore_json) -> bool:
    try:
        r = requests.post(f"{API_BASE}/admin/register/character",
            data={"show_id": show_id, "character_id": character_id, "name": name,
                  "role": role, "persona_prompt": persona_prompt, "emoji": emoji, "lore_json": lore_json},
            timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(str(e)); return False


def fetch_episodes(show_id=None) -> list[dict]:
    try:
        params = {}
        if show_id: params["show_id"] = show_id
        r = requests.get(f"{API_BASE}/admin/episodes", params=params, timeout=5)
        r.raise_for_status()
        return r.json().get("episodes", [])
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "page": "chat",
        "selected_show": None,
        "selected_character": None,
        "episode_number": 1,
        "session_id": None,
        # chat_messages: list of dicts {role, content, intent, intent_confidence}
        "chat_messages": [],
        "show_lore": False,
        "show_ep_context": False,
        "memory_turn_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style="padding: 0.5rem 0 1rem;">
            <div style="font-family:'Playfair Display',serif;font-size:22px;color:#e8a045;">Zupee Studio</div>
            <div style="font-size:12px;color:#5a5650;margin-top:2px;">Drama AI Companion v2 · Cohere</div>
        </div>""", unsafe_allow_html=True)

    # Cohere health indicator
    health = check_health()
    cohere_ok = health.get("cohere_key_configured", False)
    cohere_icon = "🟢" if cohere_ok else "🔴"
    st.markdown(
        f"<div style='font-size:12px;color:#5a5650;margin-bottom:12px;'>"
        f"{cohere_icon} Cohere · <code style='color:#9b9690'>{health.get('cohere_model','?')}</code></div>",
        unsafe_allow_html=True
    )
    if not cohere_ok:
        st.warning("Cohere API key not set. Add `COHERE_API_KEY` to your `.env` file.", icon="⚠️")

    st.markdown("---")
    page = st.radio("Navigate", ["🎬 Watch & Chat", "📤 Upload Script", "🗂️ Admin Panel"],
                    label_visibility="collapsed")
    st.session_state.page = page

    if st.session_state.selected_character and "Chat" in page:
        show = st.session_state.selected_show
        char = st.session_state.selected_character
        st.markdown("---")
        st.markdown(
            f"<div style='font-size:11px;color:#5a5650;text-transform:uppercase;letter-spacing:.5px;'>Chatting with</div>"
            f"<div style='font-size:16px;font-family:\"Playfair Display\",serif;color:#f0ede8;margin:4px 0;'>"
            f"{char.get('emoji','🎭')} {char.get('name','')}</div>"
            f"<div style='font-size:11px;color:#e8a045;'>{show.get('name','')}</div>",
            unsafe_allow_html=True
        )

        ep = st.slider("📺 Your episode", 1, 20, st.session_state.episode_number, key="ep_slider")
        st.session_state.episode_number = ep

        st.markdown(
            f"<div style='font-size:11px;color:#5a5650;margin-top:8px;'>Memory turns: "
            f"<strong style='color:#9b9690'>{st.session_state.memory_turn_count}</strong></div>",
            unsafe_allow_html=True
        )
        if st.session_state.session_id:
            sid_short = st.session_state.session_id[-12:]
            st.markdown(
                f"<div style='font-size:10px;color:#5a5650;margin-top:4px;'>Session: "
                f"<code style='color:#3d6e5a'>…{sid_short}</code></div>",
                unsafe_allow_html=True
            )

        if st.button("🗑️ Clear memory"):
            if st.session_state.session_id:
                reset_memory(st.session_state.session_id)
                st.session_state.chat_messages = []
                st.session_state.memory_turn_count = 0
                st.toast("Memory cleared", icon="🗑️")
                st.rerun()

        if st.button("🔄 New session"):
            st.session_state.session_id = None
            st.session_state.chat_messages = []
            st.session_state.memory_turn_count = 0
            st.rerun()

        if st.button("← Change character"):
            st.session_state.selected_character = None
            st.session_state.session_id = None
            st.session_state.chat_messages = []
            st.rerun()

        if st.button("← Change show"):
            st.session_state.selected_show = None
            st.session_state.selected_character = None
            st.session_state.session_id = None
            st.session_state.chat_messages = []
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Page: Watch & Chat
# ─────────────────────────────────────────────────────────────

def render_chat_page():
    shows = fetch_shows()

    # ── Show selector ─────────────────────────────────────────
    if not st.session_state.selected_show:
        st.markdown("## Choose a Drama")
        st.markdown("<p style='color:#9b9690;margin-top:-8px;margin-bottom:1.5rem;'>Select a show to begin chatting with its characters</p>", unsafe_allow_html=True)
        if not shows:
            st.warning("No shows registered yet. Run `python scripts/ingest_data.py` first.")
            return
        cols = st.columns(min(len(shows), 3))
        for i, show in enumerate(shows):
            with cols[i % 3]:
                chars = list(show.get("characters", {}).values())
                char_names = " · ".join(c["name"] for c in chars)
                st.markdown(
                    f"""<div style="background:#111118;border:1px solid rgba(255,255,255,0.08);
                         border-radius:12px;padding:16px;margin-bottom:8px;">
                        <div style="font-family:'Playfair Display',serif;font-size:20px;color:#f0ede8;margin-bottom:4px;">{show['name']}</div>
                        <div style="font-size:11px;color:#e8a045;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">{show['genre']}</div>
                        <div style="font-size:13px;color:#9b9690;line-height:1.5;margin-bottom:10px;">{show.get('description','')}</div>
                        <div style="font-size:11px;color:#5a5650;">Characters: {char_names}</div>
                    </div>""", unsafe_allow_html=True)
                if st.button(f"Enter {show['name']}", key=f"show_{show['show_id']}"):
                    st.session_state.selected_show = show
                    st.rerun()
        return

    show = st.session_state.selected_show

    # ── Character selector ────────────────────────────────────
    if not st.session_state.selected_character:
        st.markdown(f"## {show['name']}")
        st.markdown(f"<p style='color:#9b9690;margin-top:-8px;margin-bottom:1.5rem;'>{show['genre']}</p>", unsafe_allow_html=True)
        st.markdown("### Who do you want to talk to?")
        chars = list(show.get("characters", {}).values())
        if not chars:
            st.warning("No characters registered for this show.")
            return
        cols = st.columns(min(len(chars), 3))
        for i, char in enumerate(chars):
            with cols[i % 3]:
                # Check if a session exists for this character
                sessions = fetch_sessions(show_id=show["show_id"], character_id=char["character_id"])
                session_note = f"💬 {sessions[0]['total_turns']} turns" if sessions else "New conversation"
                st.markdown(
                    f"""<div style="background:#111118;border:1px solid rgba(255,255,255,0.08);
                         border-radius:12px;padding:16px;text-align:center;margin-bottom:8px;">
                        <div style="font-size:36px;margin-bottom:8px;">{char.get('emoji','🎭')}</div>
                        <div style="font-family:'Playfair Display',serif;font-size:17px;color:#f0ede8;margin-bottom:4px;">{char['name']}</div>
                        <div style="font-size:11px;color:#e8a045;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">{char['role']}</div>
                        <div style="font-size:11px;color:#5a5650;">{session_note}</div>
                    </div>""", unsafe_allow_html=True)

                col_new, col_resume = st.columns(2)
                with col_new:
                    if st.button("New", key=f"new_{char['character_id']}"):
                        st.session_state.selected_character = char
                        st.session_state.session_id = f"{show['show_id']}__{char['character_id']}__{uuid.uuid4().hex[:12]}"
                        st.session_state.chat_messages = []
                        st.session_state.memory_turn_count = 0
                        st.rerun()
                with col_resume:
                    if sessions:
                        if st.button("Resume", key=f"resume_{char['character_id']}"):
                            st.session_state.selected_character = char
                            st.session_state.session_id = sessions[0]["session_id"]
                            st.session_state.chat_messages = []
                            st.session_state.memory_turn_count = sessions[0].get("total_turns", 0)
                            st.rerun()
        return

    char = st.session_state.selected_character

    # ── Chat interface ────────────────────────────────────────
    col_h, col_a = st.columns([3, 1])
    with col_h:
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:12px;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.07);">
                <span style="font-size:32px;">{char.get('emoji','🎭')}</span>
                <div>
                    <div style="font-family:'Playfair Display',serif;font-size:22px;color:#f0ede8;">{char['name']}</div>
                    <div style="font-size:12px;color:#9b9690;">{show['name']} · {char['role']} · <span style="color:#4caf6e;">● online</span></div>
                </div>
            </div>""", unsafe_allow_html=True)
    with col_a:
        if st.button("✦ Lore"):
            st.session_state.show_lore = not st.session_state.show_lore
        if st.button("📺 Context"):
            st.session_state.show_ep_context = not st.session_state.show_ep_context

    if st.session_state.show_ep_context:
        st.markdown(
            f'<div class="ep-context"><strong style="color:#e8a045;">Episode {st.session_state.episode_number} context</strong><br>'
            f'Spoiler fence active — character will not reveal events beyond Ep. {st.session_state.episode_number}.</div>',
            unsafe_allow_html=True)

    if st.session_state.show_lore:
        lore = char.get("lore", {})
        if lore:
            with st.expander("✦ Behind-the-scenes lore", expanded=True):
                for k, v in lore.items():
                    st.markdown(f"**{k}**  \n{v}")

    st.markdown("---")

    # Render messages
    if not st.session_state.chat_messages:
        resume_note = ""
        if st.session_state.memory_turn_count > 0:
            resume_note = f"<br><span style='font-size:12px;color:#3d6e5a;'>✓ Resuming conversation — {st.session_state.memory_turn_count} turns in memory</span>"
        st.markdown(
            f'<div style="text-align:center;padding:2rem 0;color:#5a5650;font-style:italic;">'
            f'Connected with {char["name"]}.{resume_note}</div>',
            unsafe_allow_html=True)

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            badge = intent_badge(msg.get("intent", "general")) if msg.get("intent") else ""
            st.markdown(
                f'<div class="user-msg">👤 {msg["content"]}{badge}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="char-msg">'
                f'<div class="char-name-label">{char.get("emoji","🎭")} {char["name"]}</div>'
                f'{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True)

    # Memory indicator
    if st.session_state.memory_turn_count > 0:
        st.markdown(
            f'<div class="memory-indicator">🧠 {st.session_state.memory_turn_count} turns in memory · '
            f'session <code style="color:#3d6e5a">…{(st.session_state.session_id or "")[-10:]}</code></div>',
            unsafe_allow_html=True)

    # Suggestion chips (first 4 turns only)
    if len(st.session_state.chat_messages) < 4:
        suggestions = [
            "What's on your mind right now?",
            "Tell me something you've never told anyone.",
            "What are you most afraid of?",
            "Do you trust me?",
        ]
        s_cols = st.columns(len(suggestions))
        for i, sugg in enumerate(suggestions):
            with s_cols[i]:
                if st.button(sugg, key=f"s{i}", use_container_width=True):
                    _process_message(sugg, show, char)

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Message", placeholder=f"Message {char['name']}…", label_visibility="collapsed")
        _, col_send = st.columns([6, 1])
        with col_send:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)
    if submitted and user_input.strip():
        _process_message(user_input.strip(), show, char)


def _process_message(user_input: str, show: dict, char: dict):
    # Ensure session_id
    if not st.session_state.session_id:
        st.session_state.session_id = f"{show['show_id']}__{char['character_id']}__{uuid.uuid4().hex[:12]}"

    # Optimistic UI: show user message immediately
    st.session_state.chat_messages.append({
        "role": "user", "content": user_input, "intent": None
    })

    with st.spinner(f"{char['name']} is thinking…"):
        resp = send_chat(
            show_id=show["show_id"],
            character_id=char["character_id"],
            episode_number=st.session_state.episode_number,
            user_message=user_input,
            session_id=st.session_state.session_id,
        )

    if resp:
        # Update user message with intent
        if st.session_state.chat_messages:
            st.session_state.chat_messages[-1]["intent"] = resp.get("intent", "general")

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": resp.get("reply", "…"),
            "intent": resp.get("intent", "general"),
        })
        st.session_state.session_id = resp.get("session_id", st.session_state.session_id)
        st.session_state.memory_turn_count = resp.get("memory_turn_count", 0)

        if resp.get("spoiler_protected"):
            st.toast("🔒 Spoiler protection active", icon="🔒")

    st.rerun()


# ─────────────────────────────────────────────────────────────
# Page: Upload Script
# ─────────────────────────────────────────────────────────────

def render_upload_page():
    st.markdown("## Upload Episode Script")
    st.markdown(
        "<p style='color:#9b9690;margin-top:-8px;margin-bottom:1rem;'>"
        "Upload <strong style='color:#e8a045'>one file per episode</strong>. "
        "The system automatically reads the script, detects which characters "
        "are present in each scene, and indexes only what each character witnessed. "
        "Characters absent from a scene will have no knowledge of it."
        "</p>",
        unsafe_allow_html=True,
    )

    shows = fetch_shows()
    if not shows:
        st.warning("Register a show first.")
        return

    show_options = {s["name"]: s for s in shows}

    # ── Primary tab: Episode upload ───────────────────────────
    tab_ep, tab_override = st.tabs(["📺 Episode Upload (Recommended)", "🔧 Character Override"])

    with tab_ep:
        st.markdown(
            """<div style="background:rgba(232,160,69,0.06);border-left:3px solid #e8a045;
            border-radius:0 8px 8px 0;padding:10px 14px;font-size:13px;color:#9b9690;margin-bottom:1rem;">
            <strong style="color:#e8a045;">How it works</strong><br>
            Upload the full episode script once. The system will:<br>
            &nbsp;① Split it into scene blocks<br>
            &nbsp;② Detect which characters are present in each block<br>
            &nbsp;③ Index only the witnessed scenes into each character's knowledge base<br>
            Characters absent from a scene cannot narrate it.
            </div>""",
            unsafe_allow_html=True,
        )

        sel_show_name = st.selectbox("Show", list(show_options.keys()), key="ep_show")
        sel_show = show_options[sel_show_name]

        chars = list(sel_show.get("characters", {}).values())
        if not chars:
            st.warning("Register characters for this show first — the system needs them to detect presence.")
            return

        # Show registered characters as info
        char_pills = " &nbsp;".join(
            f"<span style='background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);"
            f"border-radius:10px;padding:2px 10px;font-size:12px;color:#9b9690;'>"
            f"{c.get('emoji','🎭')} {c['name']}</span>"
            for c in chars
        )
        st.markdown(
            f"<div style='margin-bottom:1rem;font-size:12px;color:#5a5650;'>Registered characters: "
            f"{char_pills}</div>",
            unsafe_allow_html=True,
        )

        ep = st.number_input("Episode Number", min_value=1, max_value=999, value=1, key="ep_num")
        uploaded = st.file_uploader(
            "Episode Script (.txt or .md)",
            type=["txt", "md"],
            key="ep_file",
            help="Upload the full episode script. Include all scenes, stage directions, and dialogue.",
        )

        if uploaded and st.button("🚀 Ingest Episode", type="primary", use_container_width=True):
            with st.spinner(f"Parsing episode {ep}, detecting character presence, embedding…"):
                result = upload_episode(
                    uploaded.read(),
                    uploaded.name,
                    sel_show["show_id"],
                    ep,
                )

            if result and result.get("success"):
                st.success(f"✅ Episode {ep} ingested successfully!")

                # Show per-character breakdown
                ingested = result.get("characters_ingested", {})
                absent = result.get("characters_absent", [])

                st.markdown("**Character knowledge distribution:**")
                cols = st.columns(max(len(ingested) + len(absent), 1))
                all_chars = list(ingested.items()) + [(c, None) for c in absent]

                for i, (cid, chunks) in enumerate(all_chars):
                    with cols[i % len(cols)]:
                        char_data = next((c for c in chars if c["character_id"] == cid), {})
                        emoji = char_data.get("emoji", "🎭")
                        name = char_data.get("name", cid)
                        if chunks is not None:
                            st.markdown(
                                f"<div style='background:rgba(61,156,138,0.1);border:1px solid rgba(61,156,138,0.3);"
                                f"border-radius:8px;padding:10px;text-align:center;'>"
                                f"<div style='font-size:20px'>{emoji}</div>"
                                f"<div style='font-size:12px;color:#f0ede8;margin:2px 0'>{name}</div>"
                                f"<div style='font-size:11px;color:#3d9c8a'>✓ {chunks} chunks</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div style='background:rgba(90,86,80,0.1);border:1px solid rgba(90,86,80,0.3);"
                                f"border-radius:8px;padding:10px;text-align:center;'>"
                                f"<div style='font-size:20px;opacity:.4'>{emoji}</div>"
                                f"<div style='font-size:12px;color:#5a5650;margin:2px 0'>{name}</div>"
                                f"<div style='font-size:11px;color:#5a5650'>absent this ep.</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                st.markdown(
                    f"<div style='font-size:12px;color:#5a5650;margin-top:8px;'>"
                    f"Total chunks: {result.get('total_chunks_stored', 0)}</div>",
                    unsafe_allow_html=True,
                )
            elif result:
                st.error(result.get("message", "Unknown error"))

    with tab_override:
        st.markdown(
            "<p style='color:#9b9690;font-size:13px;margin-bottom:1rem;'>"
            "Use this ONLY for manual corrections or to add character-specific background lore "
            "not present in the main script. The episode upload tab is recommended for normal use."
            "</p>",
            unsafe_allow_html=True,
        )

        sel_show_name2 = st.selectbox("Show", list(show_options.keys()), key="ov_show")
        sel_show2 = show_options[sel_show_name2]
        chars2 = list(sel_show2.get("characters", {}).values())
        if not chars2:
            st.warning("No characters registered for this show.")
        else:
            char_options2 = {c["name"]: c for c in chars2}
            sel_char_name = st.selectbox("Character", list(char_options2.keys()), key="ov_char")
            sel_char = char_options2[sel_char_name]
            ep2 = st.number_input("Episode Number", min_value=1, max_value=999, value=1, key="ov_ep")
            uploaded2 = st.file_uploader("Script File (.txt or .md)", type=["txt", "md"], key="ov_file")

            if uploaded2 and st.button("Upload Override", key="ov_btn"):
                with st.spinner("Ingesting character-specific script…"):
                    result2 = upload_script_override(
                        uploaded2.read(), uploaded2.name,
                        sel_show2["show_id"], sel_char["character_id"], ep2,
                    )
                if result2 and result2.get("success"):
                    st.success(result2["message"])

    # ── Ingested episodes summary ─────────────────────────────
    st.markdown("---")
    st.markdown("### Ingested Episodes")

    sel_show_view = show_options.get(
        st.selectbox("Show", list(show_options.keys()), key="view_show"),
        list(show_options.values())[0] if show_options else None,
    )
    if sel_show_view:
        episodes = fetch_episodes(show_id=sel_show_view["show_id"])
        if not episodes:
            st.info("No episodes ingested yet for this show.")
        else:
            # Group by episode number
            by_ep: dict[int, list] = {}
            for ep_rec in episodes:
                n = ep_rec["episode_number"]
                by_ep.setdefault(n, []).append(ep_rec)

            for ep_n in sorted(by_ep.keys()):
                recs = by_ep[ep_n]
                chars_summary = " · ".join(
                    f"`{r['character_id']}` ({r['chunk_count']} chunks)"
                    for r in sorted(recs, key=lambda x: x["character_id"])
                )
                st.markdown(f"**Ep. {ep_n}** — {chars_summary}")


# ─────────────────────────────────────────────────────────────
# Page: Admin Panel
# ─────────────────────────────────────────────────────────────

def render_admin_page():
    st.markdown("## Admin Panel")
    tab_show, tab_char, tab_mem = st.tabs(["Register Show", "Register Character", "Memory Sessions"])

    with tab_show:
        st.markdown("### Register a New Show")
        with st.form("reg_show"):
            show_id = st.text_input("Show ID (slug)", placeholder="maut-ki-ghati")
            name = st.text_input("Display Name", placeholder="Maut Ki Ghati")
            genre = st.text_input("Genre", placeholder="Thriller · Mystery")
            description = st.text_area("Description", height=80)
            if st.form_submit_button("Register Show"):
                if show_id and name and genre:
                    if register_show_api(show_id, name, genre, description):
                        st.success(f"Show '{name}' registered!")
                        st.cache_data.clear()

    with tab_char:
        st.markdown("### Register a Character")
        shows = fetch_shows()
        if not shows:
            st.warning("Register a show first.")
        else:
            show_map = {s["name"]: s["show_id"] for s in shows}
            sel = st.selectbox("Show", list(show_map.keys()), key="admin_show_sel")
            with st.form("reg_char"):
                character_id = st.text_input("Character ID (slug)", placeholder="priya")
                name = st.text_input("Character Name", placeholder="Priya Sharma")
                role = st.text_input("Role", placeholder="Investigative Journalist")
                emoji = st.text_input("Emoji", value="🎭", max_chars=2)
                persona = st.text_area("Persona Prompt", height=150,
                    placeholder="Describe voice, personality, backstory, speaking style...")
                lore_json = st.text_area("Lore (JSON)", value="{}", height=80)
                if st.form_submit_button("Register Character"):
                    if character_id and name and role and persona:
                        if register_character_api(show_map[sel], character_id, name, role, persona, emoji, lore_json):
                            st.success(f"Character '{name}' registered!")
                            st.cache_data.clear()

    with tab_mem:
        st.markdown("### Conversation Memory Sessions")
        shows = fetch_shows()
        show_map = {"All": None} | {s["name"]: s["show_id"] for s in shows}
        sel_show = st.selectbox("Filter by Show", list(show_map.keys()), key="mem_show_filter")
        sessions = fetch_sessions(show_id=show_map[sel_show])
        if not sessions:
            st.info("No memory sessions found.")
        else:
            for sess in sessions:
                col_info, col_del = st.columns([4, 1])
                with col_info:
                    st.markdown(
                        f"**{sess['character_id']}** · {sess['show_id']} · "
                        f"{sess['total_turns']} turns · "
                        f"`{sess['session_id'][-20:]}`")
                with col_del:
                    if st.button("Delete", key=f"del_{sess['session_id']}"):
                        reset_memory(sess["session_id"])
                        st.rerun()


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────
page = st.session_state.page
if "Chat" in page:
    render_chat_page()
elif "Upload" in page:
    render_upload_page()
elif "Admin" in page:
    render_admin_page()