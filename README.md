# 🎭 Zupee Drama AI Companion

> RAG-powered AI companion backend for Zupee Studio micro-dramas.
> Chat with characters from your favourite shows — with full spoiler protection.

---

## Architecture Overview

```
User Message
     │
     ▼
FastAPI /chat endpoint
     │
     ▼
ChatService.chat()
     ├── retrieve()          ← RAG: embed query → FAISS search → spoiler filter
     ├── build_system_prompt()  ← persona + retrieved script excerpts
     └── generate_reply()    ← Cohere LLM call
          │
          ▼
     Character Response
```

**RAG Pipeline:**
```
Script Upload (.txt/.md)
     │
     ├── clean_script_text()     ← unicode fix, whitespace normalise
     ├── split_into_chunks()     ← 400-word overlapping chunks
     ├── embed_texts()           ← sentence-transformers (all-MiniLM-L6-v2)
     └── upsert_chunks()         ← FAISS index per (show_id, character_id)
```

---

## Project Structure

```
ai-drama-chat/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entrypoint
│   │   ├── api/
│   │   │   ├── chat.py              # POST /chat, GET /shows
│   │   │   └── upload.py            # POST /upload, /admin/*
│   │   ├── core/
│   │   │   ├── config.py            # All settings (pydantic-settings)
│   │   │   └── logging.py           # Structured logging (structlog)
│   │   ├── rag/
│   │   │   ├── ingest.py            # Script → chunks → embeddings → FAISS
│   │   │   ├── embedder.py          # sentence-transformers wrapper
│   │   │   ├── retriever.py         # Query → embed → search → filter
│   │   │   ├── filters.py           # Spoiler fence + deduplication
│   │   │   ├── prompt_builder.py    # System prompt assembler
│   │   │   └── generator.py         # Cohere LLM call + retry
│   │   ├── models/
│   │   │   └── chat.py              # Pydantic schemas
│   │   ├── services/
│   │   │   ├── chat_service.py      # Chat orchestrator
│   │   │   └── ingest_service.py    # Upload orchestrator
│   │   ├── db/
│   │   │   ├── vector_store.py      # FAISS index management
│   │   │   └── metadata_store.py    # JSON metadata (shows/characters)
│   │   └── utils/
│   │       └── text_utils.py        # Cleaning, chunking, helpers
│   │
│   ├── scripts/
│   │   └── ingest_data.py           # Seed script (register + ingest samples)
│   ├── data/
│   │   ├── raw/                     # Place your script files here
│   │   └── processed/               # Auto-generated embeddings + metadata
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   └── app.py                       # Streamlit UI
│
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Quick Start (Local — Recommended for Development)

### Prerequisites
- Python 3.10+
- An Cohere API key → https://dashboard.cohere.com/api-keys

---

### Step 1 — Clone / Navigate

```bash
cd ai-drama-chat
```

---

### Step 2 — Backend Setup

```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 3 — Configure Environment

```bash
# Copy and edit the .env file
cp .env .env.local   # optional — or just edit .env directly
```

Open `.env` and set your API key:

```env
COHERE_API_KEY=sk-ant-your-actual-key-here
DEBUG=true
```

All other defaults work out of the box for local development.

---

### Step 4 — Seed Sample Data

This registers the 3 sample shows (Maut Ki Ghati, Forbidden Love, Ghost in Love),
all 6 characters with their persona prompts, and ingests sample episode scripts:

```bash
# From backend/ directory
python scripts/ingest_data.py
```

Expected output:
```
Registering show: maut-ki-ghati
Registering character: priya
Ingesting sample script for priya ep.1 — 12 chunks created
Registering character: vikram
...
Seed complete.
```

---

### Step 5 — Start the Backend

```bash
# From backend/ directory (with venv activated)
uvicorn app.main:app --reload --port 8000
```

Verify it's running:
```
open http://localhost:8000/docs
```
You should see the interactive Swagger UI with all endpoints.

---

### Step 6 — Start the Frontend

Open a **new terminal**:

```bash
# From the project root
cd frontend

# Install Streamlit (if not already installed)
pip install streamlit requests

# Run the UI
streamlit run app.py
```

Open in browser: **http://localhost:8501**

---

## Full Workflow

### Chat with a character

1. Open **http://localhost:8501**
2. Click **🎬 Watch & Chat**
3. Select a show (e.g. Maut Ki Ghati)
4. Select a character (e.g. Priya)
5. Use the episode slider in the sidebar to set your current episode
6. Type your message and press **Send ➤**

The character will respond using:
- Their persona prompt (voice, personality)
- Relevant script excerpts from episodes ≤ your current episode
- Full conversation history

---

### Upload a real episode script

1. Prepare a `.txt` or `.md` file with your episode script
2. Go to **📤 Upload Script**
3. Select the show and character
4. Set the episode number
5. Upload the file

The system will:
- Clean and chunk the text (~400 words per chunk, 80-word overlap)
- Embed each chunk using `all-MiniLM-L6-v2`
- Store in FAISS (indexed per character)
- Record the episode in metadata for spoiler protection

---

### Register a new show via API

```bash
# Register show
curl -X POST http://localhost:8000/api/v1/admin/register/show \
  -F "show_id=khooni-flat" \
  -F "name=Khooni Flat Ka Rahasya" \
  -F "genre=Horror · Mystery" \
  -F "description=A Mumbai flat with a bloody past"

# Register character
curl -X POST http://localhost:8000/api/v1/admin/register/character \
  -F "show_id=khooni-flat" \
  -F "character_id=detective-sharma" \
  -F "name=Detective Sharma" \
  -F "role=Investigating Officer" \
  -F 'persona_prompt=You are Detective Sharma, a seasoned police detective in Khooni Flat Ka Rahasya...' \
  -F "emoji=🔎"

# Upload a script
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@data/raw/khooni_flat_ep1.txt" \
  -F "show_id=khooni-flat" \
  -F "character_id=detective-sharma" \
  -F "episode_number=1"
```

---

### Chat via API (direct)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "show_id": "maut-ki-ghati",
    "character_id": "priya",
    "episode_number": 1,
    "user_message": "What did you find in that abandoned mill?",
    "history": []
  }'
```

---

## Docker Deployment

### Step 1 — Set your API key

```bash
export COHERE_API_KEY=sk-ant-your-key-here
```

### Step 2 — Build and run

```bash
docker-compose up --build
```

Services:
- Backend: **http://localhost:8000**
- Frontend: **http://localhost:8501**
- API Docs: **http://localhost:8000/docs**

### Step 3 — Seed data inside container

```bash
docker exec drama-ai-backend python scripts/ingest_data.py
```

### Stop

```bash
docker-compose down
```

Data is persisted in the `drama_data` Docker volume across restarts.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send a message to a character |
| `GET`  | `/api/v1/shows` | List all registered shows |
| `GET`  | `/api/v1/shows/{show_id}` | Get show details |
| `GET`  | `/api/v1/health` | Liveness check |
| `POST` | `/api/v1/upload` | Upload episode script |
| `POST` | `/api/v1/admin/register/show` | Register a show |
| `POST` | `/api/v1/admin/register/character` | Register a character |
| `GET`  | `/api/v1/admin/episodes` | List ingested episodes |

Full interactive docs: **http://localhost:8000/docs**

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `Cohere_KEY` | — | **Required.** Your cohere API key |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `400` | Words per RAG chunk |
| `CHUNK_OVERLAP` | `80` | Word overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of chunks to retrieve per query |
| `MAX_TOKENS` | `512` | Max tokens in LLM response |
| `SPOILER_PROTECTION_ENABLED` | `true` | Enable episode spoiler fence |
| `DEBUG` | `false` | Enable debug logging + hot reload |
| `PORT` | `8000` | Backend port |

---

## Spoiler Protection

The spoiler fence works at two levels:

1. **Vector store level** — `vector_store.search()` passes `max_episode` to filter
   chunks before they are even scored by FAISS.

2. **Application level** — `filters.apply_spoiler_fence()` double-checks every
   returned chunk and logs a warning if anything slipped through.

3. **Prompt level** — The system prompt explicitly instructs the LLM:
   > "You must NOT reveal events from episodes beyond Episode N."

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'faiss'`**
```bash
pip install faiss-cpu
```

**`sentence_transformers` download fails**
The model downloads on first run (~90 MB). Ensure internet access or pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**`AuthenticationError` from Coherence**
Check your `Coherence_key` in `.env`. It must start with `sk-ant-`.

**Backend not reachable from frontend**
Ensure both are running. The frontend expects the backend at `http://localhost:8000`.
If running in Docker, services communicate via the Docker network automatically.

**Empty chat responses**
Run the seed script first: `python scripts/ingest_data.py`
Characters can still chat without indexed scripts, but responses will be
persona-only (no RAG context).

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API Framework | FastAPI + Uvicorn |
| LLM | cohere |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (faiss-cpu) |
| Config | pydantic-settings |
| Logging | structlog |
| Retry | tenacity |
| Frontend | Streamlit |
| Container | Docker + docker-compose |