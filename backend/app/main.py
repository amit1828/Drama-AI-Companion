"""
main.py
FastAPI application entrypoint.

Startup sequence:
  1. Setup structured logging
  2. Create required data directories
  3. Pre-load sentence-transformer embedding model (warm-up)
  4. Verify Cohere API key is present
  5. Mount API routers
  6. Configure CORS
"""

from __future__ import annotations
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.chat import router as chat_router
from app.api.upload import router as upload_router


# ─────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    setup_logging()
    logger = get_logger("startup")
    settings = get_settings()

    logger.info("app_starting", name=settings.app_name, version=settings.app_version)

    # Ensure required directories exist
    for d in [
        settings.raw_dir,
        settings.processed_dir,
        settings.vector_store_dir,
        settings.memory_store_dir,
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Warm up the sentence-transformer embedding model
    # (downloads ~90 MB on first run, then cached)
    try:
        from app.rag.embedder import embed_query
        embed_query("warmup")
        logger.info("embedding_model_warmed_up", model=settings.embedding_model)
    except Exception as e:
        logger.warning("embedding_warmup_failed", error=str(e))

    # Verify Cohere API key is configured
    if settings.cohere_api_key:
        logger.info(
            "cohere_configured",
            model=settings.cohere_model,
            key_prefix=settings.cohere_api_key[:8] + "...",
        )
    else:
        logger.error(
            "cohere_api_key_missing",
            hint="Set COHERE_API_KEY in your .env file. "
                 "Get a free key at https://dashboard.cohere.com/api-keys",
        )

    logger.info("app_ready", host=settings.host, port=settings.port)
    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("app_shutdown")


# ─────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "RAG-powered AI companion backend for Zupee Studio micro-dramas. "
            "Powered by Cohere command-r-plus. "
            "Chat with characters — with persistent memory, intent detection, "
            "and episode spoiler protection."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",    # Streamlit default
            "http://localhost:3000",    # React dev server
            "http://localhost:5173",    # Vite dev server
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────
    app.include_router(chat_router,   prefix="/api/v1")
    app.include_router(upload_router, prefix="/api/v1")

    return app


app = create_app()


# ─────────────────────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
