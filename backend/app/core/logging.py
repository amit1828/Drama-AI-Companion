"""
core/logging.py
Structured logging with structlog — consistent JSON logs in prod,
pretty colored output in dev/debug mode.
"""

import logging
import sys
import structlog
from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
    ]

    if settings.debug:
        # Human-friendly colored output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # JSON for production / log aggregators
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        #wrapper_class=structlog.make_filtering_bound_logger(
        #    logging.DEBUG if settings.debug else logging.INFO
        #),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        #logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    return structlog.get_logger(name)