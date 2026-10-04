"""
logger.py
---------
Centralized logging system for the Intelligent Face Tracker.

Handles:
  - Rotating file-based event logging (events.log)
  - Console output (optional)
  - Structured log messages for all system events
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime


# ── Module-level logger reference ──────────────────────────────────────────────
_app_logger: logging.Logger | None = None


def setup_logger(log_file: str = "logs/events.log",
                 log_level: str = "INFO",
                 max_bytes: int = 10 * 1024 * 1024,   # 10 MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Configure and return the application logger.

    Args:
        log_file:     Path to the rotating log file.
        log_level:    Logging level string (DEBUG / INFO / WARNING / ERROR).
        max_bytes:    Max size of each log file before rotation.
        backup_count: How many backup log files to keep.

    Returns:
        Configured logging.Logger instance.
    """
    global _app_logger
    if _app_logger is not None:
        return _app_logger

    # Ensure log directory exists
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("FaceTracker")
    logger.setLevel(level)

    # ── Formatter ──────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── File handler (rotating) ────────────────────────────────────────────────
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    _app_logger = logger
    logger.info("=" * 70)
    logger.info("Intelligent Face Tracker -- Logging Initialised")
    logger.info(f"Log file : {os.path.abspath(log_file)}")
    logger.info(f"Log level: {log_level.upper()}")
    logger.info("=" * 70)
    return logger


def get_logger() -> logging.Logger:
    """Return the already-configured logger (or create a default one)."""
    global _app_logger
    if _app_logger is None:
        return setup_logger()
    return _app_logger


# ── Convenience event-logging helpers ─────────────────────────────────────────

def log_face_entry(face_id: str, is_new: bool = True):
    logger = get_logger()
    status = "NEW REGISTRATION" if is_new else "RE-ENTRY"
    logger.info(f"[ENTRY]      face_id={face_id} | status={status}")


def log_face_exit(face_id: str, duration_sec: float = 0.0):
    logger = get_logger()
    logger.info(f"[EXIT]       face_id={face_id} | visible_for={duration_sec:.1f}s")


def log_embedding_generated(face_id: str):
    logger = get_logger()
    logger.info(f"[EMBEDDING]  face_id={face_id} | embedding generated & stored")


def log_face_recognized(face_id: str, similarity: float):
    logger = get_logger()
    logger.info(f"[RECOGNIZED] face_id={face_id} | similarity={similarity:.4f}")


def log_face_tracked(face_id: str, frame_no: int, bbox: tuple):
    logger = get_logger()
    logger.debug(f"[TRACKING]   face_id={face_id} | frame={frame_no} | bbox={bbox}")


def log_system_event(message: str, level: str = "INFO"):
    logger = get_logger()
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(f"[SYSTEM]     {message}")
