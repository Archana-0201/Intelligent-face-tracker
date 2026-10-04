"""
database.py
-----------
Database layer for the Intelligent Face Tracker.

Responsibilities:
  - SQLite schema creation and migration
  - CRUD operations for Faces, Events (entry/exit), and Visitor counts
  - Thread-safe connection management
  - Resilient to unexpected interruptions (WAL mode + transactions)
"""

import sqlite3
import os
import threading
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any


class Database:
    """
    Thread-safe SQLite database manager.

    Schema
    ------
    faces         – registered unique faces with embedding blob
    events        – entry / exit events per face, with image path
    visitor_stats – daily unique visitor counts (materialised view)
    """

    _lock = threading.Lock()

    def __init__(self, db_path: str = "data/face_tracker.db"):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._local = threading.local()   # per-thread connection
        self._init_schema()

    # ── Connection management ──────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return (or create) a per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # WAL mode for crash resilience + better read concurrency
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return self._local.conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._get_conn()

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_schema(self):
        """Create all tables if they don't already exist."""
        with self._lock:
            c = self.conn
            c.executescript("""
                CREATE TABLE IF NOT EXISTS faces (
                    id          TEXT PRIMARY KEY,
                    created_at  TEXT NOT NULL,
                    embedding   BLOB,
                    face_image  TEXT,          -- path to first-seen face image
                    visit_count INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id     TEXT NOT NULL,
                    event_type  TEXT NOT NULL CHECK(event_type IN ('entry','exit')),
                    timestamp   TEXT NOT NULL,
                    image_path  TEXT,
                    frame_no    INTEGER,
                    similarity  REAL,
                    FOREIGN KEY (face_id) REFERENCES faces(id)
                );

                CREATE TABLE IF NOT EXISTS visitor_stats (
                    stat_date           TEXT PRIMARY KEY,
                    unique_visitors     INTEGER DEFAULT 0,
                    total_entries       INTEGER DEFAULT 0,
                    total_exits         INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_events_face_id   ON events(face_id);
                CREATE INDEX IF NOT EXISTS idx_events_type      ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            """)
            c.commit()

    # ── Faces ──────────────────────────────────────────────────────────────────

    def register_face(self, face_id: str, embedding: bytes,
                      face_image_path: str = "") -> bool:
        """
        Insert a new face record. Returns True if newly inserted, False if duplicate.
        """
        with self._lock:
            try:
                self.conn.execute(
                    "INSERT INTO faces (id, created_at, embedding, face_image) "
                    "VALUES (?, ?, ?, ?)",
                    (face_id, datetime.now().isoformat(), embedding, face_image_path)
                )
                self.conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False   # face_id already exists

    def get_all_embeddings(self) -> List[Tuple[str, bytes]]:
        """Return list of (face_id, embedding_blob) for all registered faces."""
        rows = self.conn.execute(
            "SELECT id, embedding FROM faces WHERE embedding IS NOT NULL"
        ).fetchall()
        return [(r["id"], r["embedding"]) for r in rows]

    def get_recent_entry_images(self, per_face: int = 3) -> List[Tuple[str, str]]:
        """Return recent logged entry-image paths for each registered face."""
        rows = self.conn.execute(
            "SELECT face_id, image_path FROM events "
            "WHERE event_type = 'entry' AND image_path IS NOT NULL AND image_path != '' "
            "ORDER BY timestamp DESC"
        ).fetchall()
        counts: Dict[str, int] = {}
        images = []
        for row in rows:
            face_id = row["face_id"]
            if counts.get(face_id, 0) >= per_face:
                continue
            images.append((face_id, row["image_path"]))
            counts[face_id] = counts.get(face_id, 0) + 1
        return images

    def get_face(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Return face record as dict, or None if not found."""
        row = self.conn.execute(
            "SELECT * FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        return dict(row) if row else None

    def increment_visit_count(self, face_id: str):
        """Increment the visit counter for a returning face."""
        with self._lock:
            self.conn.execute(
                "UPDATE faces SET visit_count = visit_count + 1 WHERE id = ?",
                (face_id,)
            )
            self.conn.commit()

    def get_unique_visitor_count(self) -> int:
        """Return total number of distinct registered faces (unique visitors)."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM faces").fetchone()
        return row["cnt"] if row else 0

    # ── Events ─────────────────────────────────────────────────────────────────

    def log_event(self, face_id: str, event_type: str, image_path: str = "",
                  frame_no: int = 0, similarity: float = 1.0):
        """
        Record an entry or exit event.

        Args:
            face_id:    Unique face identifier.
            event_type: 'entry' or 'exit'.
            image_path: Path to the cropped face image saved to disk.
            frame_no:   Video frame number where event occurred.
            similarity: Recognition similarity score (1.0 for new registrations).
        """
        with self._lock:
            ts = datetime.now().isoformat()
            self.conn.execute(
                "INSERT INTO events (face_id, event_type, timestamp, "
                "image_path, frame_no, similarity) VALUES (?,?,?,?,?,?)",
                (face_id, event_type, ts, image_path, frame_no, similarity)
            )
            self.conn.commit()
        self._update_daily_stats(event_type)

    def get_events_for_face(self, face_id: str) -> List[Dict[str, Any]]:
        """Return all events for a given face_id."""
        rows = self.conn.execute(
            "SELECT * FROM events WHERE face_id = ? ORDER BY timestamp",
            (face_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent N events."""
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Visitor stats ──────────────────────────────────────────────────────────

    def _update_daily_stats(self, event_type: str):
        """Upsert today's visitor statistics row."""
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            existing = self.conn.execute(
                "SELECT * FROM visitor_stats WHERE stat_date = ?", (today,)
            ).fetchone()

            if existing is None:
                self.conn.execute(
                    "INSERT INTO visitor_stats (stat_date, unique_visitors, "
                    "total_entries, total_exits) VALUES (?,?,?,?)",
                    (today,
                     self.get_unique_visitor_count(),
                     1 if event_type == "entry" else 0,
                     1 if event_type == "exit" else 0)
                )
            else:
                entry_delta = 1 if event_type == "entry" else 0
                exit_delta  = 1 if event_type == "exit" else 0
                self.conn.execute(
                    "UPDATE visitor_stats SET "
                    "unique_visitors = ?, "
                    "total_entries = total_entries + ?, "
                    "total_exits   = total_exits   + ? "
                    "WHERE stat_date = ?",
                    (self.get_unique_visitor_count(), entry_delta, exit_delta, today)
                )
            self.conn.commit()

    def get_stats_summary(self) -> Dict[str, Any]:
        """Return a summary of all-time and today's stats."""
        today = datetime.now().strftime("%Y-%m-%d")
        unique = self.get_unique_visitor_count()
        today_row = self.conn.execute(
            "SELECT * FROM visitor_stats WHERE stat_date = ?", (today,)
        ).fetchone()
        total_events = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM events"
        ).fetchone()
        return {
            "unique_visitors_total": unique,
            "today": dict(today_row) if today_row else {},
            "total_events_logged": total_events["cnt"] if total_events else 0,
        }

    def close(self):
        """Close the database connection gracefully."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
