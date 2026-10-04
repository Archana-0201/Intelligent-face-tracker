"""
face_registry.py
----------------
Manages the mapping of track IDs to face IDs.

Responsibilities:
  - Auto-generate unique face IDs for new faces (FACE_<YYYYMMDD>_<uuid4[:8]>)
  - Coordinate between recogniser (embedding match) and database (persistence)
  - Keep an accurate count of unique visitors (never double-counts)
  - Provide a "pending recognition" queue for tracks that haven't been
    identified yet (recognition runs every N frames for speed)
"""

import uuid
import os
import cv2
from datetime import datetime
from typing import Optional, Set, Dict
import numpy as np

from src.database import Database
from src.face_recognizer import FaceRecognizer
from src.logger import get_logger, log_system_event

logger = get_logger()


def _generate_face_id() -> str:
    """Generate a unique, human-readable face ID."""
    date_str = datetime.now().strftime("%Y%m%d")
    uid      = uuid.uuid4().hex[:8].upper()
    return f"FACE_{date_str}_{uid}"


class FaceRegistry:
    """
    Single source of truth for face identity management.

    Internal state
    --------------
    _track_to_face: dict[int, str]
        Maps active tracker track_id → registered face_id.

    _registered: set[str]
        Set of all face IDs ever registered (unique visitors).

    _pending: set[int]
        Track IDs that need recognition on the next recognition cycle.
    """

    def __init__(self, db: Database, recogniser: FaceRecognizer):
        self._db         = db
        self._recogniser = recogniser

        self._track_to_face: Dict[int, str]  = {}
        self._registered:    Set[str]        = set()
        self._pending:       Set[int]        = set()
        self._last_embedding: Dict[str, np.ndarray] = {}  # face_id → emb

        # Bootstrap from DB (handles restarts mid-stream)
        self._bootstrap()

    # ── Bootstrap ──────────────────────────────────────────────────────────────

    def _bootstrap(self):
        """Load existing embeddings and face IDs from the database."""
        rows = self._db.get_all_embeddings()
        self._recogniser.load_embeddings_from_db(rows)
        for face_id, _ in rows:
            self._registered.add(face_id)
        image_references = 0
        for face_id, image_path in self._db.get_recent_entry_images(per_face=3):
            if not os.path.isfile(image_path):
                continue
            image = cv2.imread(image_path)
            embedding = self._recogniser.get_embedding(image)
            if embedding is not None:
                self._recogniser.add_reference_embedding(face_id, embedding)
                image_references += 1
        logger.info(f"Loaded {image_references} references from logged entry photos.")
        log_system_event(f"Registry bootstrapped with {len(self._registered)} faces.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_face_id(self, track_id: int) -> Optional[str]:
        """Return the face_id for a given track, or None if not yet identified."""
        return self._track_to_face.get(track_id)

    def mark_pending(self, track_id: int):
        """Mark a track as needing recognition."""
        self._pending.add(track_id)

    def has_pending(self) -> bool:
        return bool(self._pending)

    def try_recognise(self, track_id: int,
                      face_img: np.ndarray,
                      frame_no: int = 0) -> tuple[str, bool, float]:
        """
        Attempt to recognise or register a face.

        Algorithm:
          1. Extract ArcFace embedding from face_img.
          2. Compare against all registered embeddings.
          3a. Match found → return existing face_id (returning visitor).
          3b. No match   → register as new face, return new face_id.

        Args:
            track_id: Tracker-assigned track ID.
            face_img: Cropped BGR face image.
            frame_no: Current frame number.

        Returns:
            (face_id, is_new, similarity)
              is_new = True  → first time this person was seen
              is_new = False → recognised returning visitor
        """
        self._pending.discard(track_id)

        current_face_id = self._track_to_face.get(track_id)
        if current_face_id is not None:
            return current_face_id, False, 0.0

        embedding = self._recogniser.get_embedding(face_img)
        if embedding is None:
            # Can't extract embedding (too blurry / small) — defer
            logger.debug(f"[REGISTRY] track={track_id} embedding failed at frame={frame_no}")
            return "", False, 0.0

        matched_id, similarity = self._recogniser.find_match(embedding)

        if matched_id is not None:
            # ── Returning visitor ──────────────────────────────────────────
            self._track_to_face[track_id] = matched_id
            if matched_id not in self._registered:
                self._registered.add(matched_id)
            logger.info(f"[REGISTRY] track={track_id} → RECOGNISED face_id={matched_id} "
                        f"sim={similarity:.3f}")
            return matched_id, False, similarity
        else:
            # ── New visitor ────────────────────────────────────────────────
            face_id = _generate_face_id()
            self._track_to_face[track_id] = face_id
            self._registered.add(face_id)
            self._recogniser.register_embedding(face_id, embedding)
            self._last_embedding[face_id] = embedding

            # Persist to DB
            emb_bytes = FaceRecognizer.embedding_to_bytes(embedding)
            try:
                self._db.register_face(face_id, emb_bytes)
            except Exception:
                pass

            logger.info(f"[REGISTRY] track={track_id} → NEW face_id={face_id} "
                        f"(total unique={self.unique_count})")
            return face_id, True, 0.0

    def on_track_exit(self, track_id: int):
        """Clean up tracking state when a track exits."""
        self._track_to_face.pop(track_id, None)
        self._pending.discard(track_id)

    @property
    def unique_count(self) -> int:
        """Total number of unique faces registered so far."""
        return len(self._registered)
