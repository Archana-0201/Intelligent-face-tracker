"""
face_recognizer.py
------------------
InsightFace / ArcFace-based facial embedding and recognition module.

Responsibilities:
  - Generate 512-D facial embeddings using InsightFace's ArcFace model
  - Compare a query embedding against the registered face database
  - Return the best-matching face_id or None (new face)
  - Persist and reload embeddings from the database (numpy ↔ bytes)
"""

import os
import io
import numpy as np
from typing import Optional, Tuple, List
import cv2

from src.logger import get_logger, log_embedding_generated, log_face_recognized
from src.config_loader import get_config

logger = get_logger()


class FaceRecognizer:
    """
    ArcFace-based face recogniser.

    Uses InsightFace's `buffalo_sc` pack (lightweight ArcFace model) which
    operates purely on CPU via ONNX Runtime. No GPU required.

    Embedding matching uses cosine similarity. Faces with similarity above
    `similarity_threshold` are considered the same person.
    """

    def __init__(self):
        cfg = get_config()
        rec = cfg.recognition

        self.model_name   = rec.get("model_name", "buffalo_sc")
        self.sim_thr      = float(rec.get("similarity_threshold", 0.45))
        self.min_face_sz  = int(rec.get("min_face_size", 30))

        # Multiple pose/lighting references can belong to the same identity.
        self._embeddings: dict[str, list[np.ndarray]] = {}

        self._model = None
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self):
        """Initialise InsightFace ArcFace model."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.error("insightface not installed. Run: pip install insightface")
            raise

        # buffalo_sc = lightweight model pack (detection + recognition)
        # det_size must be a tuple; set small to speed up CPU inference
        self._app = FaceAnalysis(
            name=self.model_name,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info(f"FaceRecognizer loaded: InsightFace model='{self.model_name}'")

    # ── Embedding extraction ───────────────────────────────────────────────────

    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-D ArcFace embedding from a face crop.

        Args:
            face_img: BGR face crop (should be at least 30×30 px).

        Returns:
            Normalised numpy array of shape (512,) or None if detection fails.
        """
        if face_img is None or face_img.size == 0:
            return None
        h, w = face_img.shape[:2]
        if h < self.min_face_sz or w < self.min_face_sz:
            return None

        # InsightFace expects BGR; resize to a sensible size
        img = cv2.resize(face_img, (112, 112))

        faces = self._app.get(img)
        if not faces:
            # Fallback: try with the full crop resized larger
            img2 = cv2.resize(face_img, (160, 160))
            faces = self._app.get(img2)

        if not faces:
            return None

        # Take the face with the highest detection score
        best = max(faces, key=lambda f: f.det_score)
        emb = best.normed_embedding   # shape (512,), already L2-normalised
        return emb.astype(np.float32)

    # ── Embedding serialisation ────────────────────────────────────────────────

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """Serialise a numpy embedding array to raw bytes for DB storage."""
        buf = io.BytesIO()
        np.save(buf, embedding)
        return buf.getvalue()

    @staticmethod
    def bytes_to_embedding(raw: bytes) -> np.ndarray:
        """Deserialise bytes from DB back to a numpy array."""
        buf = io.BytesIO(raw)
        return np.load(buf)

    # ── In-memory registry ─────────────────────────────────────────────────────

    def load_embeddings_from_db(self, db_rows: List[Tuple[str, bytes]]):
        """
        Populate in-memory registry from DB rows.

        Args:
            db_rows: List of (face_id, embedding_bytes) tuples.
        """
        for face_id, raw in db_rows:
            if raw:
                self._embeddings.setdefault(face_id, []).append(
                    self.bytes_to_embedding(raw).astype(np.float32)
                )
        logger.info(f"Loaded {len(self._embeddings)} face embeddings from DB.")

    def register_embedding(self, face_id: str, embedding: np.ndarray):
        """Add a new embedding to the in-memory registry."""
        self.add_reference_embedding(face_id, embedding)
        log_embedding_generated(face_id)

    def add_reference_embedding(self, face_id: str, embedding: np.ndarray):
        """Add a distinct reference embedding for an existing identity."""
        references = self._embeddings.setdefault(face_id, [])
        if any(float(np.dot(embedding, reference)) >= 0.995 for reference in references):
            return
        references.append(embedding.astype(np.float32))

    # ── Matching ───────────────────────────────────────────────────────────────

    def find_match(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Compare query embedding against all registered faces.

        Uses cosine similarity (dot product of L2-normalised vectors).

        Args:
            query: Shape (512,) L2-normalised embedding.

        Returns:
            (face_id, similarity) if best match ≥ threshold, else (None, 0.0).
        """
        if not self._embeddings:
            return None, 0.0

        best_id   = None
        best_sim  = -1.0

        for fid, references in self._embeddings.items():
            # Keep the strongest cosine match across each identity's references.
            sim = max(float(np.dot(query, emb)) for emb in references)
            if sim > best_sim:
                best_sim = sim
                best_id  = fid

        if best_sim >= self.sim_thr:
            log_face_recognized(best_id, best_sim)
            return best_id, best_sim

        return None, best_sim

    @property
    def registered_count(self) -> int:
        return len(self._embeddings)
