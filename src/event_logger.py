"""
event_logger.py
---------------
Handles face entry/exit event logging:

  1. Saves a cropped face image to the structured local folder:
       logs/entries/YYYY-MM-DD/<face_id>_<timestamp>.jpg
       logs/exits/YYYY-MM-DD/<face_id>_<timestamp>.jpg

  2. Records the event in SQLite via the Database layer.

  3. Calls the application logger for text-based tracing.
"""

import os
import cv2
import time
import numpy as np
from datetime import datetime
from typing import Optional

from src.database import Database
from src.logger import get_logger, log_face_entry, log_face_exit
from src.config_loader import get_config

logger = get_logger()


class EventLogger:
    """
    Persists face entry/exit events to disk (images) and DB (metadata).

    Folder structure:
        logs/
          entries/
            YYYY-MM-DD/
              <face_id>_<HH-MM-SS-ffffff>.jpg
          exits/
            YYYY-MM-DD/
              <face_id>_<HH-MM-SS-ffffff>.jpg
    """

    def __init__(self, db: Database):
        self._db  = db
        cfg = get_config()
        log_cfg = cfg.logging_cfg

        self._log_root   = os.path.abspath("logs")
        self._img_fmt    = log_cfg.get("image_format", "jpg")
        self._img_qual   = int(log_cfg.get("image_quality", 90))

        # Track entry times to compute dwell time on exit
        self._entry_times: dict[str, float] = {}

    # ── Image saving ───────────────────────────────────────────────────────────

    def _save_image(self, face_img: np.ndarray, event_type: str,
                    face_id: str) -> str:
        """
        Save a face crop to the appropriate dated subfolder.

        Returns:
            Absolute path of the saved image.
        """
        today  = datetime.now().strftime("%Y-%m-%d")
        ts_str = datetime.now().strftime("%H-%M-%S-%f")
        folder = "entries" if "entr" in event_type.lower() else "exits"
        subdir = os.path.join(self._log_root, folder, today)
        os.makedirs(subdir, exist_ok=True)

        filename = f"{face_id}_{ts_str}.{self._img_fmt}"
        filepath = os.path.join(subdir, filename)

        encode_params = []
        if self._img_fmt.lower() in ("jpg", "jpeg"):
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._img_qual]
        cv2.imwrite(filepath, face_img, encode_params)
        return filepath

    # ── Public event methods ───────────────────────────────────────────────────

    def on_entry(self, face_id: str, face_img: Optional[np.ndarray],
                 frame_no: int = 0, similarity: float = 1.0,
                 is_new: bool = True):
        """
        Called when a face enters the camera view.

        Args:
            face_id:    Unique identifier for this face.
            face_img:   Cropped BGR face image.
            frame_no:   Video frame number.
            similarity: Match score (1.0 for new registrations).
            is_new:     True if this is the first time this face is seen.
        """
        self._entry_times[face_id] = time.time()

        img_path = ""
        if face_img is not None and face_img.size > 0:
            img_path = self._save_image(face_img, "entr", face_id)

        self._db.log_event(face_id, "entry", img_path, frame_no, similarity)
        log_face_entry(face_id, is_new=is_new)
        logger.info(f"[EVENT]  entry | face_id={face_id} | frame={frame_no} "
                    f"| img={os.path.basename(img_path)}")

    def on_exit(self, face_id: str, face_img: Optional[np.ndarray],
                frame_no: int = 0):
        """
        Called when a face exits the camera view (track deleted).

        Args:
            face_id:   Unique identifier for this face.
            face_img:  Last-seen cropped face image.
            frame_no:  Video frame number.
        """
        dwell = time.time() - self._entry_times.pop(face_id, time.time())

        img_path = ""
        if face_img is not None and face_img.size > 0:
            img_path = self._save_image(face_img, "exit", face_id)

        self._db.log_event(face_id, "exit", img_path, frame_no, similarity=1.0)
        log_face_exit(face_id, dwell)
        logger.info(f"[EVENT]  exit  | face_id={face_id} | frame={frame_no} "
                    f"| dwell={dwell:.1f}s | img={os.path.basename(img_path)}")
