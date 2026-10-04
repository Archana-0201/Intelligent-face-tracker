"""
visualiser.py
-------------
OpenCV drawing utilities for the real-time display overlay.

Draws:
  - Bounding boxes (green for new, orange for recognised)
  - Face ID labels with confidence scores
  - FPS counter
  - Unique visitor count banner
  - Active track count
"""

import cv2
import time
import numpy as np
from typing import List, Tuple, Optional

from src.config_loader import get_config
from src.tracker import Track


class Visualiser:
    """Draws the HUD overlay on each processed frame."""

    def __init__(self):
        cfg = get_config()
        ui  = cfg.ui

        self._show_fps         = bool(ui.get("show_fps", True))
        self._show_face_id     = bool(ui.get("show_face_id", True))
        self._show_bbox        = bool(ui.get("show_bbox", True))
        self._show_count       = bool(ui.get("show_visitor_count", True))
        self._font_scale       = float(ui.get("font_scale", 0.6))

        # Colors: BGR format
        self._color_new   = tuple(ui.get("bbox_color_new",   [0, 255, 0]))
        self._color_known = tuple(ui.get("bbox_color_known", [255, 165, 0]))
        self._color_text  = (255, 255, 255)
        self._color_bg    = (30, 30, 30)

        self._fps_history: List[float] = []
        self._prev_time  = time.time()

    # ── Per-frame drawing ──────────────────────────────────────────────────────

    def draw(self,
             frame: np.ndarray,
             tracks: List[Track],
             unique_count: int,
             frame_no: int = 0) -> np.ndarray:
        """
        Draw HUD elements onto the frame (in-place + returns frame).

        Args:
            frame:        BGR frame from the video source.
            tracks:       List of active Track objects.
            unique_count: Total unique visitors detected so far.
            frame_no:     Current frame number.

        Returns:
            Annotated frame.
        """
        frame = self._draw_fps(frame)
        frame = self._draw_banner(frame, unique_count, len(tracks))

        if self._show_bbox:
            for track in tracks:
                if track.misses > 0:
                    continue
                frame = self._draw_track(frame, track)

        return frame

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _draw_track(self, frame: np.ndarray, track: Track) -> np.ndarray:
        try:
            bbox = np.asarray(track.bbox, dtype=float).reshape(-1)
            if bbox.size < 4:
                return frame
            x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
        except Exception:
            return frame

        # Clamp to frame dimensions
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)

        if x2 <= x1 or y2 <= y1:
            return frame

        color = self._color_known if track.face_id else self._color_new
        thickness = 3

        # Bounding box
        try:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), max(3, thickness + 1), color, -1)
        except Exception:
            return frame

        # Label
        if self._show_face_id:
            label = track.face_id if track.face_id else f"Track-{track.track_id}"
            short = label[-12:] if len(label) > 12 else label   # truncate for display
            label_text = f"{short}"
            try:
                (tw, th), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 1
                )
            except Exception:
                return frame
            lx = x1
            ly = max(y1 - 6, th + 4)
            # Background pill
            try:
                cv2.rectangle(frame, (lx, ly - th - 8), (lx + tw + 12, ly + baseline + 4),
                              (20, 20, 20), -1)
                cv2.rectangle(frame, (lx, ly - th - 8), (lx + tw + 12, ly + baseline + 4),
                              color, 2)
                cv2.putText(frame, label_text, (lx + 6, ly - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.55, self._font_scale),
                            (255, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                return frame

        return frame

    def _draw_fps(self, frame: np.ndarray) -> np.ndarray:
        if not self._show_fps:
            return frame
        now  = time.time()
        fps  = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _draw_banner(self, frame: np.ndarray,
                     unique_count: int, active_count: int) -> np.ndarray:
        if not self._show_count:
            return frame
        h, w = frame.shape[:2]
        banner_h = 40
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        text = (f"  Unique Visitors: {unique_count}   |   "
                f"Active Tracks: {active_count}")
        cv2.putText(frame, text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def create_summary_frame(self, stats: dict, frame_size=(640, 360)) -> np.ndarray:
        """Create a dark summary card showing DB stats."""
        w, h = frame_size
        img  = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)

        lines = [
            "── FACE TRACKER SUMMARY ──",
            "",
            f"Unique Visitors (total):  {stats.get('unique_visitors_total', 0)}",
            f"Total Events Logged:      {stats.get('total_events_logged', 0)}",
            "",
            f"Today ({stats.get('today', {}).get('stat_date', '-')})",
            f"  Entries: {stats.get('today', {}).get('total_entries', 0)}",
            f"  Exits:   {stats.get('today', {}).get('total_exits', 0)}",
        ]
        y = 40
        for line in lines:
            color = (100, 220, 255) if "──" in line else (200, 200, 200)
            cv2.putText(img, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            y += 32
        return img
