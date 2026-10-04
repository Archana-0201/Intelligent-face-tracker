"""
tracker.py
----------
Face tracker that fuses YOLO detections with a lightweight IoU + Kalman Filter
based tracker (similar in spirit to SORT / ByteTrack but dependency-free).

Responsibilities:
  - Maintain a set of active tracks across frames
  - Associate new detections to existing tracks via IoU matching
  - Detect entry (new track created) and exit (track not updated for N frames)
  - Provide stable track IDs that map to recognised face IDs
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter

from src.logger import get_logger, log_face_tracked
from src.config_loader import get_config

logger = get_logger()


# ── Kalman state: [cx, cy, s, r, cx_dot, cy_dot, s_dot]
# cx,cy = centre; s = area; r = aspect ratio (kept constant)
def _make_kalman(bbox: Tuple[int, int, int, int]) -> KalmanFilter:
    """Create a Kalman filter initialised from a bbox (x1,y1,x2,y2)."""
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.array([
        [1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,0,1,0,0,0,1],
        [0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1],
    ], dtype=float)
    kf.H = np.array([
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0],
    ], dtype=float)
    kf.R[2:, 2:] *= 10.
    kf.P[4:, 4:] *= 1000.
    kf.P *= 10.
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.
    cy = (y1 + y2) / 2.
    s  = (x2 - x1) * (y2 - y1)
    r  = (x2 - x1) / float(y2 - y1 + 1e-6)
    kf.x[:4] = np.array([[cx], [cy], [s], [r]])
    return kf


def _bbox_from_state(x: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert a Kalman state vector back to (x1, y1, x2, y2)."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Kalman state is too short to form a bbox: {arr.shape}")

    cx, cy, s, r = [float(v) for v in arr[:4]]
    w = np.sqrt(max(0.0, s * r))
    h = s / (w + 1e-6)
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return x1, y1, x2, y2


def _iou(a: Tuple, b: Tuple) -> float:
    """Compute Intersection-over-Union for two bboxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-6)


@dataclass
class Track:
    """Single face track."""
    track_id:   int
    face_id:    Optional[str]   = None   # set once recognised
    kf:         KalmanFilter    = field(default=None, repr=False)
    bbox:       Tuple[int,int,int,int] = (0,0,0,0)
    age:        int             = 0      # frames since creation
    hits:       int             = 1      # consecutive matched frames
    misses:     int             = 0      # consecutive unmatched frames
    is_new:     bool            = True   # fired entry event?
    is_active:  bool            = True


class FaceTracker:
    """
    Lightweight SORT-inspired multi-face tracker.

    Each detected bounding box is matched to existing tracks via IoU.
    Unmatched detections start new tracks; unmatched tracks accumulate misses
    until they are deleted (exit event).
    """

    def __init__(self):
        cfg = get_config()
        trk = cfg.tracking

        self.max_disappeared = int(trk.get("max_disappeared", 30))
        self.iou_thr         = float(trk.get("iou_threshold", 0.3))
        self.kalman_enabled  = bool(trk.get("kalman_enabled", True))

        self._tracks:   Dict[int, Track] = {}
        self._next_tid: int              = 1

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, detections: List[Tuple[int,int,int,int,float]],
               frame_no: int = 0
               ) -> Tuple[List[Track], List[Track]]:
        """
        Update the tracker with new detections.

        Args:
            detections: List of (x1,y1,x2,y2,conf) from the detector.
            frame_no:   Current frame number (for logging).

        Returns:
            (active_tracks, exited_tracks)
              active_tracks  – list of Track objects currently visible
              exited_tracks  – tracks that have just been removed (exit event)
        """
        # 1. Predict next positions via Kalman
        if self.kalman_enabled:
            for t in self._tracks.values():
                if t.kf:
                    t.kf.predict()
                    t.bbox = tuple(int(v) for v in _bbox_from_state(t.kf.x))

        # 2. Build cost matrix (negative IoU)
        track_ids  = list(self._tracks.keys())
        det_bboxes = [(d[0], d[1], d[2], d[3]) for d in detections]

        matches, unmatched_dets, unmatched_trks = self._associate(
            track_ids, det_bboxes
        )

        # 3. Update matched tracks
        for tid, det_idx in matches:
            t = self._tracks[tid]
            bbox = tuple(int(v) for v in det_bboxes[det_idx])
            t.bbox   = bbox
            t.misses = 0
            t.hits  += 1
            t.age   += 1
            if self.kalman_enabled and t.kf:
                cx = (bbox[0]+bbox[2])/2.; cy = (bbox[1]+bbox[3])/2.
                s  = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                r  = (bbox[2]-bbox[0]) / float(bbox[3]-bbox[1]+1e-6)
                t.kf.update(np.array([[cx],[cy],[s],[r]]))
            log_face_tracked(t.face_id or str(tid), frame_no, bbox)

        # 4. Mark unmatched tracks
        for tid in unmatched_trks:
            t = self._tracks[tid]
            t.misses += 1
            t.age    += 1

        # 5. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox = tuple(int(v) for v in det_bboxes[det_idx])
            kf   = _make_kalman(bbox) if self.kalman_enabled else None
            new_track = Track(
                track_id=self._next_tid,
                kf=kf,
                bbox=bbox,
                is_new=True,
            )
            self._tracks[self._next_tid] = new_track
            self._next_tid += 1

        # 6. Collect exited tracks and remove them
        exited = []
        to_remove = []
        for tid, t in self._tracks.items():
            if t.misses > self.max_disappeared:
                exited.append(t)
                to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]

        active = list(self._tracks.values())
        return active, exited

    def assign_face_id(self, track_id: int, face_id: str):
        """Assign a recognised face_id to an existing track."""
        if track_id in self._tracks:
            self._tracks[track_id].face_id = face_id

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    # ── Association (greedy IoU matching) ─────────────────────────────────────

    def _associate(self, track_ids: List[int],
                   det_bboxes: List[Tuple]
                   ) -> Tuple[List, List, List]:
        """
        Match detections to tracks greedily by IoU (descending similarity).

        Returns:
            (matches, unmatched_det_indices, unmatched_track_ids)
        """
        if not track_ids or not det_bboxes:
            return [], list(range(len(det_bboxes))), list(track_ids)

        iou_matrix = np.zeros((len(track_ids), len(det_bboxes)), dtype=float)
        for i, tid in enumerate(track_ids):
            for j, dbbox in enumerate(det_bboxes):
                iou_matrix[i, j] = _iou(self._tracks[tid].bbox, dbbox)

        # Globally optimal one-to-one assignment avoids greedy identity swaps.
        row_indices, col_indices = linear_sum_assignment(1.0 - iou_matrix)
        pairs = []
        matched_trk = set()
        matched_det = set()
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] < self.iou_thr:
                continue
            pairs.append((track_ids[row], col))
            matched_trk.add(row)
            matched_det.add(col)

        unmatched_dets = [j for j in range(len(det_bboxes)) if j not in matched_det]
        unmatched_trks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_trk]

        return pairs, unmatched_dets, unmatched_trks

    @property
    def active_count(self) -> int:
        return len(self._tracks)
