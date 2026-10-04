"""
pipeline.py
-----------
Central orchestration pipeline for the Intelligent Face Tracker.

Processing loop (per frame):
  1. Read frame from VideoSource
  2. [Every N frames] Run YOLOv8 face detection
  3. Update Kalman-based tracker with detections
  4. For new tracks → try recognition / registration via FaceRegistry
  5. Log entry events for newly assigned face IDs
  6. Log exit events for tracks that disappeared
  7. Annotate frame with Visualiser
  8. Display / write output

The `frame_skip` parameter from config.json controls detection frequency.
"""

import os
import sys
import signal
import cv2
import numpy as np
from typing import Set

from src.config_loader import get_config
from src.logger        import setup_logger, get_logger, log_system_event
from src.database      import Database
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_registry import FaceRegistry
from src.tracker       import FaceTracker, Track
from src.event_logger  import EventLogger
from src.visualiser    import Visualiser
from src.video_source  import VideoSource


class Pipeline:
    """
    End-to-end face tracking pipeline.

    Orchestrates all sub-modules and manages the main processing loop.
    Handles graceful shutdown on SIGINT / SIGTERM.
    """

    def __init__(self, config_path: str = "config.json"):
        # ── Config & logging ───────────────────────────────────────────────────
        self._cfg = get_config(config_path)
        log_cfg   = self._cfg.logging_cfg
        setup_logger(
            log_file  = log_cfg.get("log_file", "logs/events.log"),
            log_level = log_cfg.get("log_level", "INFO"),
        )
        self._logger = get_logger()

        # ── Sub-modules ────────────────────────────────────────────────────────
        self._db         = Database(self._cfg.database.get("path", "data/face_tracker.db"))
        self._detector   = FaceDetector()
        self._recogniser = FaceRecognizer()
        self._registry   = FaceRegistry(self._db, self._recogniser)
        self._tracker    = FaceTracker()
        self._ev_logger  = EventLogger(self._db)
        self._visualiser = Visualiser()

        # ── Runtime state ──────────────────────────────────────────────────────
        self._running       = True
        self._frame_no      = 0
        self._tracks_seen:  Set[int]  = set()   # track_ids ever entered
        self._entry_fired:  Set[str]  = set()   # face_ids that fired entry event
        self._rec_interval  = max(1, self._cfg.frame_skip * 3)  # recognition cadence

        # Graceful shutdown
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(self):
        """Start the processing loop."""
        log_system_event("Pipeline starting ...")
        display_enabled = self._cfg.video.get("display_output", True)

        try:
            with VideoSource() as vs:
                for frame in vs:
                    if not self._running:
                        break
                    self._frame_no = vs.frame_no
                    annotated = self._process_frame(frame)

                    if self._cfg.video.get("save_output_video", False):
                        vs.write_frame(annotated)

                    vs.show(annotated)

                    # 'q' to quit, 's' to print stats
                    if display_enabled:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        elif key == ord("s"):
                            self._print_stats()

        except FileNotFoundError as e:
            self._logger.error(str(e))
        except Exception as e:
            self._logger.exception(f"Unhandled error in pipeline: {e}")
        finally:
            self._finalise()

    # ── Frame processing ───────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the full pipeline.

        Returns:
            Annotated BGR frame ready for display / recording.
        """
        fn = self._frame_no

        # 1. Detect faces (respects frame_skip internally)
        detections = self._detector.detect(frame)

        # 2. Update tracker
        active_tracks, exited_tracks = self._tracker.update(detections, fn)

        # 3. Handle newly disappeared tracks (EXIT events)
        for exited in exited_tracks:
            face_id = exited.face_id
            if face_id and face_id in self._entry_fired:
                face_crop = self._detector.crop_face(frame, exited.bbox)
                self._ev_logger.on_exit(face_id, face_crop, fn)
                self._entry_fired.discard(face_id)
            self._registry.on_track_exit(exited.track_id)

        # 4. Recognise / register new / unidentified tracks
        for track in active_tracks:
            # Mark brand-new tracks as needing recognition
            if track.track_id not in self._tracks_seen:
                self._tracks_seen.add(track.track_id)
                self._registry.mark_pending(track.track_id)

            # Attempt recognition on the scheduled interval
            if (track.face_id is None and
                    fn % self._rec_interval == 0):
                face_crop = self._detector.crop_face(frame, track.bbox)
                if face_crop is not None:
                    face_id, is_new, similarity = self._registry.try_recognise(
                        track.track_id, face_crop, fn
                    )
                    if face_id:
                        # Assign face_id to the tracker track object
                        self._tracker.assign_face_id(track.track_id, face_id)
                        track.face_id = face_id   # update local ref too

                        # Fire entry event once per face_id per session
                        if face_id not in self._entry_fired:
                            self._entry_fired.add(face_id)
                            if not is_new:
                                self._db.increment_visit_count(face_id)
                            entry_crop = self._detector.crop_face(frame, track.bbox)
                            self._ev_logger.on_entry(
                                face_id, entry_crop, fn, similarity, is_new
                            )

        # 5. Draw HUD
        annotated = self._visualiser.draw(
            frame.copy(), active_tracks,
            unique_count=self._registry.unique_count,
            frame_no=fn
        )

        return annotated

    # ── Stats & shutdown ───────────────────────────────────────────────────────

    def _print_stats(self):
        stats = self._db.get_stats_summary()
        self._logger.info(f"[STATS] {stats}")
        print("\n" + "="*50)
        print(f"  Unique Visitors : {stats['unique_visitors_total']}")
        print(f"  Total Events    : {stats['total_events_logged']}")
        print(f"  Today           : {stats.get('today', {})}")
        print("="*50 + "\n")

    def _finalise(self):
        """Print final summary and clean up."""
        stats = self._db.get_stats_summary()
        log_system_event(f"Pipeline stopped after {self._frame_no} frames.")
        log_system_event(f"Final stats: {stats}")
        self._db.close()
        print("\n" + "="*60)
        print("  INTELLIGENT FACE TRACKER -- SESSION COMPLETE")
        print("="*60)
        print(f"  Frames processed   : {self._frame_no}")
        print(f"  Unique visitors    : {stats['unique_visitors_total']}")
        print(f"  Total events logged: {stats['total_events_logged']}")
        print(f"  DB path            : {os.path.abspath(self._db.db_path)}")
        print("="*60 + "\n")

    def _shutdown(self, sig, frame):
        """Signal handler for graceful shutdown."""
        log_system_event(f"Shutdown signal received ({sig}). Stopping …")
        self._running = False
