"""
video_source.py
---------------
Unified video source abstraction supporting:
  - Local video files (MP4, AVI, MOV, …)
  - RTSP camera streams
  - Webcam (index 0)

Automatically switches between sources based on config.json settings.
"""

import cv2
import os
import subprocess
from typing import Optional, Tuple
import numpy as np

from src.config_loader import get_config
from src.logger import get_logger, log_system_event

logger = get_logger()


class VideoSource:
    """
    Context-manager-compatible video source wrapper.

    Usage:
        with VideoSource() as vs:
            for frame in vs:
                process(frame)
    """

    def __init__(self):
        cfg       = get_config()
        vid_cfg   = cfg.video

        self._use_rtsp     = bool(vid_cfg.get("use_rtsp", False))
        self._rtsp_url     = str(vid_cfg.get("rtsp_url", ""))
        self._file_path    = str(vid_cfg.get("source", "data/sample_video.mp4"))
        self._display      = bool(vid_cfg.get("display_output", True))
        self._save_output  = bool(vid_cfg.get("save_output_video", False))
        self._output_path  = str(vid_cfg.get("output_video", "output/output_tracked.mp4"))

        self._cap:    Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter]  = None
        self._frame_no = 0

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.release()

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise StopIteration
        self._frame_no += 1
        return frame

    # ── Open / Release ─────────────────────────────────────────────────────────

    def _open(self):
        """Open the video source (file, RTSP, or webcam)."""
        if self._use_rtsp and self._rtsp_url:
            source = self._rtsp_url
            log_system_event(f"Opening RTSP stream: {source}")
            self._cap = cv2.VideoCapture(source)
        elif str(self._file_path).strip() in ("0", "webcam") or str(self._file_path).isdigit():
            cam_idx = int(self._file_path) if str(self._file_path).isdigit() else 0
            log_system_event(f"Opening live webcam (device {cam_idx})")
            self._cap = cv2.VideoCapture(cam_idx)
        else:
            source = os.path.abspath(self._file_path)
            if not os.path.exists(source):
                raise FileNotFoundError(
                    f"Video file not found: {source}\n"
                    f"Please set the correct path in config.json -> video.source"
                )
            log_system_event(f"Opening video file: {source}")
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self._file_path}")

        log_system_event(
            f"Video opened | FPS={self.fps:.1f} | "
            f"Resolution={self.width}x{self.height} | "
            f"Total frames~{self.total_frames}"
        )

        if self._save_output:
            self._init_writer()

    def _init_writer(self):
        """Initialise the output video writer."""
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._output_path, fourcc, max(self.fps, 20.0),
            (self.width, self.height)
        )
        log_system_event(f"Output video writer → {self._output_path}")

    def release(self):
        """Release all resources gracefully."""
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._writer:
            self._writer.release()
            self._writer = None
            self._transcode_output_for_browser()
        cv2.destroyAllWindows()
        log_system_event("Video source released.")

    def _transcode_output_for_browser(self):
        """Convert the OpenCV MP4V output to browser-compatible H.264."""
        if not self._save_output or not os.path.isfile(self._output_path):
            return

        temp_path = os.path.splitext(self._output_path)[0] + "_h264.mp4"
        try:
            import imageio_ffmpeg

            result = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", self._output_path, "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", temp_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not os.path.isfile(temp_path):
                raise RuntimeError(result.stderr.strip() or "ffmpeg produced no output")
            os.replace(temp_path, self._output_path)
            log_system_event("Output video converted to browser-compatible H.264.")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Could not convert output video to H.264: {e}")

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        return 25.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 640

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 480

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else -1

    @property
    def frame_no(self) -> int:
        return self._frame_no

    # ── Write helper ───────────────────────────────────────────────────────────

    def write_frame(self, frame: np.ndarray):
        """Write an annotated frame to the output video."""
        if self._writer:
            self._writer.write(frame)

    def show(self, frame: np.ndarray, window_name: str = "Face Tracker"):
        """Display frame in an OpenCV window (only if display_output=true)."""
        if self._display:
            cv2.imshow(window_name, frame)
