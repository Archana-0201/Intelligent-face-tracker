"""
face_detector.py
----------------
YOLOv8-based face detection module.

Uses the `ultralytics` YOLOv8n-face model (or a user-supplied custom path) to
detect faces in video frames.  Returns normalised bounding boxes, confidence
scores, and (optional) landmark arrays.

Frame-skip logic
----------------
The configurable `frame_skip` parameter (from config.json) controls how many
frames are skipped between full detection runs.  On skipped frames the last
detection result is returned unchanged, significantly reducing CPU load.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional
import cv2

from src.logger import get_logger
from src.config_loader import get_config

logger = get_logger()


class FaceDetector:
    """
    Real-time face detector backed by YOLOv8-face.

    Attributes:
        model_path (str):  Path to the YOLO weights file.
        conf_thr   (float):Minimum detection confidence.
        iou_thr    (float):NMS IoU threshold.
        frame_skip (int):  Skip N-1 frames between detections.
    """

    # Public download URL for the lightweight yolov8n-face weights
    MODEL_URL = (
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/"
        "yolov8n-face.pt"
    )

    def __init__(self):
        cfg = get_config()
        det = cfg.detection

        self.model_path  = det.get("model_path", "models/yolov8n-face.pt")
        self.conf_thr    = float(det.get("confidence_threshold", 0.25))
        self.iou_thr     = float(det.get("iou_threshold", 0.45))
        self.frame_skip  = int(det.get("frame_skip", 2))
        self.input_size  = int(det.get("input_size", 1280))

        self._model = None
        self._frame_counter = 0
        self._last_detections: List[Tuple] = []

        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self):
        """Load YOLOv8-face model, or fallback to high-speed ONNX face detector if torch is unavailable."""
        self._onnx_detector = None
        self._use_onnx = False

        # Attempt 1: Try YOLOv8 via ultralytics
        try:
            from ultralytics import YOLO
            abs_model_path = os.path.abspath(self.model_path)
            if not os.path.exists(abs_model_path):
                abs_model_path = "yolov8n.pt"
            self._model = YOLO(abs_model_path)
            model_names = self._model.names if hasattr(self._model, 'names') else {}
            self._is_face_model = any('face' in str(v).lower() for v in model_names.values())
            if not self._is_face_model:
                raise ValueError(
                    f"Expected a face detector, but {abs_model_path} contains classes {model_names}. "
                    "Replace it with the face checkpoint using download_models.py."
                )
            logger.info(f"FaceDetector loaded (YOLO): {abs_model_path}")
            return
        except Exception as e:
            logger.warning(f"Could not load YOLO via PyTorch ({e}). Falling back to ONNX Face Detector.")

        # Attempt 2: Use InsightFace's built-in ONNX detector & ArcFace recogniser
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(self.input_size, self.input_size))
            self._insightface_app = app
            self._use_onnx = True
            logger.info("FaceDetector loaded: ONNX InsightFace (SCRFD + ArcFace) (CPU)")
        except Exception as e2:
            logger.error(f"Failed to load both YOLO and ONNX face detectors: {e2}")
            raise

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """
        Detect faces in a BGR frame and extract normalized ArcFace embeddings.

        Implements the configured frame-skip: only runs inference every
        `frame_skip` frames; returns cached results on intermediate frames.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of (x1, y1, x2, y2, confidence, embedding) tuples.
        """
        self._frame_counter += 1

        # Return cached result on skipped frames
        if self._frame_counter % self.frame_skip != 0:
            return self._last_detections

        detections = []

        if self._use_onnx and hasattr(self, '_insightface_app'):
            try:
                faces = self._insightface_app.get(frame)
                if faces:
                    for f in faces:
                        conf = float(f.det_score)
                        if conf >= self.conf_thr:
                            x1, y1, x2, y2 = [int(v) for v in f.bbox]
                            # Filter out tiny specks (< 30px)
                            if (x2 - x1) >= 28 and (y2 - y1) >= 28:
                                emb = f.normed_embedding.astype(np.float32) if hasattr(f, 'normed_embedding') and f.normed_embedding is not None else None
                                detections.append((x1, y1, x2, y2, conf, emb))
            except Exception as e:
                logger.error(f"Error in ONNX face detection: {e}")
        else:
            # YOLO detection
            results = self._model.predict(
                source=frame,
                conf=self.conf_thr,
                iou=self.iou_thr,
                imgsz=self.input_size,
                max_det=1000,
                verbose=False,
                stream=False,
            )
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append((x1, y1, x2, y2, conf, None))

        self._last_detections = detections
        return detections

    def crop_face(self, frame: np.ndarray,
                  bbox: Tuple[int, int, int, int],
                  margin: float = 0.15) -> Optional[np.ndarray]:
        """
        Crop and return a face region from the frame with an optional margin.

        Args:
            frame:  Source BGR image.
            bbox:   (x1, y1, x2, y2) bounding box.
            margin: Fractional margin to add around the box.

        Returns:
            Cropped BGR face image, or None if bbox is invalid.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin), int(bh * margin)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
