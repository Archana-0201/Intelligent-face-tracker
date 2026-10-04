# -*- coding: utf-8 -*-
"""
download_models.py
------------------
Pre-download all required AI models before running the tracker.

Run this ONCE after installing requirements:
    D:\\python313\\python.exe download_models.py

Downloads:
  1. YOLOv8n-face weights -> models/yolov8n-face.pt
  2. InsightFace buffalo_sc model pack -> ~/.insightface/models/buffalo_sc/

Total download size: ~95 MB
"""

import os
import sys
import shutil


def download_yolo_face():
    """Download and validate the YOLOv8n face-specific checkpoint."""
    dest = os.path.abspath("models/yolov8n-face.pt")
    os.makedirs("models", exist_ok=True)
    candidate = dest + ".download"

    if os.path.exists(dest):
        try:
            from ultralytics import YOLO
            names = YOLO(dest).names
            if any("face" in str(name).lower() for name in names.values()):
                print("[OK]  Verified face detector:", dest)
                return
            print("[WARN] Existing checkpoint is not a face detector:", names)
        except Exception as e:
            print("[WARN] Existing checkpoint could not be validated:", e)

    print("Downloading YOLOv8n-face ...")
    try:
        import requests
        url = (
            "https://github.com/akanametov/yolo-face/releases/download/"
            "1.0.0/yolov8n-face.pt"
        )
        response = requests.get(url, stream=True, timeout=90)
        response.raise_for_status()
        with open(candidate, "wb") as model_file:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    model_file.write(chunk)

        from ultralytics import YOLO
        names = YOLO(candidate).names
        if not any("face" in str(name).lower() for name in names.values()):
            raise ValueError(f"Downloaded weights are not face-specific: {names}")
        if os.path.exists(dest):
            backup = dest.replace(".pt", "-previous.pt")
            if not os.path.exists(backup):
                shutil.copy2(dest, backup)
        os.replace(candidate, dest)
        print("[OK]  Verified face checkpoint saved:", dest)
    except ImportError:
        print("[ERROR] Install ultralytics and requests before downloading.")
        sys.exit(1)
    except Exception as e:
        if os.path.exists(candidate):
            os.remove(candidate)
        print("[ERROR] Could not download a verified face model:", e)
        sys.exit(1)


def download_insightface():
    """Trigger InsightFace model download."""
    print("Downloading InsightFace buffalo_sc models ...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(320, 320))
        print("[OK]  InsightFace models ready.")
    except ImportError:
        print("[ERROR] insightface not installed. Run: pip install insightface")
        sys.exit(1)
    except Exception as e:
        print("[WARN] InsightFace download encountered:", e)
        print("       Models may download automatically on first run.")


if __name__ == "__main__":
    print("=" * 55)
    print("  Intelligent Face Tracker - Model Downloader")
    print("=" * 55)
    download_yolo_face()
    download_insightface()
    print()
    print("All models ready! You can now run:")
    print("  D:\\python313\\python.exe main.py")
    print("  -- or --")
    print("  run.bat")
    print("=" * 55)
