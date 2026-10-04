# 🎯 Intelligent Face Tracker with Auto-Registration & Visitor Counting

> **Real-time AI-powered face detection, recognition, tracking, and unique visitor analytics**

---

## 📋 Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [AI Planning Document](#ai-planning-document)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Compute Estimates](#compute-load-estimates)
- [Sample Output](#sample-output)
- [Assumptions](#assumptions)
- [Demo Video](#demo-video)

---

## ✅ Features

| Module | Description |
|--------|-------------|
| 🔍 **Face Detection** | YOLOv8n-face model with configurable confidence & frame-skip |
| 🧠 **Face Recognition** | InsightFace ArcFace 512-D embeddings, cosine similarity matching |
| 🏃 **Multi-Face Tracking** | SORT-inspired Kalman Filter tracker with IoU-based assignment |
| 📝 **Auto-Registration** | New faces get unique IDs (e.g. `FACE_20261003_A3F2B1C9`) and stored immediately |
| 📸 **Event Logging** | Entry and exit events with cropped face images saved to `logs/entries/` and `logs/exits/` |
| 🗄️ **SQLite Database** | WAL-mode crash-resilient DB tracking faces, events, and daily stats |
| 📊 **Visitor Counting** | Counts registered identities; recognition accuracy depends on face visibility and image quality |
| 🖥️ **Visual HUD** | Real-time bounding boxes, face IDs, FPS, and visitor count overlay |
| 🌐 **Interactive Dashboard** | Streamlit web UI with video uploader, face gallery, SQLite explorer & video playback |
| 🎥 **RTSP Support** | Switch from file to live camera with one config flag |
| ⚙️ **Config-driven** | All parameters exposed in `config.json` — zero code changes needed |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     VIDEO SOURCE                               │
│          (MP4 file  /  RTSP stream  /  Webcam)                 │
└───────────────────────────┬────────────────────────────────────┘
                            │ frame (BGR)
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  FACE DETECTOR  (YOLOv8n-face)                                 │
│  • Runs every `frame_skip` frames (configurable)               │
│  • Returns list of (x1,y1,x2,y2,conf) bounding boxes          │
└───────────────────────────┬────────────────────────────────────┘
                            │ detections
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  FACE TRACKER  (Kalman + IoU assignment)                       │
│  • Maintains stable track IDs across frames                    │
│  • Outputs: active_tracks, exited_tracks                       │
└──────────────┬─────────────────────┬──────────────────────────┘
               │ new tracks          │ exited tracks
               ▼                     ▼
┌──────────────────────┐   ┌──────────────────────────────────┐
│  FACE RECOGNIZER     │   │  EVENT LOGGER (Exit)             │
│  (InsightFace ArcFace│   │  • Save exit face crop           │
│   512-D embedding)   │   │  • DB log_event('exit')          │
│  • Match vs registry │   │  • events.log entry              │
│  • or: NEW → register│   └──────────────────────────────────┘
└──────────┬───────────┘
           │ face_id (new or existing)
           ▼
┌────────────────────────────────────────────────────────────────┐
│  FACE REGISTRY                                                 │
│  • Maps track_id → face_id                                     │
│  • Unique visitor set (never double-counts)                    │
│  • Coordinates DB writes + recogniser cache                    │
└──────────┬─────────────────────────────────────────────────────┘
           │ on new face_id assigned
           ▼
┌────────────────────────────────────────────────────────────────┐
│  EVENT LOGGER (Entry)                                          │
│  • Save entry face crop → logs/entries/YYYY-MM-DD/             │
│  • DB log_event('entry')                                       │
│  • events.log entry                                            │
└────────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────┐
│  VISUALISER                                                    │
│  • Draw bboxes, face IDs, FPS, visitor count on frame          │
└────────────────────────────────────────────────────────────────┘
           │
           ▼
  Display window  /  Output MP4  /  events.log  /  SQLite DB
```

---

## 🧠 AI Planning Document

### Phase 1 — Planning

**Goal:** Build a real-time face tracking system that:
1. Detects faces using state-of-the-art YOLO
2. Generates discriminative embeddings via ArcFace
3. Tracks faces across frames with a Kalman Filter
4. Logs every entry/exit with images to disk and metadata to DB
5. Accurately counts unique visitors (no double-counting)

**Key Design Decisions:**
- **Separation of concerns:** Each module (detect / recognise / track / log) is independent and testable
- **Frame-skip for performance:** Detection only runs every N frames; tracker fills the gaps
- **Recognition on entry only:** ArcFace inference is expensive — run once per new track, not every frame
- **DB-first persistence:** SQLite with WAL mode ensures events survive crashes
- **Configurable via JSON:** No code changes for tuning thresholds or frame-skip

### Phase 2 — Feature List

| # | Feature | Priority |
|---|---------|----------|
| 1 | YOLOv8 face detection with configurable frame-skip | P0 |
| 2 | InsightFace ArcFace 512-D embedding extraction | P0 |
| 3 | Cosine similarity matching against registered faces | P0 |
| 4 | Auto-registration with unique face IDs | P0 |
| 5 | Kalman Filter multi-face tracker | P0 |
| 6 | Entry event: 1 per unique face, with image + DB record | P0 |
| 7 | Exit event: 1 per track disappearance, with image + DB | P0 |
| 8 | Rotating events.log file | P0 |
| 9 | Unique visitor count (DB-backed) | P0 |
| 10 | RTSP stream support | P1 |
| 11 | Output video recording | P1 |
| 12 | Real-time HUD overlay (FPS, count, bboxes) | P1 |
| 13 | Daily visitor stats in DB | P1 |
| 14 | CLI flags (--stats, --reset-db, --video, --rtsp) | P2 |

### Phase 3 — Implementation Order

```
1. config.json + ConfigLoader   → Foundation
2. logger.py                    → Observability
3. database.py                  → Persistence layer
4. face_detector.py             → YOLO detection
5. face_recognizer.py           → ArcFace embeddings
6. tracker.py                   → Kalman tracking
7. face_registry.py             → Identity management
8. event_logger.py              → Entry/exit events
9. video_source.py              → Video I/O
10. visualiser.py               → HUD display
11. pipeline.py                 → Orchestration
12. main.py                     → CLI entry point
```

---

## 🛠️ Tech Stack

| Module | Technology |
|--------|-----------|
| Face Detection | YOLOv8n-face (Ultralytics) |
| Face Recognition | InsightFace ArcFace (buffalo_sc model pack) |
| Tracking | Custom SORT (Kalman Filter + IoU, using filterpy) |
| Backend | Python 3.10+ |
| Database | SQLite (WAL mode, thread-safe) |
| Configuration | JSON (`config.json`) |
| Logging | Python `logging` module (rotating file + console) |
| Video I/O | OpenCV (`cv2`) |
| Embedding Math | NumPy (cosine similarity) |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10 or newer
- pip 23+
- (Optional) NVIDIA GPU + CUDA 11.8+ for faster inference

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd "Intelligent Face Tracker"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** On first run, the YOLOv8n-face model (~6 MB) and InsightFace
> `buffalo_sc` models (~85 MB) are downloaded by the model setup step below.

### 3. Download and verify the models
```bash
python download_models.py
```

This downloads the face-specific YOLO checkpoint and prepares InsightFace.
The app rejects a YOLO checkpoint that is not labeled as a face detector.

### 4. Add your video file
Place a video file at:
```
data/sample_video1.mp4
```
The hackathon sample is available in the
[provided Google Drive folder](https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing).
Or update `config.json` `video.source` to point to your file. The dashboard
lists supported video files found directly inside `data/`.

### 5. Run
```bash
python main.py
```

Press **`q`** to quit, **`s`** to print live statistics.

---

## ⚙️ Configuration

### `config.json` — Sample Structure

```json
{
  "video": {
    "source": "data/sample_video1.mp4",
    "rtsp_url": "",
    "use_rtsp": false,
    "display_output": true,
    "output_video": "output/output_tracked.mp4",
    "save_output_video": true
  },
  "detection": {
    "model_path": "models/yolov8n-face.pt",
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "frame_skip": 2,
    "input_size": 1280
  },
  "recognition": {
    "model_name": "buffalo_sc",
    "similarity_threshold": 0.45,
    "embedding_size": 512,
    "min_face_size": 30
  },
  "tracking": {
    "max_disappeared": 30,
    "max_distance": 0.6,
    "iou_threshold": 0.3,
    "kalman_enabled": true
  },
  "database": {
    "type": "sqlite",
    "path": "data/face_tracker.db"
  },
  "logging": {
    "log_file": "logs/events.log",
    "log_level": "INFO",
    "image_format": "jpg",
    "image_quality": 90
  },
  "ui": {
    "show_fps": true,
    "show_face_id": true,
    "show_bbox": true,
    "show_visitor_count": true,
    "bbox_color_new": [0, 255, 0],
    "bbox_color_known": [255, 165, 0],
    "font_scale": 0.6
  }
}
```

**Key configurable parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `detection.frame_skip` | Skip N-1 frames between YOLO detection runs | `2` |
| `detection.confidence_threshold` | Minimum face detection confidence | `0.25` |
| `detection.input_size` | YOLO inference size; higher values help with smaller/distant faces at greater processing cost | `1280` |
| `recognition.similarity_threshold` | Minimum cosine similarity to consider a match | `0.45` |
| `tracking.max_disappeared` | Frames before a track is considered exited | `30` |
| `video.use_rtsp` | Switch to RTSP live stream mode | `false` |

---

## 🚀 Usage

```bash
# Default run (uses config.json)
python main.py

# Launch the Interactive Web Dashboard
streamlit run app.py
# (or double-click run_dashboard.bat on Windows)

# Override video source
python main.py --video path/to/video.mp4

# Live webcam mode
python main.py --webcam

# Use RTSP live stream
python main.py --rtsp rtsp://192.168.1.100:554/stream

# Headless mode (no display window)
python main.py --no-display

# Print DB statistics and exit
python main.py --stats

# Reset/wipe the database
python main.py --reset-db
```

---

## 📁 Project Structure

```
Intelligent Face Tracker/
├── main.py                  # CLI entry point
├── app.py                   # Streamlit dashboard
├── config.json              # All configuration parameters
├── requirements.txt         # Python dependencies
├── download_models.py       # Download and validate model weights
├── README.md                # This file
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py     # Singleton JSON config loader
│   ├── logger.py            # Rotating file + console logger
│   ├── database.py          # SQLite (WAL-mode) persistence layer
│   ├── face_detector.py     # YOLOv8n-face detection + frame-skip
│   ├── face_recognizer.py   # InsightFace ArcFace embeddings + matching
│   ├── tracker.py           # Kalman Filter multi-face tracker (SORT)
│   ├── face_registry.py     # Identity management + unique visitor count
│   ├── event_logger.py      # Entry/exit image saving + DB logging
│   ├── visualiser.py        # OpenCV HUD overlay
│   ├── video_source.py      # Video file / RTSP / webcam abstraction
│   └── pipeline.py          # Main orchestration loop
│
├── models/
│   └── yolov8n-face.pt      # Auto-downloaded on first run
│
├── data/
│   ├── sample_video1.mp4    # Hackathon sample video
│   └── face_tracker.db      # SQLite database (auto-created)
│
├── logs/
│   ├── events.log           # Rotating text event log
│   ├── entries/
│   │   └── YYYY-MM-DD/      # Entry face crops per day
│   └── exits/
│       └── YYYY-MM-DD/      # Exit face crops per day
│
└── output/
    └── output_tracked.mp4   # Annotated output video
```

---

## 📊 Compute Load Estimates

| Execution path | Measured / expected compute |
|---|---|
| CPU-only (tested Windows machine) | The 60-second, 1502-frame, 2688×1520 sample took about 4.5–6 minutes including H.264 encoding (roughly 4.5–6× slower than real time). CPU utilization percentage was not benchmarked. |
| GPU | Ultralytics can use CUDA with a compatible GPU/PyTorch installation. InsightFace is configured for CPU ONNX inference, so recognition remains CPU-bound. GPU utilization and end-to-end speed were not benchmarked. |

The default 1280-pixel inference size favors small faces over throughput.
Runtime varies with hardware, video resolution, and `frame_skip`; real-time
processing is not guaranteed.

---

## 📸 Sample Output

### Console / Log Output
```
2026-10-04 10:32:54 | INFO | FaceTracker | Video opened | FPS=25.0 | Resolution=2688x1520 | Total frames~1502
2026-10-04 | INFO | FaceTracker | [REGISTRY] track=1 → NEW face_id=<generated face ID> (total unique=1)
2026-10-04 | INFO | FaceTracker | [EVENT] entry | face_id=<generated face ID> | frame=6 | img=<saved entry crop>
2026-10-04 10:38:32 | INFO | FaceTracker | Pipeline stopped after 1502 frames.
2026-10-04 10:38:32 | INFO | FaceTracker | Final stats: unique_visitors_total=11, total_events_logged=22
```

The clean sample run registered 11 visitor IDs across the clip and logged 12
entry and 10 exit events. These are counts across the video, not simultaneous
people in one frame.

### Database Schema (SQLite)

```sql
-- Registered faces
SELECT id, created_at, visit_count FROM faces;
-- <face_id> | <created_at> | <visit_count>

-- Events
SELECT face_id, event_type, timestamp, image_path FROM events;
-- <face_id> | entry | <timestamp> | logs/entries/YYYY-MM-DD/<face_id>_<time>.jpg
-- <face_id> | exit  | <timestamp> | logs/exits/YYYY-MM-DD/<face_id>_<time>.jpg

-- Daily stats
SELECT * FROM visitor_stats;
-- 2026-10-04 | 11 unique | 12 entries | 10 exits
```

### Log Image Structure
```
logs/
  entries/
    YYYY-MM-DD/
      <face_id>_<time>.jpg  ← variable-size face crop with margin
  exits/
    YYYY-MM-DD/
      <face_id>_<time>.jpg
```

---

## 📝 Assumptions

1. **Face size**: Faces smaller than 30×30 pixels are skipped during recognition (too blurry for reliable embedding).
2. **Single camera**: The system is designed for a single camera feed at a time.
3. **Lighting**: Assumes reasonable indoor lighting. Extreme low-light or backlit conditions may reduce accuracy.
4. **Similarity threshold**: Default of 0.45 cosine similarity. Higher values are stricter and may cause more re-registrations; lower values may increase false matches. Tune using labeled footage.
5. **RTSP stream**: For live RTSP use, set `video.use_rtsp: true` and provide the URL in `video.rtsp_url`.
6. **InsightFace models**: Auto-downloaded on first run to `~/.insightface/`. Internet required on first launch.
7. **Frame-skip**: Detection runs every `frame_skip` frames (default 2). Tracker Kalman prediction fills intermediate frames.
8. **Unique visitor definition**: The count is the number of registered identities in the persistent database. A recognized re-entry is logged as an event without creating another identity; recognition can still split one person into multiple IDs when image quality is poor.
9. **Video FPS**: If source FPS is unavailable (some RTSP streams), defaults to 25 FPS.
10. **Database persistence**: The SQLite DB survives restarts; registered embeddings and recent entry-photo references are loaded at startup. Resetting the DB clears registered identities and event records.

---

## 🎬 Demo Video

Link: https://youtu.be/X_Mjx37kbfQ

---

## 👨‍💻 Author

Built using AI-assisted development (Google Antigravity / Gemini) for the Katomaran Hackathon 2026.

All generated code was reviewed, understood, and validated by the author.
Key prompts and design decisions are documented in this README for interview reference.

---

This project is a part of a hackathon run by https://katomaran.com
