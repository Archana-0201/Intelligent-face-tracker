"""
app.py
------
Streamlit Interactive Dashboard for the Intelligent Face Tracker.

Features:
  - Real-time Analytics (Unique Visitors, Total Entries, Total Exits)
  - Video Selector / Uploader (Select sample video, upload custom file, or webcam/RTSP)
  - One-click Pipeline Execution with progress feedback
  - Face Gallery: Visual cards of all registered visitors with their cropped photos & IDs
  - Event Log Explorer: Searchable SQLite events database
  - Output Video Player: Watch the annotated tracking results
  - Live system log viewer (events.log)
"""

import streamlit as st
import os
import sqlite3
import pandas as pd
import sys
import json
from PIL import Image
from datetime import datetime

# Set page layout
st.set_page_config(
    page_title="Intelligent Face Tracker | Katomaran Hackathon",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.path.abspath("data/face_tracker.db")
LOG_PATH = os.path.abspath("logs/events.log")
OUTPUT_VIDEO_PATH = os.path.abspath("output/output_tracked.mp4")

# ── Custom CSS for Modern Dark UI ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #eef5fb 0%, #eaf2f8 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .stSidebar {
        width: 310px !important;
        min-width: 310px !important;
        max-width: 310px !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 0.8rem;
        padding-left: 0.7rem;
        padding-right: 0.7rem;
    }
    .stFileUploader > div {
        min-height: 70px;
        min-width: 100%;
        width: 100%;
    }
    .stVideo {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #d7e7f8;
    }
    .metric-card {
        background: linear-gradient(135deg, #11253b 0%, #0b1f2d 100%);
        border: 1px solid #2b3c4d;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 5px 12px rgba(15, 23, 42, 0.12);
        min-height: 110px;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 6px;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #a6c0d9;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .face-card {
        background: #f8fbff;
        border: 1px solid #dfeaf6;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


def get_db_stats():
    """Fetch summary metrics from the SQLite database."""
    if not os.path.exists(DB_PATH):
        return {"unique_visitors": 0, "entries": 0, "exits": 0, "events": 0}
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM faces")
        unique_cnt = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM events WHERE event_type='entry'")
        entry_cnt = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM events WHERE event_type='exit'")
        exit_cnt = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM events")
        total_events = c.fetchone()[0]
        conn.close()
        return {
            "unique_visitors": unique_cnt,
            "entries": entry_cnt,
            "exits": exit_cnt,
            "events": total_events
        }
    except Exception:
        return {"unique_visitors": 0, "entries": 0, "exits": 0, "events": 0}


def get_faces():
    """Fetch registered faces with metadata."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, visit_count, face_image FROM faces ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_events(limit=100):
    """Fetch recent entry/exit events."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            f"SELECT event_id, face_id, event_type, timestamp, similarity, image_path "
            f"FROM events ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎯 Intelligent Face Tracker & Visitor Counter")
st.caption("AI-driven unique visitor counter with Auto-Registration, ArcFace Recognition, and SORT Tracking")

# ── Sidebar Controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")

    video_options = []
    # Find existing videos in data/
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_options.append(os.path.join("data", f))

    source_type = st.radio("Video Source", ["Sample Video", "Upload Video", "Live Webcam", "RTSP Stream"])

    selected_video = None
    if source_type == "Sample Video":
        if video_options:
            selected_video = st.selectbox("Select Video File", video_options)
        else:
            st.info("No sample videos found in data/ folder.")
    elif source_type == "Upload Video":
        uploaded = st.file_uploader("Upload MP4 / AVI", type=["mp4", "avi", "mov"])
        if uploaded:
            dest = os.path.join("data", uploaded.name)
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            selected_video = dest
            st.success(f"Uploaded: {uploaded.name}")
    elif source_type == "Live Webcam":
        selected_video = "0"
    elif source_type == "RTSP Stream":
        selected_video = st.text_input("RTSP URL", placeholder="rtsp://admin:password@ip:554/stream")

    st.markdown("---")
    st.subheader("Thresholds")
    frame_skip = st.slider("Frame Skip (Detection frequency)", 1, 10, 2)
    conf_thr = st.slider("Detection Confidence", 0.05, 0.9, 0.25, 0.05)
    sim_thr = st.slider("Face Similarity Threshold (ArcFace)", 0.2, 0.8, 0.45, 0.05)

    st.markdown("---")
    col_run, col_reset = st.columns(2)
    run_btn = col_run.button("🚀 Run Tracker", use_container_width=True, type="primary")
    reset_btn = col_reset.button("🗑️ Reset DB", use_container_width=True)

    if reset_btn:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        from src.database import Database
        Database(DB_PATH).close()
        st.success("Database reset!")
        st.rerun()

# ── Run Pipeline if Triggered ──────────────────────────────────────────────────
if run_btn and selected_video:
    st.info(f"Starting tracking pipeline on `{selected_video}`...")
    progress_bar = st.progress(0, text="Initializing AI models (SCRFD + ArcFace)...")

    # Persist the current sliders into the runtime config before executing.
    runtime_cfg = {
        "video": {
            "source": str(selected_video),
            "rtsp_url": str(selected_video) if source_type == "RTSP Stream" else "",
            "use_rtsp": source_type == "RTSP Stream",
            "display_output": False,
            "output_video": "output/output_tracked.mp4",
            "save_output_video": True,
        },
        "detection": {
            "model_path": "models/yolov8n-face.pt",
            "confidence_threshold": float(conf_thr),
            "iou_threshold": 0.45,
            "frame_skip": int(frame_skip),
            "input_size": 1280,
        },
        "recognition": {
            "model_name": "buffalo_sc",
            "similarity_threshold": float(sim_thr),
            "embedding_size": 512,
            "min_face_size": 30,
        },
        "tracking": {
            "max_disappeared": 30,
            "max_distance": 0.6,
            "iou_threshold": 0.3,
            "kalman_enabled": True,
        },
        "database": {"type": "sqlite", "path": "data/face_tracker.db"},
        "logging": {"log_file": "logs/events.log", "log_level": "INFO", "image_format": "jpg", "image_quality": 90},
        "ui": {"show_fps": True, "show_face_id": True, "show_bbox": True, "show_visitor_count": True, "bbox_color_new": [0,255,0], "bbox_color_known": [255,165,0], "font_scale": 0.6},
    }

    runtime_path = os.path.abspath("config_runtime.json")
    with open(runtime_path, "w") as f:
        json.dump(runtime_cfg, f, indent=2)

    import subprocess
    cmd = [
        sys.executable,
        "main.py",
        "--config", runtime_path,
        "--video", str(selected_video),
        "--no-display",
    ]

    with st.spinner("Processing video frames..."):
        p = subprocess.run(cmd, capture_output=True, text=True)
    progress_bar.progress(100, text="Processing Complete!")
    if p.returncode == 0:
        st.success("Video processed successfully!")
    else:
        st.error(f"Process completed with notice:\n{p.stderr or p.stdout}")
    st.rerun()

# ── KPI Dashboard Cards ────────────────────────────────────────────────────────
stats = get_db_stats()
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Unique Visitors</div>
        <div class="metric-val">{stats['unique_visitors']}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Entries</div>
        <div class="metric-val" style="color: #4ade80;">{stats['entries']}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Exits</div>
        <div class="metric-val" style="color: #f87171;">{stats['exits']}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Events Logged</div>
        <div class="metric-val" style="color: #c084fc;">{stats['events']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Content Tabs ──────────────────────────────────────────────────────────
tab_gallery, tab_events, tab_video, tab_logs = st.tabs([
    "👥 Registered Faces Gallery",
    "📋 Event Log Table",
    "🎬 Output Video Playback",
    "📝 Raw events.log"
])

# ── Tab 1: Face Gallery ────────────────────────────────────────────────────────
with tab_gallery:
    st.subheader("Auto-Registered Unique Visitors")
    faces = get_faces()
    if not faces:
        st.info("No faces registered yet. Run the tracker on a video to detect and register visitors.")
    else:
        # Display in rows of 4 columns
        cols = st.columns(4)
        for idx, face in enumerate(faces):
            col = cols[idx % 4]
            with col:
                st.markdown('<div class="face-card">', unsafe_allow_html=True)
                # Find matching face image in logs/entries
                img_path = face.get("face_image")
                if not img_path or not os.path.exists(img_path):
                    fid = face["id"]
                    best_img = None
                    best_area = 0
                    if os.path.exists("logs/entries"):
                        for root, _, files in os.walk("logs/entries"):
                            for filename in files:
                                if not filename.startswith(fid):
                                    continue
                                candidate = os.path.join(root, filename)
                                try:
                                    with Image.open(candidate) as candidate_image:
                                        width, height = candidate_image.size
                                    if width * height > best_area:
                                        best_img = candidate
                                        best_area = width * height
                                except Exception:
                                    continue
                    img_path = best_img

                if img_path and os.path.exists(img_path):
                    try:
                        im = Image.open(img_path)
                        st.image(im, width=min(im.width * 2, 240))
                        if im.width < 96 or im.height < 96:
                            st.caption(f"Low source detail: {im.width} x {im.height} px")
                    except Exception:
                        st.text("👤 [No Image Preview]")
                else:
                    st.text("👤 [No Image Saved]")

                st.markdown(f"**ID:** `{face['id']}`")
                created = face.get("created_at", "")[:19].replace("T", " ")
                st.caption(f"First Seen: {created}")
                st.caption(f"Visits: **{face.get('visit_count', 1)}**")
                st.markdown('</div>', unsafe_allow_html=True)

# ── Tab 2: Events Table ────────────────────────────────────────────────────────
with tab_events:
    st.subheader("System Event Log (Entry & Exit Events)")
    df_events = get_events()
    if df_events.empty:
        st.info("No events logged yet.")
    else:
        st.dataframe(df_events, use_container_width=True)

# ── Tab 3: Video Playback ──────────────────────────────────────────────────────
with tab_video:
    st.subheader("Annotated Output Video")
    if os.path.exists(OUTPUT_VIDEO_PATH) and os.path.getsize(OUTPUT_VIDEO_PATH) > 1000:
        with open(OUTPUT_VIDEO_PATH, "rb") as video_file:
            st.video(video_file.read(), format="video/mp4")
    else:
        st.info("Output video not yet generated or video output saving is turned off in config.json.")

# ── Tab 4: Raw Logs ────────────────────────────────────────────────────────────
with tab_logs:
    st.subheader("Live events.log")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        log_text = "".join(lines[-100:])  # last 100 lines
        st.text_area("Last 100 log lines", log_text, height=400)
    else:
        st.info("No logs/events.log found yet.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "This project is a part of a hackathon run by <a href='https://katomaran.com' target='_blank' style='color:#38bdf8;'>https://katomaran.com</a>"
    "</div>",
    unsafe_allow_html=True
)
