"""
main.py
-------
Entry point for the Intelligent Face Tracker.

Usage:
    python main.py                      # uses config.json defaults
    python main.py --config my.json     # custom config file
    python main.py --video path/to.mp4  # override video source
    python main.py --rtsp rtsp://...    # use RTSP stream
    python main.py --stats              # print DB stats and exit
    python main.py --reset-db           # wipe and reinitialise DB
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligent Face Tracker — Auto-Registration & Visitor Counting"
    )
    parser.add_argument("--config",  default="config.json",
                        help="Path to configuration JSON (default: config.json)")
    parser.add_argument("--video",   default=None,
                        help="Override video source path from config")
    parser.add_argument("--webcam",  action="store_true",
                        help="Use live webcam (device 0)")
    parser.add_argument("--rtsp",    default=None,
                        help="RTSP stream URL (overrides config)")
    parser.add_argument("--stats",   action="store_true",
                        help="Print visitor stats from DB and exit")
    parser.add_argument("--reset-db", action="store_true",
                        help="Delete and reinitialise the database, then exit")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV display window")
    return parser.parse_args()


def patch_config(args):
    """Apply CLI overrides to config.json in-memory."""
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    if args.webcam:
        cfg["video"]["source"]   = "0"
        cfg["video"]["use_rtsp"] = False
    elif args.video:
        cfg["video"]["source"]   = args.video
        cfg["video"]["use_rtsp"] = False

    if args.rtsp:
        cfg["video"]["rtsp_url"] = args.rtsp
        cfg["video"]["use_rtsp"] = True

    if args.no_display:
        cfg["video"]["display_output"] = False

    # Write back to a temp config so ConfigLoader picks it up
    tmp = config_path.replace(".json", "_runtime.json")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    return tmp


def print_stats(config_path: str):
    """Print DB statistics without starting the video pipeline."""
    from src.config_loader import get_config
    from src.database      import Database
    cfg = get_config(config_path)
    db  = Database(cfg.database.get("path", "data/face_tracker.db"))
    stats = db.get_stats_summary()
    db.close()

    print("\n" + "="*55)
    print("  INTELLIGENT FACE TRACKER -- DATABASE STATISTICS")
    print("="*55)
    print(f"  Unique Visitors (all-time) : {stats['unique_visitors_total']}")
    print(f"  Total Events Logged        : {stats['total_events_logged']}")
    today = stats.get("today", {})
    if today:
        print(f"\n  Today ({today.get('stat_date', '-')})")
        print(f"    Entries : {today.get('total_entries', 0)}")
        print(f"    Exits   : {today.get('total_exits', 0)}")
    print("="*55 + "\n")


def reset_database(config_path: str):
    """Delete and reinitialise the SQLite database."""
    from src.config_loader import get_config
    cfg     = get_config(config_path)
    db_path = os.path.abspath(cfg.database.get("path", "data/face_tracker.db"))
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"[RESET] Database deleted: {db_path}")
    from src.database import Database
    Database(db_path)
    print(f"[RESET] Database reinitialised: {db_path}")


def main():
    args        = parse_args()
    config_path = patch_config(args)   # apply CLI overrides, returns path

    if args.stats:
        print_stats(config_path)
        return

    if args.reset_db:
        reset_database(config_path)
        return

    # ── Normal pipeline run ────────────────────────────────────────────────────
    from src.pipeline import Pipeline
    pipeline = Pipeline(config_path=config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
