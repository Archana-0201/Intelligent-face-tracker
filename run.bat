@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  Intelligent Face Tracker — Launcher
REM  Uses D:\python313 where all AI packages are installed.
REM  Usage: run.bat [any main.py arguments]
REM  Examples:
REM    run.bat
REM    run.bat --video data\sample_video.mp4
REM    run.bat --rtsp rtsp://192.168.1.100:554/stream
REM    run.bat --stats
REM    run.bat --no-display
REM ─────────────────────────────────────────────────────────────────────────

set PYTHON=D:\python313\python.exe

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║    INTELLIGENT FACE TRACKER — Starting          ║
echo  ╚══════════════════════════════════════════════════╝
echo.

%PYTHON% main.py %*
pause
