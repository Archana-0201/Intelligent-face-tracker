@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  Intelligent Face Tracker — Streamlit Dashboard Launcher
REM  Launches the interactive dashboard in your web browser.
REM ─────────────────────────────────────────────────────────────────────────

set PYTHON=D:\python313\python.exe

echo.
echo  ======================================================
echo    Launching Intelligent Face Tracker Dashboard...
echo  ======================================================
echo.

%PYTHON% -m streamlit run app.py
pause
