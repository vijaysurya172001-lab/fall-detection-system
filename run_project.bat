@echo off
title Fall Detection System - Startup
echo ==================================================
echo      Elderly Care System - Fall Detection
echo ==================================================
echo.

echo [1/3] Step 1: Installing Required AI Libraries...
echo (This may take a minute if running for the first time)
pip install flask opencv-python mediapipe numpy scikit-learn pandas joblib

echo.
echo [2/3] Step 2: Verifying System Logic...
if not exist app.py (
    echo [ERROR] app.py not found! Make sure you extracted the files.
    pause
    exit
)

echo.
echo [3/3] Step 3: Starting the Dashboard...
echo.
echo --------------------------------------------------
echo ONCE RUNNING: Open Chrome and go to:
echo http://127.0.0.1:5050
echo --------------------------------------------------
echo.

python app.py

echo.
echo [INFO] System has stopped.
pause
