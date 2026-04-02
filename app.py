"""
FALL DETECTION DASHBOARD - MAIN BACKEND
This module serves as the primary web server for the Elderly Care System.
It integrates the camera feed with AI detection and provides a real-time 
web-based dashboard for monitoring.
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
import winsound
import os
import tempfile
from pose_detector import PoseDetector
from utils import detect_activity_from_pose

# --- INITIALIZATION ---
# Initialize Flask app to serve HTML and Static files
app = Flask(__name__, 
            template_folder='dashboard/templates',
            static_folder='dashboard/static')

# Initialize the PoseDetector (MediaPipe AI)
detector = PoseDetector()

# Global variables to manage camera state across web requests
camera = None
monitoring = False
current_activity = "Inactive"

def gen_frames():
    """
    VIDEO STREAM GENERATOR
    This function handles the heavy lifting: 
    1. Opens the webcam.
    2. Captures frames.
    3. Runs AI pose detection.
    4. Decides the activity (Sitting, Standing, etc.).
    5. Yields the processed image to the web browser.
    """
    global camera, monitoring, current_activity
    print(">>> gen_frames: Initializing Camera...")
    
    # Use DirectShow for Windows - often fixes blank screen/camera lock issues
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Warmup time for camera to stabilize (CRITICAL for some webcams)
    time.sleep(2.0)
    
    if not camera or not camera.isOpened():
        print("!!! ERROR: Could not open webcam via DirectShow. Trying default...")
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("!!! FATAL ERROR: No webcam found.")
        monitoring = False
        return

    # Quick Warmup: Capture and discard first 5 frames to ensure bright image
    for i in range(5):
        camera.read()
    
    print(">>> gen_frames: Camera ready, starting stream...")
    last_fall_time = 0
    
    # Track sound timing for posture to avoid constant noise
    if not hasattr(gen_frames, "last_posture_time"):
        gen_frames.last_posture_time = 0
    
    while monitoring:
        success, frame = camera.read()
        if not success:
            print("!!! gen_frames: Failed to read frame.")
            break
            
        try:
            # Step 1: Detect human pose landmarks using MediaPipe AI
            frame = detector.find_pose(frame)
            
            # Step 2: Use geometric logic to classify activity (1-11 scale)
            id, name = detect_activity_from_pose(detector.results)
            current_activity = name
            
            # --- Step 3: Draw Visual Feedback on the screen ---
            # Create a professional status box at the top left
            cv2.rectangle(frame, (20, 20), (500, 100), (0, 0, 0), -1)      # Black box
            cv2.rectangle(frame, (20, 20), (500, 100), (255, 255, 255), 2) # White border
            
            color = (0, 255, 0) # Green for safe states
            curr_t = time.time()
            
            # Check if it's ANY type of Fall (IDs 1 to 5)
            if 1 <= id <= 5 or "Fall" in name:
                color = (0, 0, 255) # Red for danger
                cv2.putText(frame, f"ALERT: {name.upper()}", (40, 60), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                
                # RECOVERY ALERT: 1000Hz Beep
                if curr_t - last_fall_time > 2:
                    winsound.Beep(1000, 250)
                    last_fall_time = curr_t
            
            elif "Posture" in name or id == 8:
                # Posture color coding
                posture_color = (255, 255, 255) # Default white
                if "Good" in name:
                    posture_color = (0, 255, 0) # Green
                    # Sound removed for Good Posture as requested
                else:
                    posture_color = (0, 165, 255) # Orange
                    if curr_t - gen_frames.last_posture_time > 5:
                        winsound.Beep(500, 300)
                        gen_frames.last_posture_time = curr_t
                
                cv2.putText(frame, f"STATE: {name.upper()}", (40, 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, posture_color, 2)
            
            elif id == 10: # Jumping
                cv2.putText(frame, "MOTION: JUMPING", (40, 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
            
            elif id == 9: # Picking up
                cv2.putText(frame, "STATE: PICKING UP", (40, 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
            
            else:
                # Normal State Label (Standing, Walking, Laying)
                cv2.putText(frame, f"ACTIVITY: {name.upper()}", (40, 65), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            # Live Status Indicator (Blowing dot)
            cv2.circle(frame, (35, 62), 8, color, -1)
            
        except Exception as e:
            print(f"!!! Detection Error: {e}")

        # Step 4: Convert the processed OpenCV frame into a JPEG for the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        # Step 5: Yield the frame as a multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    # CLEANUP: Release camera when browser closes or monitoring stops
    if camera:
        camera.release()
        camera = None
    print(">>> gen_frames: Stream finished.")

# --- WEB ROUTES ---

@app.route('/get_status')
def get_status():
    """Returns the current detected activity as JSON for dashboard updates."""
    return jsonify(status=current_activity)

@app.route('/')
def index():
    """Serves the main Dashboard HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint that serves the live MJPEG video stream to the dashboard."""
    global monitoring
    monitoring = True
    print(">>> Route: /video_feed accessed")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_monitoring')
def stop_monitoring():
    """Stops the camera and the tracking loop safely."""
    global monitoring, camera
    monitoring = False
    if camera:
        camera.release()
        camera = None
    print(">>> Route: /stop_monitoring accessed")
    return jsonify(status="stopped")

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """
    Handles image uploads for manual posture detection.
    Processes the uploaded file and returns the detected activity.
    """
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    try:
        # Convert the file stream into an OpenCV image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify(error="Invalid image format"), 400

        # Perform detection with reset=True to ensure deterministic output for static images
        processed_img = detector.find_pose(img)
        id, name = detect_activity_from_pose(detector.results, reset=True)
        
        # Update global status so main dashboard reflects manual detection
        global current_activity
        current_activity = name
        
        print(f">>> Manual Detection: {name}")
        return jsonify(status=name)
        
    except Exception as e:
        print(f"!!! Upload Error: {e}")
        return jsonify(error=str(e)), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Handles video uploads for sequence-based posture detection.
    Scans the video and returns the most critical state found.
    """
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    temp_path = None
    try:
        # Save video to a temporary file for OpenCV to read
        fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        file.save(temp_path)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify(error="Could not open video file"), 400

        highest_priority_name = "No Human Detected"
        highest_priority_id = 100 # Lower is better? No, let's use a logic priority

        # Detection Results Mapping
        # Priority: Fall (1-5) > Bad Posture (11/8 bad) > Good Posture > Walking/Standing
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for speed
            for _ in range(4):
                cap.read()

            detector.find_pose(frame)
            id, name = detect_activity_from_pose(detector.results, reset=False)

            # Determine Priority
            # If we find ANY Fall, we stop and return it immediately
            if 1 <= id <= 5 or "Fall" in name:
                highest_priority_name = name
                break
            
            # If we find Bad Posture, track it but keep looking for falls
            if "Bad Posture" in name:
                highest_priority_name = name
            
            # If we haven't found anything yet, set to current name
            if highest_priority_name == "No Human Detected":
                highest_priority_name = name

        cap.release()
        
        # Update global status
        global current_activity
        current_activity = highest_priority_name
        
        print(f">>> Video Detection Segment: {highest_priority_name}")
        return jsonify(status=highest_priority_name)

    except Exception as e:
        print(f"!!! Video Upload Error: {e}")
        return jsonify(error=str(e)), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

import argparse

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Add support for custom port via command line: python app.py --port 8000
    parser = argparse.ArgumentParser(description="Run the Fall Detection Dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the server on")
    args = parser.parse_args()

    print("--------------------------------------------------")
    print("DASHBOARD STARTING...")
    print(f"Open your browser at: http://127.0.0.1:{args.port}")
    print("--------------------------------------------------")
    
    # Threaded=True allows simultaneous streaming and status polling
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
