import cv2
import numpy as np
import time
import winsound  # For alarm sound on Windows

def detect_activity_from_pose(landmarks):
    """
    Rule-based activity detection using geometric analysis of pose landmarks.
    More reliable than ML models trained on synthetic data.
    
    Returns: (activity_id, activity_name)
    """
    if landmarks is None or not landmarks.pose_landmarks:
        return 20, "Unknown"
    
    lm = landmarks.pose_landmarks.landmark
    
    # Key body points (MediaPipe indices)
    nose = lm[0]
    left_shoulder = lm[11]
    right_shoulder = lm[12]
    left_hip = lm[23]
    right_hip = lm[24]
    left_knee = lm[25]
    right_knee = lm[26]
    left_ankle = lm[27]
    right_ankle = lm[28]
    
    # Calculate center points
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_center_y = (left_hip.y + right_hip.y) / 2
    knee_center_y = (left_knee.y + right_knee.y) / 2
    ankle_center_y = (left_ankle.y + right_ankle.y) / 2
    
    # Calculate body orientation
    torso_height = abs(shoulder_center_y - hip_center_y)
    total_height = abs(nose.y - ankle_center_y)
    
    # Check if person is horizontal (laying down)
    horizontal_spread = max(abs(nose.x - left_ankle.x), abs(nose.x - right_ankle.x))
    vertical_spread = abs(nose.y - ankle_center_y)
    
    if horizontal_spread > vertical_spread * 1.2 and nose.y > 0.5:
        return 11, "Laying"
    
    # Check if sitting (hips and knees at similar height, knees bent)
    hip_knee_diff = abs(hip_center_y - knee_center_y)
    knee_ankle_diff = abs(knee_center_y - ankle_center_y)
    
    # Sitting: knees are close to hips vertically, ankles below knees
    if hip_knee_diff < 0.15 and knee_ankle_diff > 0.1 and hip_center_y > 0.4:
        return 8, "Sitting"
    
    # Check if standing (vertical alignment, good posture)
    if torso_height > 0.2 and total_height > 0.5:
        # Check if legs are relatively straight
        if knee_center_y < hip_center_y + 0.3 and ankle_center_y < 0.9:
            # Check for walking motion (one leg forward)
            leg_spread = abs(left_ankle.x - right_ankle.x)
            if leg_spread > 0.15:
                return 6, "Walking"
            else:
                return 7, "Standing"
    
    # Check for bending/picking up (nose close to hips)
    if abs(nose.y - hip_center_y) < 0.25 and torso_height < 0.15:
        return 9, "Picking up object"
    
    # Check for jumping (feet off ground or compressed pose)
    if ankle_center_y < 0.7 or (nose.y < 0.3 and total_height < 0.5):
        return 10, "Jumping"
    
    # Check for falling (diagonal orientation, rapid position change)
    body_angle = abs(shoulder_center_y - hip_center_y) / (abs(left_shoulder.x - left_hip.x) + 0.01)
    if body_angle < 2.0 and nose.y > 0.4:  # Body is tilted
        if nose.x < shoulder_center_y:
            return 1, "Falling forward (hands)"
        else:
            return 3, "Falling backwards"
    
    # Default to standing if upright
    if total_height > 0.4:
        return 7, "Standing"
    
    return 20, "Unknown"


def main():
    from pose_detector import PoseDetector
    
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    p_time = 0
    
    print("Starting Fall Detection with Rule-Based Detection. Press 'q' to quit.")
    print("Try: Standing, Sitting, Walking, Laying down, Bending over, Jumping")

    while True:
        success, img = cap.read()
        if not success:
            break

        # 1. Detect Skeleton
        img = detector.find_pose(img)
        
        # 2. Detect Activity using geometric rules
        activity_id, activity_name = detect_activity_from_pose(detector.results)

        # 3. Display
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # Visual and audio enhancement for fall
        color = (0, 255, 0)
        if "Falling" in activity_name:
            color = (0, 0, 255)
            cv2.putText(img, "WARNING: FALL DETECTED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # Play alarm sound (beep at 2500 Hz for 500ms)
            try:
                winsound.Beep(2500, 500)  # Frequency: 2500 Hz, Duration: 500ms
            except:
                pass  # If sound fails, continue without it

        cv2.putText(img, f"Activity: {activity_name}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"FPS: {int(fps)}", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Add quit instructions on screen
        cv2.putText(img, "Press 'Q' or 'ESC' to quit", (50, img.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Fall Detection System", img)

        # Multiple exit options for better usability
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, or ESC
            print("Exiting Fall Detection System...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
