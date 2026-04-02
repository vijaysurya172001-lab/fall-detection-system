"""
ACTIVITY DETECTION GEOMETRY & LOGIC
This module contains the math and logic to identify human positions 
based on AI-detected pose landmarks, following the standard 11-activity schema.
"""

import time
import math

def detect_activity_from_pose(landmarks, reset=False):
    """
    HUMAN ACTIVITY CLASSIFIER (11 Categories)
    
    Mapped Categories:
    1: Falling forward (hands)
    2: Falling forward (knees)
    3: Falling backwards
    4: Falling sideways
    5: Falling sitting
    6: Walking
    7: Standing
    8: Sitting
    9: Picking up object
    10: Jumping
    11: Laying
    """
    if landmarks is None or not landmarks.pose_landmarks:
        return 20, "No Human Detected"
    
    lm = landmarks.pose_landmarks.landmark
    
    # --- KEY BODY POINTS ---
    nose = lm[0]
    left_shoulder, right_shoulder = lm[11], lm[12]
    left_hip, right_hip = lm[23], lm[24]
    left_knee, right_knee = lm[25], lm[26]
    left_ankle, right_ankle = lm[27], lm[28]
    left_wrist, right_wrist = lm[15], lm[16]
    
    # --- CENTER POINTS & RATIOS ---
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    hip_x = (left_hip.x + right_hip.x) / 2
    knee_y = (left_knee.y + right_knee.y) / 2
    ankle_y = (left_ankle.y + right_ankle.y) / 2
    
    # --- VISIBILITY CHECKS ---
    # Stricter for motion: We need clear confidence in at least one full leg
    # or both knees/ankles to trust upright positions.
    lower_body_visible = (
        (left_knee.visibility > 0.5 and left_ankle.visibility > 0.5) or
        (right_knee.visibility > 0.5 and right_ankle.visibility > 0.5) or
        (left_knee.visibility > 0.5 and right_knee.visibility > 0.5)
    )

    torso_len = abs(shoulder_y - hip_y)
    leg_len = abs(hip_y - ankle_y)
    total_h = abs(nose.y - ankle_y)
    
    # If lower body is not visible, total height is unreliable
    if not lower_body_visible:
        total_h = torso_len * 2 # Estimate total height based on torso
    
    curr_t = time.time()
    
    # --- MOTION TRACKING ---
    if reset or not hasattr(detect_activity_from_pose, "prev_nose_y"):
        detect_activity_from_pose.prev_nose_y = nose.y
        detect_activity_from_pose.prev_time = curr_t
        detect_activity_from_pose.prev_v = 0
    
    dt = curr_t - detect_activity_from_pose.prev_time
    delta_y = nose.y - detect_activity_from_pose.prev_nose_y
    
    velocity_y = 0
    if dt > 0:
        velocity_y = delta_y / dt
    
    # Update persistent state
    detect_activity_from_pose.prev_nose_y = nose.y
    detect_activity_from_pose.prev_time = curr_t

    # 10. JUMPING: Sudden upward velocity followed by airborne state
    # Added visibility safety to prevent noise in headshots from triggering jumping
    if lower_body_visible and velocity_y < -1.5 and ankle_y < 0.8:
        return 10, "Jumping"

    # 11. LAYING: Horizontal orientation OR Longitudinal (on a bed)
    horiz_span = abs(nose.x - left_ankle.x)
    vert_span = abs(nose.y - ankle_y)
    
    is_laying = False
    # Case A: Person is horizontal (Side-view)
    if horiz_span > vert_span * 1.5:
        is_laying = True
    # Case B: Person is longitudinal (Top-down bed view)
    # Check if the torso is flat (shoulders and hips have similar Y-tilt)
    # AND the body is significantly horizontal in the frame
    elif horiz_span > 0.4 and abs(left_shoulder.y - right_shoulder.y) < 0.1:
        is_laying = True

    if is_laying:
        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
        nose_off = abs(nose.x - shoulder_x)
        if shoulder_tilt < 0.2 and nose_off < 0.2:
            return 11, "Laying (Good Posture)"
        else:
            return 11, "Laying (Bad Posture)"

    # --- FALLING OVERVIEW ---
    is_falling = False
    
    # Check if the person is at the bottom of the frame (Ground Level)
    # Refined: nose.y threshold lowered to 0.5 to catch falls further from camera
    on_ground = nose.y > 0.5 and total_h < 0.4 and horiz_span > vert_span * 1.1

    # Rule 1: Sudden drop in head position
    if velocity_y > 1.2 and nose.y > 0.3: is_falling = True 
    
    # Rule 2: Body tilted/horizontal (using aspect ratio)
    body_angle = torso_len / (abs(left_shoulder.x - left_hip.x) + 0.01)
    if body_angle < 1.0 and nose.y > 0.4: is_falling = True 

    # Rule 3: Direct ground-level detection (Priority)
    # Horizontal body at bottom of frame
    if on_ground or (total_h < 0.35 and nose.y > 0.5 and horiz_span > 0.4):
        is_falling = True

    if is_falling:
        # 5. Falling sitting: If hips are low but torso is somewhat upright
        if hip_y > 0.7 and torso_len > 0.15:
            return 5, "Falling sitting"
        
        # 3. Falling backwards: Head behind hips/shoulders
        if nose.x > shoulder_x + 0.1:
            return 3, "Falling backwards"
        
        # 4. Falling sideways: Significant tilt to left or right
        if abs(left_shoulder.y - right_shoulder.y) > 0.15:
            return 4, "Falling sideways"
            
        # 1 & 2. Falling forward
        if nose.x < shoulder_x - 0.05:
            # 1. Forward (hands): Wrists are out in front
            if left_wrist.y < nose.y or right_wrist.y < nose.y:
                return 1, "Falling forward (hands)"
            # 2. Forward (knees): Wrists not protecting, knees leading
            return 2, "Falling forward (knees)"
        
        return 1, "Fall Detected"

    # 9. PICKING UP OBJECT: High hip, low head/wrists
    if hip_y < 0.6 and (left_wrist.y > 0.8 or right_wrist.y > 0.8) and torso_len < 0.2:
        return 9, "Picking up object"

    # --- UPRIGHT POSITION PRIORITY (Standing/Walking) ---
    # Smart Guard: Stand/Walk only if legs are significantly long compared to the torso
    # AND the head is not too low (unless full body is clearly visible far away)
    is_upright = (leg_len > torso_len * 1.1) and (nose.y < 0.45)
    
    # If the person is very close (large torso), we are even stricter
    if torso_len > 0.4:
        is_upright = (leg_len > torso_len * 1.3)

    # 6. WALKING: Significant leg movement detected
    # Refined: Relaxed span (0.12) if body is clearly upright
    if lower_body_visible and is_upright and abs(left_ankle.x - right_ankle.x) > 0.12 and total_h > 0.45:
        return 6, "Walking"

    # 7. STANDING: Upright position
    # Refined: Ratio-based guard to ensure person is actually on their feet
    if lower_body_visible and is_upright and total_h > 0.45 and nose.y < 0.4:
        return 7, "Standing"

    # 8. SITTING & POSTURE
    # Refined logic: Clear evidence of sitting (knee-hip alignment)
    is_sitting = False
    if abs(hip_y - knee_y) < 0.15 and total_h < 0.6: 
        is_sitting = True
    
    if is_sitting:
        if hip_y > 0.5: 
            shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
            nose_off = abs(nose.x - shoulder_x)
            # Tightened to 0.07 for better sensitivity
            if shoulder_tilt < 0.07 and nose_off < 0.07:
                return 8, "Sitting (Good Posture)"
            else:
                return 8, "Sitting (Bad Posture)"

    # Final fallback: Logic-based priority
    # If hip is low and body is compressed -> Sitting
    if hip_y > 0.5:
        # Check posture of the fallback sitting (Tightened to 0.07)
        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
        nose_off = abs(nose.x - shoulder_x)
        if shoulder_tilt < 0.07 and nose_off < 0.07:
            return 8, "Sitting (Good Posture)"
        else:
            return 8, "Sitting (Bad Posture)"
        
    return 7, "Standing"
