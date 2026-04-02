"""
POSE DETECTOR MODULE
This module integrates Google's MediaPipe AI to detect human body parts.
It essentially "finds" the person in the video and extracts their skeleton coordinates.
"""

import cv2
import mediapipe as mp

class PoseDetector:
    """
    MEDIAPIPE WRAPPER CLASS
    This class wraps the complexity of MediaPipe into a simple 'find_pose' function.
    """
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the MediaPipe Pose AI engine with specific confidence settings.
        """
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # MediaPipe Drawing requirements
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        
        # Initialize the actual AI model
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def find_pose(self, img, draw=True):
        """
        AI INFERENCE FUNCTION
        This takes a raw image, runs it through the neural network, 
        and extracts 33 skeleton points.
        """
        # MediaPipe needs RGB images, but OpenCV captures in BGR
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run AI detection
        self.results = self.pose.process(imgRGB)
        
        # Draw the skeleton (lines and dots) back onto the original image
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, 
                    self.mpPose.POSE_CONNECTIONS,
                    # Custom styling: Red dots and White lines
                    self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
        return img

    def get_points(self):
        """
        Extracts raw coordinate points into a list if needed for advanced training.
        """
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = [img_height, img_width, img_channels] # Placeholder for context
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList
