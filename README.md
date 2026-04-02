# Fall Detection and Activity Recognition

This project implements a fall detection and activity recognition system using human skeleton features, as described in the paper "Fall Detection and Activity Recognition Using Human Skeleton Features" (2021).

## Architecture

1.  **Pose Estimation**: Uses MediaPipe Pose to extract landmarks.
2.  **Feature Extraction**: Maps 33 MediaPipe landmarks to the 17 standard COCO keypoints (x, y, confidence) used in the paper.
3.  **Classification**: Implements Random Forest (RF), Support Vector Machine (SVM), Multilayer Perceptron (MLP), and K-Nearest Neighbors (KNN).

## Getting Started

### Prerequisites
- **Python 3.8-3.11** (Required for MediaPipe)
- Webcam
- (Note: Current system Python 3.7.4 is too old for the Pose Detector module)
```bash
pip install -r requirements.txt
```

### Usage
1.  **Train Mock Models**: Generate a baseline model (since the full UP-FALL dataset is proprietary/separate).
    ```bash
    python train_mock.py
    ```
2.  **Run Real-time Detection**:
    ```bash
    python main.py
    ```

## Activity Labels
- Falling (Forward, Backwards, Sideways, Sitting)
- Walking
- Standing
- Sitting
- Laying
- Picking up object
- Jumping
