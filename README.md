# Fall Detection System

This project is a real-time posture monitoring and fall detection system developed using Python, OpenCV, and MediaPipe. The system detects human posture from a live camera feed and classifies falls using trained machine learning models.

## Features

- Real-time human pose detection
- Fall detection using machine learning models
- Live camera monitoring
- Multiple trained models (KNN, SVM, Random Forest, MLP)
- Alert generation for detected falls

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Machine Learning (Scikit-learn)
- NumPy
- Pandas

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the project:

python main.py

## Project Structure

app.py
classifier.py
main.py
pose_detector.py
test_camera.py
requirements.txt
fall_model_KNN.pkl
fall_model_SVM.pkl
fall_model_RF.pkl
fall_model_MLP.pkl
