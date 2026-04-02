import numpy as np
from classifier import ActivityClassifier, LABELS
import joblib

def generate_mock_data():
    """
    Generates realistic skeleton features for training.
    Creates distinct patterns for each activity based on biomechanics.
    """
    X = []
    y = []
    
    # Generate 200 samples per class for better training
    activities = {
        1: "Falling forward (hands)",
        2: "Falling forward (knees)", 
        3: "Falling backwards",
        4: "Falling sideways",
        5: "Falling sitting",
        6: "Walking",
        7: "Standing",
        8: "Sitting",
        9: "Picking up object",
        10: "Jumping",
        11: "Laying"
    }
    
    for label, activity_name in activities.items():
        for _ in range(200):
            # Initialize base features (17 keypoints * 3 = 51 features)
            # Format: [x1, y1, vis1, x2, y2, vis2, ...]
            features = np.zeros(51)
            
            # Set visibility to 1 for all points
            features[2::3] = 1.0
            
            if label == 7:  # Standing - vertical alignment, head high
                features[0::3] = 0.5 + np.random.normal(0, 0.05, 17)  # x centered
                features[1::3] = np.linspace(0.2, 0.9, 17) + np.random.normal(0, 0.03, 17)  # y vertical
                
            elif label == 6:  # Walking - slight lean, varied positions
                features[0::3] = 0.5 + np.random.normal(0, 0.1, 17)  # x with variation
                features[1::3] = np.linspace(0.25, 0.85, 17) + np.random.normal(0, 0.05, 17)
                
            elif label == 8:  # Sitting - torso upright, legs bent
                features[0::3] = 0.5 + np.random.normal(0, 0.04, 17)
                # Upper body high, lower body lower
                upper_y = np.linspace(0.3, 0.6, 9)
                lower_y = np.linspace(0.6, 0.8, 8)
                features[1::3] = np.concatenate([upper_y, lower_y]) + np.random.normal(0, 0.03, 17)
                
            elif label == 11:  # Laying - horizontal orientation
                features[0::3] = np.linspace(0.2, 0.8, 17) + np.random.normal(0, 0.05, 17)  # x spread
                features[1::3] = 0.7 + np.random.normal(0, 0.05, 17)  # y flat
                
            elif label in [1, 2, 3, 4, 5]:  # Falling - diagonal/tilted
                if label == 1:  # Forward fall (hands)
                    features[0::3] = np.linspace(0.3, 0.7, 17) + np.random.normal(0, 0.05, 17)
                    features[1::3] = np.linspace(0.4, 0.8, 17) + np.random.normal(0, 0.05, 17)
                elif label == 2:  # Forward fall (knees)
                    features[0::3] = np.linspace(0.35, 0.65, 17) + np.random.normal(0, 0.05, 17)
                    features[1::3] = np.linspace(0.5, 0.85, 17) + np.random.normal(0, 0.05, 17)
                elif label == 3:  # Backward fall
                    features[0::3] = np.linspace(0.7, 0.3, 17) + np.random.normal(0, 0.05, 17)
                    features[1::3] = np.linspace(0.3, 0.7, 17) + np.random.normal(0, 0.05, 17)
                elif label == 4:  # Sideways fall
                    features[0::3] = np.linspace(0.2, 0.8, 17) + np.random.normal(0, 0.05, 17)
                    features[1::3] = 0.6 + np.random.normal(0, 0.08, 17)
                else:  # Falling sitting
                    features[0::3] = 0.5 + np.random.normal(0, 0.06, 17)
                    features[1::3] = np.linspace(0.4, 0.75, 17) + np.random.normal(0, 0.05, 17)
                    
            elif label == 9:  # Picking up object - bent over
                features[0::3] = 0.5 + np.random.normal(0, 0.05, 17)
                # Upper body bent down
                upper_y = np.linspace(0.5, 0.7, 9)
                lower_y = np.linspace(0.7, 0.9, 8)
                features[1::3] = np.concatenate([upper_y, lower_y]) + np.random.normal(0, 0.04, 17)
                
            elif label == 10:  # Jumping - compressed or extended
                features[0::3] = 0.5 + np.random.normal(0, 0.06, 17)
                if np.random.rand() > 0.5:  # Extended
                    features[1::3] = np.linspace(0.1, 0.8, 17) + np.random.normal(0, 0.04, 17)
                else:  # Compressed
                    features[1::3] = np.linspace(0.4, 0.9, 17) + np.random.normal(0, 0.04, 17)
            
            X.append(features)
            y.append(label)
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Generating realistic training data...")
    X, y = generate_mock_data()
    
    print(f"Generated {len(X)} samples across {len(set(y))} activity classes")
    
    for mtype in ['RF', 'SVM', 'MLP', 'KNN']:
        print(f"Training {mtype} model...")
        clf = ActivityClassifier(model_type=mtype)
        clf.train(X, y)
    
    print("All models trained and saved successfully!")
