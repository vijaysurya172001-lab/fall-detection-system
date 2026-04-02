from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

class ActivityClassifier:
    def __init__(self, model_type='RF'):
        self.model_type = model_type
        self.model = self._initialize_model()
        self.model_path = f"fall_model_{model_type}.pkl"

    def _initialize_model(self):
        if self.model_type == 'RF':
            return RandomForestClassifier(n_estimators=100)
        elif self.model_type == 'SVM':
            return SVC(probability=True)
        elif self.model_type == 'MLP':
            return MLPClassifier(hidden_layer_sizes=(64, 32))
        elif self.model_type == 'KNN':
            return KNeighborsClassifier(n_neighbors=5)
        else:
            return RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def predict(self, features):
        if features is None or len(features) == 0:
            return "Unknown"
        # Reshape for single prediction
        features = features.reshape(1, -1)
        pred = self.model.predict(features)[0]
        return pred

# Labels from paper
LABELS = {
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
    11: "Laying (Good Posture)",
    20: "No Human Detected"
}
