import joblib
import numpy as np

def load_rf_model(model_path):
    """
    Load the Random Forest model from the specified path.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load Random Forest model: {e}")

def predict_rf(model, input_features):
    try:
        prediction = model.predict(input_features)
        return prediction
    except Exception as e:
        raise RuntimeError(f"Random Forest prediction failed: {e}")

