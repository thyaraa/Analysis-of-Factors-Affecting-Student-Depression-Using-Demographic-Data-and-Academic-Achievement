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

#def predict_rf(model, input_features):
 #   try:
  #      prediction = model.predict(input_features)
  #      return prediction
  #  except Exception as e:
   #     raise RuntimeError(f"Random Forest prediction failed: {e}")
def predict_rf(model, input_features, total_features=12):
    """
    Predict using the Random Forest model with padded features.

    Parameters:
        model: Loaded Random Forest model.
        input_features (numpy.ndarray): Input features as a 2D array.
        total_features (int): Total number of features the model expects.

    Returns:
        array: Predicted class.
    """
    try:
        # Debug: Print input shape before modification
        print(f"Input features shape before modification: {input_features.shape}")

        # Pad input features to match the model's expected input size
        if input_features.shape[1] < total_features:
            padding = np.zeros((input_features.shape[0], total_features - input_features.shape[1]))
            input_features = np.hstack((input_features, padding))
        elif input_features.shape[1] > total_features:
            input_features = input_features[:, :total_features]

        # Debug: Print input shape after modification
        print(f"Final input shape for prediction: {input_features.shape}")

        # Predict using the model
        prediction = model.predict(input_features)
        return prediction
    except Exception as e:
        raise RuntimeError(f"Random Forest prediction failed: {e}")
