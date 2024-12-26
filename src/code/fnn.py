import tensorflow as tf
import numpy as np

# Functions for FFNN

def load_fnn_model(model_path):
    """
    Load the FFNN model from the specified path.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load FFNN model: {e}")

def predict_fnn(model, input_features, total_features=12):
    """
    Predict using the FFNN model with padded features.

    Parameters:
        model (tf.keras.Model): Loaded FFNN model.
        input_features (numpy.ndarray): Input features as a 2D array.
        total_features (int): Total number of features the model expects.

    Returns:
        int: Predicted class index.
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
        print(f"Final input features: {input_features}")

        # Predict using the model
        prediction = model.predict(input_features)
        return np.argmax(prediction, axis=1)
    except Exception as e:
        raise RuntimeError(f"FFNN prediction failed: {e}")