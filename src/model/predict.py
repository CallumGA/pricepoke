import torch
import joblib
import pandas as pd
import numpy as np

from network import PricePredictor


def predict(model, scaler, input_features):
    """
    Makes a prediction on a single sample of data.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        scaler (StandardScaler): The fitted scaler object.
        input_features (pd.Series): A pandas Series containing the features for one card.

    Returns:
        tuple: A tuple containing the predicted class (bool) and the prediction probability (float).
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Convert features to numpy array, reshape for a single sample, and scale
    features_np = input_features.to_numpy(dtype="float32").reshape(1, -1)
    features_scaled = scaler.transform(features_np)

    # Convert to PyTorch tensor
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        logit = model(features_tensor)
        # Convert logit to probability using sigmoid
        probability = torch.sigmoid(logit).item()
        # Get predicted class by rounding the probability
        predicted_class = bool(round(probability))

    return predicted_class, probability


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATH = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/models/price_predictor.pth"
    SCALER_PATH = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/models/encoders/scaler.pkl"
    DATA_PATH = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/processed/pokemon_final_with_labels.csv"
    SAMPLE_INDEX = 150  # The row index of the card we want to predict

    # --- Load Data and Model ---
    # Load the full dataset to get feature names and a sample
    full_data = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in full_data.columns if c != "y"]

    # Load the scaler
    scaler = joblib.load(SCALER_PATH)

    # Load the model
    # We need the input_size to initialize the model architecture before loading the weights
    input_size = len(feature_columns)
    model = PricePredictor(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH))

    # --- Make a Prediction on a Sample ---
    # Get a sample card's features and its true label
    sample_features = full_data.loc[SAMPLE_INDEX, feature_columns]
    true_label = bool(full_data.loc[SAMPLE_INDEX, "y"])

    # Get the prediction
    predicted_class, probability = predict(model, scaler, sample_features)

    # --- Display Results ---
    print("--- Prediction Report ---")
    print(f"Analyzing sample card at index: {SAMPLE_INDEX}")
    print(f"Model Prediction: '{predicted_class}' (Will it rise 30% in 6 months?)")
    print(f"Prediction Confidence: {probability:.2%}")
    print(f"Actual Result: '{true_label}'")
    print("-----------------------")