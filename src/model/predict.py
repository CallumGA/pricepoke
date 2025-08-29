import torch
import joblib
import pandas as pd
import numpy as np

from network import PricePredictor


def predict(model, scaler, input_features):

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
    MODEL_PATH = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/models/price_predictor.pth"
    SCALER_PATH = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/models/encoders/scaler.pkl"
    DATA_PATH = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/processed/pokemon_final_with_labels.csv"
    # The TCGPlayer ID for the card you want to predict
    TCGPLAYER_ID_TO_PREDICT = 283798  # Mimikyu from Trick or Trade BOOster Bundle

    # --- Load Data and Model ---
    # We load the final processed data file, as it contains the features in the exact
    # format the model was trained on.
    full_data = pd.read_csv(DATA_PATH)

    # Check if the tcgplayer_id column exists
    if 'tcgplayer_id' not in full_data.columns:
        print(f"Error: 'tcgplayer_id' column not found in {DATA_PATH}.")
        print("Please ensure your final processed CSV contains this identifier column.")
        exit()

    # Find the specific card's data by its ID
    card_data_row = full_data[full_data['tcgplayer_id'] == TCGPLAYER_ID_TO_PREDICT]

    if card_data_row.empty:
        print(f"Error: Card with tcgplayer_id '{TCGPLAYER_ID_TO_PREDICT}' not found in the dataset.")
        exit()
    
    # Take the first row if multiple entries exist for the same ID
    card_sample = card_data_row.iloc[0]

    # --- Define Feature Columns (MUST MATCH TRAINING SCRIPT) ---
    # We explicitly define which columns are identifiers and which is the target,
    # so we can isolate the feature columns the model expects.
    # Add any other non-feature/identifier columns to this list.
    identifier_cols = ['tcgplayer_id'] # e.g., ['tcgplayer_id', 'card_name']
    target_col = 'y'
    feature_columns = [c for c in full_data.columns if c not in identifier_cols and c != target_col]

    # Load the scaler
    scaler = joblib.load(SCALER_PATH)

    # Load the model
    input_size = len(feature_columns)
    model = PricePredictor(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH))

    # --- Make a Prediction on a Sample ---
    # Get a sample card's features and its true label
    sample_features = card_sample[feature_columns]
    true_label = bool(card_sample[target_col])

    # Get the prediction
    predicted_class, probability = predict(model, scaler, sample_features)

    # --- Display Results ---
    print("--- Prediction Report ---")
    # Try to display card name if available for better context
    card_name_display = f" (ID: {TCGPLAYER_ID_TO_PREDICT})"
    if 'name' in card_sample and pd.notna(card_sample['name']):
        card_name_display = f" ({card_sample['name']}, ID: {TCGPLAYER_ID_TO_PREDICT})"

    print(f"Analyzing Card: {card_name_display}")
    print(f"Model Prediction: '{predicted_class}' (Will it rise 30% in 6 months?)")
    print(f"Prediction Confidence: {probability:.2%}")
    print(f"Actual Result in Dataset: '{true_label}'")
    print("-----------------------")