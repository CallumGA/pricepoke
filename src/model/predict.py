import torch
import joblib
import pandas as pd
import numpy as np
import argparse

from network import PricePredictor
import config


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
    # --- Argument Parsing ---
    # Set up the script to accept a command-line argument for the card ID
    parser = argparse.ArgumentParser(description="Predict Pokémon card price trends.")
    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help="The TCGPlayer ID of the card to predict.",
    )
    args = parser.parse_args()
    tcgplayer_id_to_predict = args.id

    # --- Load Data and Model ---
    # We load the final processed data file, as it contains the features in the exact
    # format the model was trained on.
    full_data = pd.read_csv(config.INPUT_CSV_PATH)

    # Check if the tcgplayer_id column exists
    if 'tcgplayer_id' not in full_data.columns:
        print(f"Error: 'tcgplayer_id' column not found in {config.INPUT_CSV_PATH}.")
        print("Please ensure your final processed CSV contains this identifier column.")
        exit()

    # Find the specific card's data by its ID
    card_data_row = full_data[full_data['tcgplayer_id'] == tcgplayer_id_to_predict]

    if card_data_row.empty:
        print(f"Error: Card with tcgplayer_id '{tcgplayer_id_to_predict}' not found in the dataset.")
        exit()
    
    # Take the first row if multiple entries exist for the same ID
    card_sample = card_data_row.iloc[0]

    # --- Define Feature Columns (MUST MATCH TRAINING SCRIPT) ---
    # We explicitly define which columns are identifiers and which is the target,
    # so we can isolate the feature columns the model expects.
    feature_columns = [c for c in full_data.columns if c not in config.IDENTIFIER_COLS and c != config.TARGET_COL]

    # Load the scaler
    scaler = joblib.load(config.SCALER_PATH)

    # Load the model
    input_size = len(feature_columns)
    model = PricePredictor(input_size=input_size)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

    # --- Make a Prediction on a Sample ---
    # Get a sample card's features and its true label
    sample_features = card_sample[feature_columns]
    true_label = bool(card_sample[config.TARGET_COL])

    # Get the prediction
    predicted_class, probability = predict(model, scaler, sample_features)

    # --- Display Results ---
    print("--- Prediction Report ---")
    # Try to display card name if available for better context
    card_name_display = f" (ID: {tcgplayer_id_to_predict})"
    if 'name' in card_sample and pd.notna(card_sample['name']):
        card_name_display = f" ({card_sample['name']}, ID: {tcgplayer_id_to_predict})"

    print(f"Analyzing Card: {card_name_display}")
    print(f"Model Prediction: '{predicted_class}' (Will it rise 30% in 6 months?)")

    # Calculate and display the confidence in the *actual* prediction made
    if predicted_class:  # If the prediction was 'True'
        confidence_in_prediction = probability
    else:  # If the prediction was 'False'
        confidence_in_prediction = 1 - probability

    print(f"Confidence in this Prediction: {confidence_in_prediction:.2%}")
    print(f"Actual Result in Dataset: '{true_label}'")
    print("-----------------------")