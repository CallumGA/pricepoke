import gradio as gr
import torch
import joblib
import pandas as pd
import numpy as np
import os
import json
import re
from safetensors.torch import load_file
from typing import List, Tuple

# The network.py file containing the PricePredictor class must be in the same directory
from network import PricePredictor

# --- Constants for file paths in the Hugging Face Space repo ---
# You will need to upload your 'model' and 'data' directories to your Space.
MODEL_DIR = "model"
DATA_DIR = "data"
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
DATA_PATH = os.path.join(DATA_DIR, "pokemon_final_with_labels.csv")


# --- Model and Data Loading ---
# These functions are adapted from your original predict.py script.

def load_model_and_config(model_dir: str) -> Tuple[torch.nn.Module, List[str]]:
    """Loads a trained model and its configuration from a directory."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    model = PricePredictor(input_size=model_config["input_size"])
    weights_path = os.path.join(model_dir, "model.safetensors")
    model.load_state_dict(load_file(weights_path))
    model.eval()
    return model, model_config["feature_columns"]

def perform_prediction(model: torch.nn.Module, scaler, input_features: pd.Series) -> Tuple[bool, float]:
    """Performs a prediction on a single sample of card features."""
    features_np = input_features.to_numpy(dtype="float32").reshape(1, -1)
    features_scaled = scaler.transform(features_np)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = model(features_tensor)
        probability = torch.sigmoid(logit).item()
        predicted_class = bool(round(probability))

    return predicted_class, probability

# --- Pre-load all assets to avoid reloading on each prediction ---
# This part of the script runs once when the Gradio app starts.
try:
    model, feature_columns = load_model_and_config(MODEL_DIR)
    scaler = joblib.load(SCALER_PATH)
    full_data = pd.read_csv(DATA_PATH)

    # Create a user-friendly list of cards for the dropdown: "Card Name (ID: 12345)"
    full_data['display_name'] = full_data.apply(
        lambda row: f"{row['name']} (ID: {row['tcgplayer_id']})", axis=1
    )
    card_choices = sorted(full_data['display_name'].unique().tolist())
    ASSETS_LOADED = True
except FileNotFoundError as e:
    print(f"Error loading necessary files: {e}")
    print("Please make sure you have uploaded the 'model' and 'data' directories to your Hugging Face Space.")
    card_choices = ["Error: Model or data files not found. Check logs."]
    ASSETS_LOADED = False


# --- Gradio Prediction Function ---

def predict_price_trend(card_display_name: str) -> str:
    """
    This is the main function called by the Gradio interface.
    It takes the selected card's display name, finds its data, and returns a formatted prediction.
    """
    if not ASSETS_LOADED:
        return "## ❌ Application Error\nAssets could not be loaded. Please check the logs on Hugging Face Spaces for details. You may need to upload your `model` and `data` directories."

    # Extract tcgplayer_id from the display name string, e.g., "Charizard (ID: 123)" -> 123
    try:
        tcgplayer_id = int(re.search(r'\(ID: (\d+)\)', card_display_name).group(1))
    except (AttributeError, ValueError):
        return f"## ⚠️ Input Error\nCould not parse ID from '{card_display_name}'. Please select a valid card from the dropdown."

    card_data = full_data[full_data['tcgplayer_id'] == tcgplayer_id]
    if card_data.empty:
        # This is a defensive check; it shouldn't happen if the dropdown is built from the same data.
        return f"## ⚠️ Internal Error\nCould not find data for ID {tcgplayer_id}. Please restart the Space or select another card."

    card_sample = card_data.iloc[0]
    sample_features = card_sample[feature_columns]

    # Perform prediction
    predicted_class, probability = perform_prediction(model, scaler, sample_features)

    # --- Format the Output ---
    prediction_text = "**RISE**" if predicted_class else "**NOT RISE**"
    confidence = probability if predicted_class else 1 - probability

    # Get true label for comparison if it exists (using the target from your config)
    target_col = 'price_will_rise_30_in_6m' # NOTE: Assumed target column name. Change if yours is different.
    true_label_text = ""
    if target_col in card_sample and pd.notna(card_sample[target_col]):
        true_label = bool(card_sample[target_col])
        true_label_text = f"\n- **Actual Result in Dataset:** The price did **{'RISE' if true_label else 'NOT RISE'}**."

    # Build the markdown response
    output = f"""
    ## 🔮 Prediction Report for {card_sample['name']}
    - **Prediction:** The model predicts the card's price will {prediction_text} by 30% in the next 6 months.
    - **Confidence:** {confidence:.2%}
    {true_label_text}
    """
    return output


# --- Create and Launch the Gradio Interface ---
iface = gr.Interface(
    fn=predict_price_trend,
    inputs=gr.Dropdown(
        choices=card_choices,
        label="Select a Pokémon Card",
        info="Choose a card from the dataset to predict its price trend."
    ),
    outputs=gr.Markdown(),
    title="PricePoke: Pokémon Card Price Trend Predictor",
    description="""
    Select a Pokémon card to predict whether its market price will increase by 30% or more over the next 6 months.
    This model was trained on historical TCGPlayer market data.
    """,
    examples=[[card_choices[0]] if card_choices and ASSETS_LOADED else []],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()