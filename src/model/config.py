from pathlib import Path

# --- Path Configuration ---
# Use pathlib for OS-agnostic path handling. This makes your code work on Mac, Windows, or Linux.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
ENCODER_DIR = MODEL_DIR / "encoders"

# Ensure directories exist when the config is imported
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
INPUT_CSV_PATH = DATA_DIR / "pokemon_final_with_labels.csv"
MODEL_SAVE_PATH = MODEL_DIR / "price_predictor.pth"
SCALER_PATH = ENCODER_DIR / "scaler.pkl"


# --- Data Configuration ---
# Define columns that are not features for the model.
# 'name' is included so it's available for reporting but excluded from training.
IDENTIFIER_COLS = ['tcgplayer_id', 'name']
TARGET_COL = 'y'