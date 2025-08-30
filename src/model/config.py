from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
ENCODER_DIR = MODEL_DIR / "encoders"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV_PATH = DATA_DIR / "pokemon_final_with_labels.csv"
MODEL_SAVE_DIR = MODEL_DIR / "price_predictor_model"
SCALER_PATH = ENCODER_DIR / "scaler.pkl"

IDENTIFIER_COLS = ['tcgplayer_id', 'name']
TARGET_COL = 'y'