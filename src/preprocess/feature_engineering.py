import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import json

# TODO: clean this up significantly

CATEGORICAL_COLS_DEFAULT = [
    "rarity",
    "energyType",
    "cardType",
    "variant",
    "expCodeTCGP",
]

NUMERIC_COLS_DEFAULT = [
    "rawPrice",
    "gradedPriceTen",
    "gradedPriceNine",
    "rawPrice_missing",
    "gradedPriceTen_missing",
    "gradedPriceNine_missing",
    "age_days",
    "first_raw",
    "price_ratio_to_first",
    "log_raw",
    "log_g10",
    "log_g9",
    "price_vs_rolling_avg",
]

IDENTIFIER_COLS = [
    "cardId",
    "idTCGP",
    "name",
    "expIdTCGP",
    "expName",
    "expCardNumber",
    "pokedex",
    "variants",
    "date",
]

TARGET_COLS = [
    "y",
    "y_point",
    "y_ever",
]


def _present_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def build_feature_frame(df: pd.DataFrame,
                        categorical_cols: list[str] | None = None,
                        numeric_cols: list[str] | None = None,
                        fit_encoder: bool = True,
                        encoder: OneHotEncoder | None = None,
                        scaler: StandardScaler | None = None,
                        max_onehot_cardinality: int = 30) -> tuple[pd.DataFrame, OneHotEncoder | None, StandardScaler]:
    cat_cols = _present_columns(df, categorical_cols or CATEGORICAL_COLS_DEFAULT)
    num_cols = _present_columns(df, numeric_cols or NUMERIC_COLS_DEFAULT)

    base = df.drop(columns=_present_columns(df, IDENTIFIER_COLS + TARGET_COLS), errors="ignore").copy()

    onehot_cols = []
    skipped_high_card_cols = []
    for c in cat_cols:
        try:
            unique_ct = base[c].nunique(dropna=True)
        except Exception:
            unique_ct = max_onehot_cardinality + 1
        if unique_ct <= max_onehot_cardinality:
            onehot_cols.append(c)
        else:
            skipped_high_card_cols.append(c)

    flag_cols = [c for c in num_cols if c.endswith("_missing")]
    scale_cols = [c for c in num_cols if c not in flag_cols]
    base[scale_cols] = base[scale_cols].fillna(0.0)

    if fit_encoder:
        if onehot_cols:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(base[onehot_cols])
        else:
            encoder = None
            encoded = np.empty((len(base), 0))
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaled = scaler.fit_transform(base[scale_cols]) if scale_cols else np.empty((len(base), 0))
    else:
        if scaler is None:
            raise ValueError("When fit_encoder=False, scaler must be provided.")
        if onehot_cols:
            if encoder is None:
                raise ValueError("When fit_encoder=False and onehot_cols present, encoder must be provided.")
            encoded = encoder.transform(base[onehot_cols])
        else:
            encoded = np.empty((len(base), 0))
        scaled = scaler.transform(base[scale_cols]) if scale_cols else np.empty((len(base), 0))

    encoded = np.asarray(encoded)
    scaled = np.asarray(scaled)
    if encoded.ndim == 1:
        encoded = encoded.reshape(-1, 1)
    if scaled.ndim == 1:
        scaled = scaled.reshape(-1, 1)

    flags = base[flag_cols].to_numpy() if flag_cols else np.empty((len(base), 0))

    cat_feature_names = list(encoder.get_feature_names_out(onehot_cols)) if onehot_cols and encoder is not None else []
    feature_names = list(scale_cols) + list(flag_cols) + cat_feature_names

    parts = []
    if scaled.size:
        parts.append(scaled)
    if flags.size:
        parts.append(flags)
    if encoded.size:
        parts.append(encoded)
    features = np.hstack(parts) if parts else np.empty((len(base), 0))
    X = pd.DataFrame(features, columns=feature_names, index=df.index)

    if skipped_high_card_cols:
        print(f"Skipped high-cardinality categoricals (>{max_onehot_cardinality} unique): {skipped_high_card_cols}")

    return X, encoder, scaler


def save_artifacts(encoder: OneHotEncoder | None, scaler: StandardScaler, out_dir: str, onehot_cols: list[str], scale_cols: list[str], flag_cols: list[str], max_onehot_cardinality: int) -> tuple[str | None, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    enc_path = os.path.join(out_dir, "onehot_encoder.pkl")
    sc_path = os.path.join(out_dir, "standard_scaler.pkl")
    cfg_path = os.path.join(out_dir, "encoding_config.json")
    if encoder is not None:
        joblib.dump(encoder, enc_path)
    else:
        enc_path = None
    joblib.dump(scaler, sc_path)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "onehot_cols": onehot_cols,
            "scale_cols": scale_cols,
            "flag_cols": flag_cols,
            "max_onehot_cardinality": max_onehot_cardinality,
        }, f)
    return enc_path, sc_path, cfg_path


def main():
    # Configuration is now hardcoded to simplify the pipeline, removing the need for command-line arguments.
    class Config:
        in_path = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/processed/cleaned_sales.csv"
        out_path = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/processed/pokemon_final.csv"
        artifacts_dir = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/models/encoders"
        max_onehot_card = 30

    args = Config()

    df = pd.read_csv(args.in_path, low_memory=False)

    # --- Create Historical Features (MUST be done before deduplication) ---
    # We use the full time-series data to calculate trend-based features.
    # This gives the model a "memory" of the card's recent price momentum.
    if "rawPrice" in df.columns and "idTCGP" in df.columns and "date" in df.columns:
        print("Creating historical trend features...")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # We must sort by date to ensure the rolling window is correct
        df.sort_values(by=["idTCGP", "date"], inplace=True)

        # Calculate a 7-day rolling average price for each card.
        df['rolling_avg_price_7d'] = df.groupby('idTCGP')['rawPrice'].transform(
            lambda x: x.rolling(window=7, min_periods=2).mean()
        )

        # Calculate the ratio of the current price to the rolling average.
        df['price_vs_rolling_avg'] = df['rawPrice'] / df['rolling_avg_price_7d']

        # Replace infinite values (from division by zero) and fill NaNs.
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['price_vs_rolling_avg'].fillna(1.0, inplace=True)

        # We can now drop the intermediate rolling average column.
        df.drop(columns=['rolling_avg_price_7d'], inplace=True)
        print("Historical trend features created.\n")

    # --- Deduplicate Data to Keep Only the Most Recent Entry Per Card ---
    if "date" in df.columns and "idTCGP" in df.columns:
        print("Deduplicating data to keep only the most recent entry per card...")
        df.dropna(subset=["date"], inplace=True) # Drop rows where date could not be parsed
        df.sort_values(by=["idTCGP", "date"], inplace=True)

        initial_rows = len(df)
        df.drop_duplicates(subset=["idTCGP"], keep="last", inplace=True)
        final_rows = len(df)
        print(f"Deduplication complete. Processed {initial_rows} rows down to {final_rows} unique cards.\n")
    numeric_to_coerce = [
        "rawPrice",
        "gradedPriceTen",
        "gradedPriceNine",
        "rawPrice_missing",
        "gradedPriceTen_missing",
        "gradedPriceNine_missing",
        "age_days",
        "first_raw",
    ]
    for col in numeric_to_coerce:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "first_raw" in df.columns and "rawPrice" in df.columns:
        df["price_ratio_to_first"] = np.where(
            (df["first_raw"].notna()) & (df["first_raw"] > 0) & df["rawPrice"].notna(),
            df["rawPrice"] / df["first_raw"],
            np.nan,
        )
    if "rawPrice" in df.columns:
        df["log_raw"] = np.log1p(df["rawPrice"].astype(float))
    if "gradedPriceTen" in df.columns:
        df["log_g10"] = np.log1p(df["gradedPriceTen"].astype(float))
    if "gradedPriceNine" in df.columns:
        df["log_g9"] = np.log1p(df["gradedPriceNine"].astype(float))

    X, encoder, scaler = build_feature_frame(df, max_onehot_cardinality=args.max_onehot_card)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # Also emit a sidecar file that includes the identifier, features, and the primary label `y`
    if 'y' in df.columns and 'idTCGP' in df.columns:
        out_with_labels = args.out_path[:-4] + "_with_labels.csv" if args.out_path.lower().endswith(".csv") else args.out_path + "_with_labels.csv"
        
        # Prepare identifier columns, renaming for consistency with other scripts
        # We include 'name' here so the prediction script can display it.
        id_cols = df[['idTCGP', 'name']].rename(columns={'idTCGP': 'tcgplayer_id'})

        # Prepare the target column
        # This is a more robust way to clean the target column.
        # It warns if any values are being silently changed, which is crucial for data integrity.
        y_series = pd.to_numeric(df['y'], errors='coerce')

        y_clean = y_series.fillna(0).astype('int64')

        # Combine identifier, features, and target into the final dataframe
        X_with_id_and_y = pd.concat([
            id_cols.reset_index(drop=True),
            X.reset_index(drop=True),
            y_clean.reset_index(drop=True).to_frame('y')
        ], axis=1)
        X_with_id_and_y.to_csv(out_with_labels, index=False)
        print(f"Wrote features+labels to: {out_with_labels} (targets: ['y'])")

    cat_cols = _present_columns(df, CATEGORICAL_COLS_DEFAULT)
    num_cols = _present_columns(df, NUMERIC_COLS_DEFAULT)
    flag_cols = [c for c in num_cols if c.endswith("_missing")]
    scale_cols = [c for c in num_cols if c not in flag_cols]
    onehot_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= args.max_onehot_card]

    enc_path, sc_path, cfg_path = save_artifacts(encoder, scaler, args.artifacts_dir, onehot_cols, scale_cols, flag_cols, args.max_onehot_card)

    if 'y' in df.columns:
        labeled_out = args.out_path[:-4] + "_with_labels.csv" if args.out_path.lower().endswith(".csv") else args.out_path + "_with_labels.csv"
        print(f"Wrote features+labels to: {labeled_out}")
    print(f"Saved encoder to: {enc_path}")
    print(f"Saved scaler to:  {sc_path}")
    print(f"Saved config to: {cfg_path}")
    nnz = (X.to_numpy() != 0).sum()
    total = X.size
    density = nnz / total if total else 0.0
    print(f"Feature nonzero density: {density:.4f}")


if __name__ == "__main__":
    main()
