import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import ast

"""
TODO: Clean the data

This module provides a configurable cleaning pipeline with one function per
bullet in the TODO list:
  - remove special characters
  - drop rows that are 80% empty
  - remove duplicate rows
  - make sure dates are consistent
  - normalize categorical values (e.g., condition strings)
  - look for outliers via statistical methods to remove them
  - ensure correct data types

Each function contains basic, safe defaults and clearly marked TODOs for you
to customize to your dataset (column names, patterns, thresholds, etc.).
"""


# =============================
# Configuration (edit as needed)
# =============================
@dataclass
class CleanConfig:
    # Columns that contain dates to be parsed/normalized
    date_columns: List[str] = field(default_factory=list)
    # Optional explicit date format (e.g., "%Y-%m-%d"); if None, pandas will infer
    date_format: Optional[str] = None
    # Optional timezone string (e.g., "UTC"); if None, leave tz-naive
    timezone: Optional[str] = None

    # === Merge configuration (data <-> prices) ===
    # Key column names
    data_card_id_col: str = "cardId"
    prices_card_id_col: str = "cardId"
    # Date column in prices and data (for normalization)
    prices_date_col: str = "date"
    data_release_date_col: Optional[str] = "releaseDate"

    # Variant handling
    data_variants_col: Optional[str] = "variants"  # list-like string, e.g., "[\"Holofoil\"]"
    prices_variant_col: Optional[str] = "variant"   # string per row in prices
    # Canonical variant mapping (case-insensitive keys)
    variant_map: Dict[str, str] = field(default_factory=lambda: {
        "normal": "Normal",
        "reverse holo": "Reverse Holofoil",
        "reverse holofoil": "Reverse Holofoil",
        "holo": "Holofoil",
        "holofoil": "Holofoil",
        "foil": "Holofoil",
        "non-holo": "Normal",
        "": "Normal",
        "none": "Normal",
        "na": "Normal",
    })


    # Keep rows with at least this ratio of non-null values; 0.20 ≈ "drop rows 80% empty"
    min_non_null_ratio: float = 0.20

    # Subset of columns to consider when dropping duplicates (None → all columns)
    duplicates_subset: Optional[List[str]] = None
    # Keep the first occurrence when dropping duplicates (alternatives: "last" or False to mark only)
    keep_duplicate: str | bool = "first"

    # Dtype overrides (pandas dtype strings), e.g., {"price": "float64", "qty": "Int64"}
    dtype_overrides: Dict[str, str] = field(default_factory=dict)

    # Patterns of special characters to remove from string columns
    # NOTE: These are applied via regex replace with "" (empty string)
    special_char_patterns: List[str] = field(
        default_factory=lambda: [r"[\r\n\t]", r"\u200b", r"\ufeff"]
    )

    # Price columns present in prices file to coerce to numeric
    price_columns: List[str] = field(default_factory=lambda: [
        "rawPrice", "gradedPriceTen", "gradedPriceNine"
    ])

    # Outlier handling strategy: None | "zscore" | "iqr"
    outlier_strategy: Optional[str] = None
    # Numeric columns to evaluate for outliers (None → all numeric)
    outlier_columns: Optional[List[str]] = None
    # Thresholds for strategies
    z_thresh: float = 3.0
    iqr_k: float = 1.5


# =============================
# Cleaning steps (customize)
# =============================

def remove_special_characters(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Remove configured special characters from all string columns.

    TODO: If certain columns should be excluded (e.g., IDs), filter here.
    """
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    cleaned = df.copy()
    if not len(str_cols):
        return cleaned
    for col in str_cols:
        series = cleaned[col].astype("string")
        for pat in cfg.special_char_patterns:
            series = series.str.replace(pat, "", regex=True)
        series = series.str.replace(r"\s+", " ", regex=True).str.strip()
        cleaned[col] = series
    return cleaned


def drop_sparse_rows(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Drop rows that are ≥ (1 - min_non_null_ratio) null.

    With the default 0.20, this drops rows that are 80% empty.
    """
    thresh = max(1, int(cfg.min_non_null_ratio * df.shape[1]))
    return df.dropna(axis=0, thresh=thresh)


def remove_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Remove exact duplicate rows.

    TODO: If you want to dedupe on a key subset (e.g., ["card_id", "date", "condition"]),
    set cfg.duplicates_subset accordingly.
    """
    return df.drop_duplicates(subset=cfg.duplicates_subset, keep=cfg.keep_duplicate)


def ensure_date_consistency(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Parse/normalize date columns to a consistent dtype and timezone.

    - If cfg.date_format is provided, use it; otherwise, infer.
    - If cfg.timezone is provided, localize or convert to that timezone.
    """
    cleaned = df.copy()
    for col in cfg.date_columns:
        if col not in cleaned.columns:
            continue
        # Parse
        cleaned[col] = pd.to_datetime(
            cleaned[col],
            format=cfg.date_format,
            errors="coerce",
            utc=(cfg.timezone == "UTC"),
        )
        # Localize/convert
        if cfg.timezone and cfg.timezone != "UTC":
            # If tz-naive, localize; else convert
            if cleaned[col].dt.tz is None:
                cleaned[col] = cleaned[col].dt.tz_localize(cfg.timezone)
            else:
                cleaned[col] = cleaned[col].dt.tz_convert(cfg.timezone)
    return cleaned




# =============================
# Cleaning steps (customize)
# =============================

def drop_useless_columns(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    cleaned = df.copy()
    for col in ["description", "variantMap", "price"]:
        if col in cleaned.columns:
            cleaned = cleaned.drop(columns=[col])
    return cleaned


def _normalize_variant_value(val: Optional[str], cfg: CleanConfig) -> Optional[str]:
    if val is None:
        return None
    key = str(val).strip().lower()
    return cfg.variant_map.get(key, val if isinstance(val, str) else None)

def _parse_variants_list(val: Optional[str]) -> List[str]:
    """Parse a list-like string such as '["Normal","Reverse Holofoil"]' into a Python list.
    Falls back to a single-item list if parsing fails.
    """
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # Fallback: split on commas
    return [item.strip() for item in s.split(",") if item.strip()]

def normalize_and_explode_variants(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """From a dataset that has a list-like variants column (`cfg.data_variants_col`),
    produce one row per variant with a new column `variant`.
    If no variants exist, keep a single row with variant=None.
    """
    if not cfg.data_variants_col or cfg.data_variants_col not in df.columns:
        out = df.copy()
        out["variant"] = None
        return out

    tmp = df.copy()
    parsed = tmp[cfg.data_variants_col].apply(_parse_variants_list)
    # If a row has no variants, use [None] so explode keeps one row
    parsed = parsed.apply(lambda lst: lst if lst else [None])
    tmp = tmp.assign(_variants_parsed=parsed).explode("_variants_parsed", ignore_index=True)
    tmp = tmp.rename(columns={"_variants_parsed": "variant"})
    # Normalize variant values to canonical labels
    tmp["variant"] = tmp["variant"].apply(lambda v: _normalize_variant_value(v, cfg))
    return tmp

def coerce_price_columns_numeric(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cfg.price_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned

def merge_data_and_prices(data_df: pd.DataFrame, prices_df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Merge card metadata (data_df) with prices (prices_df) without creating duplicate
    cartesian products. Strategy:
      1) Clean basics (special chars, dates, numeric prices)
      2) Normalize/align `cardId` columns and `variant` values (explode data variants)
      3) Drop exact duplicate price observations by [date, cardId, variant]
      4) Left-join prices onto expanded data on [cardId, variant]; if `variant` is missing
         in prices, interpret as `Normal` for matching (configurable via cfg.variant_map)
    """
    # Shallow copies
    data = data_df.copy()
    prices = prices_df.copy()

    # Normalize column names of keys
    if cfg.data_card_id_col not in data.columns:
        raise KeyError(f"Missing data key column: {cfg.data_card_id_col}")
    if cfg.prices_card_id_col not in prices.columns:
        raise KeyError(f"Missing prices key column: {cfg.prices_card_id_col}")

    # Normalize and coerce dates
    # For data (releaseDate)
    if cfg.data_release_date_col and cfg.data_release_date_col in data.columns:
        data[cfg.data_release_date_col] = pd.to_datetime(data[cfg.data_release_date_col], errors="coerce", utc=True)
    # For prices (transaction date)
    if cfg.prices_date_col in prices.columns:
        prices[cfg.prices_date_col] = pd.to_datetime(prices[cfg.prices_date_col], errors="coerce", utc=True)

    # Price columns to numeric
    prices = coerce_price_columns_numeric(prices, cfg)

    # Normalize variant columns
    # Prices: normalize text; empty/NaN => map to canonical 'Normal'
    if cfg.prices_variant_col and cfg.prices_variant_col in prices.columns:
        prices[cfg.prices_variant_col] = prices[cfg.prices_variant_col].apply(
            lambda v: _normalize_variant_value(v if pd.notna(v) else "", cfg)
        )
    else:
        prices[cfg.prices_variant_col or "variant"] = None

    # Data: explode variants -> column `variant`
    data_expanded = normalize_and_explode_variants(data, cfg)

    # Drop exact duplicate price observations (keep first)
    dedupe_subset = [cfg.prices_date_col, cfg.prices_card_id_col]
    if cfg.prices_variant_col:
        dedupe_subset.append(cfg.prices_variant_col)
    prices = prices.drop_duplicates(subset=dedupe_subset, keep="first")

    # Prepare join keys
    left_on = [cfg.data_card_id_col, "variant"]
    right_on = [cfg.prices_card_id_col, cfg.prices_variant_col or "variant"]

    merged = pd.merge(
        data_expanded,
        prices,
        how="left",
        left_on=left_on,
        right_on=right_on,
        suffixes=("", "_price")
    )

    return merged


def drop_rows_missing_prices(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Drop rows where all key price-related columns (including date) are missing."""
    required_cols = ["date", "rawPrice", "gradedPriceTen", "gradedPriceNine"]
    cleaned = df.copy()
    present_cols = [c for c in required_cols if c in cleaned.columns]
    if not present_cols:
        return cleaned
    return cleaned.dropna(subset=present_cols, how="all")


def detect_and_remove_outliers(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Detect and remove outliers using basic statistical heuristics.

    Strategies:
      - zscore: drop rows where |z| > cfg.z_thresh
      - iqr: drop rows outside [Q1 - k*IQR, Q3 + k*IQR]

    TODO: Validate chosen columns and thresholds against your data distribution.
    """
    if cfg.outlier_strategy is None:
        return df

    cleaned = df.copy()
    num_cols = cfg.outlier_columns or list(cleaned.select_dtypes(include="number").columns)
    if not num_cols:
        return cleaned

    if cfg.outlier_strategy == "zscore":
        # Compute per-column z-scores and filter
        for col in num_cols:
            col_series = cleaned[col]
            mean = col_series.mean()
            std = col_series.std(ddof=0)
            if std == 0 or pd.isna(std):
                continue
            z = (col_series - mean).abs() / std
            cleaned = cleaned[z <= cfg.z_thresh]
        return cleaned

    if cfg.outlier_strategy == "iqr":
        for col in num_cols:
            q1 = cleaned[col].quantile(0.25)
            q3 = cleaned[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - cfg.iqr_k * iqr
            upper = q3 + cfg.iqr_k * iqr
            cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
        return cleaned

    # Unknown strategy → return unchanged
    return cleaned


def ensure_correct_dtypes(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Coerce columns to configured pandas dtypes where specified.

    TODO: Extend with smarter inference if desired.
    """
    cleaned = df.copy()
    for col, dtype in (cfg.dtype_overrides or {}).items():
        if col in cleaned.columns:
            try:
                cleaned[col] = cleaned[col].astype(dtype)
            except Exception:
                # Leave as-is if coercion fails; you can add logging or error handling here.
                pass
    return cleaned


# =============================
# Orchestrator
# =============================

def clean_data(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Run the full cleaning pipeline according to the provided config.

    Order of operations is aligned with the TODO list.
    """
    # If you need to merge card metadata with price history first, use `merge_data_and_prices` externally
    # and then pass the merged DataFrame into this function.
    steps = [
        remove_special_characters,
        drop_useless_columns,
        drop_sparse_rows,
        remove_duplicates,
        ensure_date_consistency,
        drop_rows_missing_prices,
        detect_and_remove_outliers,
        ensure_correct_dtypes,
    ]
    cleaned = df.copy()
    for step in steps:
        cleaned = step(cleaned, cfg)
    return cleaned.reset_index(drop=True)


# =============================
# Example usage (adjust or remove)
# =============================
if __name__ == "__main__":
    import pandas as pd
    data_df = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/raw/pokemon_data.csv")
    prices_df = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/raw/pokemon_prices.csv")

    # Basic config: set date columns and types
    cfg = CleanConfig(
        date_columns=["releaseDate", "date"],
        timezone="UTC",
        dtype_overrides={
            # Example: coerce price columns if present in merged frame
            # "rawPrice": "float64",
            # "gradedPriceTen": "float64",
            # "gradedPriceNine": "float64",
        },
        outlier_strategy=None,   # or "iqr" / "zscore" with `outlier_columns=["rawPrice"]`
        outlier_columns=["rawPrice", "gradedPriceTen", "gradedPriceNine"],
    )

    # === Merge step (avoids duplicates) ===
    merged = merge_data_and_prices(data_df, prices_df, cfg)

    # === Clean the merged dataset ===
    cleaned = clean_data(merged, cfg)

    # Save result
    cleaned.to_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/raw/pokemon_merged_cleaned.csv", index=False)
    print("Wrote merged, cleaned CSV to: /Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/raw/pokemon_merged_cleaned.csv")
