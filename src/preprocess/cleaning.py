
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    # Column that holds condition/grade labels and a normalization map
    condition_column: Optional[str] = None
    # Map various spellings → canonical form, e.g., {"nm": "Near Mint", "near-mint": "Near Mint"}
    condition_map: Dict[str, str] = field(default_factory=dict)

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


def normalize_categorical_values(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Normalize categorical labels (e.g., card condition values) using a mapping.

    Example:
        cfg.condition_column = "condition"
        cfg.condition_map = {
            "nm": "Near Mint",
            "near mint": "Near Mint",
            "near-mint": "Near Mint",
            "n/m": "Near Mint",
        }

    NOTE: Pre-normalizes keys to lowercase, stripped form for robust matching.
    """
    cleaned = df.copy()
    if cfg.condition_column and cfg.condition_column in cleaned.columns and cfg.condition_map:
        norm_map = {str(k).strip().lower(): v for k, v in cfg.condition_map.items()}
        col = cleaned[cfg.condition_column].astype("string").str.strip().str.lower()
        cleaned[cfg.condition_column] = col.map(norm_map).fillna(cleaned[cfg.condition_column])
    return cleaned


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
    steps = [
        remove_special_characters,
        drop_sparse_rows,
        remove_duplicates,
        ensure_date_consistency,
        normalize_categorical_values,
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
    # Example scaffold (no I/O by default). Replace with your own paths.
    # df = pd.read_csv("/path/to/raw.csv")
    # cfg = CleanConfig(
    #     date_columns=["sale_date"],
    #     timezone="UTC",
    #     condition_column="condition",
    #     condition_map={
    #         "nm": "Near Mint",
    #         "near-mint": "Near Mint",
    #         "near mint": "Near Mint",
    #     },
    #     dtype_overrides={"price": "float64"},
    #     outlier_strategy="iqr",
    #     outlier_columns=["price"],
    # )
    # cleaned = clean_data(df, cfg)
    # cleaned.to_csv("/path/to/cleaned.csv", index=False)
    pass
