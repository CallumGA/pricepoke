import ast
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# TODO: CLEANUP AND COMMENT EXPLANATIONS!

@dataclass
class CleanConfig:
    date_columns: List[str] = field(default_factory=list)
    date_format: Optional[str] = None
    timezone: Optional[str] = None
    data_card_id_col: str = "cardId"
    prices_card_id_col: str = "cardId"
    prices_date_col: str = "date"
    data_release_date_col: Optional[str] = "releaseDate"
    set_code_col: str = "expCodeTCGP"
    set_name_col: Optional[str] = "expName"
    data_variants_col: Optional[str] = "variants"
    prices_variant_col: Optional[str] = "variant"
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
    min_non_null_ratio: float = 0.20
    duplicates_subset: Optional[List[str]] = None
    keep_duplicate: str | bool = "first"
    dtype_overrides: Dict[str, str] = field(default_factory=dict)
    special_char_patterns: List[str] = field(default_factory=lambda: [r"[\r\n\t]", r"\u200b", r"\ufeff"])
    price_columns: List[str] = field(default_factory=lambda: [
        "rawPrice", "gradedPriceTen", "gradedPriceNine"
    ])
    price_imputation_strategy: str = "zero"
    add_missingness_flags: bool = True
    outlier_strategy: Optional[str] = None
    outlier_columns: Optional[List[str]] = None
    z_thresh: float = 3.0
    iqr_k: float = 1.5


def remove_special_characters(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    out = df.copy()
    if not len(string_cols):
        return out
    for col in string_cols:
        s = out[col].astype("string")
        for pattern in cfg.special_char_patterns:
            s = s.str.replace(pattern, "", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        out[col] = s
    return out


def drop_sparse_rows(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    threshold = max(1, int(cfg.min_non_null_ratio * df.shape[1]))
    return df.dropna(axis=0, thresh=threshold)


def remove_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    return df.drop_duplicates(subset=cfg.duplicates_subset, keep=cfg.keep_duplicate)


def ensure_date_consistency(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    for col in cfg.date_columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_datetime(
            out[col],
            format=cfg.date_format,
            errors="coerce",
            utc=(cfg.timezone == "UTC"),
        )
        if cfg.timezone and cfg.timezone != "UTC":
            if out[col].dt.tz is None:
                out[col] = out[col].dt.tz_localize(cfg.timezone)
            else:
                out[col] = out[col].dt.tz_convert(cfg.timezone)
    return out


def impute_release_dates(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    rd_col = cfg.data_release_date_col
    if not rd_col or rd_col not in out.columns:
        return out
    set_col = None
    if cfg.set_code_col and cfg.set_code_col in out.columns:
        set_col = cfg.set_code_col
    elif cfg.set_name_col and cfg.set_name_col in out.columns:
        set_col = cfg.set_name_col
    out["releaseDate_missing"] = out[rd_col].isna().astype("Int8")
    if set_col is None:
        return out

    def _fill_group(s: pd.Series) -> pd.Series:
        if s.isna().all():
            return s
        if not s.mode(dropna=True).empty:
            mode_val = s.mode(dropna=True).iloc[0]
            return s.fillna(mode_val)
        return s.fillna(s.min())

    out[rd_col] = out.groupby(set_col)[rd_col].transform(_fill_group)
    return out


def add_age_features(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    rd_col = cfg.data_release_date_col
    dt_col = cfg.prices_date_col
    if rd_col in out.columns and dt_col in out.columns:
        out["age_days"] = (out[dt_col] - out[rd_col]).dt.days
    return out


def drop_useless_columns(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    for col in ["description", "variantMap", "price", "img", "releaseDate"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def _normalize_variant_value(val: Optional[str], cfg: CleanConfig) -> Optional[str]:
    if val is None:
        return None
    key = str(val).strip().lower()
    return cfg.variant_map.get(key, val if isinstance(val, str) else None)


def _parse_variants_list(val: Optional[str]) -> List[str]:
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
    return [item.strip() for item in s.split(",") if item.strip()]


def normalize_and_explode_variants(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    if not cfg.data_variants_col or cfg.data_variants_col not in df.columns:
        out = df.copy()
        out["variant"] = None
        return out
    tmp = df.copy()
    parsed = tmp[cfg.data_variants_col].apply(_parse_variants_list)
    parsed = parsed.apply(lambda lst: lst if lst else [None])
    tmp = tmp.assign(_variants_parsed=parsed).explode("_variants_parsed", ignore_index=True)
    tmp = tmp.rename(columns={"_variants_parsed": "variant"})
    tmp["variant"] = tmp["variant"].apply(lambda v: _normalize_variant_value(v, cfg))
    return tmp


def coerce_price_columns_numeric(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    for col in cfg.price_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def merge_data_and_prices(data_df: pd.DataFrame, prices_df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    data = data_df.copy()
    prices = prices_df.copy()

    # Basic validations
    if cfg.data_card_id_col not in data.columns:
        raise KeyError(f"Missing data key column: {cfg.data_card_id_col}")
    if cfg.prices_card_id_col not in prices.columns:
        raise KeyError(f"Missing prices key column: {cfg.prices_card_id_col}")

    # Parse dates
    if cfg.data_release_date_col and cfg.data_release_date_col in data.columns:
        data[cfg.data_release_date_col] = pd.to_datetime(data[cfg.data_release_date_col], errors="coerce", utc=True)
    if cfg.prices_date_col in prices.columns:
        prices[cfg.prices_date_col] = pd.to_datetime(prices[cfg.prices_date_col], errors="coerce", utc=True)

    # Normalize numeric types
    prices = coerce_price_columns_numeric(prices, cfg)

    # Normalize variant column in prices
    variant_col = cfg.prices_variant_col or "variant"
    if variant_col in prices.columns:
        prices[variant_col] = prices[variant_col].apply(lambda v: _normalize_variant_value(v, cfg))
    else:
        prices[variant_col] = None

    # Normalize and explode data variants
    data_expanded = normalize_and_explode_variants(data, cfg)

    # Drop duplicate price entries
    dedupe_subset = [cfg.prices_date_col, cfg.prices_card_id_col]
    if variant_col in prices.columns:
        dedupe_subset.append(variant_col)
    prices = prices.drop_duplicates(subset=dedupe_subset, keep="first")

    # === Pass 1: Strict merge on cardId + variant
    merged_strict = pd.merge(
        data_expanded,
        prices,
        how="left",
        left_on=[cfg.data_card_id_col, "variant"],
        right_on=[cfg.prices_card_id_col, variant_col],
        suffixes=("", "_price")
    )

    # Identify cardIds that did not match prices in pass 1
    unmatched_card_ids = merged_strict[merged_strict["rawPrice"].isna()][cfg.data_card_id_col].unique()

    # === Pass 2: Fallback loose merge on just cardId
    fallback_rows = data_expanded[data_expanded[cfg.data_card_id_col].isin(unmatched_card_ids)]
    fallback_merged = pd.merge(
        fallback_rows,
        prices,
        how="left",
        left_on=[cfg.data_card_id_col],
        right_on=[cfg.prices_card_id_col],
        suffixes=("", "_price")
    )

    # Mark fallback rows
    fallback_merged["fallback_merge"] = True
    merged_strict["fallback_merge"] = False

    # Drop rows from strict that were also fallback targets
    merged_strict = merged_strict[~merged_strict[cfg.data_card_id_col].isin(unmatched_card_ids)]

    # Combine both
    merged_all = pd.concat([merged_strict, fallback_merged], ignore_index=True)

    return merged_all


def drop_rows_missing_prices(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    required = ["date", "rawPrice", "gradedPriceTen", "gradedPriceNine"]
    out = df.copy()
    present = [c for c in required if c in out.columns]
    if not present:
        return out
    return out.dropna(subset=present, how="all")


def remove_fixed_price_rows(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """
    Removes rows where rawPrice, gradedPriceTen, and gradedPriceNine are all 20.0,
    which might indicate placeholder or invalid data.
    """
    price_cols = ["rawPrice", "gradedPriceTen", "gradedPriceNine"]
    out = df.copy()
    # Check if all required columns exist before applying the filter
    if not all(col in out.columns for col in price_cols):
        return out

    # Create a boolean mask for rows where all specified columns are 20.0
    mask = (out["rawPrice"] == 20.0) & \
           (out["gradedPriceTen"] == 20.0) & \
           (out["gradedPriceNine"] == 20.0)

    # Invert the mask to keep rows that do NOT meet the condition
    return out[~mask]


def add_missing_flags_and_impute_prices(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in (cfg.price_columns or []) if c in out.columns]
    if not cols:
        return out
    for col in cols:
        missing = out[col].isna() | out[col].eq(0.0)
        if cfg.add_missingness_flags:
            out[f"{col}_missing"] = missing.astype("Int8")
        if cfg.price_imputation_strategy == "median":
            median_val = out.loc[~missing, col].median(skipna=True)
            fill_val = median_val if pd.notna(median_val) else 0.0
        else:
            fill_val = 0.0
        out.loc[missing, col] = fill_val
    return out


def detect_and_remove_outliers(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    if cfg.outlier_strategy is None:
        return df
    out = df.copy()
    numeric_cols = cfg.outlier_columns or list(out.select_dtypes(include="number").columns)
    if not numeric_cols:
        return out
    if cfg.outlier_strategy == "zscore":
        for col in numeric_cols:
            col_series = out[col]
            mean = col_series.mean()
            std = col_series.std(ddof=0)
            if std == 0 or pd.isna(std):
                continue
            z = (col_series - mean).abs() / std
            out = out[z <= cfg.z_thresh]
        return out
    if cfg.outlier_strategy == "iqr":
        for col in numeric_cols:
            q1 = out[col].quantile(0.25)
            q3 = out[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - cfg.iqr_k * iqr
            upper = q3 + cfg.iqr_k * iqr
            out = out[(out[col] >= lower) & (out[col] <= upper)]
        return out
    return out


def ensure_correct_dtypes(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in (cfg.dtype_overrides or {}).items():
        if col in out.columns:
            try:
                out[col] = out[col].astype(dtype)
            except Exception:
                pass
    return out


def clean_data(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    steps = [
        remove_special_characters,
        drop_useless_columns,
        drop_sparse_rows,
        remove_duplicates,
        ensure_date_consistency,
        impute_release_dates,
        add_age_features,
        remove_fixed_price_rows,
        drop_rows_missing_prices,
        add_missing_flags_and_impute_prices,
        detect_and_remove_outliers,
        ensure_correct_dtypes,
    ]
    out = df.copy()
    for func in steps:
        out = func(out, cfg)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    data_df = pd.read_csv("/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/raw/pokemon_data.csv")
    prices_df = pd.read_csv("/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/raw/pokemon_prices.csv")

    config = CleanConfig(
        date_columns=["releaseDate", "date"],
        timezone="UTC",
        dtype_overrides={},
        outlier_strategy=None,
        outlier_columns=["rawPrice", "gradedPriceTen", "gradedPriceNine"],
        price_imputation_strategy="zero",
        add_missingness_flags=True,
    )

    merged_df = merge_data_and_prices(data_df, prices_df, config)
    cleaned_df = clean_data(merged_df, config)

    THRESH_PCT = 0.30

    if 'date' in cleaned_df.columns:
        if cleaned_df['date'].dtype.kind in {'O', 'U', 'S'}:
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], utc=True, errors='coerce')

    cleaned_df = cleaned_df.sort_values(['idTCGP', 'date'])

    if 'rawPrice_missing' in cleaned_df.columns:
        _raw_valid = (~cleaned_df['rawPrice_missing'].astype(bool)) & (cleaned_df['rawPrice'].fillna(0) > 0)
    else:
        _raw_valid = cleaned_df['rawPrice'].notna() & (cleaned_df['rawPrice'] > 0)

    _first_idx = cleaned_df[_raw_valid].groupby('idTCGP')['date'].idxmin()
    _first_raw_map = cleaned_df.loc[_first_idx, ['idTCGP', 'rawPrice']].set_index('idTCGP')['rawPrice'].to_dict()

    cleaned_df['first_raw'] = cleaned_df['idTCGP'].map(_first_raw_map)

    _label_mask = _raw_valid & cleaned_df['first_raw'].notna()
    cleaned_df.loc[_label_mask, 'y_point'] = (
        cleaned_df.loc[_label_mask, 'rawPrice'] >= (1.0 + THRESH_PCT) * cleaned_df.loc[_label_mask, 'first_raw']
    ).astype('Int8')

    cleaned_df['y_point'] = cleaned_df['y_point'].astype('Int8')

    cleaned_df['y_ever'] = (
        cleaned_df
        .sort_values(['idTCGP', 'date'])
        .groupby('idTCGP')['y_point']
        .transform(lambda s: (s.fillna(0).astype('Int8').cumsum() > 0).astype('Int8'))
    )

    cleaned_df['y'] = cleaned_df['y_ever'].astype('Int8')

    out_path = "/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/processed/cleaned_sales.csv"
    cleaned_df = cleaned_df.sort_values("date")
    cleaned_df.to_csv(out_path, index=False)

    try:
        vc = cleaned_df['y'].value_counts(dropna=False)
        print("Label distribution (y):\n", vc.to_string())
        vc_point = cleaned_df['y_point'].value_counts(dropna=False)
        print("Label distribution (y_point):\n", vc_point.to_string())
        vc_ever = cleaned_df['y_ever'].value_counts(dropna=False)
        print("Label distribution (y_ever):\n", vc_ever.to_string())
    except Exception as e:
        print("Label diagnostics error:", e)

    print(f"Wrote merged, cleaned CSV to: {out_path}")
