"""Loading and cleaning helpers for the Steam game price prediction project."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATASET_URL = "https://zenodo.org/records/15121027"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "steam-games.csv"
TARGET_COLUMN = "price"
FEATURE_COLUMNS = ["genres", "launch_year", "reviews", "general_reviews"]


def parse_price(value: Any) -> float:
    """Convert common EU/Spanish currency formats to a float in euros.

    Examples handled: 4.99, '4,99 €', '1.234,56 €', '€ 4.99', 'Free', 'Gratis'.
    Missing/unknown values return NaN. Free games are zero.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "n/a", "na", "-", "—"}:
        return np.nan
    if any(token in text for token in ("free", "gratis", "gratuito", "gratuit", "sin coste", "sin cargo")):
        return 0.0

    text = text.replace("\u00a0", " ").replace("'", "")
    text = re.sub(r"[^0-9,\.\-]", "", text)
    if not text or text in {"-", ".", ","}:
        return np.nan

    # When both symbols occur, the rightmost symbol is treated as decimal.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # Dataset is collected from a Spanish-language store, where comma decimals are common.
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            text = parts[0].replace(".", "") + "." + parts[1]
        elif len(parts) > 2:
            text = "".join(parts[:-1]) + "." + parts[-1]
        else:
            text = "".join(parts)
    elif text.count(".") > 1:
        parts = text.split(".")
        if len(parts[-1]) in (1, 2):
            text = "".join(parts[:-1]) + "." + parts[-1]
        else:
            text = "".join(parts)
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_count(value: Any) -> float:
    """Parse review counts such as 1.234 or '1,234 reviews' into a number."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    digits = re.sub(r"[^0-9]", "", str(value))
    return float(digits) if digits else np.nan


def _normalise_column_names(columns) -> list[str]:
    return [re.sub(r"\s+", "_", str(c).strip().lower()) for c in columns]


def read_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read CSV with delimiter sniffing to tolerate comma/semicolon variants."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python scripts/download_data.py"
        )
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    except Exception:
        df = pd.read_csv(path, encoding_errors="replace")
    if df.shape[1] == 1:
        for sep in (";", ",", "\t"):
            try:
                candidate = pd.read_csv(path, sep=sep, encoding_errors="replace")
                if candidate.shape[1] > 1:
                    df = candidate
                    break
            except Exception:
                continue
    df.columns = _normalise_column_names(df.columns)
    return df


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned frame with the columns needed by the first model.

    The target is `price` (observed current price in euros). `discount` and
    `original_price` are intentionally excluded from model inputs to reduce target leakage.
    """
    df = raw.copy()
    df.columns = _normalise_column_names(df.columns)

    aliases = {
        "genre": "genres",
        "release_date": "launch_date",
        "launchdate": "launch_date",
        "review_count": "reviews",
        "general_review": "general_reviews",
        "current_price": "price",
        "final_price": "price",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected a '{TARGET_COLUMN}' column. Available columns: {list(df.columns)}"
        )

    if "is_error" in df.columns:
        err = pd.to_numeric(df["is_error"], errors="coerce")
        df = df.loc[err.fillna(0).eq(0)].copy()

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(parse_price)
    df["reviews"] = df["reviews"].map(parse_count) if "reviews" in df else np.nan

    if "genres" not in df:
        df["genres"] = "Unknown"
    if "general_reviews" not in df:
        df["general_reviews"] = "Unknown"
    if "launch_date" not in df:
        df["launch_date"] = np.nan

    # Extract year from dates with common Spanish/English formats; fallback to any 19xx/20xx year.
    date_text = df["launch_date"].astype(str)
    parsed_dates = pd.to_datetime(date_text, errors="coerce", dayfirst=True)
    fallback_year = date_text.str.extract(r"\b((?:19|20)\d{2})\b", expand=False)
    df["launch_year"] = pd.to_numeric(parsed_dates.dt.year, errors="coerce").astype(float)
    missing_year = df["launch_year"].isna() & fallback_year.notna()
    df.loc[missing_year, "launch_year"] = pd.to_numeric(fallback_year[missing_year], errors="coerce")

    df["genres"] = df["genres"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["genres"].isin(["", "nan", "None"]), "genres"] = "Unknown"
    df["general_reviews"] = df["general_reviews"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["general_reviews"].isin(["", "nan", "None"]), "general_reviews"] = "Unknown"
    df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0).clip(lower=0)
    df["launch_year"] = pd.to_numeric(df["launch_year"], errors="coerce")
    valid_year = df["launch_year"].between(1970, 2100)
    df.loc[~valid_year, "launch_year"] = np.nan
    if df["launch_year"].notna().any():
        median_year = float(df["launch_year"].median())
        df["launch_year"] = df["launch_year"].fillna(median_year)
    else:
        df["launch_year"] = 2000.0
    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace([np.inf, -np.inf], np.nan)
    df = df.loc[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] >= 0)].copy()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable records remained after cleaning. Check the dataset and price format.")
    return df


def model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model input matrix and target series."""
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required cleaned columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y
