"""Model training, comparison and artifact-saving functions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data_utils import FEATURE_COLUMNS, TARGET_COLUMN, clean_dataset, model_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "steam-games.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
REPORTS_PATH = ROOT / "reports"
RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["launch_year", "reviews"]
    categorical_features = ["genres", "general_reviews"]
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, Any]:
    return {
        "Median Baseline": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE, max_depth=12, min_samples_leaf=3
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=180, random_state=RANDOM_STATE, min_samples_leaf=2,
            max_features=0.9, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=180, random_state=RANDOM_STATE, min_samples_leaf=2,
            max_features=0.9, n_jobs=-1
        ),
    }


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])


def train_and_evaluate(data_path: str | Path = DEFAULT_DATA_PATH,
                       model_path: str | Path = MODEL_PATH,
                       metadata_path: str | Path = METADATA_PATH,
                       reports_path: str | Path = REPORTS_PATH) -> dict[str, Any]:
    """Train models, select by training-only 5-fold CV MAE, and test once."""
    from data_utils import read_dataset

    raw = read_dataset(data_path)
    cleaned = clean_dataset(raw)
    X, y = model_features(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows = []
    best_name = None
    best_cv_mae = float("inf")
    best_pipeline = None

    for name, estimator in candidate_models().items():
        pipeline = make_pipeline(estimator)
        neg_mae_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1
        )
        mean_cv_mae = float(-np.mean(neg_mae_scores))
        std_cv_mae = float(np.std(-neg_mae_scores, ddof=1))
        comparison_rows.append({
            "model": name,
            "cv_mae_mean_eur": mean_cv_mae,
            "cv_mae_std_eur": std_cv_mae,
            "selection_metric": "5-fold CV MAE (lower is better)",
        })
        if mean_cv_mae < best_cv_mae:
            best_cv_mae = mean_cv_mae
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None and best_name is not None
    best_pipeline.fit(X_train, y_train)
    predictions = best_pipeline.predict(X_test)
    metrics = {
        "selected_model": best_name,
        "test_rows": int(len(X_test)),
        "test_mae_eur": float(mean_absolute_error(y_test, predictions)),
        "test_rmse_eur": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(X_train)),
        "clean_rows": int(len(cleaned)),
        "dropped_rows_from_raw": int(len(raw) - len(cleaned)),
        "target": TARGET_COLUMN,
        "target_units": "EUR (dataset snapshot price)",
        "features": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "selection_method": "Lowest mean 5-fold cross-validation MAE on training split",
    }

    reports_path = Path(reports_path)
    reports_path.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(comparison_rows).sort_values("cv_mae_mean_eur")
    comparison["selected"] = comparison["model"].eq(best_name)
    comparison.to_csv(reports_path / "model_comparison.csv", index=False)
    pd.DataFrame({"actual_price_eur": y_test.to_numpy(), "predicted_price_eur": predictions}).to_csv(
        reports_path / "test_predictions.csv", index=False
    )
    with open(reports_path / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    metadata = {
        **metrics,
        "genre_options": sorted(cleaned["genres"].astype(str).unique().tolist()),
        "review_options": sorted(cleaned["general_reviews"].astype(str).unique().tolist()),
        "launch_year_min": int(max(1970, np.floor(cleaned["launch_year"].min()))),
        "launch_year_max": int(min(2100, np.ceil(cleaned["launch_year"].max()))),
        "reviews_max": int(max(100, np.ceil(cleaned["reviews"].quantile(0.99)))),
        "dataset_source": "https://zenodo.org/records/15121027",
        "dataset_collection_note": "Steam games collected in March 2025 from the Spanish-language store.",
    }
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return {"comparison": comparison, "metrics": metrics, "metadata": metadata, "cleaned": cleaned}
