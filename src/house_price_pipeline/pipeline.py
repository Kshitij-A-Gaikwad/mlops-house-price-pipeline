from pathlib import Path
import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = [
    "Area_sqft", "Bedrooms", "Bathrooms", "Age_Years",
    "Distance_to_Center_km", "Neighborhood_Score", "Has_Parking"
]
CATEGORICAL = ["Property_Condition", "City_Zone"]
MISSING_TEXT = {"", "na", "n/a", "null", "none", "nan", "?", "unknown"}
BOUNDS = {
    "Area_sqft": (1, 100000),
    "Bedrooms": (0, 100),
    "Bathrooms": (0, 100),
    "Age_Years": (0, 1000),
    "Distance_to_Center_km": (0, 10000),
    "Neighborhood_Score": (0, 10),
    "Has_Parking": (0, 1),
}


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_schema(df, target, id_column, incoming=False, batch_column="Batch"):
    df = df.copy()

    # Normalize common textual missing-value placeholders.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        values = df[col].astype("string").str.strip()
        df[col] = values.mask(values.str.lower().isin(MISSING_TEXT), pd.NA)

    required = set([id_column] + NUMERIC + CATEGORICAL)
    required.add(batch_column if incoming else target)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[id_column].isna().any() or df[id_column].duplicated().any():
        raise ValueError(f"ID column '{id_column}' contains missing or duplicate values.")

    if not incoming:
        if df[target].isna().any():
            raise ValueError(f"Target column '{target}' contains missing values.")
        y = pd.to_numeric(df[target], errors="coerce")
        if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError(f"Target column '{target}' must contain finite numeric values.")
        if (y < 0).any():
            raise ValueError(f"Target column '{target}' cannot be negative.")
        df[target] = y

    for col in NUMERIC:
        original = df[col]
        values = pd.to_numeric(original, errors="coerce")
        invalid_text = values.isna() & original.notna()
        if invalid_text.any():
            raise ValueError(f"Feature '{col}' contains non-numeric values.")
        nonnull = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(nonnull).all():
            raise ValueError(f"Feature '{col}' contains infinite values.")
        low, high = BOUNDS[col]
        if ((values.dropna() < low) | (values.dropna() > high)).any():
            raise ValueError(f"Feature '{col}' contains values outside allowed range [{low}, {high}].")
        df[col] = values

    if incoming:
        batch = pd.to_numeric(df[batch_column], errors="coerce")
        if batch.isna().any() or not np.isfinite(batch.to_numpy(dtype=float)).all():
            raise ValueError(f"Batch column '{batch_column}' must contain finite numeric values.")
        df[batch_column] = batch

    return df


def validate_training_data(df, target, id_column):
    validate_schema(df, target, id_column)
    return True


def validate_incoming_data(df, target, id_column, batch_column="Batch"):
    return validate_schema(df, target, id_column, True, batch_column)


def build_model(X, config):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = RandomForestRegressor(
        n_estimators=int(config["training"]["n_estimators"]),
        max_depth=config["training"]["max_depth"],
        random_state=int(config["project"]["random_state"]),
        n_jobs=-1,
    )
    return Pipeline([("preprocessing", prep), ("model", model)])


def train_and_evaluate(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    data_path = Path(cfg["data"]["train_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = validate_schema(
        pd.read_csv(data_path),
        cfg["data"]["target"],
        cfg["data"]["id_column"],
    )
    target = cfg["data"]["target"]
    identifier = cfg["data"]["id_column"]
    X = df.drop(columns=[target, identifier])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(cfg["training"]["test_size"]),
        random_state=int(cfg["project"]["random_state"]),
    )
    model = build_model(X_train, cfg)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
        "target": target,
        "model_type": cfg["training"]["model_type"],
        "random_state": int(cfg["project"]["random_state"]),
    }

    # Reject weak candidates before writing the approved model file.
    if metrics["r2"] < float(cfg["evaluation"]["minimum_r2"]):
        raise RuntimeError(f"Model quality gate failed: R2={metrics['r2']:.4f}")
    if metrics["rmse"] > float(cfg["evaluation"]["maximum_rmse"]):
        raise RuntimeError(f"Model quality gate failed: RMSE={metrics['rmse']:.4f}")

    model_path = Path(cfg["outputs"]["model_path"])
    metrics_path = Path(cfg["outputs"]["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic replacement avoids partial model files.
    with tempfile.NamedTemporaryFile(dir=model_path.parent, delete=False) as f:
        tmp_model = Path(f.name)
    try:
        joblib.dump(model, tmp_model)
        os.replace(tmp_model, model_path)
    finally:
        if tmp_model.exists():
            tmp_model.unlink()

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="House price training pipeline")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    print(json.dumps(train_and_evaluate(args.config), indent=2))
