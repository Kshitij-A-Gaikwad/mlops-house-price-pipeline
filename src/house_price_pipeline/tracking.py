
import json
import os
import subprocess
import sys

import mlflow
from mlflow import MlflowClient

ROOT = "/content/mlops-house-price-pipeline"
os.chdir(ROOT)

sys.path.insert(0, os.path.join(ROOT, "src"))

from house_price_pipeline.pipeline import load_config

TRACKING_DB = os.path.join(ROOT, "mlflow.db")
mlflow.set_tracking_uri("sqlite:///" + TRACKING_DB)

EXPERIMENT_NAME = "house-price-pipeline"
mlflow.set_experiment(EXPERIMENT_NAME)

# Execute the actual pipeline so its validation and quality gates apply.
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "src.house_price_pipeline.pipeline",
    ],
    cwd=ROOT,
    env={
        **os.environ,
        "PYTHONPATH": os.path.join(ROOT, "src"),
    },
    capture_output=True,
    text=True,
)

print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("Training pipeline failed; MLflow run not recorded.")

with open(os.path.join(ROOT, "artifacts/metrics.json")) as f:
    metrics = json.load(f)

config = load_config(os.path.join(ROOT, "configs/config.yaml"))

with mlflow.start_run(run_name="random-forest-baseline") as run:
    mlflow.log_params({
        "model_type": metrics["model_type"],
        "random_state": metrics["random_state"],
        "target": metrics["target"],
        "train_rows": metrics["train_rows"],
        "test_rows": metrics["test_rows"],
        "feature_count": metrics["feature_count"],
        "test_size": config["training"]["test_size"],
        "n_estimators": config["training"]["n_estimators"],
        "max_depth": config["training"]["max_depth"],
    })

    mlflow.log_metrics({
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    })

    mlflow.set_tags({
        "project": "house-price-pipeline",
        "stage": "baseline",
        "quality_gate": "passed",
    })

    mlflow.log_artifact(
        os.path.join(ROOT, "artifacts/house_price_model.joblib"),
        artifact_path="model-artifacts",
    )
    mlflow.log_artifact(
        os.path.join(ROOT, "artifacts/metrics.json"),
        artifact_path="evaluation",
    )
    mlflow.log_artifact(
        os.path.join(ROOT, "configs/config.yaml"),
        artifact_path="configuration",
    )

    print("\nMLflow run logged successfully.")
    print("Run ID:", run.info.run_id)
    print("Experiment:", EXPERIMENT_NAME)
    print("R²:", metrics["r2"])
    print("RMSE:", metrics["rmse"])
    print("MAE:", metrics["mae"])
