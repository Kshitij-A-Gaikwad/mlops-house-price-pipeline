# Task 1 — Problem Framing and ML Lifecycle Map

## 1. Problem statement
Build a reproducible machine-learning system that predicts house prices
in lakhs from property attributes. The system must validate incoming data,
train and evaluate a model, prevent unsafe model promotion, register approved
models, expose predictions through a service, and monitor incoming batches.

## 2. ML problem
- Problem type: Supervised regression
- Prediction target: Price_Lakhs
- Model: RandomForestRegressor
- Training data: capstone_house_prices_train.csv
- Incoming data: capstone_incoming_batches.csv
- Incoming evaluation: Three batches, evaluated separately

## 3. Success criteria
- Data validation rejects invalid schemas, impossible values and duplicate IDs.
- Model evaluation meets the configured minimum R² and maximum RMSE.
- A failed data or model gate does not replace the production model.
- Training runs, configuration and data versions are traceable.
- CI tests run automatically on pushes and pull requests.
- The model can be served through a containerised API.
- Operational metrics, data drift and performance drift are reported
  for all three incoming batches.

Numerical model thresholds are defined in configs/config.yaml.
Operational and drift alert thresholds must be documented and justified
before monitoring results are interpreted.

## 4. Lifecycle map

Raw data
  -> Ingest
  -> Validate data
  -> Build preprocessing features
  -> Train model
  -> Evaluate model
  -> Quality gate
       | failed -> stop; retain current production model
       | passed -> log run and register candidate
  -> Approve and promote model
  -> Build versioned deployment image
  -> Serve predictions
  -> Monitor API metrics and incoming batches
  -> Alert, investigate and retrain when required

## 5. Main risks and controls
- Invalid or incomplete data: schema, range and missingness checks.
- Poor candidate model: numeric quality gate before promotion.
- Accidental production replacement: promote only after gates pass.
- Irreproducible results: versioned code, YAML configuration, data and
  MLflow run metadata.
- Runtime failures: automated tests, API error handling and CI.
- Distribution changes: batch-level drift and performance monitoring.

## 6. Evidence to capture
- Passing and failing CI runs.
- Data-validation and model-quality gate failures.
- MLflow run, registered model version and lineage.
- Container build/release and rollback evidence.
- API operational metrics and monitoring results for all three batches.
