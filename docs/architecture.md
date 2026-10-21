# House Price MLOps Architecture

```mermaid
flowchart TD
    A[Versioned Training Data] --> B[Data Ingestion]
    B --> C[Schema and Quality Validation]
    C -->|Valid| D[Preprocessing and Training]
    C -->|Invalid| X[Reject Run and Log Failure]
    D --> E[Model Evaluation]
    E --> F{Quality Gate}
    F -->|Fail| Y[Preserve Production Model]
    F -->|Pass| G[MLflow Tracking and Registry]
    G --> H[Approved Model]
    H --> I[Container Build and Release]
    I --> J[Prediction API]
    K[Three Incoming Batches] --> L[Batch Validation and Prediction]
    J --> M[Operational Monitoring]
    L --> N[Data Drift and Model Quality Monitoring]
    M --> O[Monitoring Evidence]
    N --> O
    O --> P[Investigate and Retrain if Needed]
    P --> B
```

## Component responsibilities

- **Data ingestion:** loads configured training and incoming datasets.
- **Validation:** rejects invalid schema, identifiers, ranges and values.
- **Training:** fits the configured regression pipeline.
- **Evaluation:** checks candidate performance against configured thresholds.
- **MLflow:** records experiments and model versions.
- **Release:** packages an approved model for serving.
- **API:** exposes predictions.
- **Monitoring:** records service metrics and evaluates all three incoming batches.
- **Safe failure:** a failed gate must not replace the current production model.
