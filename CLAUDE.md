# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Production-grade MLOps pipeline for prepaid subscriber churn prediction. There are two datasets: Expresso Telecom (Senegal, real, ~18.75% churn) and a CTGAN-generated Moroccan synthetic set (2.7% churn). Target metric: ROC-AUC.

## Commands

```bash
# Install (use project venv)
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test
pytest tests/test_features.py::test_pipeline_fit_transform

# Lint / type-check
ruff check src/ tests/
mypy src/

# Full pipeline for both datasets
dvc repro

# Local API (auth explicitly disabled for dev)
CHURN_AUTH_DISABLED=1 uvicorn src.api.app:app --reload

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Architecture

```
configs/base.yaml + configs/{dataset}.yaml   # shared defaults + per-dataset overlay (tuned params)
src/
  utils/config.py      # load_config(dataset), dataset_paths(dataset), setup_mlflow — use these,
                       # never hardcode data/ or models/ paths
  data/                # ingestion (dtypes, records_to_frame), validation
  features/
    build_features.py  # FeaturePipeline (cross-fitted target encoding) + get_model_features()
    run_pipeline.py    # validate → split holdout → fit on train → data/features/{dataset}/
  models/
    fitting.py         # early stopping on an inner split; refit with fixed rounds
    train.py           # CV → refit → OOF calibration → frozen thresholds → models/{dataset}/model.pkl
    calibrate.py       # CalibratedChurnModel, fit_calibration (OOF scores only)
    evaluate_run.py    # holdout evaluation with frozen thresholds
    tune.py            # Optuna; writes configs/{dataset}.yaml, never a model
    artifacts.py       # load_artifacts(dataset) — shared by API and predict CLI
  api/                 # FastAPI app, strict schemas, fail-closed auth, rate limit, SQLite log
  monitoring/drift.py  # PSI + KS statistic vs. holdout baseline; prediction drift
tests/                 # conftest.make_raw_frame() builds synthetic raw data with real signal
```

## Evaluation rules (do not regress)

- The labeled holdout is split off in `run_pipeline.py` **before** anything is fit. Training, tuning, early stopping, calibration, and threshold selection must never read `holdout_features.parquet`.
- Early stopping uses an inner split of the training fold (`src/models/fitting.py`), never the scored fold.
- Target encoders are cross-fitted in `fit_transform`. Call `transform` only on data the pipeline was not fit on.
- Calibrators are fit on out-of-fold predictions. Thresholds are chosen on cross-fitted calibrated OOF predictions and stored in the artifact.
- Feature lists come from the training frame (`get_model_features`) and are reused for every other split.

## Feature pipeline (`src/features/build_features.py`)

1. `MissingIndicatorAdder` — `{col}_missing` flags before imputation
2. `ServiceAbsenceEncoder` — `n_services_absent`, `is_ghost_subscriber`
3. `ZeroImputer` — MNAR usage columns → 0
4. `MedianImputer` — MAR columns → train median
5. `NumericFeatureEngineer` — `regularity_rate`, `recharge_per_freq`, `total_calls`, …
6. `TenureEncoder` — ordinal 0–10 + `is_new_subscriber`
7. `MRGEncoder` — YES/NO → `mrg_flag`
8. `TargetEncoder(cols=["REGION"])` — James–Stein smoothed, cross-fitted
9. `TopPackEncoder` — rare-pack collapsing + frequency + cross-fitted target encoding

`get_model_features(df)` drops the structural columns (IDs, target, raw categoricals, `ARPU_SEGMENT`, `REGULARITY`). It then drops constant columns and exact duplicates, judged from the data itself.

## MLflow

Experiment `moroccan_prepaid_churn`; tracking URI from `configs/base.yaml` (`sqlite:///mlruns.db`). Always go through `setup_mlflow(cfg)`.

## Code style

- Line length 100, ruff rules: `E, F, I, N, UP, ANN`
- Strict mypy — all functions need full type annotations
- Python 3.11+ syntax (`list[str]` not `List[str]`, `X | Y` unions)
- Data: never commit raw data; track via DVC. Strip notebook outputs before committing.
