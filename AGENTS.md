# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

Production-grade MLOps pipeline for prepaid subscriber churn prediction using the Expresso Telecom dataset (Senegal). Target metric: ROC-AUC. Class imbalance ~18% churn.

## Commands

```bash
# Install (use project venv)
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test
pytest tests/test_features.py::test_pipeline_fit_transform

# Lint
ruff check src/ tests/

# Type-check
mypy src/

# MLflow UI
mlflow ui --backend-store-uri mlruns
```

## Architecture

```
configs/base.yaml          # Single source of truth for all hyperparameters and paths
src/
  data/
    ingestion.py           # load_train / load_test with explicit dtypes; reads from data/raw/expresso/
    validation.py          # validate_raw() → ValidationReport; schema + domain rules
  features/
    build_features.py      # FeaturePipeline + individual sklearn-compatible transformers
  models/                  # (planned) training scripts and model wrappers
  utils/
    logging.py             # loguru setup; call setup_logger() at entry points
tests/
  test_features.py         # Unit tests for every transformer in build_features.py
data/                      # Versioned via DVC, not git
  raw/expresso/            # Train.csv, Test.csv, VariableDefinitions.csv
  processed/ features/     # Generated artifacts
```

### Feature pipeline (`src/features/build_features.py`)

`FeaturePipeline` chains transformers in order — all fit on train only, applied to both splits:

1. `MissingIndicatorAdder` — adds `{col}_missing` binary flags before imputation
2. `ZeroImputer` — MNAR usage columns (DATA_VOLUME, call cols, FREQ_TOP_PACK) → 0
3. `MedianImputer` — MAR columns (MONTANT, REVENUE, etc.) → train median
4. `NumericFeatureEngineer` — derived features: `regularity_rate`, `recharge_per_freq`, `revenue_per_freq`, `total_calls`, `n_active_call_types`, `is_inactive`, `has_data`, `has_calls`
5. `TenureEncoder` — ordinal 0–10 + `is_new_subscriber` flag
6. `MRGEncoder` — YES/NO → binary `mrg_flag`
7. `TargetEncoder(cols=["REGION"])` — James–Stein smoothed target encoding
8. `TopPackEncoder` — rare-pack collapsing (min_freq=200) + frequency encode + target encode

`get_model_features(df)` returns the final feature list after dropping `user_id`, `CHURN`, and raw categorical columns (`REGION`, `TENURE`, `MRG`, `TOP_PACK`, `ARPU_SEGMENT`).

### Config

All paths, hyperparameters, and service settings live in `configs/base.yaml`. Load with PyYAML at pipeline entry points rather than hardcoding values.

### MLflow

Experiment name: `moroccan_prepaid_churn`, tracking URI: `mlruns/` (local). Log params, metrics, and artifacts per run. `mlruns/` is gitignored.

### Code style

- Line length 100, ruff rules: `E, F, I, N, UP, ANN` (ANN101/ANN102 ignored)
- Strict mypy — all functions need full type annotations
- Python 3.11+ syntax (`list[str]` not `List[str]`, `X | Y` unions)
- Data: never commit raw data; track via DVC. Strip notebook outputs before committing.
