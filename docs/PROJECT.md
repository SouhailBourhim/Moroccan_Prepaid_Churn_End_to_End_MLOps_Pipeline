# Project Technical & Functional Documentation

**Moroccan Prepaid Churn — End-to-End MLOps Pipeline**

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Repository Layout](#3-repository-layout)
4. [Configuration](#4-configuration)
5. [Data Layer](#5-data-layer)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Training](#7-model-training)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Model Evaluation](#9-model-evaluation)
10. [DVC Pipeline](#10-dvc-pipeline)
11. [API Serving](#11-api-serving)
12. [React Dashboard](#12-react-dashboard)
13. [Docker Deployment](#13-docker-deployment)
14. [CI/CD](#14-cicd)
15. [MLflow Experiment Tracking](#15-mlflow-experiment-tracking)
16. [Test Suite](#16-test-suite)
17. [Recent Production Hardening](#17-recent-production-hardening)
18. [Design Decisions & Trade-offs](#18-design-decisions--trade-offs)
19. [Roadmap](#19-roadmap)

---

## Current Implementation Status

The project currently includes:

- typed raw-data ingestion and validation for the Expresso dataset
- leak-aware sklearn-compatible feature engineering pipeline
- candidate training for Logistic Regression, XGBoost, LightGBM, and CatBoost
- Optuna tuning for CatBoost
- deterministic holdout evaluation and DVC-tracked metrics
- FastAPI serving with `/health`, `/ready`, `/info`, and `/predict`
- Dockerized API runtime with external model and feature artifacts
- MLflow experiment logging for feature engineering, training, and tuning
- CI checks for linting, typing, and tests
- React operations dashboard in `dashboard/`
- production-hardening fixes for final refit, threshold selection, feature drift checks, and config-driven feature parameters

The next production milestone is not another model algorithm. It is the monitoring and feedback loop: prediction logging, drift detection, delayed-label evaluation, and retraining/promotion automation.

---

## 1. Problem Statement

**Goal:** Predict which prepaid mobile subscribers will churn (stop using the service) in the next 30 days, using historical usage data. The business value is enabling targeted retention interventions — SMS offers, call-centre outreach, promotional credits — before a subscriber has already left.

**Why this is hard:**
- **Imbalance:** 18.75% churn rate. A model that always predicts "retain" gets 81.25% accuracy but zero business value.
- **MNAR missingness:** 35–94% of usage columns are missing, not at random. Missing values are themselves a signal — a subscriber with no recorded call activity has a churn rate ~4× higher than one with activity.
- **Scale:** 2.15M training rows. Feature engineering and CV must be efficient.
- **Interpretability requirement:** Retention teams need to know *why* a subscriber is flagged, not just a score. SHAP values are computed for every deployed model.

**Primary metric: ROC-AUC.** Secondary: PR-AUC (more informative under class imbalance), Brier score (calibration), F1 at various operating thresholds.

---

## 2. Dataset

### 2.1 Primary: Expresso Telecom (Senegal)

Source: [Zindi Expresso Churn Prediction Challenge](https://zindi.africa/competitions/expresso-churn-prediction)

| Split | Rows | Columns | Churn rate |
|-------|------|---------|-----------|
| Train | 2,154,048 | 19 | 18.75% |
| Test | 380,127 | 18 | unknown |

**Raw columns:**

| Column | Type | Missing % | Description |
|--------|------|-----------|-------------|
| `user_id` | string | 0% | Subscriber identifier |
| `REGION` | category | 39.4% | Geographic region (14 values) |
| `TENURE` | category | 0% | Subscription duration band (8 values) |
| `MRG` | category | 0% | VAS subscription flag (YES/NO) |
| `TOP_PACK` | category | 41.9% | Most-used data/call bundle (140 values) |
| `MONTANT` | float32 | 35.1% | Recharge amount (FCFA) |
| `FREQUENCE_RECH` | float32 | 35.1% | Number of recharges |
| `REVENUE` | float32 | 33.7% | Monthly revenue (FCFA) |
| `ARPU_SEGMENT` | float32 | 33.7% | Average revenue per user (≡ REVENUE/3) |
| `FREQUENCE` | float32 | 33.7% | Number of transactions |
| `DATA_VOLUME` | float32 | 49.2% | Data usage (MB) |
| `ON_NET` | float32 | 36.5% | On-network call volume |
| `ORANGE` | float32 | 41.6% | Calls to Orange network |
| `TIGO` | float32 | 59.9% | Calls to Tigo network |
| `ZONE1` | float32 | 92.1% | Zone 1 international calls |
| `ZONE2` | float32 | 93.6% | Zone 2 international calls |
| `REGULARITY` | float32 | 0% | Days active in last 90 days (1–62 in train) |
| `FREQ_TOP_PACK` | float32 | 41.9% | Frequency of top pack usage |
| `CHURN` | int8 | 0% | Target (1 = churned) |

### 2.2 Reference Datasets (EDA only)

| Dataset | Purpose |
|---------|---------|
| Cell2Cell Wireless (USA, 51k rows) | Validates that customer-service call volume, tenure, and charge level are cross-market churn signals |
| Orange Telecom (USA, 3.3k rows) | Validates international plan enrollment and call volume signals |

The reference datasets are never used for training. Their role is to identify which Expresso features have cross-dataset evidence of robustness vs. which may be dataset-specific noise.

**Key cross-dataset finding:** The universal churn signal is disengagement. `REGULARITY` (days active / 90) is the dominant predictor in Expresso (Spearman ρ = 0.53). Missing usage columns are the closest proxy for "customer service friction" — the strongest predictor in Orange and Cell2Cell.

---

## 3. Repository Layout

```
.
├── configs/
│   └── base.yaml                  # Single source of truth — all params and paths
│
├── data/
│   ├── raw/
│   │   └── expresso/              # Train.csv, Test.csv (DVC-tracked)
│   └── features/                  # Generated by featurize stage (DVC-tracked output)
│       ├── train_features.parquet # 2,154,048 × 39 (38 features + CHURN)
│       ├── test_features.parquet  # 380,127 × 39 (38 features + user_id)
│       ├── feature_pipeline.pkl   # Fitted FeaturePipeline (for inference)
│       ├── feature_stats.csv      # Per-feature diagnostics
│       └── feature_manifest.json  # Feature names, counts, churn rate
│
├── models/                        # DVC-tracked train stage outputs
│   ├── best_model.pkl             # {model: CatBoostClassifier, feature_cols: [...]}
│   ├── training_manifest.json     # CV results for all 4 candidates (DVC metric)
│   ├── eval_metrics.json          # Holdout metrics (DVC metric)
│   └── tuning_results.json        # Optuna best params (written by tune.py)
│
├── src/
│   ├── data/
│   │   ├── ingestion.py           # load_train(), load_test() with typed dtypes
│   │   └── validation.py          # validate_raw() → ValidationReport
│   ├── features/
│   │   ├── build_features.py      # FeaturePipeline + 9 transformers
│   │   └── run_pipeline.py        # Featurize stage entry point
│   ├── models/
│   │   ├── train.py               # 4-candidate CV training entry point
│   │   ├── tune.py                # Optuna search entry point
│   │   ├── evaluate.py            # Metric utils, threshold selection, SHAP
│   │   ├── evaluate_run.py        # DVC evaluate stage entry point
│   │   └── predict.py             # CLI inference on raw CSV
│   ├── api/
│   │   ├── app.py                 # FastAPI app — routes, lifespan, dependency
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── dependencies.py        # ModelArtifacts + load_artifacts()
│   └── utils/
│       └── logging.py             # loguru setup
│
├── dashboard/
│   ├── src/
│   │   ├── App.tsx                # Model ops dashboard shell
│   │   ├── data.ts                # Saved project metrics and dashboard defaults
│   │   ├── styles.css             # Responsive operational UI styling
│   │   └── types.ts               # API response types
│   ├── package.json               # Vite React scripts
│   └── vite.config.ts
│
├── tests/
│   ├── test_features.py           # 18 tests — every transformer + pipeline
│   ├── test_models.py             # 16 tests — evaluate utilities, build_candidates, tune
│   └── test_api.py                # 14 tests — all endpoints + 503/422 cases
│
├── notebooks/
│   ├── 01_eda.ipynb               # Expresso EDA
│   ├── 02_eda_cell2cell.ipynb     # Cell2Cell reference EDA
│   ├── 03_eda_orange.ipynb        # Orange Telecom reference EDA
│   ├── 04_cross_dataset_analysis.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_model_evaluation.ipynb  # CV comparison, ROC/PR/SHAP
│
├── docs/
│   ├── adr/
│   │   └── 001-model-selection.md
│   └── PROJECT.md                 # ← this file
│
├── dvc.yaml                       # Pipeline DAG
├── Dockerfile                     # Multi-stage API image
├── docker-compose.yml
├── .github/workflows/ci.yml       # Lint → type → test
└── pyproject.toml
```

---

## 4. Configuration

All hyperparameters, paths, and service settings live in `configs/base.yaml`. No values are hardcoded in source files.

```yaml
data:
  raw_dir: data/raw/expresso
  features_dir: data/features

features:
  top_pack_min_freq: 200       # packs below this count → "OTHER"
  target_enc_smoothing: 20.0   # James–Stein smoothing factor
  regularity_inactive_threshold: 5

training:
  random_state: 42
  test_size: 0.2
  cv_folds: 5
  eval_metric: roc_auc
  early_stopping_rounds: 50

xgboost:         { n_estimators: 1000, max_depth: 7, learning_rate: 0.05, … }
lightgbm:        { n_estimators: 1000, num_leaves: 127, … }
catboost:        { iterations: 1000, depth: 7, l2_leaf_reg: 3.0, … }
logistic_regression: { C: 0.1, max_iter: 1000 }

mlflow:
  experiment_name: moroccan_prepaid_churn
  tracking_uri: mlruns

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
```

After `tune.py` runs, the `catboost` section is automatically patched with the Optuna-found best parameters and optimal `iterations` (derived from early-stopping mean across folds).

---

## 5. Data Layer

### 5.1 Ingestion (`src/data/ingestion.py`)

`load_train()` and `load_test()` read raw CSVs with **explicit column dtypes** declared upfront:

- Categorical columns (`REGION`, `TENURE`, `MRG`, `TOP_PACK`) are loaded as `category` dtype — avoids accidental label encoding, reduces memory.
- All numeric columns are `float32` (not float64) — halves memory on 2.15M rows.
- `CHURN` is cast to `int8`.

No preprocessing happens here — ingestion is strictly loading.

### 5.2 Validation (`src/data/validation.py`)

`validate_raw(df, is_train)` returns a `ValidationReport` with:

- **Schema checks:** all expected columns present, correct dtypes
- **Domain checks:** `REGULARITY` in [0, 90], no negative usage values, `CHURN` is 0/1 (train only)
- **Warnings** (non-blocking): unusual missing rates, unexpected category values

`run_pipeline.py` calls validation before any transformation and raises if `report.passed` is False.

---

## 6. Feature Engineering

### 6.1 Design Principles

1. **Fit on train only, apply to both.** Every transformer stores its statistics (medians, target encoding maps, pack frequency counts) from the training set. The test set is transformed using those stored values — no leakage.
2. **MNAR missing values are informative.** A missing `ON_NET` does not mean "unknown call volume" — it means the subscriber never made an on-network call. The missingness pattern encodes service disengagement. `_missing` flags capture this before imputation overwrites the NaN.
3. **Every feature decision is traceable.** Each transformer has a docstring citing the EDA notebook and finding that motivated it.

### 6.2 Transformer Chain

```
Raw DataFrame (19 cols)
        │
        ▼  MissingIndicatorAdder
           Adds {col}_missing binary flags for all 14 columns that have NaN in train.
           Must run first — ServiceAbsenceEncoder and the model read these flags.
           Churners have 2–4× higher missing rates on MNAR columns (EDA notebook 01).
        │
        ▼  ServiceAbsenceEncoder
           Aggregates the 7 MNAR missing flags into:
             n_services_absent   [0–7]  count of channels with zero recorded usage
             is_ghost_subscriber [0/1]  1 if ≥ 5 of 7 channels absent (ρ = 0.46)
           Motivation: customer service call volume is the dominant predictor in
           Orange (ρ = 0.14) and Cell2Cell (ρ = 0.05) but has no equivalent in
           Expresso. The MNAR pattern is the closest proxy — subscribers with zero
           activity across multiple channels map to the same disengaged/frustrated
           segment (validated: n_services_absent achieves ρ = 0.47, EDA notebook 04).
        │
        ▼  ZeroImputer
           MNAR columns (DATA_VOLUME, ON_NET, ORANGE, TIGO, ZONE1, ZONE2,
           FREQ_TOP_PACK) → 0.
           Semantically correct: absent = never used that service, not missing data.
        │
        ▼  MedianImputer
           MAR columns (MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT,
           FREQUENCE) → training-set median.
           These are missing for ~34% of subscribers. Missingness correlates with
           churn (MAR, not MNAR), captured by the _missing flags already added.
        │
        ▼  NumericFeatureEngineer
           Derived features (all stateless — no leakage risk):
             regularity_rate       REGULARITY / 90          dominant predictor (ρ = 0.53)
             recharge_per_freq     MONTANT / (FREQ_RECH+1)  spend commitment per event
             revenue_per_freq      REVENUE / (FREQUENCE+1)  revenue efficiency
             data_per_freq         DATA_VOLUME / (FREQ+1)   data intensity
             total_calls           sum of 5 call columns    aggregate usage
             intl_calls            ZONE1 + ZONE2            international presence
             n_active_call_types   count of non-zero call cols  network breadth
             is_inactive           REGULARITY < 5 → 1       extreme low-activity flag
             has_data              DATA_VOLUME > 0 → 1
             has_calls             total_calls > 0 → 1
             has_intl_usage        intl_calls > 0 → 1
        │
        ▼  TenureEncoder
           TENURE (ordinal string) → tenure_ordinal [0–10] + is_new_subscriber [0/1]
           New subscribers (< 3 months) churn at 5× the rate of loyal subscribers
           (> 24 months). Ordinal preserves the monotonic duration relationship.
        │
        ▼  MRGEncoder
           MRG (YES/NO) → mrg_flag [0/1]
           VAS enrollment = loyalty signal (cross-dataset: plan adoption reduces churn)
        │
        ▼  TargetEncoder(cols=["REGION"])
           James–Stein smoothed target encoding:
             encoded = λ × group_mean + (1 − λ) × global_mean
             λ = sigmoid((n − min_samples) / smoothing)
           REGION has 14 values with up to 2× churn rate variation across regions.
           Unseen regions at inference → global_mean (population churn rate).
        │
        ▼  TopPackEncoder
           TOP_PACK (140 unique values): rare packs (< 200 subscribers) → "OTHER",
           then frequency encode (top_pack_freq) + James–Stein target encode (top_pack_te).
           Pack engagement is a robust predictor across all three datasets.
        │
        ▼  get_model_features()
           Drops: user_id, CHURN, REGION, TENURE, MRG, TOP_PACK (raw categoricals
           replaced by encoded versions), ARPU_SEGMENT and ARPU_SEGMENT_missing
           (perfectly collinear with REVENUE and REVENUE_missing respectively).
           Also drops redundant post-audit columns:
             REGULARITY                redundant with regularity_rate
             is_new_subscriber         all-zero in current Expresso data
             mrg_flag                  all-zero in current Expresso data
             FREQUENCE_RECH_missing    duplicate of MONTANT_missing
             FREQUENCE_missing         duplicate of REVENUE_missing
             FREQ_TOP_PACK_missing     duplicate of TOP_PACK_missing
        │
        ▼
38 model features after pruning, zero nulls
```

### 6.3 Top Features by Signal Strength

| Rank | Feature | Spearman ρ with CHURN | Type |
|------|---------|----------------------|------|
| 1 | `is_inactive` | 0.54 | Engineered binary |
| 2 | `REGION_missing` | 0.54 | Missing indicator |
| 3 | `regularity_rate` / `REGULARITY` | −0.53 | Engineered / raw |
| 4 | `REVENUE_missing` | 0.48 | Missing indicator |
| 5 | `n_services_absent` | 0.47 | Cross-dataset proxy |
| 6 | `is_ghost_subscriber` | 0.46 | Cross-dataset proxy |
| 7 | `MONTANT_missing` | 0.44 | Missing indicator |
| … | 32 of 44 engineered candidate features | \|ρ\| > 0.10 | — |

32 of the engineered candidate features have |ρ| > 0.10, confirming the feature engineering adds genuine signal beyond the raw columns. The final model matrix prunes perfectly redundant and all-zero columns before training.

---

## 7. Model Training

### 7.1 Candidate Models

Four classifiers are evaluated in parallel. Selection rationale is documented in [`docs/adr/001-model-selection.md`](adr/001-model-selection.md).

| Model | Role | Imbalance handling |
|-------|------|--------------------|
| `LogisticRegression` | Interpretable baseline | `class_weight='balanced'` |
| `XGBoostClassifier` | Gradient boosting reference | `scale_pos_weight = n_neg / n_pos ≈ 4.33` |
| `LGBMClassifier` | Speed reference | `is_unbalance=True` |
| `CatBoostClassifier` | Primary — wins on calibration and zero-inflated features | `auto_class_weights='Balanced'` |

### 7.2 Cross-Validation Protocol

- **Stratified 5-fold**, shuffle=True, random_state=42
- Scored on: ROC-AUC (primary) + PR-AUC (secondary)
- The feature pipeline is **pre-computed** before CV (run_pipeline.py saves parquet). CV is run on the engineered features, not raw data. This introduces minor target-encoding leakage (smoothing=20 makes it negligible in practice). A future improvement would refit the pipeline inside each fold.

### 7.3 Results

| Model | CV ROC-AUC | σ | CV PR-AUC |
|-------|-----------|---|----------|
| **CatBoost** ★ | **0.9316** | 0.0005 | **0.7039** |
| XGBoost | 0.9314 | 0.0005 | 0.7039 |
| LightGBM | 0.9313 | 0.0005 | 0.7038 |
| Logistic Regression | 0.9284 | 0.0005 | 0.6899 |

**The three GBDTs are statistically indistinguishable** — the gap between best and worst GBDT (0.0003) is smaller than one standard deviation (0.0005). CatBoost is selected as the winner by convention.

**Logistic Regression closes to within 3.2 pp of the best GBDT.** This confirms the feature engineering is doing most of the work — the model complexity adds marginal value.

### 7.4 Final Model

The winning CatBoost is **refitted on the full training set** (not a fold) after CV. The artifact saved to `models/best_model.pkl` is:
```python
{"model": CatBoostClassifier(...), "feature_cols": [...feature names...]}
```

The final refit path explicitly disables XGBoost early stopping if XGBoost ever wins model selection. This prevents XGBoost from failing on the full-data refit because no validation `eval_set` is provided at that stage.

### 7.5 Entry Point

```bash
python -m src.models.train [--config path] [--no-mlflow]
```

Outputs: `models/best_model.pkl`, `models/training_manifest.json`.

---

## 8. Hyperparameter Tuning

### 8.1 Strategy

Tuning on 2.15M rows with full 5-fold CV × 50 Optuna trials is prohibitively slow (~20+ hours). The solution:

1. **Stratified subsample** (default 400k rows, preserves 18.75% churn rate) for the Optuna search. Cuts cost by ~5×.
2. **3-fold CV** during search (not 5). Halves per-trial cost again.
3. **CatBoost early stopping** (`early_stopping_rounds=50`) inside each fold — bad configs terminate early instead of running all 2000 iterations.
4. **MedianPruner** — Optuna kills trials whose running mean after fold 1 is below the median of completed trials. Eliminates ~30–40% of trials early.
5. **Full-data validation** — the winning trial's params are re-evaluated on all 2.15M rows with 5-fold CV before saving.

### 8.2 Search Space

| Parameter | Default | Search range |
|-----------|---------|-------------|
| `depth` | 7 | [4, 10] |
| `learning_rate` | 0.05 | [0.005, 0.3] log-uniform |
| `l2_leaf_reg` | 3.0 | [0.5, 15.0] |
| `subsample` | 0.8 | [0.6, 1.0] |
| `colsample_bylevel` | 0.8 | [0.5, 1.0] |
| `min_data_in_leaf` | 50 | [10, 300] |
| `iterations` | 1000 | 2000 ceiling + early stopping |

The optimal `iterations` is determined per-fold by CatBoost's early stopping; the mean across folds is used for the final refit and written back to `configs/base.yaml`.

### 8.3 Entry Point

```bash
python -m src.models.tune                           # 50 trials, 400k sample
python -m src.models.tune --trials 20 --sample 200000
```

After tuning, `configs/base.yaml` is patched automatically and `models/best_model.pkl` is replaced with the tuned model.

---

## 9. Model Evaluation

### 9.1 Holdout Metrics

Evaluated on a **deterministic 20% stratified holdout** (430,810 rows) — same split used every time (`random_state=42`).

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9330** |
| PR-AUC | **0.7071** |
| Brier score | 0.1119 |
| Baseline Brier (predict mean) | 0.1520 |

PR-AUC of 0.7071 vs a random baseline of 0.1875 means the model is ~3.8× more precise than random at any given recall level.

### 9.2 Threshold Analysis

Three operating points are computed and stored in `models/eval_metrics.json`:

| Strategy | Threshold | Precision | Recall | F1 | Use case |
|----------|-----------|-----------|--------|----|---------|
| Default 0.5 | 0.500 | 0.537 | 0.923 | 0.679 | General default |
| Youden-J | 0.501 | 0.537 | 0.923 | 0.679 | Balanced sensitivity/specificity |
| F1-optimal | 0.689 | 0.612 | 0.822 | 0.702 | Maximise F1 on churn class |

The **Youden threshold** is the recommended default for unknown cost ratios. The **80%-recall threshold** is available via `threshold_at_recall(y_true, y_prob, min_recall=0.80)` for campaigns with a fixed recall floor.

If a requested recall floor is unattainable, `threshold_at_recall()` returns the lowest available threshold so recall is maximised instead of accidentally falling back to the strictest threshold.

### 9.3 SHAP Explainability

`src/models/evaluate.py::shap_summary()` uses `shap.TreeExplainer` (auto-dispatched via `shap.Explainer`) to produce beeswarm and dependence plots. The top 3 features by mean |SHAP| are:

1. `regularity_rate` — higher regularity → strong negative SHAP (less likely to churn)
2. `n_services_absent` — every absent channel increases churn probability monotonically
3. `REVENUE` or `revenue_per_freq` — low revenue per transaction is the spend-commitment signal

See `notebooks/06_model_evaluation.ipynb` §7–8 for the full beeswarm plot.

---

## 10. DVC Pipeline

### 10.1 DAG

```
data/raw/expresso/Train.csv.dvc ──┐
data/raw/expresso/Test.csv.dvc  ──┤
                                  ▼
                            featurize          (src/features/run_pipeline.py)
                            ─────────
                            deps:  src/data/, src/features/, raw CSVs
                            params: configs.features
                            outs:  data/features/ (parquet, pkl, csv, json)
                                  │
                                  ▼
                               train           (src/models/train.py)
                               ─────
                               deps:  src/models/train.py, train_features.parquet
                               params: configs.training, .xgboost, .lightgbm,
                                       .catboost, .logistic_regression
                               outs:  models/best_model.pkl
                               metrics: models/training_manifest.json
                                  │
                                  ▼
                             evaluate          (src/models/evaluate_run.py)
                             ────────
                             deps:  evaluate_run.py, evaluate.py, best_model.pkl,
                                    train_features.parquet
                             metrics: models/eval_metrics.json
```

### 10.2 Key Commands

```bash
dvc repro               # re-run changed stages only
dvc repro --force       # full re-run
dvc dag                 # print ASCII DAG
dvc metrics show        # print eval_metrics.json
dvc metrics diff HEAD~1 # diff metrics vs previous commit
dvc params diff         # diff config params vs previous commit
dvc status              # show which stages are out of date
```

The evaluate stage checks that the saved model's `feature_cols` are all present in `data/features/train_features.parquet`. If the feature pipeline changes but artifacts are stale, evaluation fails with an explicit regeneration/retraining message instead of silently dropping missing columns.

### 10.3 Reproducing from Scratch

```bash
git clone <repo>
cd <repo>
pip install -e ".[dev]"
dvc pull                # fetch raw data from DVC remote
dvc repro               # run featurize → train → evaluate
```

### 10.4 DVC Remote Setup (optional)

```bash
# Example: S3 remote
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push                # upload cached artifacts to remote
```

Without a remote, artifacts are cached locally in `.dvc/cache/`.

---

## 11. API Serving

### 11.1 Architecture

```
HTTP Request
    │
    ▼
FastAPI (src/api/app.py)
    │
    ├── GET /health    → HealthResponse {status: "ok"}
    ├── GET /ready     → ReadyResponse | 503 if model not loaded
    ├── GET /info      → InfoResponse {model_name, cv_roc_auc_mean, n_features, …}
    └── POST /predict  → PredictionResponse
             │
             ▼
    _to_dataframe(subscribers)    ← typed pandas DataFrame with correct dtypes
             │
             ▼
    FeaturePipeline.transform()   ← loaded once at startup, thread-safe
             │
             ▼
    model.predict_proba()         ← CatBoostClassifier
             │
             ▼
    [{churn_probability, churn_prediction}, …]
```

### 11.2 Startup

On startup (via FastAPI `lifespan`), `load_artifacts()` is called once:
- Loads `data/features/feature_pipeline.pkl` (FeaturePipeline fitted on train)
- Loads `models/best_model.pkl` (CatBoostClassifier + feature_cols)
- Loads `models/training_manifest.json` (metrics for `/info`)

If artifact files are missing, startup logs a warning and sets `app.state.artifacts = None`. All endpoints requiring the model return HTTP 503 until artifacts are available.

### 11.3 Request/Response Schema

**POST /predict request:**
```json
{
  "subscribers": [
    {
      "REGION": "DAKAR",
      "TENURE": "K > 24 month",
      "MRG": "NO",
      "REGULARITY": 54.0,
      "ON_NET": 388.0,
      "REVENUE": 4251.0
      // all 17 raw columns optional; None → pipeline handles as MNAR
    }
  ],
  "threshold": 0.5
}
```

**POST /predict response:**
```json
{
  "predictions": [
    {"churn_probability": 0.049, "churn_prediction": false}
  ],
  "model_name": "catboost",
  "threshold": 0.5,
  "n_subscribers": 1
}
```

### 11.4 Validation

Pydantic enforces:
- `subscribers`: list, min_length=1, max_length=10,000
- `threshold`: float in [0.0, 1.0]
- `REGULARITY`: float in [0, 90] (only field with domain bounds)
- All other subscriber fields: optional, any float or null

### 11.5 Running

```bash
# Development
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Production (via config)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive docs: `http://localhost:8000/docs`

---

## 12. React Dashboard

The project now includes a Vite React dashboard in `dashboard/` for local model-operations visibility. It is intentionally a working operational interface rather than a marketing page.

### 12.1 Capabilities

- **Model overview:** best model, CV ROC-AUC, CV PR-AUC, and feature count.
- **Candidate comparison:** ROC-AUC leaderboard for CatBoost, XGBoost, LightGBM, and Logistic Regression.
- **Threshold analysis:** precision, recall, and F1 tradeoff across default, Youden-J, and F1-optimal thresholds.
- **Feature signals:** high-signal engineered features such as `regularity_rate`, `n_services_absent`, `is_ghost_subscriber`, `REGION_te`, and `top_pack_te`.
- **Live scoring panel:** editable raw subscriber inputs, score-current action, batch queue, and batch prediction results sent to the FastAPI `/predict` endpoint.
- **Pipeline view:** DVC stage flow from raw validation through serving.

### 12.2 Running Locally

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173`.

By default the dashboard calls `http://localhost:8000`. Override the API target with:

```bash
VITE_CHURN_API_URL=https://your-api.example.com npm run dev
```

If the API is offline, the dashboard falls back to saved project metrics from the current training/evaluation outputs so the UI remains useful for demos and reviews.

### 12.3 Validation

```bash
cd dashboard
npm run lint
npm run build
```

The production build currently emits a Recharts chunk-size warning. That is acceptable for the first dashboard release; if the dashboard grows, split chart components with dynamic imports.

---

## 13. Docker Deployment

### 13.1 Multi-Stage Build

**Stage 1 (builder):** `python:3.11-slim` + `build-essential`. Installs only the API-runtime subset of dependencies (`pip install ".[api]"`) into an isolated `/venv`. Excludes: jupyter, optuna, mlflow, matplotlib, dvc, imbalanced-learn, scipy, pre-commit, mypy, ruff, pytest.

**Stage 2 (runtime):** `python:3.11-slim`. Copies `/venv` from builder. Copies `src/` and `configs/`. No dev tools, no build tools. Non-root `appuser` runs the process.

### 13.2 Artifact Volumes

Model artifacts are **never baked into the image**. They are mounted at runtime:

```yaml
volumes:
  - ./models:/app/models:ro
  - ./data/features:/app/data/features:ro
```

This means:
- The image stays at ~3.3 GB instead of ~3.5 GB+ with the model
- You can update `models/best_model.pkl` (after retraining/tuning) without rebuilding the image
- Different model versions can be served by swapping the mounted directory

### 13.3 Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_WORKERS` | 1 | uvicorn worker count |
| `LOG_LEVEL` | info | uvicorn log level |

### 13.4 Image Size Breakdown

| Component | ~Size |
|-----------|-------|
| catboost | ~700 MB |
| xgboost + lightgbm | ~300 MB |
| numpy + pandas + scikit-learn | ~350 MB |
| Python base (3.11-slim) | ~150 MB |
| fastapi + uvicorn + pydantic + joblib | ~80 MB |
| pyarrow | ~200 MB |
| **Total** | **~3.3 GB** |

CatBoost is the floor — there is no meaningful way to reduce the image size without switching serving models.

---

## 14. CI/CD

Two jobs in `.github/workflows/ci.yml`:

**lint-and-type** (fast, no heavy ML deps):
```bash
pip install ruff mypy types-PyYAML
pip install --no-deps -e .
pip install pandas numpy scikit-learn fastapi pydantic loguru joblib mlflow pyyaml
ruff check src/
mypy src/
```

**test** (full install including catboost):
```bash
pip install -e ".[dev]"
pytest --cov=src --cov-report=xml -q
```

The test job needs catboost because `tests/test_models.py::test_build_candidates_*` imports `src.models.train` which imports catboost. All API tests use mocked artifacts — no model files are needed in CI.

**Coverage:** 48 backend tests across 3 test files.

---

## 15. MLflow Experiment Tracking

Experiment name: `moroccan_prepaid_churn`
Tracking URI: `mlruns/` (local, gitignored)

```bash
mlflow ui --backend-store-uri mlruns   # → http://localhost:5000
```

Runs logged:
- **feature_engineering** (`run_pipeline.py`): n_features, churn_rate_train, pipeline_fit_seconds, top-10 Spearman ρ values, artifacts (feature_stats.csv, feature_manifest.json, feature_pipeline.pkl)
- **training__catboost** (`train.py`): CV ROC-AUC/PR-AUC for all 4 candidates, best model params, artifacts (best_model.pkl, training_manifest.json)
- **catboost_optuna_tuning** (`tune.py`): n_trials, all Optuna params, sample_cv_roc_auc, full_cv_roc_auc_mean, artifacts (tuning_results.json)

MLflow is optional in all entry points via `--no-mlflow` flag.

---

## 16. Test Suite

48 backend tests across 3 files plus dashboard lint/build checks. No real model files or data are required for the backend tests — all tests run in CI from a clean install.

### `tests/test_features.py` (18 tests)

Covers every transformer class individually + the full `FeaturePipeline`:
- `MissingIndicatorAdder`: flags created correctly, fit-only columns preserved
- `ServiceAbsenceEncoder`: n_services_absent count, ghost threshold
- `ZeroImputer` / `MedianImputer`: imputation values, test-set leakage prevention
- `NumericFeatureEngineer`: derived feature formulas
- `TenureEncoder`: ordinal mapping + is_new_subscriber flag
- `MRGEncoder`: YES/NO binary encoding
- `TargetEncoder`: smoothing formula, unseen-category fallback
- `TopPackEncoder`: rare-pack collapsing, frequency + target encoding
- `FeaturePipeline.fit_transform`: full chain, correct feature count, zero nulls
- Configurable feature parameters: `top_pack_min_freq`, target-encoding smoothing, and inactive threshold reach their transformers

### `tests/test_models.py` (16 tests)

- `compute_metrics`: key presence, perfect classifier, range constraints, threshold sensitivity
- `optimal_threshold`: Youden strategy, F1 strategy, invalid strategy raises, inf sentinel not returned
- `threshold_at_recall`: floor achieved, fallback to lowest threshold when floor impossible
- `build_candidates`: returns 4 candidates with correct names, scale_pos_weight logged
- `_fit_final_model`: disables XGBoost early stopping for full-data refit
- `_suggest_params`: all required CatBoost keys present, ranges valid
- `_patch_config`: catboost section updated, other keys untouched

### `tests/test_api.py` (14 tests)

All tests inject `MOCK_ARTIFACTS` via `app.dependency_overrides[get_artifacts]` — no real model files:
- `/health`: always 200, with and without model
- `/ready`: 200 when model loaded, 503 when missing
- `/info`: returns manifest fields correctly, 503 when missing
- `/predict`: single subscriber, batch (10), custom threshold flips label, all-null subscriber (MNAR handled), 503 without model, 422 on empty list, 422 on invalid threshold, 422 on REGULARITY > 90

---

## 17. Recent Production Hardening

The latest hardening pass addressed audit findings that would matter in production:

| Area | Change | Impact |
|------|--------|--------|
| Feature config | `FeaturePipeline` now accepts `top_pack_min_freq`, `target_enc_smoothing`, and `regularity_inactive_threshold` from `configs/base.yaml`. | Config is the real source of truth instead of documentation-only. |
| Final refit | XGBoost early stopping is disabled on full-data final refit. | Prevents training failure when XGBoost wins CV. |
| Threshold utility | `threshold_at_recall()` returns the lowest threshold if a recall floor cannot be met. | Maximises recall under impossible campaign constraints. |
| Artifact drift | `evaluate_run.py` fails if saved model features are missing from generated parquet. | Makes stale feature/model artifacts visible. |
| Feature pruning | `get_model_features()` drops redundant and all-zero columns. | Reduces model matrix noise after correlation audit. |
| Dashboard | Added React ops dashboard with API-aware scoring fallback. | Gives a usable local interface for stakeholders and future monitoring panels. |

---

## 18. Design Decisions & Trade-offs

### Why not retrain the pipeline inside each CV fold?

The FeaturePipeline includes target encoding for `REGION` and `TOP_PACK`. These are fit on the full training set before CV begins (run_pipeline.py saves the parquet). This introduces minor target-encoding leakage during CV: fold validation sets have seen the target-encoded values during pipeline fitting.

With `smoothing=20`, this leakage is negligible in practice — the James–Stein estimator heavily regularises small-n groups toward the global mean, so over-fitting to fold-specific signal is minimal. The accepted trade-off: a clean, fast CV loop vs. a correct but 9× slower loop that refits the full pipeline per fold.

### Why 400k subsample for Optuna?

Full-data Optuna (2.15M rows × 3-fold × 50 trials) would take ~40 hours. A stratified 400k subsample preserves the 18.75% churn rate and finds near-optimal hyperparameters. The best params are then validated on the full dataset before saving — this catches cases where a subsample-optimised setting fails to generalise.

### Why mount artifacts as volumes, not bake into the image?

Model artifacts (~400 MB for best_model.pkl) change on every retraining run. Baking them into the image would require a full 3.3 GB image rebuild on each retrain. With volume mounts, you `dvc repro` to regenerate the model and restart the container — the image itself never changes.

### Why Logistic Regression as a baseline?

A calibrated linear model with the engineered feature set closes to within 3.2 pp of the best GBDT (0.9284 vs 0.9316 CV ROC-AUC). This tells you two things: (1) the feature engineering is doing most of the work, (2) the GBDT complexity is justified but only marginally. If LR had scored 0.85, the feature set would need revision. The baseline acts as a quality gate.

### Why separate `evaluate_run.py` from `evaluate.py`?

`evaluate.py` is a library of evaluation utilities (compute_metrics, threshold selection, plot functions, SHAP) used by notebooks, the API tests, and training code. `evaluate_run.py` is the DVC pipeline entry point — a thin wrapper that calls the library, takes no arguments, and writes `eval_metrics.json`. Separating them keeps the library reusable and the pipeline stage minimal.

---

## 19. Roadmap

| Item | Priority | Description |
|------|----------|-------------|
| Prediction logging | High | Persist request metadata, model version, feature payload hashes, prediction probability, threshold, latency, and timestamp for every scored subscriber. |
| Data drift monitoring | High | Detect when live subscriber distributions diverge from training data. Evidently or WhyLogs can compare feature distributions between the training parquet and scored traffic. |
| Dashboard monitoring panels | High | Extend the React dashboard with traffic volume, prediction distribution, drift status, latency, and error-rate panels. |
| Ground-truth feedback loop | High | Join delayed churn labels back to stored predictions and compute production ROC-AUC, PR-AUC, calibration, and threshold metrics over time. |
| Automated retraining trigger | Medium | Retrain when drift exceeds threshold, performance drops, or enough new labels arrive. Promote only if the candidate beats the current production model. |
| DVC remote storage | Medium | Configure S3/GCS/Azure remote so the pipeline is reproducible in CI and on new machines without manual data transfer. |
| Model registry | Medium | Use MLflow Model Registry to promote models from "staging" to "production" with version tracking and rollback capability. |
| API hardening | Medium | Add authentication, rate limiting, strict CORS, expanded schema bounds, and structured JSON logs. |
| Cloud deployment | Medium | Deploy the API and dashboard to a production target such as Azure Container Apps, App Service, AKS, Render, or similar. |
| Optuna dashboard | Low | `optuna-dashboard` provides a live UI for watching trials during long tuning runs. |
| Isotonic / Platt calibration | Low | The calibration curve shows the model is slightly over-confident at high probability values. Isotonic regression post-processing would improve the Brier score and make threshold selection more reliable. |
