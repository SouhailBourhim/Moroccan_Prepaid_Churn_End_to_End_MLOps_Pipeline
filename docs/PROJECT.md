# Project Technical & Functional Documentation

**Moroccan Prepaid Churn — End-to-End MLOps Pipeline**

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Datasets](#2-datasets)
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

- typed raw-data ingestion and validation for both datasets (Expresso and the Moroccan synthetic set)
- a CTGAN-based synthetic Moroccan prepaid dataset generator (`generate_moroccan_dataset.py`) that keeps the churn signal of the columns it relabels; the `--no-ctgan` fallback samples without replacement
- a leak-free evaluation protocol (§9.0): the labeled holdout is split off before anything is fit, early stopping uses an inner split, target encodings are cross-fitted, and calibration and thresholds come from out-of-fold predictions
- per-dataset configs (`configs/{dataset}.yaml` overlays) and artifacts (`data/features/{dataset}/`, `models/{dataset}/`)
- candidate training for Logistic Regression, XGBoost, LightGBM, and CatBoost, plus Optuna tuning for CatBoost and XGBoost
- FastAPI serving (v2.0.0) of the calibrated model, with strict request schemas, fail-closed API-key auth, proxy-aware rate limiting, and a SQLite prediction log that records subscriber_id and prunes old rows
- `DriftDetector` for feature and prediction drift against the holdout baseline (PSI + KS statistic)
- a Dockerized API with external model and feature artifacts and a persistent prediction-log volume
- MLflow logging for feature engineering, training, tuning, and evaluation
- CI for lint, types, tests (including an end-to-end train → evaluate test), dashboard lint/build, and a Docker build with smoke test
- a React operations dashboard with live Monitoring and Drift panels and API-key support

> Current results (§7.3, §9.1) come from `dvc repro` under the leak-free protocol. Metrics from before the fix are kept in `models/legacy/` for reference and are optimistic; see its README.

---

## 1. Problem Statement

**Goal:** Predict which prepaid mobile subscribers will churn (stop using the service) in the next 30 days, using historical usage data. The business value is enabling targeted retention interventions — SMS offers, call-centre outreach, promotional credits — before a subscriber has already left.

**Why this is hard:**
- **Imbalance:** 18.75% churn rate in Expresso; 2.7% in the Moroccan synthetic dataset. A model that always predicts "retain" gets high accuracy but zero business value.
- **MNAR missingness:** 35–94% of usage columns are missing, not at random. Missing values are themselves a signal — a subscriber with no recorded call activity has a churn rate ~4× higher than one with activity.
- **Scale:** 2.15M training rows (Expresso) / 2M rows (Moroccan). Feature engineering and CV must be efficient.
- **Interpretability requirement:** Retention teams need to know *why* a subscriber is flagged, not just a score. SHAP values are computed for every deployed model.

**Primary metric: ROC-AUC.** Secondary: PR-AUC (more informative under class imbalance), Brier score (calibration), F1 at various operating thresholds.

---

## 2. Datasets

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

### 2.3 Moroccan Synthetic Dataset

**Source:** Generated by `generate_moroccan_dataset.py` using CTGAN (SDV library v1.36.3), calibrated to ANRT 2023 Moroccan telecom market data.

| Split | Rows | File size | Churn rate |
|-------|------|-----------|-----------|
| Full (train + test) | 2,000,000 | 276 MB | 2.7% |

**Market calibration parameters:**

| Parameter | Value | Source |
|-----------|-------|--------|
| Churn rate | 2.7% / month | Moroccan industry estimate |
| Target ARPU | 27.5 MAD | ANRT 2023 |
| 4G data penetration | 74% | ANRT 2023 |
| DATA_VOLUME scale | 1.80× Expresso baseline | adjusted for 4G usage |

**Moroccan regions (12 cities, population-weighted):**

Casablanca, Rabat, Fès, Marrakech, Tanger, Agadir, Meknès, Oujda, Béni Mellal, Tétouan, Settat, Nador.

**Operators (market share):**

| Operator | Market share |
|----------|-------------|
| Inwi | 35.3% |
| Maroc_Telecom | 32.5% |
| Orange_Maroc | 32.2% |

**Generation pipeline:**

1. CTGAN fits on an 80k Expresso sample (300 epochs, batch=10,000) to learn the joint feature distribution.
2. CTGAN synthesises 2M rows with the learned distribution.
3. Post-processing calibrates the output to Moroccan market parameters:
   - Region assignment at population weight via a churn-preserving (comonotone) mapping; operator at market share (random)
   - ARPU calibration to 27.5 MAD (NaN-preserving — only non-null positive values are scaled)
   - DATA_VOLUME scale × 1.80 for 4G penetration
   - 20 Moroccan pack names mapped from Expresso packs (churn-preserving)
   - MRG VAS rate set to 12% (random; Expresso has no MRG=YES to learn from)
   - Stratified churn resampling to 2.7% (54k churners drawn from augmented churner pool; 1.946M non-churners drawn from non-churner pool)
4. Output is DVC-tracked at `data/raw/moroccan_telecom_churn.csv`.

**Column schema:** Identical to Expresso except for an added `OPERATOR` column (category). `run_pipeline.py` drops `OPERATOR` before the feature pipeline to keep the feature matrix schema consistent.

**Signal-preserving relabelling:** REGION and TOP_PACK are mapped from Expresso categories with a churn-rate-ordered (comonotone) assignment. Each output category gets exactly its target weight, and Expresso categories with similar churn rates land together, so the category–churn association survives. Which city inherits which Senegalese region's churn level is arbitrary. OPERATOR and MRG are random and carry no churn signal. New data users (to reach 74% 4G penetration) are drawn with probability proportional to REGULARITY + 1 rather than uniformly at random.

**What the dataset is:** Expresso's joint distribution, shifted to a 2.7% churn prior and relabelled with Moroccan names and units. It exercises the pipeline on Moroccan-shaped data; it is not evidence about Moroccan churn behaviour.

**Fallback without CTGAN:** `--no-ctgan` skips CTGAN and shuffles the full Expresso Train.csv **without replacement**, then applies the same post-processing. Bootstrapping a small sample with replacement (the old behaviour) put identical rows on both sides of the train/holdout split. Without replacement, the output is capped at about 1.8M rows at 2.7% churn, and the generator warns.

---

## 3. Repository Layout

```
.
├── generate_moroccan_dataset.py   # CTGAN Moroccan synthetic data generator
│
├── configs/
│   ├── base.yaml                  # Shared defaults — features, training, hyperparams, API
│   ├── expresso.yaml              # Per-dataset overlay (tuned params), deep-merged on base
│   └── moroccan.yaml
│
├── data/
│   ├── raw/
│   │   ├── expresso/              # Train.csv, Test.csv (DVC-tracked)
│   │   └── moroccan_telecom_churn.csv  # Moroccan synthetic (DVC-tracked)
│   └── features/{dataset}/        # featurize_{dataset} stage outputs
│       ├── train_features.parquet # model features + CHURN (cross-fitted target encodings)
│       ├── holdout_features.parquet  # labeled holdout: user_id + features + CHURN
│       ├── score_features.parquet # unlabeled Test.csv rows (Expresso only)
│       ├── feature_pipeline.pkl   # Fitted FeaturePipeline (for inference)
│       ├── feature_stats.csv      # Per-feature diagnostics
│       └── feature_manifest.json  # Feature names, counts, churn rates, split seed
│
├── models/
│   ├── {dataset}/
│   │   ├── model.pkl              # {model: CalibratedChurnModel, feature_cols, thresholds, …}
│   │   ├── training_manifest.json # CV results, calibration, thresholds, OOF metrics
│   │   ├── eval_metrics.json      # Holdout metrics at frozen thresholds
│   │   └── tuning_results_{model}.json
│   └── legacy/                    # Pre-fix metrics, reference only
│
├── src/
│   ├── data/
│   │   ├── ingestion.py           # load_train(), load_test(), load_moroccan()
│   │   └── validation.py          # validate_raw() → ValidationReport
│   ├── features/
│   │   ├── build_features.py      # FeaturePipeline + 9 transformers
│   │   └── run_pipeline.py        # Featurize stage: --dataset {expresso,moroccan}
│   ├── models/
│   │   ├── fitting.py             # Early stopping on an inner split; refit with fixed rounds
│   │   ├── train.py               # CV → refit → OOF calibration → frozen thresholds
│   │   ├── calibrate.py           # CalibratedChurnModel + fit_calibration (OOF scores)
│   │   ├── tune.py                # Optuna search; writes configs/{dataset}.yaml only
│   │   ├── evaluate.py            # Metric utils, threshold selection, SHAP
│   │   ├── evaluate_run.py        # Holdout evaluation with frozen thresholds
│   │   ├── artifacts.py           # ModelArtifacts + load_artifacts(dataset)
│   │   └── predict.py             # Batch inference CLI
│   ├── api/
│   │   ├── app.py                 # FastAPI app v2.0.0 — /predict /logs /drift /info /health /ready
│   │   ├── schemas.py             # Strict Pydantic models (extra fields forbidden)
│   │   ├── auth.py                # X-API-Key, fails closed
│   │   ├── rate_limit.py          # Sliding window, trusted-proxy aware, bounded memory
│   │   ├── dependencies.py        # Re-exports src.models.artifacts
│   │   └── logger.py              # PredictionLogger: SQLite WAL, windows, retention
│   ├── monitoring/
│   │   └── drift.py               # DriftDetector: PSI + KS statistic vs holdout; prediction drift
│   └── utils/
│       ├── config.py              # load_config(dataset), dataset_paths(dataset), setup_mlflow
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
│   ├── conftest.py                # make_raw_frame(): synthetic raw rows with real signal
│   ├── test_features.py           # transformers, cross-fitting, feature selection
│   ├── test_models.py             # metrics, thresholds, candidates, fitting helpers, tune
│   ├── test_training_pipeline.py  # config overlays, calibration, end-to-end train → evaluate
│   ├── test_monitoring.py         # PSI, effect-size statuses, DriftDetector on a real log
│   └── test_api.py                # endpoints, validation, auth, rate limits, real-pipeline predict
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

Configuration is layered:

- `configs/base.yaml` holds the defaults shared by every dataset.
- `configs/{dataset}.yaml` is an optional overlay, deep-merged on top by `src.utils.config.load_config(dataset)`.

Tuned hyperparameters live in the overlays, so tuning one dataset never changes another.

```yaml
# configs/base.yaml (abridged)
features:
  top_pack_min_freq: 200       # packs below this count → "OTHER"
  target_enc_smoothing: 20.0   # James–Stein smoothing factor
  target_enc_cv: 5             # folds for cross-fitted target encoding
  regularity_inactive_threshold: 5

training:
  random_state: 42
  test_size: 0.2               # labeled holdout, split off before any fitting
  cv_folds: 5
  early_stopping_rounds: 50    # on an inner split of each training fold

xgboost:   { n_estimators: 1000, max_depth: 7, learning_rate: 0.05, gamma: 0.0, … }
lightgbm:  { n_estimators: 1000, num_leaves: 127, learning_rate: 0.05, … }
catboost:  { iterations: 1000, depth: 7, learning_rate: 0.05, … }
logistic_regression: { C: 0.1, max_iter: 1000 }

mlflow:
  experiment_name: moroccan_prepaid_churn
  tracking_uri: sqlite:///mlruns.db

api:
  dataset: expresso            # served model; override with CHURN_DATASET
```

```yaml
# configs/moroccan.yaml — generated by src.models.tune
xgboost:  { n_estimators: 378, max_depth: 7, learning_rate: 0.025777, gamma: 4.9473, … }
catboost: { iterations: 552, depth: 4, learning_rate: 0.045577, … }
```

`n_estimators` / `iterations` act as ceilings: training early-stops below them and refits with the mean CV round count.

---

## 5. Data Layer

### 5.1 Ingestion (`src/data/ingestion.py`)

Three loading functions with **explicit column dtypes** declared upfront:

**`load_train()`** and **`load_test()`** — read Expresso CSVs:
- Categorical columns (`REGION`, `TENURE`, `MRG`, `TOP_PACK`) are loaded as `category` dtype.
- All numeric columns are `float32` — halves memory on 2.15M rows.
- `CHURN` is cast to `int8`.

**`load_moroccan(path: Path | str | None = None, nrows: int | None = None)`** — reads the Moroccan synthetic CSV:
- Accepts an explicit path or defaults to `configs/base.yaml → data.moroccan_raw_file`.
- Uses the same dtype dict as Expresso, plus `OPERATOR: category` for the added column.
- Raises `FileNotFoundError` with a descriptive message if the file is not present (e.g., not yet generated).

No preprocessing happens in any loader — ingestion is strictly loading.

### 5.2 Validation (`src/data/validation.py`)

`validate_raw(df, is_train)` returns a `ValidationReport` with:

- **Schema checks:** all expected columns present, correct dtypes
- **Domain checks:** `REGULARITY` in [0, 90], no negative usage values, `CHURN` is 0/1 (train only)
- **Warnings** (non-blocking): unusual missing rates, unexpected category values

`run_pipeline.py` validates every dataset (Expresso and Moroccan, plus Expresso's unlabeled Test.csv) before any transformation and raises if `report.passed` is False. It also warns about exact duplicate rows, which could otherwise land on both sides of the holdout split.

---

## 6. Feature Engineering

### 6.1 Design Principles

1. **Fit on train only, apply to both.** Every transformer stores its statistics (medians, target encoding maps, pack frequency counts) from the training set. The test set is transformed using those stored values — no leakage.
2. **MNAR missing values are informative.** A missing `ON_NET` does not mean "unknown call volume" — it means the subscriber never made an on-network call. The missingness pattern encodes service disengagement. `_missing` flags capture this before imputation overwrites the NaN.
3. **Every feature decision is traceable.** Each transformer has a docstring citing the EDA notebook and finding that motivated it.
4. **Dataset-agnostic pipeline.** The same `FeaturePipeline` handles both Expresso and Moroccan data. The only dataset-specific step in `run_pipeline.py` is dropping the Moroccan-only `OPERATOR` column before the pipeline begins.

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

32 of the engineered candidate features have |ρ| > 0.10, confirming the feature engineering adds genuine signal beyond the raw columns.

---

## 7. Model Training

### 7.1 Candidate Models

Four classifiers are evaluated in parallel. Selection rationale is documented in [`docs/adr/001-model-selection.md`](adr/001-model-selection.md).

| Model | Role | Imbalance handling |
|-------|------|--------------------|
| `LogisticRegression` | Interpretable baseline | `class_weight='balanced'` |
| `XGBoostClassifier` | Gradient boosting reference | `scale_pos_weight = n_neg / n_pos` |
| `LGBMClassifier` | Speed reference | `is_unbalance=True` |
| `CatBoostClassifier` | Primary — wins on calibration and zero-inflated features | `auto_class_weights='Balanced'` |

`scale_pos_weight` for XGBoost is computed per-dataset: ~4.33 on Expresso (18.75% churn) and ~36.0 on Moroccan data (2.7% churn).

### 7.2 Cross-Validation Protocol

- **Stratified 5-fold**, shuffle=True, random_state=42, on the **training split only**. The holdout is a separate file and is never loaded here.
- Scored on ROC-AUC (primary) and PR-AUC (secondary).
- **Early stopping** (`src/models/fitting.py`) carves a stratified 10% inner split out of each training fold. The fold being scored is never used to choose the number of trees.
- **Target encodings are cross-fitted.** `FeaturePipeline.fit_transform` encodes each training row from the other folds, so no row's label leaks into its own `REGION_te` / `top_pack_te`.
- The out-of-fold predictions of every candidate are kept; the winner's feed calibration (§9.0).
- LightGBM uses `subsample_freq=1`, because without it `subsample` has no effect.

### 7.3 Results

5-fold CV on the training split (`models/{dataset}/training_manifest.json`, `all_results`):

| Model | Expresso ROC-AUC | Expresso PR-AUC | Rounds | Moroccan ROC-AUC | Moroccan PR-AUC | Rounds |
|---|---|---|---|---|---|---|
| **XGBoost** ★ | **0.9314** ± 0.0004 | 0.7065 | 284 | **0.9217** ± 0.0013 | 0.3166 | 459 |
| LightGBM | 0.9314 ± 0.0004 | 0.7057 | 205 | 0.9202 ± 0.0014 | 0.3175 | 191 |
| CatBoost | 0.9314 ± 0.0004 | 0.7065 | 506 | 0.9211 ± 0.0015 | 0.3119 | 603 |
| Logistic Regression | 0.9284 ± 0.0003 | 0.6891 | — | 0.8981 ± 0.0016 | 0.1699 | — |

"Rounds" is the mean early-stopped boosting round count, used for the final refit. On Expresso the three boosted models are tied, and XGBoost wins on the fifth decimal. Pre-fix CV numbers (`models/legacy/`) were about 0.0002 higher, because early stopping watched the scored fold and target encodings were fit on all rows.

CatBoost was retuned on the regenerated Moroccan dataset: 50 Optuna trials on a 400k-row sample, then a 5-fold check on the full training split. That gave depth 5, learning rate 0.063 and l2_leaf_reg 10.8. Its CV ROC-AUC rose from 0.9198 to 0.9211, but XGBoost still wins model selection.

An earlier version of `tune.py` wrote the subsample's mean early-stopped round count as `iterations` / `n_estimators`. Training treated that as a hard cap, which held CatBoost at 552 rounds and XGBoost at 378. Tuning now writes a 2,000-round ceiling, and early stopping on the full training split chooses the count (603 and 459 rounds).

### 7.4 Final Model

The winner is refit on all training rows. Boosters use the **mean early-stopped round count from CV** and no early stopping. The tuned XGBoost `gamma` is now passed through; it used to be silently dropped. The model is then wrapped with its calibrator. `models/{dataset}/model.pkl` contains:

```python
{
    "model": CalibratedChurnModel(base_model, calibrator, method),
    "model_name": "catboost",
    "dataset": "expresso",
    "feature_cols": [...],
    "thresholds": {"youden": 0.31, "f1": 0.38},   # chosen on OOF predictions
    "default_threshold": 0.38,                    # F1-optimal; used by the API
    "calibration_method": "isotonic",
}
```

### 7.5 Entry Point

```bash
python -m src.models.train --dataset {expresso,moroccan} [--config path] [--no-mlflow]
```

Outputs: `models/{dataset}/model.pkl`, `models/{dataset}/training_manifest.json`.

---

## 8. Hyperparameter Tuning

### 8.1 Strategy

Tuning on millions of rows with full 5-fold CV × 50 Optuna trials is prohibitively slow. The solution:

1. **Stratified subsample** (default 400k rows) for the Optuna search. Cuts cost by ~5×.
2. **3-fold CV** during search (not 5). Halves per-trial cost again.
3. **Early stopping** inside each fold — bad configs terminate early instead of running all iterations.
4. **MedianPruner** — Optuna kills trials whose running mean after fold 1 is below the median of completed trials. Eliminates ~30–40% of trials early.
5. **Full-data validation** — the winning trial's params are re-evaluated on the full dataset with 5-fold CV before saving.

### 8.2 CatBoost Search Space

| Parameter | Default | Search range |
|-----------|---------|-------------|
| `depth` | 7 | [4, 10] |
| `learning_rate` | 0.05 | [0.005, 0.3] log-uniform |
| `l2_leaf_reg` | 3.0 | [0.5, 15.0] |
| `subsample` | 0.8 | [0.6, 1.0] |
| `colsample_bylevel` | 0.8 | [0.5, 1.0] |
| `min_data_in_leaf` | 50 | [10, 300] |
| `iterations` | 1000 | 2000 ceiling + early stopping (50 rounds) |

**Best found (regenerated Moroccan data):** depth=5, lr=0.0628, l2_leaf_reg=10.84, subsample=0.873, colsample_bylevel=0.936, min_data_in_leaf=58. The mean early-stopped round count on the sample was 424; the overlay stores a 2,000-round ceiling instead.
Full-split CV ROC-AUC: **0.9212 ± 0.0014** (50 trials, 2 pruned).

### 8.3 XGBoost Search Space

| Parameter | Default | Search range |
|-----------|---------|-------------|
| `max_depth` | 7 | [3, 10] |
| `learning_rate` | 0.05 | [0.005, 0.3] log-uniform |
| `subsample` | 0.8 | [0.5, 1.0] |
| `colsample_bytree` | 0.8 | [0.4, 1.0] |
| `min_child_weight` | 50 | [10, 300] |
| `reg_alpha` | 0.1 | [1e-3, 10] log-uniform |
| `reg_lambda` | 1.0 | [0.5, 10] |
| `gamma` | 0.0 | [0.0, 5.0] |
| `n_estimators` | 1000 | 2000 ceiling + early stopping via eval_set |

Early stopping uses an inner split of each training fold (`fit_with_early_stopping`), never the fold being scored. The mean early-stopped round count across folds is written as `n_estimators` / `iterations`.

**Best found (Moroccan data):** max_depth=7, lr=0.0258, subsample=0.662, colsample_bytree=0.455, min_child_weight=260, reg_alpha=0.103, reg_lambda=8.129, gamma=4.947, n_estimators=378.
Full CV ROC-AUC: **0.8972 ± 0.0014** (50 trials, 21 pruned).

### 8.4 Entry Points

```bash
# CatBoost tuning (default)
python -m src.models.tune --dataset moroccan --trials 50 --sample 400000

# XGBoost tuning
python -m src.models.tune --dataset moroccan --model xgboost --trials 50 --sample 400000

# Faster exploration
python -m src.models.tune --dataset expresso --model xgboost --trials 20 --sample 200000
```

After tuning, the dataset's overlay `configs/{dataset}.yaml` is rewritten with the best params, and `models/{dataset}/tuning_results_{model}.json` is written. Tuning **never writes a model**. Retrain (`dvc repro` picks up the overlay change) so the tuned params go through the same CV → refit → calibration path as every model.

---

## 9. Model Evaluation

### 9.0 Protocol

| Step | Data it may see | Never sees |
|------|-----------------|------------|
| Holdout split (stratified 20%, `run_pipeline.py`) | — | — |
| Feature pipeline fit | training split | holdout |
| Target encoding of training rows | other folds (cross-fitted) | the row's own label |
| CV model selection | training split; early stopping on an inner 10% of each fold | the scored fold |
| Final refit | all training rows, mean CV round count | holdout |
| Calibration method + fit | out-of-fold CV predictions of the winner | holdout |
| Decision thresholds | cross-fitted calibrated OOF predictions | holdout |
| Holdout evaluation (`evaluate_run.py`) | holdout, thresholds frozen | — |

**What was wrong before:**

1. The Expresso "holdout" was carved out of `train_features.parquet` *after* the final model had been refit on all of it, so it was a training-set score.
2. The calibrator was fit on rows the base model had trained on.
3. The calibration method and the Youden/F1 thresholds were picked on the same rows they were scored on.
4. Early stopping watched the scored CV fold.
5. Target encodings were fit on all training rows before CV.

All five are fixed. The old numbers are in `models/legacy/`.

### 9.1 Holdout Metrics

Holdout results of the served model (XGBoost + isotonic calibration):

| Metric | Expresso | Moroccan |
|---|---|---|
| Holdout size · churn rate | 430,810 · 18.75% | 400,000 · 2.70% |
| ROC-AUC | 0.9316 | 0.9227 |
| PR-AUC | 0.7034 | 0.3193 |
| Brier: calibrated / uncalibrated / base rate | 0.0815 / 0.1130 / 0.1524 | 0.0213 / 0.1139 / 0.0263 |
| Youden-J threshold · F1 · P · R | 0.191 · 0.677 · 0.537 · 0.917 | 0.027 · 0.225 · 0.129 · 0.870 |
| F1-optimal threshold · F1 · P · R | 0.343 · 0.699 · 0.609 · 0.820 | 0.175 · 0.375 · 0.302 · 0.494 |

The Expresso raw data was verified byte-for-byte (MD5) against the `.dvc` pointers. The Moroccan dataset was regenerated with the current generator (CTGAN, 300 epochs, 2M rows), so its pointer hash changed.

`python -m src.models.evaluate_run --dataset {expresso,moroccan}` writes `models/{dataset}/eval_metrics.json`:

| Key | Meaning |
|-----|---------|
| `roc_auc`, `pr_auc` | Ranking quality of the served (calibrated) model |
| `brier`, `uncalibrated_brier`, `baseline_brier` | Probability quality, before/after calibration, vs. predicting the base rate |
| `threshold_{youden,f1}` | Thresholds frozen in training |
| `f1_*`, `precision_*`, `recall_*` | Measured on the holdout at those frozen thresholds |

The Moroccan ROC-AUC is lower than Expresso's, but that is **not** because of the 2.7% churn rate: ROC-AUC does not depend on class prevalence. The gap reflects CTGAN fidelity and, before the generator fix, random relabelling of REGION and TOP_PACK.

### 9.2 Post-Hoc Calibration (`src/models/calibrate.py`)

Class re-weighting (`scale_pos_weight`, `Balanced`, `is_unbalance`) ranks well but inflates probabilities. With `scale_pos_weight ≈ 36` on Moroccan data, the raw Brier score was worse than predicting the base rate.

`fit_calibration(oof_scores, y, cv_folds, random_state)`:

1. For each method (isotonic, sigmoid), cross-validates a 1-D calibrator over the winner's **out-of-fold** scores and records the Brier score.
2. Picks the method with the lower CV Brier score.
3. Fits it on all OOF scores. This is sklearn's `CalibratedClassifierCV(ensemble=False)` scheme.
4. Returns the cross-fitted calibrated OOF probabilities, which are used to pick thresholds.

`CalibratedChurnModel.predict_proba` applies the calibrator to the base model's scores; `predict_raw` exposes the uncalibrated scores.

### 9.3 Threshold Analysis

Youden-J and F1-optimal thresholds are chosen in training on cross-fitted calibrated OOF probabilities and stored in the artifact. The F1 threshold is the API default. `evaluate_run.py` only *measures* them on the holdout. `threshold_at_recall(y_true, y_prob, min_recall=0.80)` is available for campaigns with a recall floor. Choose its threshold on OOF or validation data, not on the holdout.

If a requested recall floor is unattainable, `threshold_at_recall()` returns the lowest available threshold so recall is maximised instead of accidentally falling back to the strictest threshold.

### 9.4 SHAP Explainability

`src/models/evaluate.py::shap_summary()` uses `shap.TreeExplainer` to produce beeswarm and dependence plots. The top 3 features by mean |SHAP| are:

1. `regularity_rate` — higher regularity → strong negative SHAP (less likely to churn)
2. `n_services_absent` — every absent channel increases churn probability monotonically
3. `REVENUE` or `revenue_per_freq` — low revenue per transaction is the spend-commitment signal

See `notebooks/06_model_evaluation.ipynb` §7–8 for the full beeswarm plot (it explains `model.base_model`; calibration is a monotone map on top).

---

## 10. DVC Pipeline

### 10.1 DAG

One chain per dataset, with namespaced artifacts so the chains never overwrite each other:

```
featurize_expresso ──► train_expresso ──► evaluate_expresso
  raw Expresso CSVs      train_features     model.pkl + holdout_features
  → data/features/       → models/expresso/ → models/expresso/eval_metrics.json
    expresso/              model.pkl
                           training_manifest.json

featurize_moroccan ──► train_moroccan ──► evaluate_moroccan
  moroccan_telecom_churn.csv → data/features/moroccan/ → models/moroccan/
```

Each train stage depends on its `configs/{dataset}.yaml` overlay, so a tuning run triggers retraining. Generating the Moroccan CSV stays outside the DAG (it needs CTGAN); track it with `dvc add`.

### 10.2 Key Commands

```bash
dvc repro                      # re-run changed stages only (both datasets)
dvc repro evaluate_moroccan    # one chain
dvc repro --force              # full re-run
dvc dag                        # print ASCII DAG
dvc metrics show               # training manifests + eval metrics
dvc metrics diff HEAD~1        # diff metrics vs previous commit
dvc params diff                # diff config params vs previous commit
dvc status                     # show which stages are out of date

# Track new artifacts
dvc add data/raw/moroccan_telecom_churn.csv
```

The evaluate stage checks that the model's `feature_cols` are all present in the holdout parquet. If the feature pipeline changes but the artifacts are stale, it fails with an explicit regeneration message.

### 10.3 Reproducing from Scratch

```bash
git clone <repo>
cd <repo>
pip install -e ".[dev]"
dvc pull                # fetch raw data from DVC remote
dvc repro               # run featurize → train → evaluate
```

### 10.4 DVC Remote Setup

The remote is configured per machine in `.dvc/config.local`, which is gitignored, so no personal path or credential is committed. The maintainer's setup uses a folder that OneDrive syncs:

```bash
dvc remote add --local -d onedrive "C:\Users\<you>\OneDrive\dvc-store\moroccan-prepaid-churn"
dvc push                # upload raw data, features and models (~0.8 GB)
dvc status -c           # "Cache and remote are in sync"
dvc pull                # restore everything on another checkout
```

Any DVC backend works the same way (S3, GCS, Azure, SSH, a DagsHub repository). This was tested by restoring `Train.csv` into a fresh clone from the remote alone and checking it against the original MD5.

The Expresso files come from a Zindi competition, so keep the remote private. Without a remote, re-download them from Zindi and verify their MD5s against `data/raw/expresso/*.dvc`.

---

## 11. API Serving

### 11.1 Architecture

```
HTTP Request
    │
    ▼
FastAPI v2.0.0 (src/api/app.py)
    │   X-API-Key (fails closed) · per-client rate limit
    ├── GET  /health    → {status: "ok"}            (no auth)
    ├── GET  /ready     → 503 until model loaded
    ├── GET  /info      → model, dataset, calibration, default threshold, CV metrics
    ├── GET  /logs      → all-time totals + window stats + recent rows
    ├── GET  /drift     → feature drift + prediction drift vs holdout
    └── POST /predict
             ├─ strict schema validation (unknown fields → 422)
             ├─ records_to_frame() → FeaturePipeline.transform()
             ├─ CalibratedChurnModel.predict_proba()
             ├─ threshold = request.threshold or the model's F1-optimal threshold
             └─ PredictionLogger.log_request()   ← background task, after the response
```

### 11.2 Startup

On startup (FastAPI `lifespan`):

1. `load_artifacts(dataset)` (`src/models/artifacts.py`) loads `data/features/{dataset}/feature_pipeline.pkl`, `models/{dataset}/model.pkl`, and the training manifest. The dataset comes from `CHURN_DATASET`, falling back to `api.dataset` in `configs/base.yaml`. If the files are missing, model endpoints return 503.
2. `PredictionLogger(CHURN_PREDICTIONS_DB)` creates or migrates the SQLite log and prunes rows older than `CHURN_LOG_RETENTION_DAYS` (default 30).
3. `DriftDetector` samples up to 10k rows of `holdout_features.parquet` as the baseline. The holdout went through exactly the serving transform, unlike the cross-fitted training split. It also scores those rows to get the baseline prediction distribution.

### 11.3 Request/Response Schema

**POST /predict request:**
```json
{
  "subscribers": [
    {
      "subscriber_id": "sub-000123",
      "REGION": "DAKAR",
      "TENURE": "K > 24 month",
      "MRG": "NO",
      "REGULARITY": 54.0,
      "ON_NET": 388.0,
      "REVENUE": 4251.0
    }
  ]
}
```

**POST /predict response:**
```json
{
  "request_id": "6f1c…",
  "predictions": [
    {"subscriber_id": "sub-000123", "churn_probability": 0.049, "churn_prediction": false}
  ],
  "model_name": "catboost",
  "threshold": 0.38,
  "n_subscribers": 1
}
```

### 11.4 Validation

Pydantic enforces:
- `subscribers`: 1–10,000 items; unknown keys anywhere → 422, so a misspelled column cannot silently become a missing value
- `threshold`: optional float in [0, 1]; defaults to the model's F1-optimal threshold
- `TENURE`: one of the 11 known bands; `MRG`: `YES` / `NO`
- `REGULARITY`: [0, 90]; every other numeric field ≥ 0
- `subscriber_id`: optional string (≤ 128), not a model input

### 11.5 Auth, Rate Limiting, Logging

- **Auth** (`src/api/auth.py`): `CHURN_API_KEYS` (comma-separated) and the `X-API-Key` header, compared in constant time. If no keys are set and `CHURN_AUTH_DISABLED=1` is not set, protected endpoints return **503**, so the API never runs open by accident.
- **Rate limiting** (`src/api/rate_limit.py`): sliding window per client, keyed on the TCP peer address. `X-Forwarded-For` is honoured only from `CHURN_TRUSTED_PROXIES`, and then the right-most untrusted hop is used. Limits are divided by `API_WORKERS` because state is per process. Idle client buckets are evicted, so spoofed clients cannot grow memory.
- **Prediction log** (`src/api/logger.py`): every scored row is written after the response is sent:

```sql
prediction_requests (request_id TEXT PK, ts TEXT, model_name TEXT,
                     n_subscribers INT, threshold REAL, latency_ms REAL)
prediction_rows (id INTEGER PK, request_id TEXT, subscriber_idx INT,
                 churn_probability REAL, churn_prediction INT,
                 features_json TEXT, subscriber_id TEXT)   -- indexed for label joins
```

`GET /logs?hours=24&limit=50` returns all-time `total_predictions` / `total_requests`. It also returns `predictions_in_window`, `requests_in_window`, mean probability, flag rate, and mean latency for the last `hours`, plus the most recent rows.

### 11.6 Drift Detection (`src/monitoring/drift.py`)

Status is driven by **effect sizes**, not p-values. With thousands of live rows a KS p-value flags shifts far too small to matter, and testing ~40 features at p < 0.05 would raise a false WARN on almost every report.

| Metric | Applies to | OK | WARN | DRIFT |
|--------|------------|----|------|-------|
| PSI | every feature | < 0.1 | 0.1–0.2 | > 0.2 |
| KS statistic D | continuous features | < 0.1 | 0.1–0.2 | > 0.2 |

- Features with ≤ 12 distinct baseline values (binary flags, counts, ordinals) are treated as discrete: they are binned on their own values and judged by PSI only.
- The KS p-value is still reported for continuous features, for reference.
- The served `churn_probability` is compared with the model's scores on the holdout (`prediction_drift`).
- At most the 50k most recent live rows in the window are read per report.
- `INSUFFICIENT_DATA` is returned below `min_samples` (default 50).

### 11.7 Running

```bash
CHURN_AUTH_DISABLED=1 uvicorn src.api.app:app --port 8000 --reload              # development
CHURN_API_KEYS=… API_WORKERS=4 uvicorn src.api.app:app --host 0.0.0.0 --workers 4  # production
```

Interactive docs: `http://localhost:8000/docs`

---

## 12. React Dashboard

The project includes a Vite React dashboard in `dashboard/` for local model-operations visibility.

### 12.1 Capabilities

- **Model overview:** best model, CV ROC-AUC ± σ, CV PR-AUC, and feature count (live from `/info`).
- **Holdout KPIs:** Brier score, churn rate, recall and precision at the default threshold.
- **Candidate comparison:** ROC-AUC leaderboard for CatBoost, XGBoost, LightGBM, and Logistic Regression.
- **Threshold analysis:** precision, recall, and F1 tradeoff across default, Youden-J, and F1-optimal thresholds.
- **ROC and PR curves:** full curves with AUC annotations and random/baseline reference lines.
- **SHAP feature importance:** mean |SHAP| beeswarm for the top 15 features, coloured by direction (protective / risk / contextual).
- **Confusion matrix:** TP/FN/FP/TN counts with precision, recall, specificity, and F1 summary.
- **Calibration curve:** predicted vs observed churn fraction with Brier score annotation.
- **Geography:** churn rate by region, above/below average colouring; dataset profile and class imbalance bar.
- **Feature signals:** high-signal engineered features such as `regularity_rate`, `n_services_absent`, `is_ghost_subscriber`.
- **Live scoring panel:** editable raw subscriber inputs, score-current action, batch queue, and batch prediction results sent to the FastAPI `/predict` endpoint.
- **Pipeline view:** DVC stage flow from raw validation through serving.
- **Monitoring panel:** traffic KPIs for the last 24 h (predictions, requests, mean churn probability, mean latency) plus a recent-predictions table, from `GET /logs`. Includes a refresh button.
- **Drift panel:** overall status badge (OK / WARN / DRIFT / INSUFFICIENT_DATA), a prediction-drift row, and per-feature PSI / KS statistic, from `GET /drift`.

### 12.2 Running Locally

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173`. Override the API target with `VITE_CHURN_API_URL`. Set `VITE_CHURN_API_KEY` to send `X-API-Key`; it is inlined into the JS bundle, so only use it for local or internal deployments. A rejected key, or a server with auth unconfigured, is reported in the sidebar.

If the API is offline, the dashboard falls back to saved project metrics from the current training/evaluation outputs.

### 12.3 Validation

```bash
cd dashboard && npm run lint && npm run build
```

---

## 13. Docker Deployment

### 13.1 Multi-Stage Build

**Stage 1 (builder):** `python:3.11-slim` + `build-essential`. Installs `.[${MODEL_EXTRAS}]`, default `api,serve-all`. The `api` extra has no boosting library. Add `serve-xgboost`, `serve-lightgbm`, or `serve-catboost` for the served model's library. It excludes jupyter, optuna, mlflow, matplotlib, dvc, sdv, and the dev tools.

**Stage 2 (runtime):** `python:3.11-slim`. Copies `/venv`, `src/`, `configs/`. Runs as the non-root `appuser`, with a `HEALTHCHECK` on `/health`.

### 13.2 Volumes

```yaml
volumes:
  - ./models:/app/models:ro
  - ./data/features:/app/data/features:ro
  - predictions-db:/app/data/db          # prediction log survives restarts
```

Model artifacts are never baked into the image. Retrain, then restart the container.

### 13.3 Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHURN_DATASET` | `expresso` | Which dataset's model to serve |
| `CHURN_API_KEYS` | — | Required unless `CHURN_AUTH_DISABLED=1` |
| `CHURN_TRUSTED_PROXIES` | — | Proxies allowed to set `X-Forwarded-For` |
| `CHURN_PREDICTIONS_DB` | `/app/data/db/predictions.db` | Prediction log path |
| `CHURN_LOG_RETENTION_DAYS` | 30 | Prediction-log retention |
| `API_WORKERS` | 1 (compose: 4) | uvicorn workers; rate limits are divided by this |
| `LOG_LEVEL` / `LOG_FORMAT` | info / text | uvicorn level; `json` for structured logs |

### 13.4 Image Size

With `serve-all`, CatBoost (~700 MB) and XGBoost + LightGBM (~300 MB) dominate a ~3.3 GB image. Installing only the winning model's library (`--build-arg MODEL_EXTRAS=api,serve-xgboost`) removes the other two.

---

## 14. CI/CD

Four jobs in `.github/workflows/ci.yml`:

- **lint-and-type**: `ruff check src/ tests/` and `mypy src/`, without the heavy ML dependencies.
- **test**: `pip install -e ".[dev]"`, then `pytest --cov=src` and an upload to Codecov.
- **dashboard**: `npm ci`, `npm run lint`, `npm run build`.
- **docker**: builds the runtime image and smoke-tests it: `/health` is 200, and `/logs` is 503 with no API keys (auth fails closed).

---

## 15. MLflow Experiment Tracking

Experiment name: `moroccan_prepaid_churn`
Tracking URI: `sqlite:///mlruns.db` (local, gitignored), resolved once by `src.utils.config.setup_mlflow`.

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db   # → http://localhost:5000
```

Runs logged:

| Run name | Entry point | Key logged items |
|----------|-------------|-----------------|
| `feature_engineering__expresso` | `run_pipeline.py` | n_features, churn_rate_train, pipeline_fit_seconds, top-10 Spearman ρ, artifacts |
| `feature_engineering__moroccan` | `run_pipeline.py --dataset moroccan` | same as above |
| `training__{dataset}__{model}` | `train.py` | CV ROC-AUC/PR-AUC for all 4 candidates, calibration method, OOF metrics, thresholds |
| `evaluate__{dataset}` | `evaluate_run.py` | holdout metrics at frozen thresholds |
| `{model}_optuna_tuning__{dataset}` | `tune.py` | n_trials, best params, sample/full CV ROC-AUC |

MLflow is optional in all entry points via `--no-mlflow` flag.

---

## 16. Test Suite

| File | Covers |
|------|--------|
| `tests/test_features.py` | every transformer, the full pipeline, cross-fitted target encoding (a row's own label never changes its encoding), `UNKNOWN` pack handling, data-driven feature selection |
| `tests/test_models.py` | metrics and threshold utilities, `build_candidates` (incl. XGBoost `gamma`, LightGBM bagging), inner-split early stopping, refit with fixed rounds, tuning helpers |
| `tests/test_training_pipeline.py` | config overlays, namespaced paths, MLflow URI resolution, calibration, and an **end-to-end train → evaluate run** on tiny data that checks the artifact, manifest, and frozen thresholds |
| `tests/test_monitoring.py` | PSI, discrete vs continuous handling, large-sample tiny shifts stay OK, real shifts are flagged, `DriftDetector` on a real SQLite log incl. prediction drift |
| `tests/test_api.py` | all endpoints; strict validation (unknown field, negative value, bad tenure); default threshold from the artifact; subscriber_id echo/logging; `/logs` windows; retention; fail-closed auth; spoof-proof and bounded rate limiting; `/predict` through a **real** fitted pipeline and calibrated model |

`tests/conftest.py` puts the API in explicit dev-auth mode for each test and provides `make_raw_frame()`, synthetic raw rows with a real churn signal.

---

## 17. Recent Production Hardening

| Area | Change | Impact |
|------|--------|--------|
| Moroccan data generator | `generate_moroccan_dataset.py` with CTGAN + resampling fallback | Enables multi-market development without real subscriber data |
| Multi-dataset pipeline | `--dataset {expresso,moroccan}` flag in `run_pipeline.py` | Single pipeline serves both datasets; OPERATOR column handled transparently |
| Multi-model tuning | `--model {catboost,xgboost}` flag in `tune.py` | XGBoost Optuna search implemented with eval_set early stopping |
| XGBoost refit | Early stopping disabled on full-data refit | Prevents training failure when XGBoost wins model selection |
| ARPU calibration | NaN-preserving scaling in `_calibrate_monetary()` | Validation mean is accurate; no phantom zero ARPU records |
| Feature config | `FeaturePipeline` accepts `top_pack_min_freq`, `target_enc_smoothing`, `regularity_inactive_threshold` from config | Config is the real source of truth |
| Threshold utility | `threshold_at_recall()` returns lowest threshold if floor unattainable | Maximises recall under impossible campaign constraints |
| Artifact drift | `evaluate_run.py` fails if saved model features are missing from generated parquet | Makes stale artifacts visible |
| Moroccan holdout evaluation | `evaluate_run.py --dataset moroccan` → `eval_metrics_moroccan.json` | Formal holdout results (pre-fix; thresholds were tuned on the holdout — see models/legacy/) |
| Post-hoc calibration | `calibrate.py` fits isotonic regression on a separate calibration split | Pre-fix: calibrator fit on in-sample scores; superseded by OOF calibration in train.py |
| Prediction logging | `PredictionLogger` (SQLite WAL) + `/logs` endpoint | Every `/predict` call persisted; traffic and latency queryable via dashboard |
| Feature drift detection | `DriftDetector` (PSI + KS) + `/drift` endpoint | Detects distribution shift vs training baseline; INSUFFICIENT_DATA guard for cold start |
| Dashboard monitoring panels | Monitoring panel (KPIs + recent table) + Drift panel (status badge + per-feature table) | Full operational visibility in the React dashboard without external tools |
| Evaluation protocol | Holdout split before fitting; inner-split early stopping; cross-fitted target encoding; OOF calibration and thresholds | Reported metrics are true out-of-sample estimates (old numbers moved to `models/legacy/`) |
| Per-dataset artifacts | `data/features/{dataset}/`, `models/{dataset}/`, `configs/{dataset}.yaml`; one DVC chain per dataset | Datasets and tuning runs no longer overwrite each other |
| Serving the calibrated model | API and CLI load `models/{dataset}/model.pkl` with frozen thresholds | Served probabilities are calibrated; default threshold is meaningful |
| API hardening | Strict schemas, fail-closed auth, trusted-proxy rate limiting with bounded memory, per-worker limits, subscriber_id, windowed `/logs`, retention, background logging | Closes silent-input, open-by-default, and limiter-bypass gaps |
| Drift | Effect-size statuses, discrete-feature handling, holdout baseline, prediction drift | Alerts reflect meaningful shifts, not sample size |
| Generator | Signal-preserving REGION/TOP_PACK mapping; activity-weighted data adoption; no-replacement fallback | Moroccan features keep their signal; no duplicated rows across splits |

---

## 18. Design Decisions & Trade-offs

### Why cross-fit target encoding instead of refitting the pipeline per fold?

Refitting the whole `FeaturePipeline` inside every CV fold would be about 9× slower. Cross-fitting the two target encoders gives the same guarantee for the only label-dependent steps: no training row is encoded with its own label. That is the scheme `sklearn.preprocessing.TargetEncoder` uses. Unseen data (holdout, live traffic) is encoded with maps fit on the full training split.

### Why 400k subsample for Optuna?

Full-data Optuna (2M rows × 3-fold × 50 trials) would take 20–40 hours. A stratified 400k subsample finds near-optimal hyperparameters. The best params are then validated on the full dataset before saving — this catches cases where a subsample-optimised setting fails to generalise.

### Why CTGAN over resampling real rows for the Moroccan dataset?

Resampling real rows preserves the Expresso joint distribution exactly, but it can only produce as many distinct rows as Expresso has. CTGAN learns the joint distribution from an Expresso seed and can sample any number of new rows. The `--no-ctgan` fallback is kept for speed; it samples without replacement so no row is duplicated across the train/holdout split.

### Why preserve NaN in ARPU calibration?

Filling NaN values with 0 before scaling would create phantom low-ARPU records (all imputed as 0 MAD after scaling), causing the validation mean to read as ~21 MAD instead of the target 27.5 MAD. Preserving NaN through the calibration ensures only genuinely active subscribers are included in the ARPU signal.

### Why mount artifacts as volumes, not bake into the image?

Model artifacts (`models/{dataset}/model.pkl`, a few hundred MB) change on every retraining run. Baking them into the image would require a full 3.3 GB rebuild on each retrain. With volume mounts, `dvc repro` regenerates the model and a container restart picks it up — the image itself never changes.

### Why Logistic Regression as a baseline?

A linear model with the engineered feature set came within 3.2 pp of the best GBDT on the pre-fix CV (0.9284 vs 0.9316 ROC-AUC). This confirms the feature engineering is doing most of the work. If LR had scored 0.85, the feature set would need revision. The baseline acts as a quality gate.

### Why separate `evaluate_run.py` from `evaluate.py`?

`evaluate.py` is a library of evaluation utilities (metrics, threshold selection, plot functions, SHAP) used by notebooks, the API tests, and training code. `evaluate_run.py` is the DVC pipeline entry point. It is a thin wrapper that calls the library and writes `models/{dataset}/eval_metrics.json`. Separating them keeps the library reusable and the pipeline stage minimal.

---

## 19. Roadmap

| Item | Status | Description |
|------|--------|-------------|
| Leak-free evaluation protocol | ✅ Done | Holdout before fitting, inner-split early stopping, cross-fitted encoding, OOF calibration/thresholds |
| Post-hoc calibration | ✅ Done | Fit on OOF scores; served by the API |
| Prediction logging | ✅ Done | `PredictionLogger` SQLite WAL; `/logs` endpoint |
| Feature + prediction drift | ✅ Done | `DriftDetector` PSI + KS statistic vs holdout; `/drift` endpoint |
| Dashboard monitoring panels | ✅ Done | Monitoring (KPIs + table) + Drift (status + per-feature table) |
| Ground-truth feedback loop | High | Join delayed churn labels to `prediction_rows.subscriber_id`; compute production ROC-AUC and Brier over rolling windows |
| Automated retraining trigger | Medium | Retrain when drift exceeds PSI threshold or production Brier degrades; promote only if candidate beats current model |
| DVC remote storage | Medium | Configure S3/GCS/Azure remote so the pipeline is reproducible in CI and on new machines |
| Model registry | Medium | Use MLflow Model Registry to promote models from "staging" to "production" with version tracking and rollback |
| API hardening | ✅ Done | Fail-closed auth, proxy-aware rate limiting, strict schemas, JSON logs |
| Shared rate-limit store | Medium | Redis-backed limiter for exact limits across workers and replicas |
| Cloud deployment | Medium | Deploy the API and dashboard to Azure Container Apps, AKS, or Render with auto-scaling |
| Optuna dashboard | Low | `optuna-dashboard` live UI for watching trials during long tuning runs |
