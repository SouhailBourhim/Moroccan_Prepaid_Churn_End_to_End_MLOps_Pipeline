# Moroccan Prepaid Churn — End-to-End MLOps Pipeline

Production-grade churn prediction for prepaid telecom subscribers. Trains on the [Expresso Telecom dataset](https://zindi.africa/competitions/expresso-churn-prediction) (Senegal, 2.15M subscribers) and a synthetically generated Moroccan prepaid dataset (2M subscribers, CTGAN-calibrated to ANRT 2023 market data). Target metric: **ROC-AUC**. Class imbalance: **18.75% churn (Expresso) / 2.7% churn (Moroccan)**.

> **Expresso holdout — ROC-AUC: 0.9330 · PR-AUC: 0.7071 · Brier: 0.1119** (CatBoost, 20% stratified holdout)
> **Moroccan synthetic — XGBoost Optuna CV ROC-AUC: 0.8972 ± 0.0014** (tuned, 400k subsample)

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/SouhailBourhim/Moroccan_Prepaid_Churn_End_to_End_MLOps_Pipeline
cd Moroccan_Prepaid_Churn_End_to_End_MLOps_Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Pull raw data via DVC (requires a configured remote)
dvc pull
# or place Train.csv / Test.csv manually in data/raw/expresso/

# 3. Run the full pipeline (Expresso)
dvc repro

# 4. Generate the Moroccan synthetic dataset (CTGAN, ~15 min on GPU / ~2h on CPU)
python generate_moroccan_dataset.py --n-rows 2000000 --epochs 300 --output data/raw/moroccan_telecom_churn.csv
# Fast bootstrap fallback (no CTGAN, <60s):
python generate_moroccan_dataset.py --no-ctgan

# 5. Start the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs

# 6. Start the React dashboard
cd dashboard
npm install
npm run dev
# → http://localhost:5173

# 7. Or run the API via Docker
docker compose up --build
```

---

## Project Status

| Phase | Status | Key output |
|-------|--------|------------|
| EDA — Expresso | ✅ Done | `notebooks/01_eda.ipynb` |
| EDA — Cell2Cell (reference) | ✅ Done | `notebooks/02_eda_cell2cell.ipynb` |
| EDA — Orange Telecom (reference) | ✅ Done | `notebooks/03_eda_orange.ipynb` |
| Cross-dataset analysis | ✅ Done | `notebooks/04_cross_dataset_analysis.ipynb` |
| Feature engineering | ✅ Done | `src/features/`, `data/features/` |
| Model training (4-candidate CV) | ✅ Done | `src/models/train.py`, `models/best_model.pkl` |
| Hyperparameter tuning — CatBoost (Optuna) | ✅ Done | `src/models/tune.py`, `models/tuning_results.json` |
| Hyperparameter tuning — XGBoost (Optuna) | ✅ Done | `src/models/tune.py --model xgboost`, `models/tuning_results_xgboost.json` |
| Model evaluation | ✅ Done | `notebooks/06_model_evaluation.ipynb`, `models/eval_metrics.json` |
| Moroccan synthetic dataset generation | ✅ Done | `generate_moroccan_dataset.py` (CTGAN, 2M rows, DVC-tracked) |
| FastAPI serving endpoint | ✅ Done | `src/api/`, port 8000 |
| React operations dashboard | ✅ Done | `dashboard/`, port 5173 |
| Docker containerisation | ✅ Done | `Dockerfile`, `docker-compose.yml` |
| DVC reproducible pipeline | ✅ Done | `dvc.yaml` (featurize → train → evaluate) |
| CI/CD | ✅ Done | `.github/workflows/ci.yml` |
| Production monitoring loop | 🔜 Next | prediction logging, drift checks, feedback labels |

---

## Architecture

```
configs/base.yaml              ← single source of truth: all hyperparams & paths
                                 includes moroccan_raw_file path

generate_moroccan_dataset.py   ← CTGAN-based synthetic Moroccan prepaid dataset
                                 2M rows · 12 regions · 3 operators (Inwi/Maroc_Telecom/Orange_Maroc)
                                 2.7% churn · 27.5 MAD ARPU · 74% 4G penetration
                                 bootstrap fallback with --no-ctgan

data/
  raw/expresso/                ← Train.csv (247 MB), Test.csv (43 MB) — DVC-tracked
  raw/moroccan_telecom_churn.csv ← synthetic Moroccan dataset (276 MB) — DVC-tracked
  features/                   ← generated artifacts (parquet, pkl, json)

src/
  data/
    ingestion.py               ← load_train / load_test / load_moroccan with explicit dtypes
    validation.py              ← validate_raw() → ValidationReport
  features/
    build_features.py          ← FeaturePipeline (9 transformers) + get_model_features()
    run_pipeline.py            ← load → validate → fit → save → MLflow
                                 --dataset {expresso,moroccan}
  models/
    train.py                   ← 4-candidate CV training + MLflow logging
    tune.py                    ← Optuna search for CatBoost or XGBoost
                                 --model {catboost,xgboost}
    evaluate.py                ← metric utilities, threshold selection, SHAP
    evaluate_run.py            ← standalone holdout eval (DVC stage output)
    predict.py                 ← CLI inference on raw CSV
  api/
    app.py                     ← FastAPI: /health /ready /info /predict
    schemas.py                 ← Pydantic request/response models
    dependencies.py            ← ModelArtifacts loader (singleton on startup)
  utils/
    logging.py                 ← loguru setup

models/                        ← DVC-tracked artifacts
  best_model.pkl               ← fitted CatBoost + feature_cols list
  training_manifest.json       ← CV results for all 4 candidates
  eval_metrics.json            ← holdout metrics (DVC metric)
  tuning_results.json          ← CatBoost Optuna best params
  tuning_results_xgboost.json  ← XGBoost Optuna best params

notebooks/
  01–04_eda*.ipynb             ← EDA on 3 datasets + cross-dataset analysis
  05_feature_engineering.ipynb ← pipeline walkthrough + validation
  06_model_evaluation.ipynb    ← CV comparison, ROC/PR/calibration, SHAP

docs/
  adr/001-model-selection.md   ← why CatBoost + XGBoost + LightGBM + LR
  PROJECT.md                   ← full technical + functional documentation

dashboard/                     ← Vite React dashboard for model ops
  src/                         ← KPI tiles, charts, feature signals, scoring panel

Dockerfile                     ← multi-stage build (3.3 GB; catboost is the floor)
docker-compose.yml             ← mounts models/ and data/features/ as volumes
dvc.yaml                       ← featurize → train → evaluate pipeline DAG
```

---

## Pipeline

### Running end-to-end (Expresso)

```bash
dvc repro                   # re-runs only changed stages
dvc repro --force           # force full re-run
dvc dag                     # visualise the DAG
dvc metrics show            # print eval_metrics.json
dvc metrics diff HEAD~1     # compare metrics to previous commit
```

### Running stages individually

```bash
# Featurize
python -m src.features.run_pipeline --no-mlflow              # Expresso (default)
python -m src.features.run_pipeline --dataset moroccan       # Moroccan synthetic

# Train
python -m src.models.train --no-mlflow                       # 4-candidate CV

# Evaluate
python -m src.models.evaluate_run                            # holdout eval

# Tune
python -m src.models.tune --trials 50                        # CatBoost Optuna (default)
python -m src.models.tune --model xgboost --trials 50        # XGBoost Optuna

# Inference
python -m src.models.predict --threshold 0.4                 # inference on Test.csv
```

### Generating the Moroccan dataset

```bash
# Full CTGAN synthesis (recommended, requires sdv)
python generate_moroccan_dataset.py \
    --n-rows 2000000 --epochs 300 --batch-size 10000 \
    --output data/raw/moroccan_telecom_churn.csv

# Bootstrap fallback (fast, no GPU required)
python generate_moroccan_dataset.py --no-ctgan

# Track with DVC after generation
dvc add data/raw/moroccan_telecom_churn.csv
```

---

## Model Results

### Expresso Telecom dataset

Five-fold stratified CV on 2,154,048 training rows (default configs):

| Model | CV ROC-AUC | CV PR-AUC |
|-------|-----------|----------|
| **CatBoost** ★ | **0.9316 ± 0.0005** | **0.7039** |
| XGBoost | 0.9314 ± 0.0005 | 0.7039 |
| LightGBM | 0.9313 ± 0.0005 | 0.7038 |
| Logistic Regression | 0.9284 ± 0.0005 | 0.6899 |

Holdout evaluation (20% stratified split, 430,810 rows):

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9330** |
| PR-AUC | **0.7071** |
| Brier score | 0.1119 |
| F1 at Youden-J threshold (0.501) | 0.679 |
| Precision / Recall at Youden | 0.537 / 0.923 |
| F1 at F1-optimal threshold (0.689) | 0.702 |

### Moroccan synthetic dataset

Five-fold stratified CV on Moroccan synthetic data (~1.6M training rows, 2.7% churn):

| Model | CV ROC-AUC | CV PR-AUC |
|-------|-----------|----------|
| **XGBoost** ★ | **0.8963 ± 0.0017** | **0.2300** |
| CatBoost | 0.8961 ± 0.0016 | 0.2292 |
| LightGBM | 0.8841 ± 0.0015 | 0.1941 |
| Logistic Regression | 0.8782 ± 0.0015 | 0.1979 |

After Optuna tuning (50 trials, 400k stratified subsample):

| Model | Optuna CV ROC-AUC | Tuned params |
|-------|------------------|-------------|
| **XGBoost** ★ | **0.8972 ± 0.0014** | depth=7, lr=0.0258, min_child_weight=260 |
| CatBoost | 0.8957 ± 0.0015 | depth=4, lr=0.0456, l2_leaf_reg=8.92 |

Lower absolute AUC vs Expresso reflects the harder imbalance (2.7% vs 18.75% churn) and the smaller per-class signal in a synthetic dataset.

Rationale for model selection: see [`docs/adr/001-model-selection.md`](docs/adr/001-model-selection.md).

---

## API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe — always 200 |
| `/ready` | GET | Readiness probe — 503 until model loaded |
| `/info` | GET | Model name, CV AUC, feature count |
| `/predict` | POST | Batch churn scoring (1–10 000 subscribers) |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subscribers": [{
      "REGION": "DAKAR", "TENURE": "K > 24 month", "MRG": "NO",
      "REGULARITY": 54.0, "ON_NET": 388.0, "REVENUE": 4251.0
    }],
    "threshold": 0.5
  }'
```

Interactive docs: `http://localhost:8000/docs`

---

## Dashboard

The React dashboard in `dashboard/` gives a local operations view over the project:

- model KPI tiles for best model, CV ROC-AUC, CV PR-AUC, and feature count
- candidate model comparison chart
- threshold tradeoff chart for precision, recall, and F1
- high-signal feature list for engineered churn indicators
- editable subscriber scoring panel with score-current, batch queue, and batch results
- DVC stage flow showing the current reproducible pipeline

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173`. By default the dashboard calls `http://localhost:8000`; set `VITE_CHURN_API_URL` to point it at another API deployment.

```bash
VITE_CHURN_API_URL=https://your-api.example.com npm run dev
```

If the API is offline, the dashboard falls back to saved project metrics so the UI remains usable for demos and documentation.

---

## Docker

```bash
docker compose up --build      # build and start
docker compose up              # start with cached image

# Manual run
docker run -p 8000:8000 \
  -v ./models:/app/models:ro \
  -v ./data/features:/app/data/features:ro \
  -e API_WORKERS=4 \
  expresso-churn-api:latest
```

Model artifacts are **mounted at runtime**, not baked into the image — swap `models/best_model.pkl` without rebuilding.

---

## Feature Engineering

`FeaturePipeline` chains 9 sklearn-compatible transformers (fit on train, applied to both splits). The same pipeline runs on both Expresso and Moroccan data; the `--dataset moroccan` flag in `run_pipeline.py` handles 80/20 splitting and drops the Moroccan-only `OPERATOR` column before the pipeline starts.

```
Raw (19 cols)
  → MissingIndicatorAdder     {col}_missing flags  (14 cols)
  → ServiceAbsenceEncoder     n_services_absent, is_ghost_subscriber
  → ZeroImputer               MNAR usage cols → 0
  → MedianImputer             MAR financial cols → training median
  → NumericFeatureEngineer    regularity_rate, recharge_per_freq, total_calls, …
  → TenureEncoder             tenure_ordinal + is_new_subscriber
  → MRGEncoder                YES/NO → mrg_flag
  → TargetEncoder(REGION)     James–Stein smoothed → REGION_te
  → TopPackEncoder            rare collapsing + freq + target encode
  → get_model_features()      drops IDs + raw categoricals + ARPU_SEGMENT duplicate
38 model features after redundant-feature pruning, zero nulls
```

Full rationale: [`docs/PROJECT.md`](docs/PROJECT.md) § Feature Engineering.

---

## Development

```bash
pytest                              # 48 tests
pytest tests/test_features.py -v   # feature tests only
pytest tests/test_models.py -v     # model utility tests
pytest tests/test_api.py -v        # API endpoint tests

ruff check src/ tests/             # lint
mypy src/                          # type check

cd dashboard && npm run lint        # dashboard lint
cd dashboard && npm run build       # dashboard production build

mlflow ui --backend-store-uri mlruns   # experiment tracker → :5000
```

### Code conventions

- Line length 100; ruff rules `E, F, I, N, UP, ANN`
- Strict mypy — all functions fully typed, Python 3.11+ syntax
- Never commit raw data or model artifacts — track via DVC
- Strip notebook outputs before committing

---

## Datasets

| Dataset | Role | Rows | Churn rate |
|---------|------|------|-----------|
| Expresso Telecom (Senegal) | Primary — training + evaluation | 2,154,048 | 18.75% |
| Moroccan synthetic (CTGAN) | Secondary — Moroccan market training | 2,000,000 | 2.7% |
| Cell2Cell Wireless (USA) | Reference EDA only | 51,047 | 28.82% |
| Orange Telecom (USA) | Reference EDA only | 2,666 | 14.55% |

The Moroccan dataset is generated by `generate_moroccan_dataset.py` using CTGAN (SDV library) calibrated to ANRT 2023 data: 12 Moroccan city-regions with population-weighted distribution, three operators (Inwi 35.3%, Maroc_Telecom 32.5%, Orange_Maroc 32.2%), 27.5 MAD target ARPU, 74% 4G data penetration, and 2.7% monthly churn. A bootstrap fallback (`--no-ctgan`) is available when SDV is not installed.

Sources: [Zindi Expresso Challenge](https://zindi.africa/competitions/expresso-churn-prediction) · [Kaggle Cell2Cell](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom) · [Kaggle Orange](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)

---

## What's Next

The next production milestone is the monitoring and feedback loop:

1. Log prediction requests, scores, latency, model version, and timestamps.
2. Compare live feature distributions against the training baseline for drift.
3. Join future ground-truth churn labels back to stored predictions.
4. Trigger retraining when drift or performance degradation crosses a threshold.
5. Promote model versions through MLflow registry with rollback support.
6. Surface drift, traffic, and prediction distribution in the React dashboard.

After that, harden deployment with API auth, rate limiting, CORS controls, container scanning, and a real cloud deployment target.

---

For a full technical and functional walkthrough, see [`docs/PROJECT.md`](docs/PROJECT.md).
