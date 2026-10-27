# Moroccan Prepaid Churn — End-to-End MLOps Pipeline

Churn prediction for prepaid telecom subscribers, built as a production-grade pipeline. It trains on the [Expresso Telecom dataset](https://zindi.africa/competitions/expresso-churn-prediction) (Senegal, 2.15M subscribers) and on a synthetic Moroccan prepaid dataset (2M subscribers). The Moroccan data is generated with CTGAN from Expresso and calibrated to ANRT 2023 market statistics. Target metric: **ROC-AUC**. Class imbalance: **18.75% churn (Expresso) / 2.7% churn (Moroccan)**.

**Stack:** Python · CatBoost · XGBoost · LightGBM · CTGAN (SDV) · DVC · MLflow · FastAPI · React + Vite + TypeScript · Docker

Morocco has no public prepaid churn dataset. To work around that, the pipeline trains on Senegal's Expresso data and also generates a Moroccan-shaped dataset from it. Each dataset is reported against its own holdout. Be careful how you read the Moroccan results: that dataset is Expresso's joint distribution, shifted to a 2.7% churn prior and relabelled with Moroccan names and units. It exercises the pipeline on Moroccan-shaped data. It is **not** evidence about how Moroccan subscribers actually churn.

> **Results need regenerating.** The evaluation protocol was fixed: the holdout is now split off before anything is fit, early stopping uses an inner split, target encodings are cross-fitted, calibration uses out-of-fold scores, and thresholds are frozen before the holdout is scored. Run `dvc repro` to produce valid numbers in `models/{expresso,moroccan}/eval_metrics.json`. The [pre-fix numbers](#pre-fix-results-legacy) are kept below for reference only.

---

## Evaluation protocol

| Step | Data it may see | Never sees |
|------|-----------------|------------|
| Holdout split (stratified 20%) | — | — |
| Feature pipeline fit (imputers, encoders) | training split | holdout |
| Target encoding of training rows | other folds of the training split (cross-fitted) | the row's own label |
| CV model selection | training split; early stopping on an inner 10% of each fold | the fold being scored |
| Final refit | all training rows, with the mean early-stopped round count from CV | holdout |
| Calibration (isotonic vs Platt) | out-of-fold CV predictions; method picked by CV Brier on those | holdout |
| Decision thresholds (Youden-J, F1) | cross-fitted calibrated OOF predictions | holdout |
| **Holdout evaluation** | holdout, with the thresholds frozen above | — |

The API and the batch CLI serve exactly the artifact that this protocol produces: `models/{dataset}/model.pkl`, a calibrated model plus its frozen thresholds.

---

## Architecture

```
configs/
  base.yaml                    ← shared defaults (features, training, hyperparams, API)
  expresso.yaml / moroccan.yaml← per-dataset overlays (tuned hyperparams); deep-merged on base

generate_moroccan_dataset.py   ← CTGAN-based synthetic Moroccan prepaid dataset
                                 12 regions · 3 operators · 2.7% churn · 27.5 MAD ARPU · 74% 4G
                                 region/pack names mapped so their churn signal is kept
                                 --no-ctgan: real Expresso rows, no replacement (no duplicates)

data/
  raw/                         ← Expresso Train/Test.csv, moroccan_telecom_churn.csv (DVC-tracked)
  features/{dataset}/          ← train / holdout / score parquet, feature_pipeline.pkl, manifest

src/
  utils/config.py              ← load_config(dataset), dataset_paths(dataset), MLflow setup
  data/
    ingestion.py               ← loaders with explicit dtypes; records_to_frame() for API/log rows
    validation.py              ← validate_raw() → ValidationReport (run on every dataset)
  features/
    build_features.py          ← FeaturePipeline (9 transformers, cross-fitted target encoding)
                                 + data-driven get_model_features()
    run_pipeline.py            ← load → validate → split holdout → fit on train → save
  models/
    fitting.py                 ← early stopping on an inner split; refit with fixed rounds
    train.py                   ← 4-candidate CV → refit → OOF calibration → frozen thresholds
    calibrate.py               ← CalibratedChurnModel, fit_calibration() on OOF scores
    tune.py                    ← Optuna search; writes configs/{dataset}.yaml only
    evaluate.py                ← metric utilities, threshold selection, SHAP
    evaluate_run.py            ← holdout evaluation with frozen thresholds (DVC stage)
    artifacts.py               ← load_artifacts(dataset) shared by the API and predict CLI
    predict.py                 ← batch inference CLI
  api/
    app.py                     ← FastAPI: /health /ready /info /predict /logs /drift
    schemas.py                 ← strict request models (unknown fields → 422)
    auth.py                    ← X-API-Key, fails closed
    rate_limit.py              ← per-client sliding window, proxy-aware, bounded memory
    logger.py                  ← SQLite prediction log (WAL, retention, subscriber_id)
  monitoring/
    drift.py                   ← feature + prediction drift vs. holdout (PSI, KS statistic)

models/{dataset}/              ← model.pkl (DVC), training_manifest.json, eval_metrics.json,
                                 tuning_results_{model}.json
models/legacy/                 ← pre-fix metrics, kept for reference only

dashboard/                     ← Vite React operations dashboard
```

---

## Quick Start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Pull raw data via DVC (or place Train.csv / Test.csv in data/raw/expresso/)
dvc pull

# 3. Generate the Moroccan synthetic dataset (CTGAN, ~15 min GPU / ~2 h CPU)
python generate_moroccan_dataset.py --n-rows 2000000 --epochs 300
# Fast fallback without CTGAN (real Expresso rows, capped at ~1.8M to avoid duplicates):
python generate_moroccan_dataset.py --no-ctgan

# 4. Run both pipelines (featurize → train → evaluate, per dataset)
dvc repro

# 5. Start the API (local dev: auth explicitly disabled)
CHURN_AUTH_DISABLED=1 uvicorn src.api.app:app --port 8000 --reload
# → http://localhost:8000/docs

# 6. Start the dashboard
cd dashboard && npm install && npm run dev     # → http://localhost:5173
```

---

## Pipeline

```bash
dvc repro                      # both datasets; only changed stages re-run
dvc repro evaluate_expresso    # one chain
dvc dag
dvc metrics show               # models/*/training_manifest.json + eval_metrics.json
```

Running stages individually:

```bash
python -m src.features.run_pipeline --dataset expresso     # or --dataset moroccan
python -m src.models.train         --dataset expresso
python -m src.models.evaluate_run  --dataset expresso
python -m src.models.tune          --dataset moroccan --model xgboost --trials 50
python -m src.models.predict       --dataset expresso       # scores Test.csv
```

`tune` writes the best parameters to `configs/{dataset}.yaml` and `models/{dataset}/tuning_results_{model}.json`. It never writes a model. Retrain afterwards (`dvc repro` picks up the config change), so the tuned parameters go through the same CV → refit → calibration path as every other model.

---

## Model Results

Run `dvc repro`, then read `models/{dataset}/eval_metrics.json`. It contains:

- ROC-AUC, PR-AUC, and Brier score of the served (calibrated) model on the holdout
- the uncalibrated Brier score and the no-skill baseline Brier score
- F1, precision, and recall at the Youden-J and F1 thresholds frozen during training

`training_manifest.json` holds the CV comparison of all four candidates, the calibration choice, and the out-of-fold metrics.

### Pre-fix results (legacy)

These come from the old protocol and are kept in [`models/legacy/`](models/legacy/README.md). They are optimistic and must not be quoted as results:

| Dataset | Metric | Old value | Why it is not valid |
|---------|--------|-----------|---------------------|
| Expresso | "holdout" ROC-AUC / PR-AUC | 0.9330 / 0.7071 | scored on rows the final model was trained on |
| Expresso | CV ROC-AUC (CatBoost) | 0.9316 ± 0.0005 | early stopping on the scored fold; target encodings fit on all rows |
| Moroccan | holdout ROC-AUC / PR-AUC | 0.8965 / 0.2319 | real holdout, but thresholds were tuned on it |
| Moroccan | calibrated Brier | 0.0227 | calibrator fit on in-sample scores; method picked on the test split |

The lower Moroccan ROC-AUC is **not** explained by the lower churn rate. ROC-AUC does not depend on class prevalence. The gap comes from CTGAN fidelity and the random relabelling of columns in the old generator.

Rationale for the model candidates: [`docs/adr/001-model-selection.md`](docs/adr/001-model-selection.md).

---

## API

```bash
CHURN_API_KEYS=my-key uvicorn src.api.app:app --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe — always 200, no auth |
| `/ready` | GET | Readiness probe — 503 until the model is loaded |
| `/info` | GET | Model name, dataset, calibration method, default threshold, CV metrics |
| `/predict` | POST | Batch scoring (1–10 000 subscribers), calibrated probabilities, logged |
| `/logs` | GET | All-time totals + traffic stats for the last `hours`, recent rows |
| `/drift` | GET | Feature and prediction drift vs. the holdout baseline |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -H "X-API-Key: my-key" \
  -d '{
    "subscribers": [{
      "subscriber_id": "sub-000123",
      "REGION": "DAKAR", "TENURE": "K > 24 month", "MRG": "NO",
      "REGULARITY": 54.0, "ON_NET": 388.0, "REVENUE": 4251.0
    }]
  }'
```

- **Probabilities are calibrated.** Without `threshold`, labels use the F1-optimal threshold chosen in training. `/info` reports it.
- **Strict inputs.** Unknown or misspelled fields, negative amounts, and unknown `TENURE` or `MRG` values are rejected with 422. They are never silently treated as missing.
- **`subscriber_id`** is optional. It is not a model input. It is echoed back and stored in the prediction log, so churn labels can be joined back later.
- **Auth fails closed.** With neither `CHURN_API_KEYS` nor `CHURN_AUTH_DISABLED=1` set, protected endpoints return 503.
- **Rate limits** are per client and per 60 s. `X-Forwarded-For` is only trusted from `CHURN_TRUSTED_PROXIES`. Limits are divided across `API_WORKERS`.
- **Drift** status uses PSI on every feature and the KS *statistic* on continuous ones. It does not use p-values, which flag trivial shifts at large sample sizes. The drift of the served `churn_probability` is reported too.

See [`.env.example`](.env.example) for every setting.

---

## Dashboard

The React dashboard in `dashboard/` shows:

- model KPIs and the candidate comparison
- threshold trade-offs, ROC/PR/SHAP/calibration views, and geography
- an editable scoring panel
- the DVC stage flow
- live **Monitoring** (`/logs`) and **Drift** (`/drift`, including prediction drift) panels

```bash
cd dashboard
VITE_CHURN_API_URL=http://localhost:8000 VITE_CHURN_API_KEY=my-key npm run dev
```

`VITE_CHURN_API_KEY` is sent as `X-API-Key`. Vite inlines it into the JavaScript bundle, so only use it against local or internal deployments. If the API is offline or rejects the key, the dashboard says so and falls back to saved demo metrics.

---

## Docker

```bash
cp .env.example .env            # set CHURN_API_KEYS
docker compose up --build
```

- Model artifacts (`models/`, `data/features/`) are **mounted read-only at runtime**, not baked into the image.
- The prediction log lives on the `predictions-db` named volume, so monitoring and drift history survive restarts.
- `CHURN_DATASET` selects which dataset's model is served.
- The image installs every boosting library by default. To shrink it, install only the winning model's library: `MODEL_EXTRAS=api,serve-xgboost docker compose build`.

---

## Feature Engineering

`FeaturePipeline` chains 9 sklearn-compatible transformers. They are fit on the training split only. `fit_transform` cross-fits the target encoders, so no training row is encoded with its own label.

```
Raw (19 cols)
  → MissingIndicatorAdder     {col}_missing flags
  → ServiceAbsenceEncoder     n_services_absent, is_ghost_subscriber
  → ZeroImputer               MNAR usage cols → 0
  → MedianImputer             MAR financial cols → training median
  → NumericFeatureEngineer    regularity_rate, recharge_per_freq, total_calls, …
  → TenureEncoder             tenure_ordinal + is_new_subscriber
  → MRGEncoder                YES/NO → mrg_flag
  → TargetEncoder(REGION)     James–Stein smoothed, cross-fitted → REGION_te
  → TopPackEncoder            rare collapsing + freq + cross-fitted target encode
  → get_model_features()      drops IDs/raw categoricals, then constant and
                              exact-duplicate columns (data-driven, per dataset)
```

Full rationale: [`docs/PROJECT.md`](docs/PROJECT.md) § Feature Engineering.

---

## Development

```bash
pytest                              # unit, API, drift, and end-to-end train→evaluate tests
ruff check src/ tests/
mypy src/
cd dashboard && npm run lint && npm run build
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

CI runs lint and type checks, the tests, a dashboard lint and build, and a Docker build. The Docker job includes a smoke test that `/health` is up and that auth fails closed.

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
| Moroccan synthetic (CTGAN) | Secondary — pipeline exercise on Moroccan-shaped data | 2,000,000 | 2.7% |
| Cell2Cell Wireless (USA) | Reference EDA only | 51,047 | 28.82% |
| Orange Telecom (USA) | Reference EDA only | 2,666 | 14.55% |

How the Moroccan generator handles each column:

- **REGION and TOP_PACK** are mapped from Expresso categories with a churn-rate-ordered (comonotone) assignment. This keeps the region–churn and pack–churn signal while matching population and pack weights. Which city inherits which Senegalese region's churn level is arbitrary.
- **OPERATOR and MRG** are random. They carry no churn signal, and OPERATOR is dropped before modelling.
- **New data users** (to reach 74% 4G penetration) are drawn preferentially from active subscribers.

Sources: [Zindi Expresso Challenge](https://zindi.africa/competitions/expresso-churn-prediction) · [Kaggle Cell2Cell](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom) · [Kaggle Orange](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)

---

## What's Next

1. **Ground-truth feedback loop**: join delayed churn labels to `prediction_rows.subscriber_id` and track production ROC-AUC, PR-AUC, and Brier score over rolling windows.
2. **Automated retraining trigger**: retrain when drift or performance degradation crosses a threshold, and promote through the MLflow registry with rollback.
3. **Shared rate-limit store** (Redis) for exact limits across workers and replicas.
4. **Cloud deployment**: Azure Container Apps, AKS, or Render.

For a full technical walkthrough, see [`docs/PROJECT.md`](docs/PROJECT.md).
