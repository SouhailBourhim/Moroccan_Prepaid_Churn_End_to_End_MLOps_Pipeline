# Moroccan Prepaid Churn — End-to-End MLOps Pipeline

Production-grade churn prediction for prepaid telecom subscribers using the [Expresso Telecom dataset](https://zindi.africa/competitions/expresso-churn-prediction) (Senegal). Target metric: **ROC-AUC**. Class imbalance ~18.75% churn.

---

## Quick Start

```bash
# 1. Clone and set up
git clone <repo-url>
cd Moroccan_Prepaid_Churn_End_to_End_MLOps_Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Place data (see Data Setup below)
#    data/raw/expresso/Train.csv, Test.csv, VariableDefinitions.csv

# 3. Run tests
pytest

# 4. Open EDA notebooks
jupyter notebook notebooks/
```

---

## Datasets

### Primary: Expresso Telecom (Senegal)

The training and evaluation dataset. Sourced from the [Zindi Expresso Churn Challenge](https://zindi.africa/competitions/expresso-churn-prediction).

| Split | Rows | Columns | Churn Rate |
|-------|------|---------|------------|
| Train | 2,154,048 | 19 | 18.75% |
| Test | 380,127 | 18 | — |

Key characteristics: prepaid subscribers, West Africa, heavy missingness (35–94% in usage columns), MNAR patterns encode inactivity.

### Reference: Cell2Cell Wireless (USA)

Used for cross-dataset feature validation only — **not used for training**.

| Split | Rows | Columns | Churn Rate |
|-------|------|---------|------------|
| Train | 51,047 | 58 | 28.82% |
| Holdout | 20,000 | 58 | — |

Source: [Kaggle — datasets-for-churn-telecom](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)

### Reference: Orange Telecom (USA)

Used for cross-dataset feature validation only — **not used for training**.

| Split | Rows | Columns | Churn Rate |
|-------|------|---------|------------|
| Train (80%) | 2,666 | 20 | 14.55% |
| Test (20%) | 667 | 20 | — |

Source: [Kaggle — telecom-churn-datasets](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)

---

## Cross-Dataset Feature Evidence

The reference datasets are used to identify features that predict churn *regardless of operator or geography*. Features with strong signal in all three datasets are treated as robust; features that only work on Expresso may be dataset-specific noise.

### Robustness Scorecard (from `notebooks/04_cross_dataset_analysis.ipynb`)

| Feature Concept | Expresso |ρ| | Cell2Cell |ρ| | Orange |ρ| | Verdict |
|----------------|------------|------------|----------|---------|
| **Engagement / Activity** | **0.53** | 0.05 | 0.10 | Robust — universal disengagement signal |
| **Revenue / Spend Level** | 0.19 | 0.02 | 0.18 | Robust — low spend = low commitment |
| **Call Volume** | 0.20 | 0.06 | 0.17 | Robust — low usage predicts churn |
| **Plan Engagement** | 0.13 | 0.04 | 0.26 | Robust — plan adoption = loyalty |
| **Customer Service Friction** | *(no feature)* | 0.05 | **0.14** | Robust where available — strongest in Orange |
| **Tenure** | strong (DT) | strong (DT) | weak | Conditional — encoding matters |
| **International / Roaming** | 0.11 | 0.07 | mixed | Mixed — plan enrollment > actual usage |

> **Key insight**: The core churn signal is *disengagement*. Expresso's `REGULARITY` (days active / 90) captures this directly — it is the single strongest predictor (Gini importance 0.92 in a depth-6 tree). Every derived feature should amplify or complement this signal.

---

## Architecture

```
configs/base.yaml              ← single source of truth: paths, hyperparameters, services

data/
  raw/
    expresso/                  ← Train.csv, Test.csv, VariableDefinitions.csv  (DVC-tracked)
    cell2cell/                 ← cell2celltrain.csv, cell2cellholdout.csv       (DVC-tracked)
    orange/                    ← churn-bigml-80.csv, churn-bigml-20.csv         (DVC-tracked)
  processed/                   ← engineered features (generated)
  features/                    ← final model-ready arrays (generated)

src/
  data/
    ingestion.py               ← load_train / load_test with explicit dtypes
    validation.py              ← validate_raw() → ValidationReport
  features/
    build_features.py          ← FeaturePipeline + 8 sklearn-compatible transformers
  models/                      ← (planned) training scripts and model wrappers
  utils/
    logging.py                 ← loguru setup; call setup_logger() at entry points

notebooks/
  01_eda.ipynb                 ← Expresso EDA (executed)
  02_eda_cell2cell.ipynb       ← Cell2Cell EDA (executed)
  03_eda_orange.ipynb          ← Orange Telecom EDA (executed)
  04_cross_dataset_analysis.ipynb ← Cross-dataset robustness analysis (executed)

tests/
  test_features.py             ← Unit tests for every transformer

api/                           ← (planned) FastAPI inference endpoint
dashboard/                     ← (planned) monitoring dashboard
docker/                        ← (planned) containerisation
```

---

## Feature Pipeline

`FeaturePipeline` chains 8 transformers in order. All are **fit on train only** and applied identically to train and test.

```
Raw DataFrame (19 cols)
        │
        ▼
1. MissingIndicatorAdder   → adds {col}_missing binary flags before imputation
        │
        ▼
2. ZeroImputer             → MNAR columns → 0
                             (DATA_VOLUME, ON_NET, ORANGE, TIGO, ZONE1, ZONE2, FREQ_TOP_PACK)
        │
        ▼
3. MedianImputer           → MAR columns → train median
                             (MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE)
        │
        ▼
4. NumericFeatureEngineer  → derived features:
                             regularity_rate, recharge_per_freq, revenue_per_freq,
                             total_calls, n_active_call_types, is_inactive, has_data, has_calls
        │
        ▼
5. TenureEncoder           → ordinal 0–10 + is_new_subscriber flag
        │
        ▼
6. MRGEncoder              → YES/NO → mrg_flag binary
        │
        ▼
7. TargetEncoder(REGION)   → James–Stein smoothed target encoding
        │
        ▼
8. TopPackEncoder          → rare-pack collapsing (min_freq=200)
                             + frequency encode + target encode TOP_PACK
        │
        ▼
get_model_features()       → drops user_id, CHURN, REGION, TENURE, MRG, TOP_PACK, ARPU_SEGMENT
```

### Imputation strategy rationale

| Columns | Strategy | Why |
|---------|----------|-----|
| `DATA_VOLUME`, `ON_NET`, `ORANGE`, `TIGO`, `ZONE1`, `ZONE2`, `FREQ_TOP_PACK` | 0 | MNAR: missing = never used that service. Churn rate when missing is 2–4× higher, confirming non-random absence. |
| `MONTANT`, `FREQUENCE_RECH`, `REVENUE`, `ARPU_SEGMENT`, `FREQUENCE` | Train median | MAR: missing correlates with churn but is not caused by service absence. |
| `REGION`, `TOP_PACK` | Target encoding fallback to global mean | Unseen categories at inference get the population churn rate. |

---

## Data Setup

Raw data is tracked via **DVC** and is not committed to git.

```bash
# After installing DVC and configuring remote storage:
dvc pull

# Or place files manually:
data/raw/expresso/Train.csv
data/raw/expresso/Test.csv
data/raw/expresso/VariableDefinitions.csv
data/raw/cell2cell/cell2celltrain.csv
data/raw/cell2cell/cell2cellholdout.csv
data/raw/orange/churn-bigml-80.csv
data/raw/orange/churn-bigml-20.csv
```

---

## Development

### Commands

```bash
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

### Code style

- Line length 100, ruff rules: `E, F, I, N, UP, ANN` (ANN101/ANN102 ignored)
- Strict mypy — all functions need full type annotations
- Python 3.11+ syntax (`list[str]` not `List[str]`, `X | Y` unions)
- Never commit raw data; track via DVC
- Strip notebook outputs before committing

### Running the feature pipeline

```python
from src.data.ingestion import load_train, load_test
from src.features.build_features import FeaturePipeline, get_model_features

train = load_train()
test  = load_test()

X_train = train.drop(columns=["CHURN"])
y_train = train["CHURN"]

pipeline = FeaturePipeline()
X_train_fe = pipeline.fit_transform(X_train, y_train)
X_test_fe  = pipeline.transform(test)

feature_cols = get_model_features(X_train_fe)
```

### Config

All paths, hyperparameters, and service settings live in `configs/base.yaml`. Load with PyYAML at pipeline entry points rather than hardcoding values.

```python
import yaml
from pathlib import Path

with open(Path(__file__).parents[2] / "configs" / "base.yaml") as f:
    cfg = yaml.safe_load(f)
```

---

## MLflow Experiment Tracking

```bash
mlflow ui --backend-store-uri mlruns
# → http://localhost:5000
```

- Experiment name: `moroccan_prepaid_churn`
- Tracking URI: `mlruns/` (local)
- `mlruns/` is gitignored

---

## Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Expresso full EDA — distributions, missingness, correlations, feature importance | Executed |
| [02_eda_cell2cell.ipynb](notebooks/02_eda_cell2cell.ipynb) | Cell2Cell reference EDA — tenure, customer care calls, retention signals | Executed |
| [03_eda_orange.ipynb](notebooks/03_eda_orange.ipynb) | Orange Telecom reference EDA — international plan, customer service calls, charge/minutes collinearity | Executed |
| [04_cross_dataset_analysis.ipynb](notebooks/04_cross_dataset_analysis.ipynb) | Cross-dataset robustness analysis — feature ontology mapping, Spearman ρ, DT importance, priority table | Executed |

---

## Planned Work

- [ ] `src/models/` — XGBoost and LightGBM training scripts with Optuna tuning, MLflow logging, and SHAP explanations
- [ ] `api/` — FastAPI inference endpoint with Pydantic request validation
- [ ] `dashboard/` — Plotly Dash monitoring dashboard
- [ ] `docker/` — Dockerfile and docker-compose for API + MLflow
- [ ] DVC pipeline stages for reproducible end-to-end runs
