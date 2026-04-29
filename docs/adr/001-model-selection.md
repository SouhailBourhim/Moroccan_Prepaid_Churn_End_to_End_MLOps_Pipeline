# ADR 001 — Candidate Model Selection for Expresso Churn Prediction

**Status:** Accepted  
**Date:** 2026-04-29  
**Decider:** SouhailBourhim

---

## Context

We need to select candidate classifiers for predicting prepaid subscriber churn on the Expresso Telecom dataset. The evaluation metric is ROC-AUC. Key constraints:

- **Class imbalance:** ~18% churn rate (not extreme but non-trivial)
- **Feature mix:** numeric usage columns with heavy zero-inflation and missingness (MNAR pattern), ordinal tenure, binary flags, smoothed target-encoded categoricals (REGION, TOP_PACK)
- **Business requirement:** individual predictions must be explainable — retention teams need to know *why* a subscriber is flagged, not just that they are
- **Scale:** single-machine training; dataset is ~2.5M rows

---

## Decision

Train and compare four candidate classifiers via stratified 5-fold cross-validation (ROC-AUC and PR-AUC). The winner by mean CV ROC-AUC is refitted on all training data and saved.

| # | Candidate | Role | Imbalance handling |
|---|-----------|------|--------------------|
| 1 | `LogisticRegression` | Interpretable linear baseline | `class_weight='balanced'` |
| 2 | `XGBoostClassifier` | Gradient boosting — accuracy reference | `scale_pos_weight = n_neg / n_pos` |
| 3 | `LGBMClassifier` | Leaf-wise boosting — speed reference | `is_unbalance=True` |
| 4 | `CatBoostClassifier` | Robust GBDT — primary challenger | `auto_class_weights='Balanced'` |

---

## Rationale

### Why gradient boosting at all?

GBDTs consistently outperform other approaches on tabular data, including neural methods, when row counts are in the millions and features are heterogeneous (Grinsztajn et al., 2022 — *"Why tree-based models still outperform deep learning on tabular data"*; confirmed across hundreds of Kaggle competitions). This dataset is tabular, mixed-type, and single-machine — the ideal regime for GBDTs.

### Why three GBDT variants?

Each covers a distinct axis of the XGBoost–LightGBM–CatBoost tradeoff triangle:

**XGBoost** (`xgb`): level-wise tree growth, long empirical track record on churn problems, conservative bias-variance tradeoff. Used as the accuracy reference point.

**LightGBM** (`lgb`): leaf-wise growth with `num_leaves` control — significantly faster than XGBoost at the same AUC on large datasets. The `is_unbalance=True` flag reweights internally. Primary advantage is training throughput when iterating on features.

**CatBoost** (`cb`): the strongest motivation for inclusion here. Its ordered boosting avoids target leakage during training on categorical features. Even though our pipeline pre-encodes all categoricals with smoothed target encoding, CatBoost's internal loss function handles the zero-inflated numeric structure (DATA_VOLUME, call columns) better than standard boosting in practice. It also produces better-calibrated probabilities out of the box, which matters when setting a probability threshold for business action (retention budget). CatBoost is the most likely winner on this dataset.

### Why Logistic Regression?

A calibrated linear baseline has two values:

1. **Sanity check floor:** if GBDT is not meaningfully better than LR, the feature engineering is adding noise rather than signal, and we should investigate before spending more compute.
2. **Deployment option:** if the AUC gap is < 0.5 pp, the LR model is easier to audit, certify, and serve in constrained environments.

The model uses `StandardScaler` within a sklearn `Pipeline` to prevent data leakage during cross-validation.

---

## Alternatives considered

| Alternative | Reason not included |
|-------------|---------------------|
| Random Forest | Consistently 1–3 pp below GBDT on AUC in churn benchmarks; bagging parallelism offers no benefit vs. LightGBM speed |
| TabNet / neural networks | Not consistently better than GBDT at this scale; harder to explain; calibration is worse without isotonic regression post-processing |
| Stacking / blending ensemble | Adds inference latency and operational complexity; evaluate only if single-model ceiling is insufficient |
| SMOTE oversampling | `scale_pos_weight`-style weighting is strictly simpler, avoids synthetic-sample artifacts, and performs equivalently in benchmarks for moderate imbalance (< 1:10) |
| Optuna hyperparameter search | Deferred — run CV on default configs first to identify the winning model family, then tune that family specifically |

---

## Consequences

- CatBoost is a new dependency (`catboost>=1.2`); adds ~50MB to the environment.
- Training time per candidate (5-fold CV on ~2M rows): LightGBM ≈ 5 min, XGBoost ≈ 10 min, CatBoost ≈ 15 min, LogisticRegression < 1 min.
- SHAP is available for all four via `shap.Explainer` auto-dispatch (TreeExplainer for GBDTs, LinearExplainer for LR pipeline).
- **Next decision point:** after observing CV results, open ADR 002 to decide whether to run Optuna on the winning family or accept the default config.
