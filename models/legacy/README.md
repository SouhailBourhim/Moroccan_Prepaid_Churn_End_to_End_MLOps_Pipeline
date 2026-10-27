# Legacy metrics (pre-fix evaluation protocol)

These files come from the old flat `models/` layout. They are kept for reference only.
**Don't quote them as results.**

- `eval_metrics.json` (Expresso "holdout"): the 20% "holdout" was taken from
  `train_features.parquet`, which the final model had been refit on. So it is a
  training-set score, not a holdout score.
- `eval_metrics_moroccan.json`: the ROC-AUC and PR-AUC come from a real holdout. The
  thresholds, though, were tuned on that same holdout.
- `eval_metrics_moroccan_calibrated.json`: the calibrator was fit on rows the base
  model had trained on, and the calibration method and thresholds were picked on the
  test split.
- `training_manifest.json`, `tuning_results_*.json`: early stopping watched the fold
  being scored, and target encodings were fit on all training rows, so CV scores are
  slightly optimistic. The tuned XGBoost `gamma` was also never passed to the trained
  model.

Run `dvc repro` to regenerate valid metrics in `models/{expresso,moroccan}/`.
