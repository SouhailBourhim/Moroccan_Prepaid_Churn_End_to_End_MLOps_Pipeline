export type ApiInfo = {
  model_name: string;
  cv_roc_auc_mean: number;
  cv_roc_auc_std: number;
  cv_pr_auc_mean: number;
  n_features: number;
};

export type ReadyState = "checking" | "ready" | "offline";

export type Prediction = {
  churn_probability: number;
  churn_prediction: boolean;
};

export type PredictionResponse = {
  predictions: Prediction[];
  model_name: string;
  threshold: number;
  n_subscribers: number;
};
