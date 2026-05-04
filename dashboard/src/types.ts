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

export type SubscriberInput = {
  REGION: string;
  TENURE: string;
  MRG: string;
  TOP_PACK: string;
  MONTANT: string;
  FREQUENCE_RECH: string;
  REVENUE: string;
  ARPU_SEGMENT: string;
  FREQUENCE: string;
  DATA_VOLUME: string;
  ON_NET: string;
  ORANGE: string;
  TIGO: string;
  ZONE1: string;
  ZONE2: string;
  REGULARITY: string;
  FREQ_TOP_PACK: string;
};
