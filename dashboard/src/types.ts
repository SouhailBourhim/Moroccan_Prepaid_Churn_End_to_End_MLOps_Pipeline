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

export type LogsSummary = {
  total_predictions: number;
  total_requests: number;
  mean_churn_probability: number | null;
  churn_flag_rate: number | null;
  mean_latency_ms: number | null;
};

export type RecentPrediction = {
  request_id: string;
  timestamp: string;
  model_name: string | null;
  threshold: number;
  latency_ms: number | null;
  subscriber_idx: number;
  churn_probability: number;
  churn_prediction: boolean;
};

export type LogsResponse = {
  summary: LogsSummary;
  recent: RecentPrediction[];
};

export type FeatureDriftResult = {
  feature: string;
  psi: number;
  ks_statistic: number;
  ks_pvalue: number;
  status: "OK" | "WARN" | "DRIFT";
  n_live: number;
};

export type DriftResponse = {
  report_time: string;
  window_hours: number;
  n_live_predictions: number;
  n_features_checked: number;
  n_drifted: number;
  n_warned: number;
  overall_status: "OK" | "WARN" | "DRIFT" | "INSUFFICIENT_DATA";
  features: FeatureDriftResult[];
};
