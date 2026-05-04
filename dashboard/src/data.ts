export type ModelResult = {
  name: string;
  rocAuc: number;
  rocStd: number;
  prAuc: number;
};

export type ThresholdProfile = {
  name: string;
  threshold: number;
  f1: number;
  precision: number;
  recall: number;
};

export type FeatureSignal = {
  name: string;
  group: string;
  status: "active" | "watch" | "derived";
  detail: string;
};

export const modelResults: ModelResult[] = [
  { name: "CatBoost", rocAuc: 0.9316, rocStd: 0.0005, prAuc: 0.7039 },
  { name: "XGBoost", rocAuc: 0.9315, rocStd: 0.0005, prAuc: 0.7039 },
  { name: "LightGBM", rocAuc: 0.9311, rocStd: 0.0005, prAuc: 0.7026 },
  { name: "Logistic Regression", rocAuc: 0.9284, rocStd: 0.0005, prAuc: 0.6899 }
];

export const thresholdProfiles: ThresholdProfile[] = [
  {
    name: "Default",
    threshold: 0.5,
    f1: 0.6787,
    precision: 0.5366,
    recall: 0.9234
  },
  {
    name: "Youden-J",
    threshold: 0.501,
    f1: 0.6789,
    precision: 0.5368,
    recall: 0.9232
  },
  {
    name: "F1 optimal",
    threshold: 0.689,
    f1: 0.7016,
    precision: 0.6119,
    recall: 0.8221
  }
];

export const featureSignals: FeatureSignal[] = [
  {
    name: "regularity_rate",
    group: "Activity",
    status: "active",
    detail: "Normalised 90-day regularity signal"
  },
  {
    name: "n_services_absent",
    group: "Engagement",
    status: "derived",
    detail: "Counts missing MNAR service channels"
  },
  {
    name: "is_ghost_subscriber",
    group: "Risk flag",
    status: "watch",
    detail: "Flags subscribers absent from 5+ services"
  },
  {
    name: "REGION_te",
    group: "Target encoding",
    status: "active",
    detail: "James-Stein smoothed regional churn prior"
  },
  {
    name: "top_pack_te",
    group: "Plan",
    status: "active",
    detail: "Smoothed churn prior for active pack"
  },
  {
    name: "data_per_freq",
    group: "Data usage",
    status: "derived",
    detail: "Data intensity per transaction"
  }
];

export const pipelineStages = [
  "Validate raw schema",
  "Add missing indicators",
  "Impute MNAR and MAR fields",
  "Engineer usage features",
  "Encode tenure, MRG, REGION, TOP_PACK",
  "Train candidate models",
  "Evaluate holdout metrics",
  "Serve FastAPI predictions"
];

export const defaultSubscriber = {
  REGION: "DAKAR",
  TENURE: "K > 24 month",
  MRG: "NO",
  TOP_PACK: "On net 200F=Unlimited _call24H",
  MONTANT: 4250,
  FREQUENCE_RECH: 15,
  REVENUE: 4251,
  ARPU_SEGMENT: 1417,
  FREQUENCE: 17,
  DATA_VOLUME: 4,
  ON_NET: 388,
  ORANGE: 46,
  TIGO: 1,
  ZONE1: 1,
  ZONE2: 2,
  REGULARITY: 54,
  FREQ_TOP_PACK: 8
};
