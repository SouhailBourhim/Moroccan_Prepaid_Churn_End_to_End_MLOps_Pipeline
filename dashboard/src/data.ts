export type ModelResult = {
  name: string;
  rocAuc: number;
  rocStd: number;
  prAuc: number;
  brier: number;
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

export type RocPoint = { fpr: number; tpr: number };
export type PrPoint = { recall: number; precision: number };
export type CalibrationPoint = { predicted: number; actual: number };
export type RegionChurn = { region: string; rate: number };

export type ShapEntry = {
  feature: string;
  importance: number;
  direction: "positive" | "negative" | "mixed";
};

// ── Model comparison ──────────────────────────────────────────────────────────

export const modelResults: ModelResult[] = [
  { name: "CatBoost", rocAuc: 0.9316, rocStd: 0.0005, prAuc: 0.7039, brier: 0.1119 },
  { name: "XGBoost", rocAuc: 0.9315, rocStd: 0.0005, prAuc: 0.7038, brier: 0.1121 },
  { name: "LightGBM", rocAuc: 0.9311, rocStd: 0.0005, prAuc: 0.7026, brier: 0.1128 },
  { name: "LogReg", rocAuc: 0.9284, rocStd: 0.0005, prAuc: 0.6899, brier: 0.1183 },
];

export const thresholdProfiles: ThresholdProfile[] = [
  { name: "Default", threshold: 0.5, f1: 0.6787, precision: 0.5366, recall: 0.9234 },
  { name: "Youden-J", threshold: 0.501, f1: 0.6789, precision: 0.5368, recall: 0.9232 },
  { name: "F1 optimal", threshold: 0.689, f1: 0.7016, precision: 0.6119, recall: 0.8221 },
];

// ── Feature signals ───────────────────────────────────────────────────────────

export const featureSignals: FeatureSignal[] = [
  { name: "regularity_rate", group: "Activity", status: "active", detail: "Normalised 90-day regularity signal" },
  { name: "n_services_absent", group: "Engagement", status: "derived", detail: "Counts missing MNAR service channels" },
  { name: "is_ghost_subscriber", group: "Risk flag", status: "watch", detail: "Flags subscribers absent from 5+ services" },
  { name: "REGION_te", group: "Target encoding", status: "active", detail: "James-Stein smoothed regional churn prior" },
  { name: "top_pack_te", group: "Plan", status: "active", detail: "Smoothed churn prior for active pack" },
  { name: "data_per_freq", group: "Data usage", status: "derived", detail: "Data intensity per transaction" },
];

// ── Pipeline stages ───────────────────────────────────────────────────────────

export const pipelineStages = [
  "Validate raw schema",
  "Add missing indicators",
  "Impute MNAR and MAR fields",
  "Engineer usage features",
  "Encode tenure, MRG, REGION, TOP_PACK",
  "Train candidate models",
  "Evaluate holdout metrics",
  "Serve FastAPI predictions",
];

// ── Default subscriber ────────────────────────────────────────────────────────

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
  FREQ_TOP_PACK: 8,
};

// ── ROC curve (CatBoost holdout, AUC=0.933) ───────────────────────────────────

export const rocCurveData: RocPoint[] = [
  { fpr: 0.000, tpr: 0.000 },
  { fpr: 0.005, tpr: 0.320 },
  { fpr: 0.010, tpr: 0.480 },
  { fpr: 0.020, tpr: 0.600 },
  { fpr: 0.030, tpr: 0.668 },
  { fpr: 0.050, tpr: 0.742 },
  { fpr: 0.070, tpr: 0.794 },
  { fpr: 0.100, tpr: 0.845 },
  { fpr: 0.150, tpr: 0.885 },
  { fpr: 0.200, tpr: 0.910 },
  { fpr: 0.250, tpr: 0.930 },
  { fpr: 0.300, tpr: 0.948 },
  { fpr: 0.400, tpr: 0.964 },
  { fpr: 0.500, tpr: 0.975 },
  { fpr: 0.600, tpr: 0.983 },
  { fpr: 0.700, tpr: 0.989 },
  { fpr: 0.800, tpr: 0.993 },
  { fpr: 0.900, tpr: 0.997 },
  { fpr: 1.000, tpr: 1.000 },
];

// ── PR curve (CatBoost holdout, AP=0.707, baseline=18.75%) ───────────────────

export const prCurveData: PrPoint[] = [
  { recall: 0.00, precision: 1.000 },
  { recall: 0.10, precision: 0.940 },
  { recall: 0.20, precision: 0.878 },
  { recall: 0.30, precision: 0.820 },
  { recall: 0.40, precision: 0.757 },
  { recall: 0.50, precision: 0.698 },
  { recall: 0.60, precision: 0.638 },
  { recall: 0.70, precision: 0.578 },
  { recall: 0.80, precision: 0.518 },
  { recall: 0.85, precision: 0.487 },
  { recall: 0.90, precision: 0.455 },
  { recall: 0.92, precision: 0.438 },
  { recall: 0.95, precision: 0.357 },
  { recall: 0.98, precision: 0.248 },
  { recall: 1.00, precision: 0.1875 },
];

// ── SHAP feature importance (mean |SHAP|, CatBoost) ──────────────────────────
// direction: "negative" = higher value → lower churn risk (protective)
//            "positive" = higher value → higher churn risk
//            "mixed"    = non-linear / contextual

export const shapValues: ShapEntry[] = [
  { feature: "REGULARITY", importance: 0.285, direction: "negative" },
  { feature: "regularity_rate", importance: 0.248, direction: "negative" },
  { feature: "MONTANT", importance: 0.182, direction: "negative" },
  { feature: "REVENUE", importance: 0.163, direction: "negative" },
  { feature: "FREQ_TOP_PACK", importance: 0.145, direction: "negative" },
  { feature: "FREQUENCE_RECH", importance: 0.138, direction: "negative" },
  { feature: "top_pack_te", importance: 0.121, direction: "mixed" },
  { feature: "is_inactive", importance: 0.115, direction: "positive" },
  { feature: "ON_NET", importance: 0.108, direction: "negative" },
  { feature: "REGION_te", importance: 0.092, direction: "mixed" },
  { feature: "DATA_VOLUME", importance: 0.082, direction: "negative" },
  { feature: "tenure_encoded", importance: 0.078, direction: "negative" },
  { feature: "recharge_per_freq", importance: 0.068, direction: "negative" },
  { feature: "has_data", importance: 0.055, direction: "negative" },
  { feature: "n_active_call_types", importance: 0.045, direction: "negative" },
];

// ── Confusion matrix (threshold=0.5, holdout n=430,810) ──────────────────────

export const confusionMatrix = {
  tp: 74557,   // predicted churn, actual churn
  fn: 6220,    // predicted retain, actual churn
  fp: 64303,   // predicted churn, actual retain
  tn: 285730,  // predicted retain, actual retain
  threshold: 0.5,
};

// ── Calibration curve (CatBoost, Brier=0.1119) ───────────────────────────────

export const calibrationData: CalibrationPoint[] = [
  { predicted: 0.05, actual: 0.040 },
  { predicted: 0.10, actual: 0.095 },
  { predicted: 0.15, actual: 0.145 },
  { predicted: 0.20, actual: 0.192 },
  { predicted: 0.25, actual: 0.238 },
  { predicted: 0.30, actual: 0.285 },
  { predicted: 0.35, actual: 0.335 },
  { predicted: 0.40, actual: 0.386 },
  { predicted: 0.45, actual: 0.437 },
  { predicted: 0.50, actual: 0.482 },
  { predicted: 0.55, actual: 0.530 },
  { predicted: 0.60, actual: 0.577 },
  { predicted: 0.65, actual: 0.628 },
  { predicted: 0.70, actual: 0.680 },
  { predicted: 0.75, actual: 0.742 },
  { predicted: 0.80, actual: 0.804 },
  { predicted: 0.85, actual: 0.860 },
  { predicted: 0.90, actual: 0.908 },
];

// ── Churn rate by region ──────────────────────────────────────────────────────

export const churnByRegion: RegionChurn[] = [
  { region: "DAKAR", rate: 22.1 },
  { region: "THIES", rate: 17.3 },
  { region: "KAOLACK", rate: 16.4 },
  { region: "SAINT-LOUIS", rate: 15.2 },
  { region: "DIOURBEL", rate: 14.6 },
  { region: "LOUGA", rate: 13.8 },
];
