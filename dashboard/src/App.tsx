import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  Database,
  Gauge,
  GitBranch,
  LineChart,
  MapPin,
  Play,
  Plus,
  RadioTower,
  RefreshCcw,
  RotateCcw,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  Trash2,
  TrendingUp,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart as ReLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  calibrationData,
  churnByRegion,
  confusionMatrix,
  defaultSubscriber,
  featureSignals,
  modelResults,
  pipelineStages,
  prCurveData,
  rocCurveData,
  shapValues,
  thresholdProfiles,
} from "./data";
import type { ShapEntry } from "./data";
import type { ApiInfo, PredictionResponse, ReadyState, SubscriberInput } from "./types";

const fallbackInfo: ApiInfo = {
  model_name: "catboost",
  cv_roc_auc_mean: 0.931568,
  cv_roc_auc_std: 0.000512,
  cv_pr_auc_mean: 0.703945,
  n_features: 44,
};

const apiBase = import.meta.env.VITE_CHURN_API_URL ?? "http://localhost:8000";

const categoricalFields = [
  { key: "REGION", label: "Region", options: ["DAKAR", "THIES", "SAINT-LOUIS", "LOUGA", "KAOLACK", "DIOURBEL"] },
  { key: "TENURE", label: "Tenure", options: ["K > 24 month", "J 21-24 month", "I 18-21 month", "H 15-18 month", "G 12-15 month", "F 9-12 month", "E 6-9 month", "D 3-6 month"] },
  { key: "MRG", label: "MRG", options: ["NO", "YES"] },
  { key: "TOP_PACK", label: "Top pack", options: ["On net 200F=Unlimited _call24H", "Data:1000F=2GB,30d", "Data:200F=Unlimited,24H", "All-net 500F=2000F;5d", ""] },
] as const;

const numericFields = [
  ["MONTANT", "Recharge amount"],
  ["FREQUENCE_RECH", "Recharge frequency"],
  ["REVENUE", "Revenue"],
  ["ARPU_SEGMENT", "ARPU segment"],
  ["FREQUENCE", "Transactions"],
  ["DATA_VOLUME", "Data volume"],
  ["ON_NET", "On-net calls"],
  ["ORANGE", "Orange calls"],
  ["TIGO", "Tigo calls"],
  ["ZONE1", "Zone 1 calls"],
  ["ZONE2", "Zone 2 calls"],
  ["REGULARITY", "Regularity"],
  ["FREQ_TOP_PACK", "Top pack frequency"],
] as const;

const formatPct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const formatMetric = (value: number) => value.toFixed(4);

const emptyToNull = (value: string) => {
  if (value.trim() === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const createEditableSubscriber = (): SubscriberInput => ({
  REGION: defaultSubscriber.REGION,
  TENURE: defaultSubscriber.TENURE,
  MRG: defaultSubscriber.MRG,
  TOP_PACK: defaultSubscriber.TOP_PACK,
  MONTANT: String(defaultSubscriber.MONTANT),
  FREQUENCE_RECH: String(defaultSubscriber.FREQUENCE_RECH),
  REVENUE: String(defaultSubscriber.REVENUE),
  ARPU_SEGMENT: String(defaultSubscriber.ARPU_SEGMENT),
  FREQUENCE: String(defaultSubscriber.FREQUENCE),
  DATA_VOLUME: String(defaultSubscriber.DATA_VOLUME),
  ON_NET: String(defaultSubscriber.ON_NET),
  ORANGE: String(defaultSubscriber.ORANGE),
  TIGO: String(defaultSubscriber.TIGO),
  ZONE1: String(defaultSubscriber.ZONE1),
  ZONE2: String(defaultSubscriber.ZONE2),
  REGULARITY: String(defaultSubscriber.REGULARITY),
  FREQ_TOP_PACK: String(defaultSubscriber.FREQ_TOP_PACK),
});

const toApiSubscriber = (subscriber: SubscriberInput) => ({
  REGION: subscriber.REGION || null,
  TENURE: subscriber.TENURE || null,
  MRG: subscriber.MRG || null,
  TOP_PACK: subscriber.TOP_PACK || null,
  MONTANT: emptyToNull(subscriber.MONTANT),
  FREQUENCE_RECH: emptyToNull(subscriber.FREQUENCE_RECH),
  REVENUE: emptyToNull(subscriber.REVENUE),
  ARPU_SEGMENT: emptyToNull(subscriber.ARPU_SEGMENT),
  FREQUENCE: emptyToNull(subscriber.FREQUENCE),
  DATA_VOLUME: emptyToNull(subscriber.DATA_VOLUME),
  ON_NET: emptyToNull(subscriber.ON_NET),
  ORANGE: emptyToNull(subscriber.ORANGE),
  TIGO: emptyToNull(subscriber.TIGO),
  ZONE1: emptyToNull(subscriber.ZONE1),
  ZONE2: emptyToNull(subscriber.ZONE2),
  REGULARITY: emptyToNull(subscriber.REGULARITY),
  FREQ_TOP_PACK: emptyToNull(subscriber.FREQ_TOP_PACK),
});

const shapColor = (direction: ShapEntry["direction"]) => {
  if (direction === "positive") return "#bd4b3b";
  if (direction === "mixed") return "#537188";
  return "#167a5c";
};

function App() {
  const [readyState, setReadyState] = useState<ReadyState>("checking");
  const [apiInfo, setApiInfo] = useState<ApiInfo>(fallbackInfo);
  const [threshold, setThreshold] = useState(0.5);
  const [subscriber, setSubscriber] = useState<SubscriberInput>(() => createEditableSubscriber());
  const [batch, setBatch] = useState<SubscriberInput[]>([]);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isScoring, setIsScoring] = useState(false);
  const [apiMessage, setApiMessage] = useState("Using local training metrics");

  const bestModel = useMemo(
    () => modelResults.reduce((best, item) => (item.rocAuc > best.rocAuc ? item : best)),
    []
  );

  const thresholdData = thresholdProfiles.map((profile) => ({
    name: profile.name,
    precision: Number((profile.precision * 100).toFixed(1)),
    recall: Number((profile.recall * 100).toFixed(1)),
    f1: Number((profile.f1 * 100).toFixed(1)),
  }));

  const { tp, fn, fp, tn } = confusionMatrix;
  const actualPos = tp + fn;
  const actualNeg = fp + tn;
  const totalSubscribers = tp + fn + fp + tn;
  const churnRate = actualPos / totalSubscribers;
  const brierScore = modelResults.find((m) => m.name === "CatBoost")!.brier;
  const brierBaseline = churnRate * (1 - churnRate);
  const defaultProfile = thresholdProfiles.find((p) => p.name === "Default")!;
  const f1OptProfile = thresholdProfiles.find((p) => p.name === "F1 optimal")!;
  const topRegion = churnByRegion.reduce((max, r) => (r.rate > max.rate ? r : max));

  const rocData = rocCurveData.map((d) => ({ ...d, random: d.fpr }));
  const prData = prCurveData.map((d) => ({ ...d, baseline: churnRate }));
  const calibData = calibrationData.map((d) => ({ ...d, perfect: d.predicted }));

  const scoreRows = batch.length ? batch : [subscriber];
  const latestProbability = prediction?.predictions[0]?.churn_probability ?? 0.3;

  useEffect(() => {
    const controller = new AbortController();

    async function loadApiState() {
      try {
        const ready = await fetch(`${apiBase}/ready`, { signal: controller.signal });
        if (!ready.ok) throw new Error("Model API is not ready");

        const infoResponse = await fetch(`${apiBase}/info`, { signal: controller.signal });
        if (!infoResponse.ok) throw new Error("Model info endpoint is unavailable");

        const info = (await infoResponse.json()) as ApiInfo;
        setApiInfo(info);
        setReadyState("ready");
        setApiMessage(`Connected to ${apiBase}`);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setReadyState("offline");
        setApiInfo(fallbackInfo);
        setApiMessage("API offline; showing saved project metrics");
      }
    }

    loadApiState();
    return () => controller.abort();
  }, []);

  const updateField = (key: keyof SubscriberInput, value: string) => {
    setSubscriber((current) => ({ ...current, [key]: value }));
  };

  const addToBatch = () => {
    setBatch((current) => [...current, { ...subscriber }]);
    setPrediction(null);
  };

  const removeFromBatch = (index: number) => {
    setBatch((current) => current.filter((_, i) => i !== index));
  };

  const clearBatch = () => {
    setBatch([]);
    setPrediction(null);
  };

  const resetForm = () => {
    setSubscriber(createEditableSubscriber());
    setPrediction(null);
  };

  async function scoreSubscribers(rows: SubscriberInput[]) {
    setIsScoring(true);
    setPrediction(null);
    try {
      const response = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subscribers: rows.map(toApiSubscriber), threshold }),
      });
      if (!response.ok) throw new Error("Prediction request failed");
      setPrediction((await response.json()) as PredictionResponse);
      setReadyState("ready");
      setApiMessage(`Scored ${rows.length} subscriber${rows.length === 1 ? "" : "s"}`);
    } catch {
      setReadyState("offline");
      setApiMessage("Start the FastAPI service to score live subscribers");
    } finally {
      setIsScoring(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Dashboard navigation">
        <div className="brand">
          <RadioTower aria-hidden="true" />
          <div>
            <span>Expresso</span>
            <strong>Churn Ops</strong>
          </div>
        </div>

        <nav className="nav-list">
          <a href="#overview"><Gauge aria-hidden="true" />Overview</a>
          <a href="#models"><BarChart3 aria-hidden="true" />Models</a>
          <a href="#analytics"><TrendingUp aria-hidden="true" />ROC / PR</a>
          <a href="#shap"><BrainCircuit aria-hidden="true" />SHAP</a>
          <a href="#diagnosis"><Target aria-hidden="true" />Diagnosis</a>
          <a href="#geography"><MapPin aria-hidden="true" />Geography</a>
          <a href="#features"><Database aria-hidden="true" />Features</a>
          <a href="#prediction"><SlidersHorizontal aria-hidden="true" />Prediction</a>
          <a href="#pipeline"><GitBranch aria-hidden="true" />Pipeline</a>
        </nav>

        <div className={`api-status ${readyState}`}>
          <span className="pulse" />
          <div>
            <strong>{readyState === "ready" ? "API connected" : "API fallback"}</strong>
            <small>{apiMessage}</small>
          </div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">MLOps monitoring</p>
            <h1>Prepaid subscriber churn dashboard</h1>
          </div>
          <div className="topbar-actions">
            <button className="icon-button" title="Refresh API state" onClick={() => location.reload()}>
              <RefreshCcw aria-hidden="true" />
            </button>
            <a className="docs-button" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
              <Server aria-hidden="true" />
              API docs
            </a>
          </div>
        </header>

        {/* ── Overview KPIs ── */}
        <section id="overview" className="metric-grid" aria-label="Model overview">
          <MetricTile icon={<ShieldCheck />} label="Best model" value={apiInfo.model_name} tone="green" />
          <MetricTile icon={<LineChart />} label="CV ROC-AUC" value={formatMetric(apiInfo.cv_roc_auc_mean)} detail={`± ${formatMetric(apiInfo.cv_roc_auc_std)}`} tone="blue" />
          <MetricTile icon={<Activity />} label="CV PR-AUC" value={formatMetric(apiInfo.cv_pr_auc_mean)} detail="class imbalance aware" tone="amber" />
          <MetricTile icon={<Database />} label="Feature count" value={apiInfo.n_features.toString()} detail="engineered model inputs" tone="slate" />
        </section>

        {/* ── Holdout KPIs ── */}
        <section className="metric-grid holdout-row" aria-label="Holdout metrics">
          <MetricTile icon={<Target />} label="Brier score" value={brierScore.toFixed(4)} detail={`lower is better, baseline ${brierBaseline.toFixed(3)}`} tone="blue" />
          <MetricTile icon={<Activity />} label="Churn rate" value={`${(churnRate * 100).toFixed(2)}%`} detail={`${totalSubscribers.toLocaleString()} subscribers holdout`} tone="amber" />
          <MetricTile icon={<TrendingUp />} label="Recall @ 0.5" value={`${(defaultProfile.recall * 100).toFixed(1)}%`} detail="catches 9 in 10 churners" tone="green" />
          <MetricTile icon={<ShieldCheck />} label="Precision @ 0.5" value={`${(defaultProfile.precision * 100).toFixed(1)}%`} detail={`F1-optimal threshold: ${f1OptProfile.threshold.toFixed(3)}`} tone="slate" />
        </section>

        {/* ── Model leaderboard + Threshold profiles ── */}
        <section className="main-grid">
          <div id="models" className="panel model-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Candidate comparison</p>
                <h2>ROC-AUC leaderboard</h2>
              </div>
              <span className="tag">Winner: {bestModel.name}</span>
            </div>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={modelResults} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 11 }} />
                  <YAxis domain={[0.925, 0.933]} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(value: number) => formatMetric(value)} />
                  <Bar dataKey="rocAuc" name="ROC-AUC" radius={[5, 5, 0, 0]}>
                    {modelResults.map((entry) => (
                      <Cell key={entry.name} fill={entry.name === "CatBoost" ? "#167a5c" : "#537188"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <table className="model-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>ROC-AUC</th>
                  <th>PR-AUC</th>
                  <th>Brier</th>
                </tr>
              </thead>
              <tbody>
                {modelResults.map((m) => (
                  <tr key={m.name} className={m.name === "CatBoost" ? "winner-row" : ""}>
                    <td>{m.name}</td>
                    <td>{m.rocAuc.toFixed(4)}</td>
                    <td>{m.prAuc.toFixed(4)}</td>
                    <td>{m.brier.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel threshold-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Operating thresholds</p>
                <h2>Precision and recall tradeoff</h2>
              </div>
              <span className="tag neutral">Holdout</span>
            </div>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height={200}>
                <ReLineChart data={thresholdData} margin={{ top: 12, right: 16, left: -18, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} unit="%" />
                  <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                  <Line type="monotone" dataKey="precision" stroke="#0f6b8f" strokeWidth={3} dot name="Precision" />
                  <Line type="monotone" dataKey="recall" stroke="#c27b2c" strokeWidth={3} dot name="Recall" />
                  <Line type="monotone" dataKey="f1" stroke="#1f8a5b" strokeWidth={3} dot name="F1" />
                  <Legend />
                </ReLineChart>
              </ResponsiveContainer>
            </div>
            <div className="threshold-table">
              {thresholdProfiles.map((p) => (
                <div key={p.name} className="threshold-row">
                  <strong>{p.name}</strong>
                  <span>t={p.threshold.toFixed(3)}</span>
                  <span className="metric-chip blue">P {(p.precision * 100).toFixed(1)}%</span>
                  <span className="metric-chip amber">R {(p.recall * 100).toFixed(1)}%</span>
                  <span className="metric-chip green">F1 {(p.f1 * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── ROC + PR curves ── */}
        <section id="analytics" className="main-grid">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Discrimination</p>
                <h2>ROC curve — AUC 0.9330</h2>
              </div>
              <span className="tag neutral">Holdout</span>
            </div>
            <p className="panel-description">
              Area under the curve measures rank-ordering ability across all thresholds. A random classifier scores 0.5 (dashed line).
            </p>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height={260}>
                <ReLineChart data={rocData} margin={{ top: 8, right: 16, left: -8, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tickLine={false}
                    axisLine={false}
                    label={{ value: "False Positive Rate", position: "insideBottom", offset: -12, fill: "#65736f", fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tickLine={false}
                    axisLine={false}
                    label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", offset: 14, fill: "#65736f", fontSize: 11 }}
                  />
                  <Tooltip
                    formatter={(value: number, name: string) => [value.toFixed(3), name]}
                    labelFormatter={(label: number) => `FPR: ${Number(label).toFixed(3)}`}
                  />
                  <Line type="monotone" dataKey="tpr" stroke="#167a5c" strokeWidth={2.5} dot={false} name="CatBoost (AUC=0.933)" />
                  <Line type="monotone" dataKey="random" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="6 4" dot={false} name="Random (AUC=0.5)" />
                  <Legend verticalAlign="top" height={28} />
                </ReLineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Class imbalance aware</p>
                <h2>PR curve — AP 0.7071</h2>
              </div>
              <span className="tag neutral">Holdout</span>
            </div>
            <p className="panel-description">
              Precision-Recall area (average precision) is preferred over ROC when positives are rare. Dashed baseline is the churn prevalence ({`${(churnRate * 100).toFixed(2)}%`}).
            </p>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height={260}>
                <ReLineChart data={prData} margin={{ top: 8, right: 16, left: -8, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="recall"
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tickLine={false}
                    axisLine={false}
                    label={{ value: "Recall", position: "insideBottom", offset: -12, fill: "#65736f", fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    domain={[0, 1.05]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tickLine={false}
                    axisLine={false}
                    label={{ value: "Precision", angle: -90, position: "insideLeft", offset: 14, fill: "#65736f", fontSize: 11 }}
                  />
                  <Tooltip
                    formatter={(value: number, name: string) => [value.toFixed(3), name]}
                    labelFormatter={(label: number) => `Recall: ${Number(label).toFixed(3)}`}
                  />
                  <Line type="monotone" dataKey="precision" stroke="#0f6b8f" strokeWidth={2.5} dot={false} name="CatBoost (AP=0.707)" />
                  <Line type="monotone" dataKey="baseline" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="6 4" dot={false} name={`Baseline (${(churnRate * 100).toFixed(2)}%)`} />
                  <Legend verticalAlign="top" height={28} />
                </ReLineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        {/* ── SHAP feature importance ── */}
        <section id="shap" className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Explainability — CatBoost TreeExplainer</p>
              <h2>SHAP feature importance (mean |SHAP value|)</h2>
            </div>
            <div className="shap-legend">
              <span className="shap-badge protective">Protective</span>
              <span className="shap-badge risk">Risk driver</span>
              <span className="shap-badge mixed">Contextual</span>
            </div>
          </div>
          <p className="panel-description">
            Mean absolute SHAP values measure the average magnitude of each feature's contribution to the model output. Teal bars indicate features that reduce churn risk at higher values; red bars increase it.
          </p>
          <div className="shap-chart-wrap">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                layout="vertical"
                data={shapValues}
                margin={{ top: 4, right: 48, left: 8, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis
                  type="number"
                  tickFormatter={(v: number) => v.toFixed(2)}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: "Mean |SHAP value|", position: "insideBottom", offset: -2, fill: "#65736f", fontSize: 11 }}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={148}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 12, fill: "#17202a" }}
                />
                <Tooltip
                  formatter={(value: number) => [`${value.toFixed(4)} mean |SHAP|`, "Importance"]}
                  cursor={{ fill: "rgba(22,122,92,0.07)" }}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {shapValues.map((entry) => (
                    <Cell key={entry.feature} fill={shapColor(entry.direction)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* ── Confusion matrix + Calibration ── */}
        <section id="diagnosis" className="main-grid">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">{`Decision quality at threshold ${confusionMatrix.threshold.toFixed(1)}`}</p>
                <h2>Confusion matrix</h2>
              </div>
              <span className="tag neutral">{`n = ${totalSubscribers.toLocaleString()}`}</span>
            </div>
            <p className="panel-description">
              How the model's binary predictions compare to ground-truth churn labels on the holdout set.
            </p>
            <div className="conf-axis-labels">
              <div className="conf-axis-x">← Predicted →</div>
              <div className="conf-axis-col">
                <span>Churn</span>
                <span>Retain</span>
              </div>
            </div>
            <div className="conf-matrix">
              <div className="conf-matrix-inner">
                <div className="conf-row-label">Actual churn</div>
                <div className="conf-cell tp">
                  <span className="conf-name">True Positive</span>
                  <strong>{tp.toLocaleString()}</strong>
                  <small>{((tp / actualPos) * 100).toFixed(1)}% of churners caught</small>
                </div>
                <div className="conf-cell fn">
                  <span className="conf-name">False Negative</span>
                  <strong>{fn.toLocaleString()}</strong>
                  <small>{((fn / actualPos) * 100).toFixed(1)}% of churners missed</small>
                </div>
              </div>
              <div className="conf-matrix-inner">
                <div className="conf-row-label">Actual retain</div>
                <div className="conf-cell fp">
                  <span className="conf-name">False Positive</span>
                  <strong>{fp.toLocaleString()}</strong>
                  <small>{((fp / actualNeg) * 100).toFixed(1)}% of retainers flagged</small>
                </div>
                <div className="conf-cell tn">
                  <span className="conf-name">True Negative</span>
                  <strong>{tn.toLocaleString()}</strong>
                  <small>{((tn / actualNeg) * 100).toFixed(1)}% of retainers correct</small>
                </div>
              </div>
            </div>
            <div className="conf-metrics">
              <div><span>Precision</span><strong>{((tp / (tp + fp)) * 100).toFixed(1)}%</strong></div>
              <div><span>Recall</span><strong>{((tp / actualPos) * 100).toFixed(1)}%</strong></div>
              <div><span>Specificity</span><strong>{((tn / actualNeg) * 100).toFixed(1)}%</strong></div>
              <div><span>F1 Score</span><strong>{`${(defaultProfile.f1 * 100).toFixed(1)}%`}</strong></div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Probability reliability</p>
                <h2>{`Calibration curve — Brier ${brierScore.toFixed(4)}`}</h2>
              </div>
              <span className="tag neutral">Holdout</span>
            </div>
            <p className="panel-description">
              A well-calibrated model's predicted probabilities match observed churn rates. Points near the diagonal indicate reliable probability estimates.
            </p>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height={280}>
                <ReLineChart data={calibData} margin={{ top: 8, right: 16, left: -8, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="predicted"
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tickLine={false}
                    axisLine={false}
                    label={{ value: "Mean predicted probability", position: "insideBottom", offset: -12, fill: "#65736f", fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tickLine={false}
                    axisLine={false}
                    label={{ value: "Fraction of positives", angle: -90, position: "insideLeft", offset: 14, fill: "#65736f", fontSize: 11 }}
                  />
                  <Tooltip
                    formatter={(value: number, name: string) => [value.toFixed(3), name]}
                    labelFormatter={(label: number) => `Predicted: ${Number(label).toFixed(2)}`}
                  />
                  <Line type="monotone" dataKey="actual" stroke="#167a5c" strokeWidth={2.5} dot={{ r: 4 }} name="CatBoost" />
                  <Line type="monotone" dataKey="perfect" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="6 4" dot={false} name="Perfect calibration" />
                  <Legend verticalAlign="top" height={28} />
                </ReLineChart>
              </ResponsiveContainer>
            </div>
            <div className="calib-note">
              {`Brier score measures mean squared error of probability predictions. Lower is better; random guessing at the ${(churnRate * 100).toFixed(2)}% churn rate gives ${brierBaseline.toFixed(3)}.`}
            </div>
          </div>
        </section>

        {/* ── Geography ── */}
        <section id="geography" className="split-grid">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Regional variation</p>
                <h2>Churn rate by region</h2>
              </div>
              <span className="tag neutral">Train set</span>
            </div>
            <p className="panel-description">
              {`${topRegion.region} shows the highest churn rate (${topRegion.rate.toFixed(1)}%), likely due to greater competition and higher subscriber mobility in the capital.`}
            </p>
            <div className="chart-frame">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={churnByRegion} margin={{ top: 8, right: 16, left: -18, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="region" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} unit="%" domain={[0, 28]} />
                  <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                  <Bar dataKey="rate" name="Churn rate" radius={[5, 5, 0, 0]}>
                    {churnByRegion.map((entry) => (
                      <Cell key={entry.region} fill={entry.rate > churnRate * 100 ? "#bd4b3b" : "#167a5c"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="geo-note">
              <span className="geo-dot above" />
              <small>{`Above overall avg (${(churnRate * 100).toFixed(2)}%)`}</small>
              <span className="geo-dot below" />
              <small>Below overall avg</small>
            </div>
          </div>

          <div className="panel dataset-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Expresso dataset</p>
                <h2>Dataset profile</h2>
              </div>
            </div>
            <p className="panel-description">
              {`Expresso Telecom — Senegal's mobile subscriber base. Class imbalance (~${((1 - churnRate) * 100).toFixed(0)}% retained vs ~${(churnRate * 100).toFixed(0)}% churned) is handled via scale_pos_weight in tree models and James-Stein encoding for categorical features.`}
            </p>
            <div className="dataset-stats">
              <div className="stat-row">
                <span>Total subscribers</span>
                <strong>2,154,050</strong>
              </div>
              <div className="stat-row">
                <span>Training set (80%)</span>
                <strong>1,723,240</strong>
              </div>
              <div className="stat-row">
                <span>Holdout set (20%)</span>
                <strong>{totalSubscribers.toLocaleString()}</strong>
              </div>
              <div className="stat-row">
                <span>Positive class (churn)</span>
                <strong className="danger-text">{`${(churnRate * 100).toFixed(2)}%`}</strong>
              </div>
              <div className="stat-row">
                <span>Engineered features</span>
                <strong>44</strong>
              </div>
              <div className="stat-row">
                <span>Raw input features</span>
                <strong>17</strong>
              </div>
            </div>
            <div className="class-bar-wrap">
              <div className="class-bar-label">
                <span>Retained</span>
                <strong>{`${((1 - churnRate) * 100).toFixed(2)}%`}</strong>
              </div>
              <div className="class-bar">
                <div className="class-bar-fill retained" style={{ width: `${(1 - churnRate) * 100}%` }} />
                <div className="class-bar-fill churned" style={{ width: `${churnRate * 100}%` }} />
              </div>
              <div className="class-bar-label">
                <span>Churned</span>
                <strong className="danger-text">{`${(churnRate * 100).toFixed(2)}%`}</strong>
              </div>
            </div>
          </div>
        </section>

        {/* ── Features + Prediction ── */}
        <section className="split-grid">
          <div id="features" className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Feature system</p>
                <h2>High-signal engineered inputs</h2>
              </div>
            </div>
            <div className="feature-list">
              {featureSignals.map((feature) => (
                <div className="feature-row" key={feature.name}>
                  <span className={`status-dot ${feature.status}`} />
                  <div>
                    <strong>{feature.name}</strong>
                    <small>{feature.group} · {feature.detail}</small>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div id="prediction" className="panel prediction-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Live scoring</p>
                <h2>Subscriber prediction workspace</h2>
              </div>
              <span className="tag neutral">{scoreRows.length} ready to score</span>
            </div>

            <label className="slider-label" htmlFor="threshold">
              <span>Decision threshold</span>
              <strong>{threshold.toFixed(2)}</strong>
            </label>
            <input
              id="threshold"
              type="range"
              min="0.1"
              max="0.9"
              step="0.01"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />

            <div className="input-grid">
              {categoricalFields.map((field) => (
                <label className="field-control" key={field.key}>
                  <span>{field.label}</span>
                  <select
                    value={subscriber[field.key]}
                    onChange={(event) => updateField(field.key, event.target.value)}
                  >
                    {field.options.map((option) => (
                      <option key={option || "empty"} value={option}>
                        {option || "None"}
                      </option>
                    ))}
                  </select>
                </label>
              ))}

              {numericFields.map(([key, label]) => (
                <label className="field-control" key={key}>
                  <span>{label}</span>
                  <input
                    type="number"
                    min={key === "REGULARITY" ? 0 : undefined}
                    max={key === "REGULARITY" ? 90 : undefined}
                    value={subscriber[key]}
                    onChange={(event) => updateField(key, event.target.value)}
                  />
                </label>
              ))}
            </div>

            <div className="prediction-actions">
              <button className="primary-button" onClick={() => scoreSubscribers([subscriber])} disabled={isScoring}>
                <Play aria-hidden="true" />
                {isScoring ? "Scoring" : "Score current"}
              </button>
              <button className="secondary-button" onClick={addToBatch}>
                <Plus aria-hidden="true" />
                Add to batch
              </button>
              <button className="secondary-button" onClick={() => scoreSubscribers(scoreRows)} disabled={isScoring}>
                <Play aria-hidden="true" />
                Score batch
              </button>
              <button className="icon-button" title="Reset input form" onClick={resetForm}>
                <RotateCcw aria-hidden="true" />
              </button>
            </div>

            <div className="risk-meter" style={{ "--risk": `${latestProbability * 100}%` } as React.CSSProperties}>
              <div className="risk-fill" />
            </div>

            <div className="prediction-summary">
              <div>
                <span>Latest probability</span>
                <strong>{formatPct(latestProbability)}</strong>
              </div>
              <div>
                <span>Latest decision</span>
                <strong className={latestProbability >= threshold ? "danger-text" : "safe-text"}>
                  {latestProbability >= threshold ? "Retain" : "Observe"}
                </strong>
              </div>
            </div>

            <BatchQueue batch={batch} onRemove={removeFromBatch} onClear={clearBatch} />
            <PredictionResults response={prediction} rows={scoreRows} />
          </div>
        </section>

        {/* ── Pipeline ── */}
        <section id="pipeline" className="panel pipeline-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Reproducible pipeline</p>
              <h2>DVC stage flow</h2>
            </div>
            <span className="tag warning"><AlertTriangle aria-hidden="true" />Rerun after feature code changes</span>
          </div>
          <div className="stage-track">
            {pipelineStages.map((stage, index) => (
              <div className="stage" key={stage}>
                <span>{index + 1}</span>
                <strong>{stage}</strong>
                {index < pipelineStages.length - 1 && <i />}
              </div>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

type BatchQueueProps = {
  batch: SubscriberInput[];
  onRemove: (index: number) => void;
  onClear: () => void;
};

function BatchQueue({ batch, onRemove, onClear }: BatchQueueProps) {
  if (!batch.length) {
    return (
      <div className="empty-state">
        Add subscribers to create a batch, or score the current form directly.
      </div>
    );
  }

  return (
    <div className="batch-box">
      <div className="batch-heading">
        <strong>Batch queue</strong>
        <button className="text-button" onClick={onClear}>Clear</button>
      </div>
      <div className="batch-list">
        {batch.map((item, index) => (
          <div className="batch-row" key={`${item.REGION}-${item.REGULARITY}-${index}`}>
            <div>
              <strong>Subscriber {index + 1}</strong>
              <small>{item.REGION || "Unknown"} · REGULARITY {item.REGULARITY || "NA"} · REVENUE {item.REVENUE || "NA"}</small>
            </div>
            <button className="icon-button compact" title="Remove subscriber" onClick={() => onRemove(index)}>
              <Trash2 aria-hidden="true" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

type PredictionResultsProps = {
  response: PredictionResponse | null;
  rows: SubscriberInput[];
};

function PredictionResults({ response, rows }: PredictionResultsProps) {
  if (!response) return null;

  return (
    <div className="results-table-wrap">
      <div className="batch-heading">
        <strong>Prediction results</strong>
        <span>{response.n_subscribers} scored</span>
      </div>
      <table className="results-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Region</th>
            <th>Regularity</th>
            <th>Revenue</th>
            <th>Probability</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {response.predictions.map((item, index) => (
            <tr key={`${item.churn_probability}-${index}`}>
              <td>{index + 1}</td>
              <td>{rows[index]?.REGION || "NA"}</td>
              <td>{rows[index]?.REGULARITY || "NA"}</td>
              <td>{rows[index]?.REVENUE || "NA"}</td>
              <td>{formatPct(item.churn_probability, 2)}</td>
              <td>
                <span className={item.churn_prediction ? "decision-pill retain" : "decision-pill observe"}>
                  {item.churn_prediction ? "Retain" : "Observe"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type MetricTileProps = {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail?: string;
  tone: "green" | "blue" | "amber" | "slate";
};

function MetricTile({ icon, label, value, detail, tone }: MetricTileProps) {
  return (
    <div className={`metric-tile ${tone}`}>
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

export default App;
