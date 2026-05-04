import {
  Activity,
  AlertTriangle,
  BarChart3,
  Database,
  Gauge,
  GitBranch,
  LineChart,
  Play,
  RadioTower,
  RefreshCcw,
  Server,
  ShieldCheck,
  SlidersHorizontal
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart as ReLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  defaultSubscriber,
  featureSignals,
  modelResults,
  pipelineStages,
  thresholdProfiles
} from "./data";
import type { ApiInfo, PredictionResponse, ReadyState } from "./types";

const fallbackInfo: ApiInfo = {
  model_name: "catboost",
  cv_roc_auc_mean: 0.931568,
  cv_roc_auc_std: 0.000512,
  cv_pr_auc_mean: 0.703945,
  n_features: 44
};

const apiBase = import.meta.env.VITE_CHURN_API_URL ?? "http://localhost:8000";

const formatPct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const formatMetric = (value: number) => value.toFixed(4);

function App() {
  const [readyState, setReadyState] = useState<ReadyState>("checking");
  const [apiInfo, setApiInfo] = useState<ApiInfo>(fallbackInfo);
  const [threshold, setThreshold] = useState(0.5);
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
    f1: Number((profile.f1 * 100).toFixed(1))
  }));

  useEffect(() => {
    const controller = new AbortController();

    async function loadApiState() {
      try {
        const ready = await fetch(`${apiBase}/ready`, { signal: controller.signal });
        if (!ready.ok) {
          throw new Error("Model API is not ready");
        }

        const infoResponse = await fetch(`${apiBase}/info`, { signal: controller.signal });
        if (!infoResponse.ok) {
          throw new Error("Model info endpoint is unavailable");
        }

        const info = (await infoResponse.json()) as ApiInfo;
        setApiInfo(info);
        setReadyState("ready");
        setApiMessage(`Connected to ${apiBase}`);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setReadyState("offline");
        setApiInfo(fallbackInfo);
        setApiMessage("API offline; showing saved project metrics");
      }
    }

    loadApiState();
    return () => controller.abort();
  }, []);

  async function scoreExample() {
    setIsScoring(true);
    setPrediction(null);
    try {
      const response = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subscribers: [defaultSubscriber],
          threshold
        })
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }
      setPrediction((await response.json()) as PredictionResponse);
      setReadyState("ready");
      setApiMessage(`Scored with ${apiBase}`);
    } catch {
      setReadyState("offline");
      setApiMessage("Start the FastAPI service to score live subscribers");
    } finally {
      setIsScoring(false);
    }
  }

  const probability = prediction?.predictions[0]?.churn_probability ?? 0.3;

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
          <a className="active" href="#overview"><Gauge aria-hidden="true" />Overview</a>
          <a href="#models"><BarChart3 aria-hidden="true" />Models</a>
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

        <section id="overview" className="metric-grid" aria-label="Model overview">
          <MetricTile icon={<ShieldCheck />} label="Best model" value={apiInfo.model_name} tone="green" />
          <MetricTile icon={<LineChart />} label="CV ROC-AUC" value={formatMetric(apiInfo.cv_roc_auc_mean)} detail={`± ${formatMetric(apiInfo.cv_roc_auc_std)}`} tone="blue" />
          <MetricTile icon={<Activity />} label="CV PR-AUC" value={formatMetric(apiInfo.cv_pr_auc_mean)} detail="class imbalance aware" tone="amber" />
          <MetricTile icon={<Database />} label="Feature count" value={apiInfo.n_features.toString()} detail="engineered model inputs" tone="slate" />
        </section>

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
              <ResponsiveContainer width="100%" height={270}>
                <BarChart data={modelResults} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} />
                  <YAxis domain={[0.925, 0.933]} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(value: number) => formatMetric(value)} />
                  <Bar dataKey="rocAuc" radius={[5, 5, 0, 0]}>
                    {modelResults.map((entry) => (
                      <Cell key={entry.name} fill={entry.name === "CatBoost" ? "#167a5c" : "#537188"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
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
              <ResponsiveContainer width="100%" height={270}>
                <ReLineChart data={thresholdData} margin={{ top: 12, right: 16, left: -18, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} unit="%" />
                  <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                  <Line type="monotone" dataKey="precision" stroke="#0f6b8f" strokeWidth={3} dot />
                  <Line type="monotone" dataKey="recall" stroke="#c27b2c" strokeWidth={3} dot />
                  <Line type="monotone" dataKey="f1" stroke="#1f8a5b" strokeWidth={3} dot />
                </ReLineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

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
                <h2>Example subscriber</h2>
              </div>
              <button className="primary-button" onClick={scoreExample} disabled={isScoring}>
                <Play aria-hidden="true" />
                {isScoring ? "Scoring" : "Score"}
              </button>
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

            <div className="risk-meter" style={{ "--risk": `${probability * 100}%` } as React.CSSProperties}>
              <div className="risk-fill" />
            </div>

            <div className="prediction-summary">
              <div>
                <span>Churn probability</span>
                <strong>{formatPct(probability)}</strong>
              </div>
              <div>
                <span>Decision</span>
                <strong className={probability >= threshold ? "danger-text" : "safe-text"}>
                  {probability >= threshold ? "Retain" : "Observe"}
                </strong>
              </div>
            </div>

            <div className="subscriber-strip">
              <span>DAKAR</span>
              <span>K &gt; 24 month</span>
              <span>REGULARITY 54</span>
              <span>REVENUE 4251</span>
            </div>
          </div>
        </section>

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
