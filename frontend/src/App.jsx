import { useState, useEffect, useRef } from "react";

const API_BASE = "https://medical-diagnosis-ml.onrender.com";
const FIELDS = [
  { key: "Pregnancies", label: "Pregnancies", min: 0, max: 20, step: 1, unit: "count", tip: "Number of times pregnant" },
  { key: "Glucose", label: "Plasma Glucose", min: 0, max: 300, step: 1, unit: "mg/dL", tip: "2-hr oral glucose tolerance test" },
  { key: "BloodPressure", label: "Blood Pressure", min: 0, max: 150, step: 1, unit: "mm Hg", tip: "Diastolic blood pressure" },
  { key: "SkinThickness", label: "Skin Thickness", min: 0, max: 100, step: 1, unit: "mm", tip: "Triceps skin fold thickness" },
  { key: "Insulin", label: "Insulin", min: 0, max: 900, step: 1, unit: "μU/mL", tip: "2-hour serum insulin" },
  { key: "BMI", label: "BMI", min: 0, max: 70, step: 0.1, unit: "kg/m²", tip: "Body mass index" },
  { key: "DiabetesPedigreeFunction", label: "Pedigree Function", min: 0, max: 2.5, step: 0.001, unit: "score", tip: "Diabetes family history score" },
  { key: "Age", label: "Age", min: 1, max: 110, step: 1, unit: "years", tip: "Patient age" },
];

const DEFAULTS = {
  Pregnancies: 2, Glucose: 120, BloodPressure: 70,
  SkinThickness: 25, Insulin: 80, BMI: 28.5,
  DiabetesPedigreeFunction: 0.35, Age: 35,
};

const DEMO_DIABETIC = {
  Pregnancies: 6, Glucose: 148, BloodPressure: 72,
  SkinThickness: 35, Insulin: 0, BMI: 33.6,
  DiabetesPedigreeFunction: 0.627, Age: 50,
};

function RiskMeter({ score }) {
  const radius = 72;
  const stroke = 10;
  const norm = radius - stroke / 2;
  const circ = Math.PI * norm;
  const pct = Math.min(Math.max(score, 0), 100);
  const offset = circ - (pct / 100) * circ;

  const color = pct < 30 ? "#22c55e" : pct < 55 ? "#f59e0b" : pct < 75 ? "#f97316" : "#ef4444";
  const label = pct < 30 ? "Low" : pct < 55 ? "Moderate" : pct < 75 ? "High" : "Critical";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <svg width={radius * 2 + stroke} height={radius + stroke + 8} style={{ overflow: "visible" }}>
        <path
          d={`M ${stroke / 2} ${radius + stroke / 2} A ${norm} ${norm} 0 0 1 ${radius * 2 + stroke / 2} ${radius + stroke / 2}`}
          fill="none" stroke="#e5e7eb" strokeWidth={stroke} strokeLinecap="round"
        />
        <path
          d={`M ${stroke / 2} ${radius + stroke / 2} A ${norm} ${norm} 0 0 1 ${radius * 2 + stroke / 2} ${radius + stroke / 2}`}
          fill="none" stroke={color} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(.4,0,.2,1), stroke 0.4s" }}
        />
        <text x={radius + stroke / 2} y={radius - 4} textAnchor="middle"
          style={{ fontSize: 28, fontWeight: 700, fill: color, fontFamily: "Georgia, serif", transition: "fill 0.4s" }}>
          {Math.round(pct)}%
        </text>
        <text x={radius + stroke / 2} y={radius + 18} textAnchor="middle"
          style={{ fontSize: 13, fill: "#6b7280", letterSpacing: "0.08em", textTransform: "uppercase" }}>
          {label} RISK
        </text>
      </svg>
    </div>
  );
}

function ConfusionMatrix({ cm }) {
  if (!cm) return null;
  const cells = [
    { label: "True Negative", val: cm.true_negative, color: "#dcfce7", text: "#166534", sub: "Correctly identified healthy" },
    { label: "False Positive", val: cm.false_positive, color: "#fef9c3", text: "#854d0e", sub: "Healthy flagged as diabetic" },
    { label: "False Negative", val: cm.false_negative, color: "#fee2e2", text: "#991b1b", sub: "⚠ Missed diabetic patient" },
    { label: "True Positive", val: cm.true_positive, color: "#dcfce7", text: "#166534", sub: "Correctly identified diabetic" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
      {cells.map(c => (
        <div key={c.label} style={{ background: c.color, borderRadius: 10, padding: "12px 14px" }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: c.text, fontFamily: "Georgia, serif" }}>{c.val}</div>
          <div style={{ fontSize: 12, fontWeight: 600, color: c.text }}>{c.label}</div>
          <div style={{ fontSize: 11, color: c.text, opacity: 0.75, marginTop: 2 }}>{c.sub}</div>
        </div>
      ))}
    </div>
  );
}

function MetricBadge({ label, value, unit = "%", highlight }) {
  return (
    <div style={{
      background: highlight ? "#f0fdf4" : "#f9fafb",
      border: `1px solid ${highlight ? "#86efac" : "#e5e7eb"}`,
      borderRadius: 10, padding: "10px 14px", textAlign: "center"
    }}>
      <div style={{ fontSize: 20, fontWeight: 700, color: highlight ? "#15803d" : "#111827", fontFamily: "Georgia, serif" }}>
        {value}{unit}
      </div>
      <div style={{ fontSize: 11, color: "#6b7280", marginTop: 2, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {label}
      </div>
    </div>
  );
}

export default function App() {
  const [values, setValues] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("predict");
  const [submitted, setSubmitted] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/model-info`)
      .then(r => r.json())
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  const handlePredict = async () => {
    setLoading(true); setError(null); setSubmitted(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
      setTab("result");
    } catch (e) {
      setError(e.message || "Could not connect to backend. Is Flask running?");
    } finally {
      setLoading(false);
    }
  };

  const metrics = modelInfo?.metrics;
  const cm = metrics?.confusion_matrix;
  const isDiabetic = result?.prediction === 1;

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 40%, #f0fdf4 100%)",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      padding: "0 0 48px",
    }}>
      {/* Header */}
      <div style={{
        background: "rgba(255,255,255,0.85)",
        backdropFilter: "blur(12px)",
        borderBottom: "1px solid #e0e7ef",
        padding: "20px 0",
        position: "sticky", top: 0, zIndex: 10,
      }}>
        <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 22, fontFamily: "Georgia, serif" }}>🩺</span>
              <span style={{ fontSize: 20, fontWeight: 700, color: "#0f172a", letterSpacing: "-0.02em" }}>
                DiabetiqAI
              </span>
              <span style={{ background: "#dcfce7", color: "#166534", fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 20, letterSpacing: "0.04em" }}>
                SMOTE + LR
              </span>
            </div>
            <div style={{ fontSize: 12, color: "#64748b", marginTop: 2, marginLeft: 32 }}>
              Logistic Regression · Recall-optimised · Threshold 0.35
            </div>
          </div>
          {metrics && (
            <div style={{ display: "flex", gap: 16, fontSize: 13 }}>
              <span><strong style={{ color: "#0f172a" }}>{metrics.recall}%</strong> <span style={{ color: "#64748b" }}>Recall</span></span>
              <span><strong style={{ color: "#0f172a" }}>{metrics.f1_score}%</strong> <span style={{ color: "#64748b" }}>F1</span></span>
              <span><strong style={{ color: "#0f172a" }}>{metrics.accuracy}%</strong> <span style={{ color: "#64748b" }}>Accuracy</span></span>
            </div>
          )}
        </div>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px 0" }}>
        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 28, background: "#f1f5f9", borderRadius: 12, padding: 4, width: "fit-content" }}>
          {[["predict", "🔬  Predict"], ["metrics", "📊  Model Metrics"]].map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)} style={{
              padding: "8px 20px", borderRadius: 9, border: "none", cursor: "pointer", fontSize: 14, fontWeight: 600,
              background: tab === id ? "#fff" : "transparent",
              color: tab === id ? "#0f172a" : "#64748b",
              boxShadow: tab === id ? "0 1px 4px rgba(0,0,0,0.1)" : "none",
              transition: "all 0.15s",
            }}>
              {label}
            </button>
          ))}
          {result && (
            <button onClick={() => setTab("result")} style={{
              padding: "8px 20px", borderRadius: 9, border: "none", cursor: "pointer", fontSize: 14, fontWeight: 600,
              background: tab === "result" ? "#fff" : "transparent",
              color: tab === "result" ? (isDiabetic ? "#dc2626" : "#16a34a") : "#64748b",
              boxShadow: tab === "result" ? "0 1px 4px rgba(0,0,0,0.1)" : "none",
              transition: "all 0.15s",
            }}>
              {isDiabetic ? "⚠️  Result" : "✅  Result"}
            </button>
          )}
        </div>

        {/* ─── PREDICT TAB ─── */}
        {tab === "predict" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {FIELDS.map(f => (
              <div key={f.key} style={{
                background: "#fff", borderRadius: 14, padding: "16px 18px",
                border: "1px solid #e2e8f0", boxShadow: "0 1px 3px rgba(0,0,0,0.04)"
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, color: "#374151" }}>{f.label}</label>
                  <span style={{ fontSize: 12, color: "#94a3b8" }}>{f.unit}</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <input
                    type="range" min={f.min} max={f.max} step={f.step}
                    value={values[f.key]}
                    onChange={e => setValues(v => ({ ...v, [f.key]: parseFloat(e.target.value) }))}
                    style={{ flex: 1, accentColor: "#3b82f6" }}
                  />
                  <input
                    type="number" min={f.min} max={f.max} step={f.step}
                    value={values[f.key]}
                    onChange={e => setValues(v => ({ ...v, [f.key]: parseFloat(e.target.value) || 0 }))}
                    style={{
                      width: 72, padding: "4px 8px", borderRadius: 7, border: "1px solid #d1d5db",
                      fontSize: 14, fontWeight: 600, color: "#0f172a", textAlign: "right",
                    }}
                  />
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 6 }}>{f.tip}</div>
              </div>
            ))}

            {/* Actions */}
            <div style={{ gridColumn: "1 / -1", display: "flex", gap: 12, marginTop: 4 }}>
              <button onClick={handlePredict} disabled={loading} style={{
                flex: 1, padding: "14px", borderRadius: 12, border: "none", cursor: loading ? "not-allowed" : "pointer",
                background: loading ? "#93c5fd" : "#2563eb", color: "#fff",
                fontSize: 15, fontWeight: 700, letterSpacing: "-0.01em",
                boxShadow: "0 4px 12px rgba(37,99,235,0.35)", transition: "all 0.15s",
              }}>
                {loading ? "Analysing…" : "Run Prediction →"}
              </button>
              <button onClick={() => { setValues(DEMO_DIABETIC); setResult(null); }} style={{
                padding: "14px 20px", borderRadius: 12, border: "1px solid #e2e8f0",
                background: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 600, color: "#64748b",
              }}>
                Load Demo (Diabetic)
              </button>
              <button onClick={() => { setValues(DEFAULTS); setResult(null); }} style={{
                padding: "14px 20px", borderRadius: 12, border: "1px solid #e2e8f0",
                background: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 600, color: "#64748b",
              }}>
                Reset
              </button>
            </div>

            {error && (
              <div style={{
                gridColumn: "1 / -1", background: "#fef2f2", border: "1px solid #fca5a5",
                borderRadius: 10, padding: "12px 16px", color: "#b91c1c", fontSize: 13,
              }}>
                ⚠ {error}
              </div>
            )}
          </div>
        )}

        {/* ─── RESULT TAB ─── */}
        {tab === "result" && result && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Risk Meter Card */}
            <div style={{
              background: "#fff", borderRadius: 16, padding: "28px 24px", border: "1px solid #e2e8f0",
              boxShadow: "0 2px 12px rgba(0,0,0,0.06)", display: "flex", flexDirection: "column", alignItems: "center", gap: 16,
            }}>
              <RiskMeter score={result.risk_score} />
              <div style={{
                background: isDiabetic ? "#fef2f2" : "#f0fdf4",
                border: `2px solid ${isDiabetic ? "#fca5a5" : "#86efac"}`,
                borderRadius: 12, padding: "12px 24px", textAlign: "center", width: "100%",
              }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: isDiabetic ? "#dc2626" : "#16a34a", fontFamily: "Georgia, serif" }}>
                  {result.result}
                </div>
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>
                  Threshold: {(result.threshold * 100).toFixed(0)}% · Confidence: {result.confidence}%
                </div>
              </div>
              {result.note && (
                <div style={{ background: "#fffbeb", border: "1px solid #fcd34d", borderRadius: 10, padding: "10px 14px", fontSize: 12, color: "#92400e", width: "100%" }}>
                  ℹ {result.note}
                </div>
              )}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, width: "100%" }}>
                <div style={{ background: "#f0fdf4", borderRadius: 10, padding: "10px 14px", textAlign: "center" }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: "#16a34a" }}>{result.probability.non_diabetic}%</div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>Non-Diabetic</div>
                </div>
                <div style={{ background: "#fef2f2", borderRadius: 10, padding: "10px 14px", textAlign: "center" }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: "#dc2626" }}>{result.probability.diabetic}%</div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>Diabetic</div>
                </div>
              </div>
            </div>

            {/* Input summary */}
            <div style={{
              background: "#fff", borderRadius: 16, padding: "20px 22px", border: "1px solid #e2e8f0",
              boxShadow: "0 2px 12px rgba(0,0,0,0.06)",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#374151", marginBottom: 14, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Patient Input Summary
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {FIELDS.map(f => (
                  <div key={f.key} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, padding: "5px 0", borderBottom: "1px solid #f1f5f9" }}>
                    <span style={{ color: "#64748b" }}>{f.label}</span>
                    <span style={{ fontWeight: 600, color: "#0f172a" }}>{values[f.key]} {f.unit}</span>
                  </div>
                ))}
              </div>
              <button onClick={() => setTab("predict")} style={{
                marginTop: 16, width: "100%", padding: "10px", borderRadius: 10,
                border: "1px solid #bfdbfe", background: "#eff6ff", color: "#1d4ed8",
                fontSize: 13, fontWeight: 600, cursor: "pointer",
              }}>
                ← Adjust Values
              </button>
            </div>
          </div>
        )}

        {/* ─── METRICS TAB ─── */}
        {tab === "metrics" && metrics && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Key metrics */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
              <MetricBadge label="Recall ↑" value={metrics.recall} highlight />
              <MetricBadge label="F1 Score" value={metrics.f1_score} />
              <MetricBadge label="Precision" value={metrics.precision} />
              <MetricBadge label="Accuracy" value={metrics.accuracy} />
            </div>

            {/* Confusion matrix + explanations */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div style={{ background: "#fff", borderRadius: 14, padding: "20px", border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#374151", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Confusion Matrix
                </div>
                <ConfusionMatrix cm={cm} />
                <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 12, lineHeight: 1.6 }}>
                  The False Negative cell is the <strong style={{ color: "#dc2626" }}>danger zone</strong> — 
                  diabetic patients the model missed. SMOTE + threshold tuning minimises this.
                </div>
              </div>

              <div style={{ background: "#fff", borderRadius: 14, padding: "20px", border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#374151", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Engineering Decisions
                </div>
                {[
                  { icon: "⚖️", title: "SMOTE Balancing", body: `Training set expanded from ${metrics.dataset.train_orig} → ${metrics.dataset.train_smote} samples by synthesising minority diabetic cases.` },
                  { icon: "🎯", title: "Recall-Optimised Threshold", body: `Default 0.5 threshold → lowered to ${metrics.threshold}. Model flags a patient if it's ≥${(metrics.threshold * 100).toFixed(0)}% sure of diabetes.` },
                  { icon: "📐", title: "StandardScaler", body: "All features scaled to zero-mean, unit-variance so gradient descent converges efficiently across different units (mg/dL, kg/m², years)." },
                  { icon: "🔍", title: "F1 over Accuracy", body: "In imbalanced medical data, accuracy is misleading. F1 balances Precision and Recall, giving a fairer picture of model quality." },
                ].map(d => (
                  <div key={d.title} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: "1px solid #f1f5f9" }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#0f172a", marginBottom: 4 }}>{d.icon} {d.title}</div>
                    <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.6 }}>{d.body}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Dataset info */}
            <div style={{
              background: "#fff", borderRadius: 14, padding: "16px 20px", border: "1px solid #e2e8f0",
              display: "flex", gap: 24, alignItems: "center",
            }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: "#374151" }}>Dataset:</span>
              <span style={{ fontSize: 13, color: "#64748b" }}>PIMA Indians Diabetes · {metrics.dataset.total} records</span>
              <span style={{ fontSize: 13, color: "#64748b" }}>Train: {metrics.dataset.train_smote} (post-SMOTE)</span>
              <span style={{ fontSize: 13, color: "#64748b" }}>Test: {metrics.dataset.test}</span>
              <span style={{ background: "#dcfce7", color: "#166534", fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 20 }}>
                Stratified Split
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}