from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score, precision_score
)
from imblearn.over_sampling import SMOTE
import os
import json

app = Flask(__name__)
CORS(app)  # Allow React frontend to call this API

# ─────────────────────────────────────────────
# 1. DATA ENGINEERING
# ─────────────────────────────────────────────
print("Loading and preprocessing dataset...")
data = pd.read_csv("dataset/diabetes.csv")

# Medical columns where 0 is physiologically impossible → treat as missing
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols_with_zero] = data[cols_with_zero].replace(0, pd.NA)
data = data.apply(pd.to_numeric, errors="coerce")
data.fillna(data.mean(), inplace=True)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 2. SMOTE — balance the training set only
# ─────────────────────────────────────────────
print(f"Class distribution before SMOTE: {dict(y_train.value_counts())}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTE:  {dict(pd.Series(y_train_resampled).value_counts())}")

# ─────────────────────────────────────────────
# 3. SCALING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. MODEL — Logistic Regression
# ─────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# ─────────────────────────────────────────────
# 5. THRESHOLD TUNING — prioritise Recall
#    Default threshold = 0.5 → lower to 0.35 so
#    the model flags borderline diabetic cases
# ─────────────────────────────────────────────
THRESHOLD = 0.35

y_proba  = model.predict_proba(X_test_scaled)[:, 1]
y_pred   = (y_proba >= THRESHOLD).astype(int)

# ─────────────────────────────────────────────
# 6. EVALUATION METRICS
# ─────────────────────────────────────────────
cm        = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

metrics = {
    "accuracy":        round(accuracy_score(y_test, y_pred) * 100, 2),
    "recall":          round(recall_score(y_test, y_pred)   * 100, 2),
    "precision":       round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
    "f1_score":        round(f1_score(y_test, y_pred)       * 100, 2),
    "threshold":       THRESHOLD,
    "confusion_matrix": {
        "true_negative":  int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),  # missed diabetic patients — the danger zone
        "true_positive":  int(tp),
    },
    "dataset": {
        "total":       int(len(data)),
        "train_orig":  int(len(X_train)),
        "train_smote": int(len(X_train_resampled)),
        "test":        int(len(X_test)),
    },
    "class_report": classification_report(y_test, y_pred, output_dict=True),
}

print(f"\n{'='*50}")
print(f"  Accuracy  : {metrics['accuracy']}%")
print(f"  Recall    : {metrics['recall']}%  ← minimise False Negatives")
print(f"  Precision : {metrics['precision']}%")
print(f"  F1-Score  : {metrics['f1_score']}%")
print(f"  Threshold : {THRESHOLD}")
print(f"  FN (missed diabetics): {fn}")
print(f"{'='*50}\n")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "Logistic Regression + SMOTE"})


@app.route("/model-info")
def model_info():
    return jsonify({
        "model_type": "Logistic Regression",
        "improvements": ["SMOTE oversampling", "StandardScaler", f"Threshold={THRESHOLD} (recall-optimised)"],
        "features": feature_names,
        "metrics": metrics,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json

        expected_features = [
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age",
        ]

        missing = [f for f in expected_features if f not in input_data]
        if missing:
            return jsonify({"error": "Missing features", "missing": missing}), 400

        input_df     = pd.DataFrame([{f: input_data[f] for f in expected_features}])
        input_scaled = scaler.transform(input_df)

        proba      = model.predict_proba(input_scaled)[0]
        diabetic_p = float(proba[1])
        prediction = int(diabetic_p >= THRESHOLD)

        # Risk tier for the UI risk meter
        if diabetic_p < 0.30:
            risk_level = "Low"
            risk_color = "#22c55e"
        elif diabetic_p < 0.55:
            risk_level = "Moderate"
            risk_color = "#f59e0b"
        elif diabetic_p < 0.75:
            risk_level = "High"
            risk_color = "#f97316"
        else:
            risk_level = "Critical"
            risk_color = "#ef4444"

        return jsonify({
            "prediction":  prediction,
            "result":      "Diabetic" if prediction == 1 else "Non-Diabetic",
            "probability": {
                "non_diabetic": round(float(proba[0]) * 100, 2),
                "diabetic":     round(diabetic_p       * 100, 2),
            },
            "confidence":  round(float(max(proba)) * 100, 2),
            "threshold":   THRESHOLD,
            "risk_level":  risk_level,
            "risk_color":  risk_color,
            "risk_score":  round(diabetic_p * 100, 1),
            "note": (
                "Flagged due to recall-optimised threshold. "
                "Recommend follow-up test." if prediction == 1 and diabetic_p < 0.5
                else ""
            ),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)