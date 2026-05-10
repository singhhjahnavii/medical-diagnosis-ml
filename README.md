# DiabetiqAI — Upgraded Diabetes Prediction System

A production-grade diabetes prediction system with SMOTE balancing, recall-optimised logistic regression, and a React frontend with a live Risk Meter.

---

## What Changed From the Original

| Area | Before | After |
|---|---|---|
| Class Imbalance | Ignored | SMOTE on training set |
| Threshold | 0.5 (default) | **0.35** (recall-optimised) |
| Key Metric | Accuracy | **Recall + F1** |
| Frontend | Plain HTML | React + Risk Meter |
| CORS | Missing | `flask-cors` added |

---

## Project Structure

```
diabetes-app/
├── backend/
│   ├── app.py              # Flask API (upgraded)
│   ├── requirements.txt
│   └── dataset/
│       └── diabetes.csv    # PIMA Indians dataset
└── frontend/
    ├── src/
    │   └── App.jsx         # React frontend
    ├── package.json
    └── index.html
```

---

## Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
# → Running on http://localhost:5000
```

### Key Engineering Decisions

#### 1. SMOTE (Synthetic Minority Over-sampling Technique)
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```
Applied **only to the training set** to avoid data leakage. The test set remains the original distribution.

#### 2. Threshold Tuning (Recall Prioritisation)
```python
THRESHOLD = 0.35  # lowered from 0.5

y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred  = (y_proba >= THRESHOLD).astype(int)
```
In medical contexts, a **False Negative** (missing a diabetic patient) is far more dangerous than a False Positive (flagging a healthy person for an extra test). Lowering the threshold increases Recall at a small cost to Precision.

#### 3. Metrics Beyond Accuracy
```python
metrics = {
    "recall":    recall_score(y_test, y_pred),     # ← primary metric
    "f1_score":  f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "confusion_matrix": { "false_negative": fn, ... }
}
```

---

## API Endpoints

### `POST /predict`
```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Response:**
```json
{
  "prediction": 1,
  "result": "Diabetic",
  "probability": { "non_diabetic": 18.4, "diabetic": 81.6 },
  "confidence": 81.6,
  "threshold": 0.35,
  "risk_level": "Critical",
  "risk_score": 81.6,
  "note": ""
}
```

### `GET /model-info`
Returns full model metrics including confusion matrix, F1 score, Recall, and dataset sizes.

---

## Frontend Setup

### Option A — Vite + React (recommended)
```bash
cd frontend
npm create vite@latest . -- --template react
# Replace src/App.jsx with the provided App.jsx
npm install
npm run dev
# → http://localhost:5173
```

### Option B — Create React App
```bash
npx create-react-app frontend
cd frontend
# Replace src/App.js with App.jsx content
npm start
```

### Features
- **8-field slider form** — dual range+number inputs for all diagnostic features
- **Risk Meter** — SVG semicircle gauge, colour-coded Green → Red
- **Result panel** — prediction, probability bars, confidence
- **Model Metrics tab** — Confusion Matrix, F1/Recall/Precision, engineering decision cards
- **Demo mode** — one-click load of a known diabetic patient profile

---

## Why These Choices?

### Why SMOTE?
The PIMA dataset has ~35% positive (diabetic) cases. Without balancing, Logistic Regression learns to predict "Healthy" for most patients — achieving high accuracy while completely failing diabetics. SMOTE creates realistic synthetic diabetic samples so the model learns both classes equally.

### Why Lower the Threshold?
With threshold=0.5, a patient who scores 0.45 is sent home without follow-up. With threshold=0.35, that same patient gets flagged for a glucose test. The cost of an unnecessary test is far lower than the cost of a missed diagnosis.

### Why Logistic Regression?
1. **Interpretable** — coefficients show which features matter most
2. **Probabilistic** — `predict_proba()` gives confidence, not just 0/1
3. **Production-ready** — fast inference, no GPU needed
4. **Medical standard** — clinicians can understand and audit the model

---

## Talking Points for Interviews/Demos

- "I noticed the dataset had class imbalance, so I applied SMOTE to the training set only — keeping the test set unseen and realistic."
- "I moved the classification threshold from 0.5 to 0.35 to prioritise Recall, because in healthcare a False Negative is far costlier than a False Positive."
- "Instead of just reporting 78% accuracy, I highlight the Recall score and the Confusion Matrix — specifically the False Negative count — which is what a clinician would actually care about."
- "The React frontend shows a colour-coded Risk Meter driven by the model's `predict_proba()` output, not a hard 0/1 binary."