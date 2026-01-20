from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

app = Flask(__name__)

# Load and train model on startup
print("Loading and training model...")

# Load dataset
data = pd.read_csv("dataset/diabetes.csv")

# Replace 0 with NaN in medical columns
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols_with_zero] = data[cols_with_zero].replace(0, pd.NA)

# Fill missing values with column mean
data = data.apply(pd.to_numeric, errors="coerce")
data.fillna(data.mean(), inplace=True)

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Calculate accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model trained! Accuracy: {accuracy}")

# HTML template for home page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction API</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .info { background: #f0f0f0; padding: 15px; border-radius: 5px; }
        code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>🏥 Diabetes Prediction API</h1>
    <div class="info">
        <p><strong>Model Accuracy:</strong> {{ accuracy }}%</p>
        <p><strong>Status:</strong> Running ✅</p>
    </div>
    
    <h2>API Endpoints:</h2>
    <ul>
        <li><code>GET /</code> - This page</li>
        <li><code>GET /health</code> - Health check</li>
        <li><code>POST /predict</code> - Make prediction</li>
        <li><code>GET /model-info</code> - Model details</li>
    </ul>
    
    <h2>Example Prediction Request:</h2>
    <pre>
POST /predict
Content-Type: application/json

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
    </pre>
    
    <h2>Test It:</h2>
    <p>Use curl or Postman to test the API:</p>
    <pre>curl -X POST {{ url }}/predict -H "Content-Type: application/json" -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'</pre>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, 
                                 accuracy=round(accuracy * 100, 2),
                                 url=request.url_root.rstrip('/'))

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'Logistic Regression',
        'accuracy': round(accuracy * 100, 2)
    })

@app.route('/model-info')
def model_info():
    return jsonify({
        'model_type': 'Logistic Regression',
        'accuracy': round(accuracy * 100, 2),
        'features': list(X.columns),
        'dataset_size': len(data),
        'train_size': len(X_train),
        'test_size': len(X_test)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        input_data = request.json
        
        # Expected features
        expected_features = ["Pregnancies", "Glucose", "BloodPressure", 
                           "SkinThickness", "Insulin", "BMI", 
                           "DiabetesPedigreeFunction", "Age"]
        
        # Validate input
        if not all(feature in input_data for feature in expected_features):
            return jsonify({
                'error': 'Missing required features',
                'required': expected_features
            }), 400
        
        # Create DataFrame with input
        input_df = pd.DataFrame([input_data])
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Return result
        return jsonify({
            'prediction': int(prediction),
            'result': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': {
                'non_diabetic': round(float(probability[0]) * 100, 2),
                'diabetic': round(float(probability[1]) * 100, 2)
            },
            'confidence': round(float(max(probability)) * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


