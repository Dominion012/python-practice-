# Day 29 — Deploying ML Models with Flask
# Week 6 Day 4

import joblib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

# ============================================================
# TOPIC 1-2: Why Deploy + HTTP Basics
# ============================================================
# A trained model sitting in a .pth file is useless until something loads
# it and serves predictions over a network. Flask listens for HTTP requests
# (POST /predict with a JSON body) and returns JSON responses — the same
# request/response pattern used by every real API, including LLM APIs.

# ============================================================
# TOPIC 4: Load Model + Scaler ONCE at Startup
# ============================================================
# Loading happens here, at module level — NOT inside the route function.
# Flask's process stays alive between requests, so this runs once total,
# not once per request; reloading from disk on every request would add
# unnecessary latency at scale.
#
# scaler.pkl holds the EXACT StandardScaler fit during Day 27 training.
# Reusing scaler.transform() (not fit_transform) ensures new passengers are
# scaled with the same mean/std the model was trained on — a fresh scaler
# fit on different data would silently produce wrong predictions, no error.

class TitanicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(6, 16)
        self.fc2     = nn.Linear(16, 8)
        self.fc3     = nn.Linear(8, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = TitanicNet()
model.load_state_dict(torch.load('titanic_model.pth', weights_only=True))
model.eval()    # disables dropout/batchnorm randomness — always do this before inference

scaler = joblib.load('scaler.pkl')

print("Model and scaler loaded successfully")

# ============================================================
# TOPIC 3: Basic Flask App
# ============================================================
app = Flask(__name__)

@app.route('/')
def home():
    return "Titanic survival predictor is running!"

# ============================================================
# TOPIC 5 & 7: Prediction Endpoint + Input Validation
# ============================================================
# POST (not GET) because we're sending data to be processed, not
# retrieving something static. Validates input first and returns clean
# 400s for client mistakes — instead of crashing with a 500 stack trace.

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({'error': 'Request body must be JSON'}), 400

    required = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        features = [float(data[f]) for f in required]
    except (ValueError, TypeError):
        return jsonify({'error': 'All fields must be numeric'}), 400

    X = np.array([features])                  # shape (1, 6) — batch dimension required
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():                      # no gradients needed — predicting, not training
        probability = model(X_tensor).item()

    return jsonify({
        'survived': probability > 0.5,
        'probability': round(probability, 4)
    })

# ============================================================
# TOPIC 6 & 8: Run the Server (test externally via test_api.py)
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5001)
