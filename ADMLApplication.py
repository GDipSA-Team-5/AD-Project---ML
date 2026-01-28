
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return {
        "service": "AD ML Application",
        "status": "running"
    }

@app.route("/health")
def health():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = np.array([[
        data["Fill_percentage"],
        data["Month"],
        data["Is_weekend"],
        data["Days_since_last_REC"]
    ]])

    prediction = model.predict(X)[0]

    return jsonify({
        "predicted_fill": float(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

# Install Docker: https://docs.docker.com/get-started/get-docker/
# Before deploying to Azure — always test the docker container locally using the following commands:
# docker build -t ad-ml-flask .
# docker run -p 8000:8000 ad-ml-flask
# curl http://localhost:8000/health
# --- Test Method 1 ---
"""
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[1, 2, 3, 4]}'
"""
# --- Test Method 2 ---
"""
Invoke-RestMethod `
  -Uri http://localhost:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"features":[1,2,3,4]}'
"""
# to stop running docker container image, use docker ps, then docker stop <container_id>