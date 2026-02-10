
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

def estimate_days_to_threshold(pred_fill_growth, threshold=80):
    if pred_fill_growth <= 0:
        raise ValueError("Predicted fill growth must be > 0")
    return threshold / pred_fill_growth

def predict_and_calculate(container_id, collection_fill_percentage, cycle_duration_days, 
                          cycle_start_month, model, feature_columns, threshold=80):
    
    # Calculate avg daily fill growth
    avg_daily_fill_growth = collection_fill_percentage / cycle_duration_days

    x_input = pd.DataFrame([{
        "ContainerID": container_id,
        "collection_fill_percentage": collection_fill_percentage,
        "cycle_start_month": cycle_start_month,
        "cycle_duration_days": cycle_duration_days,
        "avg_daily_fill_growth": avg_daily_fill_growth
    }])

    # One-hot encode ContainerID
    x_input = pd.get_dummies(x_input, columns=["ContainerID"])
    x_input = x_input.reindex(columns=feature_columns, fill_value=0)

    # Predict next-cycle avg growth
    pred_fill_growth = model.predict(x_input)[0]

    # Calculate number of days till threshold
    days_to_threshold = estimate_days_to_threshold(pred_fill_growth, threshold)

    return {
        "predicted_next_avg_daily_growth": float(pred_fill_growth),
        "estimated_days_to_threshold": int(days_to_threshold)
    }


@app.route("/", methods=["GET"])
def home():
    return {
        "service": "AD ML Application",
        "status": "running"
    }

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    required_fields = [
    "container_id",
    "collection_fill_percentage",
    "cycle_duration_days",
    "cycle_start_month"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    if data["cycle_duration_days"] <= 0:
        return jsonify({"error": "cycle_duration_days must be > 0"}), 400

    if not (0 <= data["collection_fill_percentage"] <= 100):
        return jsonify({"error": "collection_fill_percentage must be between 0 and 100"}), 400
    
    try:
        result = predict_and_calculate(
            container_id=data["container_id"],
            collection_fill_percentage=data["collection_fill_percentage"],
            cycle_duration_days=data["cycle_duration_days"],
            cycle_start_month=data["cycle_start_month"],
            model=model,
            feature_columns=feature_columns
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if result["predicted_next_avg_daily_growth"] <= 0:
        return jsonify({"error": "Invalid prediction result"}), 500

    return jsonify(result)

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "8000"))
    app.run(host=host, port=port)

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