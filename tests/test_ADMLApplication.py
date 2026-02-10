import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import json
from unittest.mock import MagicMock

import ADMLApplication as flask_app


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        yield client


# Mock model + feature columns

@pytest.fixture(autouse=True)
def mock_model_and_features():
    mock_model = MagicMock()
    mock_model.predict.return_value = [5.0]  # avg daily growth

    mock_features = [
        "collection_fill_percentage",
        "cycle_start_month",
        "cycle_duration_days",
        "avg_daily_fill_growth"
    ]

    flask_app.model = mock_model
    flask_app.feature_columns = mock_features


# Health & Root Endpoints

def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "running"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.data.decode() == "OK"


# /predict Endpoint – Success

def test_predict_success(client):
    payload = {
        "container_id": 1,
        "collection_fill_percentage": 40,
        "cycle_duration_days": 10,
        "cycle_start_month": 6
    }

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 200
    data = response.get_json()

    assert "predicted_next_avg_daily_growth" in data
    assert "estimated_days_to_threshold" in data
    assert data["predicted_next_avg_daily_growth"] > 0
    assert data["estimated_days_to_threshold"] > 0


# /predict – Validation Errors

def test_predict_missing_body(client):
    response = client.post("/predict", data="{}", content_type="application/json")
    assert response.status_code == 400


def test_predict_missing_field(client):
    payload = {
        "container_id": 1,
        "cycle_duration_days": 10,
        "cycle_start_month": 6
    }

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 400
    assert "Missing field" in response.get_json()["error"]


def test_predict_invalid_cycle_duration(client):
    payload = {
        "container_id": 1,
        "collection_fill_percentage": 40,
        "cycle_duration_days": 0,
        "cycle_start_month": 6
    }

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 400
    assert "cycle_duration_days must be > 0" in response.get_json()["error"]


def test_predict_invalid_fill_percentage(client):
    payload = {
        "container_id": 1,
        "collection_fill_percentage": 120,
        "cycle_duration_days": 10,
        "cycle_start_month": 6
    }

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 400
    assert "collection_fill_percentage must be between 0 and 100" in response.get_json()["error"]


# Prediction Edge Cases

def test_predict_negative_model_output(client):
    flask_app.model.predict.return_value = [-1.0]

    payload = {
        "container_id": 1,
        "collection_fill_percentage": 40,
        "cycle_duration_days": 10,
        "cycle_start_month": 6
    }

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 500
    assert "Predicted fill growth must be > 0" in response.get_json()["error"]