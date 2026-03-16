# unit tests for API app
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app


def sample_payload():
    return {
        "patient_id": "TEST123",
        "age": 45,
        "gender": "F",
        "lead_time_days": 14,
        "appointment_day_of_week": 1,
        "appointment_month": 5,
        "scheduled_same_day": 0,
        "patient_total_appointments": 5,
        "patient_no_show_rate": 0.2,
        "previous_no_show": 1,
        "days_since_last_appointment": 30,
        "hypertension": 1,
        "diabetes": 0,
        "alcoholism": 0,
        "has_handicap": 0,
        "chronic_condition_count": 1,
        "has_chronic_condition": 1,
        "scholarship": 0,
        "sms_received": 1,
        "social_risk_score": 15.5,
        "neighbourhood_no_show_rate": 0.25,
        "neighbourhood_encoded": 42,
        "is_monday": 0,
        "is_friday": 0,
        "is_weekend": 0,
        "age_lead_time_interaction": 630.0,
        "sms_with_history": 0.2
    }


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model_info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert data["model_type"] == "LightGBM Classifier"
        assert "features_count" in data
        assert data["features_count"] > 0


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_payload())
        assert response.status_code == 200
        data = response.json()
        assert "no_show_probability" in data
        assert "risk_tier" in data
        assert "recommended_intervention" in data
        assert 0 <= data["no_show_probability"] <= 1
        assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_predict_invalid_age():
    payload = sample_payload()
    payload["age"] = 150
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_predict_invalid_gender():
    payload = sample_payload()
    payload["gender"] = "X"
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
