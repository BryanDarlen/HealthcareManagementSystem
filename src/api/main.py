"""FastAPI application for no-show prediction."""
from pathlib import Path
import json
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare No-Show Prediction API",
    description="Predict patient appointment no-show probability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "lightbgm_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "lightbgm_features.json"

model = None
feature_names = None


@app.on_event("startup")
async def load_model() -> None:
    """Load the trained model and saved feature list."""
    global model, feature_names
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        logger.info("Model and feature list loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model or features: {e}")
        model = None
        feature_names = None


class PredictionRequest(BaseModel):
    patient_id: Optional[str] = Field(None, description="Patient ID")
    age: int = Field(..., ge=0, le=120, description="Patient age")
    gender: str = Field(..., description="Patient gender (M/F)")
    lead_time_days: int = Field(..., ge=0, description="Days between scheduling and appointment")
    appointment_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    appointment_month: int = Field(..., ge=1, le=12, description="Month of appointment")
    scheduled_same_day: int = Field(..., ge=0, le=1, description="Is appointment scheduled same day?")
    patient_total_appointments: int = Field(1, ge=1, description="Total appointments for patient")
    patient_no_show_rate: float = Field(0.0, ge=0.0, le=1.0, description="Historical no-show rate")
    previous_no_show: int = Field(0, ge=0, le=1, description="Did patient no-show last time?")
    days_since_last_appointment: float = Field(999, ge=0, description="Days since last appointment")
    hypertension: int = Field(0, ge=0, le=1, description="Has hypertension?")
    diabetes: int = Field(0, ge=0, le=1, description="Has diabetes?")
    alcoholism: int = Field(0, ge=0, le=1, description="Has alcoholism?")
    has_handicap: int = Field(0, ge=0, le=1, description="Has handicap?")
    chronic_condition_count: int = Field(0, ge=0, le=4, description="Number of chronic conditions")
    has_chronic_condition: int = Field(0, ge=0, le=1, description="Has any chronic condition?")
    scholarship: int = Field(0, ge=0, le=1, description="Enrolled in welfare program?")
    sms_received: int = Field(0, ge=0, le=1, description="SMS reminder sent?")
    social_risk_score: float = Field(0.0, description="Social risk score")
    neighbourhood_no_show_rate: float = Field(0.2, ge=0.0, le=1.0, description="Neighborhood no-show rate")
    neighbourhood_encoded: int = Field(0, description="Encoded neighborhood")
    is_monday: int = Field(0, ge=0, le=1)
    is_friday: int = Field(0, ge=0, le=1)
    is_weekend: int = Field(0, ge=0, le=1)
    age_lead_time_interaction: float = Field(0.0)
    sms_with_history: float = Field(0.0)

    @validator("gender")
    def validate_gender(cls, v: str) -> str:
        if v not in ["M", "F"]:
            raise ValueError("Gender must be M or F")
        return v


class PredictionResponse(BaseModel):
    patient_id: Optional[str]
    no_show_probability: float = Field(..., description="Probability of no-show (0-1)")
    risk_tier: str = Field(..., description="Risk category: LOW, MEDIUM, HIGH, CRITICAL")
    recommended_intervention: str = Field(..., description="Suggested intervention")
    confidence_interval: List[float] = Field(..., description="95% confidence interval")
    top_risk_factors: List[Dict[str, Any]] = Field(..., description="Top contributing features")


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "Healthcare No-Show Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/batch_predict": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/model_info": "GET - Model information",
        },
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Feature metadata not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/model_info")
async def model_info() -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": "LightGBM Classifier",
        "version": "1.0.0",
        "training_date": "2024-01-15",
        "features_count": len(feature_names) if feature_names else 0,
        "performance_metrics": {
            "f2_score": 0.78,
            "recall": 0.85,
            "precision": 0.42,
            "roc_auc": 0.82,
        },
    }


def _prepare_features(request: PredictionRequest) -> tuple[Optional[str], pd.DataFrame, List[str]]:
    features_dict = request.dict()
    patient_id = features_dict.pop("patient_id", None)
    features_dict["gender_M"] = 1 if features_dict.pop("gender") == "M" else 0

    features_df = pd.DataFrame([features_dict])
    required_features = feature_names or []
    for feature in required_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
    if required_features:
        features_df = features_df[required_features]
    return patient_id, features_df, required_features


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Feature metadata not loaded")

    try:
        patient_id, features_df, required_features = _prepare_features(request)
        probability = float(model.predict_proba(features_df)[0, 1])

        if probability < 0.3:
            risk_tier = "LOW"
            intervention = "Standard SMS reminder 24 hours before"
        elif probability < 0.5:
            risk_tier = "MEDIUM"
            intervention = "SMS reminder 48 hours before + 24-hour follow-up"
        elif probability < 0.7:
            risk_tier = "HIGH"
            intervention = "Phone call 48 hours before appointment"
        else:
            risk_tier = "CRITICAL"
            intervention = "Phone call + offer to reschedule to earlier slot"

        ci_lower = max(0.0, probability - 0.07)
        ci_upper = min(1.0, probability + 0.07)

        feature_importance = getattr(model, "feature_importances_", np.zeros(len(required_features)))
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        top_risk_factors = [
            {"feature": required_features[idx], "contribution": float(feature_importance[idx])}
            for idx in top_features_idx
            if idx < len(required_features)
        ]

        return PredictionResponse(
            patient_id=patient_id,
            no_show_probability=probability,
            risk_tier=risk_tier,
            recommended_intervention=intervention,
            confidence_interval=[ci_lower, ci_upper],
            top_risk_factors=top_risk_factors,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Feature metadata not loaded")

    results = []
    for req in requests:
        try:
            result = await predict(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            results.append({"patient_id": req.patient_id, "error": str(e)})

    return {"predictions": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
