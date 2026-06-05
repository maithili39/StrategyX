import io
import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import FatiguePredictor
from src.db.database import get_db

app = FastAPI(
    title="OTT User Fatigue Prediction API",
    description="Production-grade REST service to predict and explain streaming service user engagement fatigue.",
    version="1.0.0"
)

# Instantiate predictor (loads model on startup)
predictor = FatiguePredictor()

class UserStatsRequest(BaseModel):
    tenure_days: int = Field(..., description="Days since user registered", example=74)
    subscription_tier: str = Field(..., description="Subscription level: Basic, Standard, Premium", example="Basic")
    avg_daily_minutes_last_7d: float = Field(..., description="Average daily viewing minutes last 7 days", example=0.0)
    avg_daily_minutes_last_30d: float = Field(..., description="Average daily viewing minutes last 30 days", example=5.0)
    sessions_last_7d: int = Field(..., description="Number of sessions last 7 days", example=3)
    sessions_last_30d: int = Field(..., description="Number of sessions last 30 days", example=13)
    avg_completion_rate: float = Field(..., description="Average completion rate of contents (0.0 - 1.0)", example=0.05)
    unique_genres_watched_30d: int = Field(..., description="Unique content genres watched last 30 days", example=4)
    days_since_last_session: int = Field(..., description="Days elapsed since the last session", example=2)
    binge_sessions_last_30d: int = Field(..., description="Number of binge sessions last 30 days", example=0)
    peak_hour_viewing_pct: float = Field(..., description="Percentage of peak-hour viewing (0.0 - 100.0)", example=87.8)
    original_content_pct: float = Field(..., description="Percentage of original content viewed (0.0 - 100.0)", example=27.4)
    recommendation_click_rate: float = Field(..., description="Click-through-rate on recommendations (0.0 - 1.0)", example=0.02)
    user_id: str = Field("UNKNOWN", description="Unique user identifier", example="U006253")

class PredictionResponse(BaseModel):
    user_id: str
    fatigue_probability: float
    is_fatigued: bool
    risk_level: str
    business_archetype: str

class ShapFeatureExplanation(BaseModel):
    feature: str
    shap_value: float
    actual_value: float

@app.get("/health")
def health_check():
    """
    Returns the health of the API service and the loading status of the ML model pipeline.
    """
    model_loaded = predictor.is_model_loaded()
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "optimal_threshold": predictor.threshold if model_loaded else None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_user_fatigue(request: UserStatsRequest, db: Session = Depends(get_db)):
    """
    Predicts the fatigue probability, risk level, flag, and business archetype for a single user.
    """
    if not predictor.is_model_loaded():
        raise HTTPException(status_code=503, detail="Machine Learning model is not trained/loaded.")
        
    try:
        user_data = request.model_dump()
        result = predictor.predict_single(user_data, db=db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/explain", response_model=list[ShapFeatureExplanation])
def explain_user_fatigue(request: UserStatsRequest):
    """
    Calculates SHAP feature contributions for a single user, showing which behaviors drive fatigue.
    """
    if not predictor.is_model_loaded():
        raise HTTPException(status_code=503, detail="Machine Learning model is not trained/loaded.")
        
    try:
        user_data = request.model_dump()
        explanation = predictor.explain_single(user_data)
        # Filter for output format
        return [
            ShapFeatureExplanation(
                feature=e["feature"],
                shap_value=e["shap_value"],
                actual_value=e["actual_value"]
            )
            for e in explanation
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

@app.post("/predict/batch")
def predict_batch_users(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Runs batch predictions on an uploaded CSV file containing raw user statistics.
    Returns a JSON array of predictions.
    """
    if not predictor.is_model_loaded():
        raise HTTPException(status_code=503, detail="Machine Learning model is not trained/loaded.")
        
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Verify columns exist
        required_cols = [
            "tenure_days", "subscription_tier", "avg_daily_minutes_last_7d",
            "avg_daily_minutes_last_30d", "sessions_last_7d", "sessions_last_30d",
            "avg_completion_rate", "unique_genres_watched_30d", "days_since_last_session",
            "binge_sessions_last_30d", "peak_hour_viewing_pct", "original_content_pct",
            "recommendation_click_rate"
        ]
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns in uploaded CSV: {missing}")
            
        results_df = predictor.predict_batch(df, db=db)
        return results_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
