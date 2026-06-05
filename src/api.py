import io
import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import FatiguePredictor
from src.db.database import get_db
from src.db.crud import update_prediction_feedback
from src.auth import get_current_user, verify_password, create_access_token, ADMIN_USERNAME, ADMIN_PASSWORD_HASH
from src.logger import api_logger

app = FastAPI(
    title="StrategyX Production Churn API",
    description="Secured high-scale REST microservice with JWT auth, structured JSON logging, and asynchronous task offloading.",
    version="2.0.0"
)

# CORS: allow React dev server (port 5173) and containerised frontend (port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate predictor (loads model on startup)
predictor = FatiguePredictor()

class UserStatsRequest(BaseModel):
    user_id: str = Field("UNKNOWN", description="Unique user identifier", example="U006253")
    tenure_days: int = Field(..., ge=0, description="Days since user registered", example=74)
    subscription_tier: str = Field(..., description="Subscription level: Basic, Standard, Premium", example="Basic")
    avg_daily_minutes_last_7d: float = Field(..., ge=0.0, description="Average daily viewing minutes last 7 days", example=0.0)
    avg_daily_minutes_last_30d: float = Field(..., ge=0.0, description="Average daily viewing minutes last 30 days", example=5.0)
    sessions_last_7d: int = Field(..., ge=0, description="Number of sessions last 7 days", example=3)
    sessions_last_30d: int = Field(..., ge=0, description="Number of sessions last 30 days", example=13)
    avg_completion_rate: float = Field(..., ge=0.0, le=1.0, description="Average completion rate of contents (0.0 - 1.0)", example=0.05)
    unique_genres_watched_30d: int = Field(..., ge=1, le=15, description="Unique content genres watched last 30 days", example=4)
    days_since_last_session: int = Field(..., ge=0, le=30, description="Days elapsed since the last session", example=2)
    binge_sessions_last_30d: int = Field(..., ge=0, description="Number of binge sessions last 30 days", example=0)
    peak_hour_viewing_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of peak-hour viewing (0.0 - 100.0)", example=87.8)
    original_content_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of original content viewed (0.0 - 100.0)", example=27.4)
    recommendation_click_rate: float = Field(..., ge=0.0, le=1.0, description="Click-through-rate on recommendations (0.0 - 1.0)", example=0.02)

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

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

def bg_batch_predict(contents: bytes, db_session: Session):
    """
    Task executed asynchronously in a background worker thread.
    Parses CSV inputs, runs batch inferences, and records predictions to DB.
    """
    try:
        df = pd.read_csv(io.BytesIO(contents))
        # Ensure we bind a new thread-specific session for safety
        results_df = predictor.predict_batch(df, db=db_session)
        api_logger.info(
            f"Asynchronous batch prediction run completed successfully.",
            extra={"extra_attrs": {"records_processed": len(results_df), "at_risk_count": int(results_df['fatigue_flag'].sum())}}
        )
    except Exception as e:
        api_logger.error(
            f"Error processing asynchronous background batch predictions: {e}",
            exc_info=True
        )

@app.post("/token", response_model=TokenResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 Password Flow Token exchange route.
    Username: strategyx_admin | Password: strategyx_password
    """
    if form_data.username != ADMIN_USERNAME or not verify_password(form_data.password, ADMIN_PASSWORD_HASH):
        api_logger.warning(
            f"Failed login attempt for user: {form_data.username}",
            extra={"extra_attrs": {"username": form_data.username}}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = create_access_token(data={"sub": form_data.username})
    api_logger.info(
        f"Access token issued for user: {form_data.username}",
        extra={"extra_attrs": {"username": form_data.username}}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/health")
def health_check():
    """
    Returns API status and ML model status. Unsecured route.
    """
    model_loaded = predictor.is_model_loaded()
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "optimal_threshold": predictor.threshold if model_loaded else None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_user_fatigue(
    request: UserStatsRequest, 
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Predicts the fatigue probability, risk level, flag, and archetype for a single user.
    Secured: Requires a valid OAuth2 Bearer token.
    """
    if not predictor.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
        
    try:
        user_data = request.model_dump()
        result = predictor.predict_single(user_data, db=db)
        
        api_logger.info(
            f"Single prediction computed for user: {result['user_id']}",
            extra={"extra_attrs": {
                "user_id": result['user_id'],
                "probability": result['fatigue_probability'],
                "risk_level": result['risk_level'],
                "archetype": result['business_archetype'],
                "requested_by": current_user
            }}
        )
        return result
    except Exception as e:
        api_logger.error(f"Prediction error for user {request.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/explain", response_model=list[ShapFeatureExplanation])
def explain_user_fatigue(
    request: UserStatsRequest,
    current_user: str = Depends(get_current_user)
):
    """
    Calculates SHAP feature contributions showing which behaviors drive fatigue.
    Secured: Requires a valid OAuth2 Bearer token.
    """
    if not predictor.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
        
    try:
        user_data = request.model_dump()
        explanation = predictor.explain_single(user_data)
        
        api_logger.info(
            f"SHAP explanation generated for user: {request.user_id}",
            extra={"extra_attrs": {"user_id": request.user_id, "requested_by": current_user}}
        )
        
        return [
            ShapFeatureExplanation(
                feature=e["feature"],
                shap_value=e["shap_value"],
                actual_value=e["actual_value"]
            )
            for e in explanation
        ]
    except Exception as e:
        api_logger.error(f"SHAP calculation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

@app.post("/predict/batch", status_code=status.HTTP_202_ACCEPTED)
def predict_batch_users(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Accepts CSV uploads of subscriber metrics.
    Executes inference asynchronously in a background thread to prevent client timeouts.
    Returns 202 Accepted immediately. Results are logged to the database.
    Secured: Requires a valid OAuth2 Bearer token.
    """
    if not predictor.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
        
    try:
        contents = file.file.read()
        
        # Quick validation of CSV format (read only headers first)
        df_headers = pd.read_csv(io.BytesIO(contents), nrows=2)
        
        required_cols = [
            "tenure_days", "subscription_tier", "avg_daily_minutes_last_7d",
            "avg_daily_minutes_last_30d", "sessions_last_7d", "sessions_last_30d",
            "avg_completion_rate", "unique_genres_watched_30d", "days_since_last_session",
            "binge_sessions_last_30d", "peak_hour_viewing_pct", "original_content_pct",
            "recommendation_click_rate"
        ]
        
        missing = [col for col in required_cols if col not in df_headers.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Uploaded CSV is missing columns: {missing}")
            
        # Parse total records submitted
        total_records = len(pd.read_csv(io.BytesIO(contents)))
        
        # Dispatch background prediction task
        background_tasks.add_task(bg_batch_predict, contents, db)
        
        api_logger.info(
            f"Background batch job scheduled.",
            extra={"extra_attrs": {
                "records_submitted": total_records,
                "requested_by": current_user
            }}
        )
        
        return {
            "status": "Accepted",
            "message": "Batch prediction task scheduled in the background. Inferences will be recorded in database.",
            "records_submitted": total_records
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        api_logger.error(f"Batch prediction trigger error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch trigger error: {str(e)}")

class FeedbackRequest(BaseModel):
    user_id: str
    conversion_success: int = Field(..., ge=0, le=1)
    retention_action_triggered: str = Field(None, description="The action blueprint that was triggered")

class FeedbackResponse(BaseModel):
    user_id: str
    status: str
    conversion_success: int
    retention_action_triggered: str = None

@app.post("/feedback", response_model=FeedbackResponse)
def submit_prediction_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Submits feedback on a subscriber's retention action conversion status.
    Updates the most recent prediction history record for analytics and retraining.
    Secured: Requires a valid OAuth2 Bearer token.
    """
    updated_record = update_prediction_feedback(
        db=db,
        user_id=request.user_id,
        conversion_success=request.conversion_success,
        retention_action=request.retention_action_triggered
    )
    if not updated_record:
        raise HTTPException(status_code=404, detail=f"No prediction record found for user {request.user_id}")
        
    api_logger.info(
        f"Logged conversion feedback for user {request.user_id}: {request.conversion_success}",
        extra={"extra_attrs": {
            "user_id": request.user_id,
            "conversion_success": request.conversion_success,
            "retention_action": request.retention_action_triggered,
            "updated_by": current_user
        }}
    )
    return {
        "user_id": updated_record.user_id,
        "status": "Success",
        "conversion_success": updated_record.conversion_success,
        "retention_action_triggered": updated_record.retention_action_triggered
    }

@app.websocket("/ws/predict")
async def websocket_predict_endpoint(websocket: WebSocket):
    """
    WebSocket channel for live, real-time subscriber diagnostics.
    Accepts user telemetry, returns fatigue classification and SHAP explainer attributes.
    """
    await websocket.accept()
    api_logger.info("Real-time prediction WebSocket client connected.")
    try:
        while True:
            # Expect client to stream telemetry JSON matching UserStatsRequest fields (minus user_id)
            data = await websocket.receive_json()
            user_id = data.get("user_id", "STREAMING_WS_USER")
            
            # Predict single
            if not predictor.is_model_loaded():
                await websocket.send_json({"error": "ML model is not loaded on server."})
                continue
                
            res = predictor.predict_single(data)
            
            # Include SHAP explanation key metrics
            shaps = predictor.explain_single(data)
            res["shap_explanation"] = shaps[:5] # Top 5 drivers
            
            await websocket.send_json(res)
    except WebSocketDisconnect:
        api_logger.info("Real-time prediction WebSocket client disconnected.")
    except Exception as e:
        api_logger.error(f"WebSocket execution error: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass

