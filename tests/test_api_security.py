import io
import os
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api import app
from src.auth import ADMIN_USERNAME

client = TestClient(app)

# Test payload matching validation schemas
VALID_USER_PAYLOAD = {
    "user_id": "U12345",
    "tenure_days": 100,
    "subscription_tier": "Premium",
    "avg_daily_minutes_last_7d": 15.0,
    "avg_daily_minutes_last_30d": 30.0,
    "sessions_last_7d": 3,
    "sessions_last_30d": 15,
    "avg_completion_rate": 0.45,
    "unique_genres_watched_30d": 5,
    "days_since_last_session": 3,
    "binge_sessions_last_30d": 2,
    "peak_hour_viewing_pct": 82.5,
    "original_content_pct": 40.0,
    "recommendation_click_rate": 0.25
}

def test_unsecured_health_check():
    """
    Verify that health check route is unsecured and accessible.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()

def test_secured_routes_require_authentication():
    """
    Verify that inference routes block access without a token.
    """
    # /predict
    response = client.post("/predict", json=VALID_USER_PAYLOAD)
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"
    
    # /explain
    response = client.post("/explain", json=VALID_USER_PAYLOAD)
    assert response.status_code == 401
    
    # /predict/batch
    response = client.post("/predict/batch")
    assert response.status_code == 401

def test_token_retrieval_success_and_failure():
    """
    Verify credential validation for OAuth2 tokens.
    """
    # 1. Correct credentials
    response = client.post("/token", data={"username": ADMIN_USERNAME, "password": "strategyx_password"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"
    
    # 2. Incorrect credentials
    response = client.post("/token", data={"username": ADMIN_USERNAME, "password": "wrong_password"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

def test_secured_inference_with_token():
    """
    Verify that authenticating with token successfully accesses the secured inference routes.
    """
    # Retrieve token
    token_response = client.post("/token", data={"username": ADMIN_USERNAME, "password": "strategyx_password"})
    token = token_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Check predict
    response = client.post("/predict", json=VALID_USER_PAYLOAD, headers=headers)
    assert response.status_code == 200
    assert response.json()["user_id"] == "U12345"
    assert "fatigue_probability" in response.json()
    assert "is_fatigued" in response.json()
    assert "risk_level" in response.json()
    assert "business_archetype" in response.json()
    
    # Check explain
    explain_response = client.post("/explain", json=VALID_USER_PAYLOAD, headers=headers)
    assert explain_response.status_code == 200
    assert isinstance(explain_response.json(), list)
    assert "feature" in explain_response.json()[0]
    assert "shap_value" in explain_response.json()[0]

def test_pydantic_validation_constraints():
    """
    Verify that out-of-bounds/negative input values trigger Pydantic validation errors (422).
    """
    token_response = client.post("/token", data={"username": ADMIN_USERNAME, "password": "strategyx_password"})
    token = token_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 1. Invalid tenure_days (negative value ge=0 constraint)
    invalid_payload = VALID_USER_PAYLOAD.copy()
    invalid_payload["tenure_days"] = -5
    response = client.post("/predict", json=invalid_payload, headers=headers)
    assert response.status_code == 422
    assert "tenure_days" in response.text
    
    # 2. Invalid completion rate (> 1.0 le=1.0 constraint)
    invalid_payload = VALID_USER_PAYLOAD.copy()
    invalid_payload["avg_completion_rate"] = 1.25
    response = client.post("/predict", json=invalid_payload, headers=headers)
    assert response.status_code == 422
    assert "avg_completion_rate" in response.text

def test_async_batch_prediction_trigger():
    """
    Verify that /predict/batch registers predictions asynchronously and returns 202 Accepted.
    """
    token_response = client.post("/token", data={"username": ADMIN_USERNAME, "password": "strategyx_password"})
    token = token_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Construct small mock CSV
    df = pd.DataFrame([VALID_USER_PAYLOAD, VALID_USER_PAYLOAD])
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    
    files = {"file": ("test_batch.csv", csv_bytes, "text/csv")}
    
    response = client.post("/predict/batch", files=files, headers=headers)
    assert response.status_code == 202
    assert response.json()["status"] == "Accepted"
    assert response.json()["records_submitted"] == 2
