import os
import sys
import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db.models import Base, DBUser, DBPrediction, DBTrainingRun
from src.db.crud import (
    bulk_insert_users, get_users_dataframe, save_prediction,
    save_batch_predictions, get_prediction_history, save_training_run, get_training_runs
)

# Set up in-memory SQLite for testing
@pytest.fixture(name="db_session")
def fixture_db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

def test_database_bulk_insert_and_dataframe(db_session):
    """
    Test bulk seeding users to database and pulling them back as a pandas DataFrame.
    """
    users_data = [
        {
            "user_id": "U001",
            "tenure_days": 100,
            "subscription_tier": "Basic",
            "avg_daily_minutes_last_7d": 10.0,
            "avg_daily_minutes_last_30d": 20.0,
            "sessions_last_7d": 2,
            "sessions_last_30d": 10,
            "avg_completion_rate": 0.5,
            "unique_genres_watched_30d": 3,
            "days_since_last_session": 4,
            "binge_sessions_last_30d": 1,
            "peak_hour_viewing_pct": 80.0,
            "original_content_pct": 30.0,
            "recommendation_click_rate": 0.1,
            "fatigue_label": 0
        },
        {
            "user_id": "U002",
            "tenure_days": 250,
            "subscription_tier": "Premium",
            "avg_daily_minutes_last_7d": 0.0,
            "avg_daily_minutes_last_30d": 5.0,
            "sessions_last_7d": 0,
            "sessions_last_30d": 8,
            "avg_completion_rate": 0.1,
            "unique_genres_watched_30d": 1,
            "days_since_last_session": 10,
            "binge_sessions_last_30d": 0,
            "peak_hour_viewing_pct": 50.0,
            "original_content_pct": 20.0,
            "recommendation_click_rate": 0.05,
            "fatigue_label": 1
        }
    ]
    
    # Seed
    bulk_insert_users(db_session, users_data)
    
    # Check count in DB
    users_in_db = db_session.query(DBUser).all()
    assert len(users_in_db) == 2
    assert users_in_db[0].user_id == "U001"
    assert users_in_db[1].subscription_tier == "Premium"
    
    # Pull as DataFrame
    df = get_users_dataframe(db_session)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "user_id" in df.columns
    assert "id" not in df.columns # Ensure internal id column is dropped
    assert df[df['user_id'] == "U002"]['fatigue_label'].values[0] == 1

def test_database_predictions_history(db_session):
    """
    Test logging single and batch predictions and fetching prediction history.
    """
    # 1. Single prediction
    pred_data = {
        "user_id": "U001",
        "fatigue_probability": 0.85,
        "is_fatigued": True,
        "risk_level": "High Risk",
        "business_archetype": "Binge-and-Leave"
    }
    
    save_prediction(db_session, pred_data)
    
    # Check prediction history count
    history = get_prediction_history(db_session)
    assert len(history) == 1
    assert history[0]["user_id"] == "U001"
    assert history[0]["risk_level"] == "High Risk"
    assert history[0]["predicted_probability"] == 0.85
    
    # 2. Batch predictions
    batch_df = pd.DataFrame([
        {
            "user_id": "U002",
            "predicted_fatigue_probability": 0.12,
            "fatigue_flag": 0,
            "risk_level": "Low Risk",
            "business_archetype": "Active & Engaged"
        },
        {
            "user_id": "U003",
            "predicted_fatigue_probability": 0.54,
            "fatigue_flag": 1,
            "risk_level": "Medium Risk",
            "business_archetype": "Waning Casual"
        }
    ])
    
    save_batch_predictions(db_session, batch_df)
    
    # Check prediction history count again
    history = get_prediction_history(db_session)
    assert len(history) == 3
    # Order should be timestamp descending (most recent first, sqlite bulk inserts are fast so order is preserved or based on pk/time)
    user_ids_in_history = [h["user_id"] for h in history]
    assert "U001" in user_ids_in_history
    assert "U002" in user_ids_in_history
    assert "U003" in user_ids_in_history

def test_database_training_runs(db_session):
    """
    Test logging and fetching training run metrics.
    """
    run_metrics = {
        "dataset_used": "base_csv",
        "num_records": 8000,
        "validation_auc": 0.8035,
        "brier_score": 0.1641,
        "optimal_threshold": 0.4827,
        "best_f1_score": 0.6697,
        "precision": 0.7008,
        "recall": 0.6413
    }
    
    save_training_run(db_session, "run_12345_uuid", run_metrics)
    
    # Get runs list
    runs = get_training_runs(db_session)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_12345_uuid"
    assert runs[0]["validation_auc"] == 0.8035
    assert runs[0]["dataset_used"] == "base_csv"
