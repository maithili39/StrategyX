from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from src.db.database import Base

class DBUser(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    tenure_days = Column(Integer, nullable=False)
    subscription_tier = Column(String, nullable=False)
    avg_daily_minutes_last_7d = Column(Float, nullable=False)
    avg_daily_minutes_last_30d = Column(Float, nullable=False)
    sessions_last_7d = Column(Integer, nullable=False)
    sessions_last_30d = Column(Integer, nullable=False)
    avg_completion_rate = Column(Float, nullable=False)
    unique_genres_watched_30d = Column(Integer, nullable=False)
    days_since_last_session = Column(Integer, nullable=False)
    binge_sessions_last_30d = Column(Integer, nullable=False)
    peak_hour_viewing_pct = Column(Float, nullable=False)
    original_content_pct = Column(Float, nullable=False)
    recommendation_click_rate = Column(Float, nullable=False)
    fatigue_label = Column(Integer, nullable=True) # Nullable because test set has no label

class DBPrediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    predicted_probability = Column(Float, nullable=False)
    fatigue_flag = Column(Integer, nullable=False)
    risk_level = Column(String, nullable=False)
    business_archetype = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class DBTrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True, nullable=False)
    dataset_used = Column(String, nullable=False)
    num_records = Column(Integer, nullable=False)
    validation_auc = Column(Float, nullable=False)
    brier_score = Column(Float, nullable=False)
    optimal_threshold = Column(Float, nullable=False)
    best_f1_score = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
