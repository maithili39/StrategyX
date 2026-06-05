import pandas as pd
from sqlalchemy.orm import Session
from src.db.database import engine, Base
from src.db.models import DBUser, DBPrediction, DBTrainingRun

def init_db():
    """
    Creates all database tables defined in the SQLAlchemy models.
    """
    Base.metadata.create_all(bind=engine)

def bulk_insert_users(db: Session, users_list: list[dict]):
    """
    Inserts a list of raw user dictionaries into the database,
    removing any existing user records with the same user_id to prevent duplicates.
    """
    if not users_list:
        return
        
    user_ids = [u['user_id'] for u in users_list]
    
    # Remove existing records
    db.query(DBUser).filter(DBUser.user_id.in_(user_ids)).delete(synchronize_session=False)
    
    # Bulk insert
    db.bulk_insert_mappings(DBUser, users_list)
    db.commit()

def get_users_dataframe(db: Session) -> pd.DataFrame:
    """
    Fetches all records from the users table and returns them as a pandas DataFrame.
    """
    query = db.query(DBUser)
    df = pd.read_sql(query.statement, db.bind)
    # Drop the internal database auto-incrementing ID column if it exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df

def save_prediction(db: Session, pred_dict: dict):
    """
    Saves a single prediction log record.
    """
    db_pred = DBPrediction(
        user_id=pred_dict['user_id'],
        predicted_probability=pred_dict['fatigue_probability'],
        fatigue_flag=int(pred_dict['is_fatigued']),
        risk_level=pred_dict['risk_level'],
        business_archetype=pred_dict['business_archetype'],
        retention_action_triggered=pred_dict.get('retention_action_triggered'),
        conversion_success=pred_dict.get('conversion_success')
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

def update_prediction_feedback(db: Session, user_id: str, conversion_success: int, retention_action: str = None):
    """
    Updates the most recent prediction feedback for a subscriber.
    """
    pred = db.query(DBPrediction).filter(DBPrediction.user_id == user_id).order_by(DBPrediction.timestamp.desc()).first()
    if pred:
        pred.conversion_success = conversion_success
        if retention_action:
            pred.retention_action_triggered = retention_action
        db.commit()
        db.refresh(pred)
        return pred
    return None

def save_batch_predictions(db: Session, pred_df: pd.DataFrame):
    """
    Bulk saves a batch of predictions from a DataFrame or list of dicts.
    """
    records = pred_df.to_dict(orient="records")
    mappings = [
        {
            "user_id": r["user_id"],
            "predicted_probability": r["predicted_fatigue_probability"],
            "fatigue_flag": int(r["fatigue_flag"]),
            "risk_level": r["risk_level"],
            "business_archetype": r["business_archetype"],
            "retention_action_triggered": r.get("retention_action_triggered"),
            "conversion_success": r.get("conversion_success")
        }
        for r in records
    ]
    db.bulk_insert_mappings(DBPrediction, mappings)
    db.commit()

def get_prediction_history(db: Session, limit: int = 100) -> list[dict]:
    """
    Retrieves the most recent predictions.
    """
    preds = db.query(DBPrediction).order_by(DBPrediction.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": p.id,
            "user_id": p.user_id,
            "predicted_probability": p.predicted_probability,
            "fatigue_flag": p.fatigue_flag,
            "risk_level": p.risk_level,
            "business_archetype": p.business_archetype,
            "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "retention_action_triggered": p.retention_action_triggered,
            "conversion_success": p.conversion_success
        }
        for p in preds
    ]

def save_training_run(db: Session, run_id: str, run_metrics: dict):
    """
    Saves metadata about a training run.
    """
    db_run = DBTrainingRun(
        run_id=run_id,
        dataset_used=run_metrics.get("dataset_used", "unknown"),
        num_records=run_metrics.get("num_records", 0),
        validation_auc=run_metrics.get("validation_auc", 0.0),
        brier_score=run_metrics.get("brier_score", 0.0),
        optimal_threshold=run_metrics.get("optimal_threshold", 0.0),
        best_f1_score=run_metrics.get("best_f1_score", 0.0),
        precision=run_metrics.get("precision", 0.0),
        recall=run_metrics.get("recall", 0.0)
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run

def get_training_runs(db: Session) -> list[dict]:
    """
    Retrieves the list of training runs sorted by execution time.
    """
    runs = db.query(DBTrainingRun).order_by(DBTrainingRun.timestamp.desc()).all()
    return [
        {
            "id": r.id,
            "run_id": r.run_id,
            "dataset_used": r.dataset_used,
            "num_records": r.num_records,
            "validation_auc": r.validation_auc,
            "brier_score": r.brier_score,
            "optimal_threshold": r.optimal_threshold,
            "best_f1_score": r.best_f1_score,
            "precision": r.precision,
            "recall": r.recall,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in runs
    ]
