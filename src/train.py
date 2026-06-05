import os
import sys
import argparse
import yaml
import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              precision_recall_curve, confusion_matrix, f1_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import OutlierCapper, engineer_features

# Database imports
from src.db.database import SessionLocal
from src.db.crud import init_db, get_users_dataframe, save_training_run

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_training(use_scaled=False, use_db=False, config_path="config.yaml"):
    config = load_config(config_path)
    
    # 1. Load Data
    db_session = None
    dataset_name = "base_csv"
    
    if use_db:
        print("Loading training data from Relational Database...")
        init_db()
        db_session = SessionLocal()
        try:
            train_df = get_users_dataframe(db_session)
            if train_df.empty:
                raise ValueError("Database 'users' table is empty. Seed the database first using 'python src/data_generator.py --num-rows 10000 --to-db'")
            dataset_name = f"database_records_{len(train_df)}"
        except Exception as e:
            db_session.close()
            raise e
    else:
        data_cfg = config['data']
        train_file = data_cfg['scaled_train_path'] if use_scaled else data_cfg['train_path']
        dataset_name = "scaled_csv" if use_scaled else "base_csv"
        
        print(f"Loading training data from: {train_file}")
        if not os.path.exists(train_file):
            if use_scaled:
                print(f"Scaled dataset {train_file} not found. Generating now...")
                from src.data_generator import generate_synthetic_data
                df = generate_synthetic_data(num_users=50000)
                train_size = int(len(df) * 0.8)
                train_df = df.iloc[:train_size]
                test_df = df.iloc[train_size:].copy().drop(columns=['fatigue_label'])
                
                # Save
                train_df.to_csv(data_cfg['scaled_train_path'], index=False)
                test_df.to_csv(data_cfg['scaled_test_path'], index=False)
            else:
                raise FileNotFoundError(f"Base training dataset {train_file} not found. Please place it in the working directory.")
        else:
            train_df = pd.read_csv(train_file)
        
    print(f"Training dataset shape: {train_df.shape}")
    
    # 2. Split into features and target before feature engineering to prevent leakage
    id_col = config['features']['id_column']
    target_col = config['features']['target_column']
    
    X = train_df.drop(columns=[target_col, id_col], errors='ignore')
    y = train_df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config['training']['test_size'], 
        stratify=y, 
        random_state=config['model']['random_state']
    )
    
    # 3. Apply Feature Engineering
    print("Engineering features...")
    X_train_fe = engineer_features(X_train)
    X_val_fe = engineer_features(X_val)
    
    # Align validation columns with training columns
    X_train_fe, X_val_fe = X_train_fe.align(X_val_fe, join='left', axis=1, fill_value=0)
    
    # Save the feature list
    feature_list = X_train_fe.columns.tolist()
    
    # 4. Initialize MLflow
    mlflow_tracking_uri = config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    mlflow_exp_name = config.get("mlflow", {}).get("experiment_name", "ott_user_fatigue_predictions")
    
    # Environment variable overrides
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_tracking_uri)
    mlflow_exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", mlflow_exp_name)
    
    import mlflow
    import mlflow.sklearn
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_exp_name)
    
    # 5. Hyperparameter Tuning with Optuna & Model Training
    print("Starting hyperparameter optimization with Optuna...")
    cv_splits = config['training']['cv_splits']
    optuna_trials = config['training']['optuna_trials']
    random_state = config['model']['random_state']
    
    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 150, 450),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma":            trial.suggest_float("gamma", 0, 3),
            "random_state": random_state,
            "eval_metric": "auc",
            "tree_method": "hist"
        }
        
        # SMOTE applied inside CV folds to prevent label leak
        pipeline = ImbPipeline([
            ("capper", OutlierCapper()),
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=random_state)),
            ("model", xgb.XGBClassifier(**params))
        ])
        
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, va_idx in cv.split(X_train_fe, y_train):
            X_tr, X_va = X_train_fe.iloc[tr_idx], X_train_fe.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            pipeline.fit(X_tr, y_tr)
            scores.append(roc_auc_score(y_va, pipeline.predict_proba(X_va)[:, 1]))
        return np.mean(scores)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials)
    
    print(f"Best CV ROC-AUC: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    # Wrap fitting and logging in an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Active Run ID: {run_id}")
        
        # Log params
        mlflow.log_params(study.best_params)
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("num_records", len(train_df))
        
        # 6. Fit Final Tuned Model on Training Split
        print("Training final tuned model pipeline...")
        final_pipeline = ImbPipeline([
            ("capper", OutlierCapper()),
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=random_state)),
            ("model", xgb.XGBClassifier(
                **study.best_params,
                random_state=random_state,
                eval_metric="auc",
                tree_method="hist"
            ))
        ])
        final_pipeline.fit(X_train_fe, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(final_pipeline, "model_pipeline")
        
        # 7. Evaluate Model on Held-out Validation Set
        val_probs = final_pipeline.predict_proba(X_val_fe)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        brier_score = brier_score_loss(y_val, val_probs)
        
        # Optimize Decision Threshold using Precision-Recall F1 maximisation
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        
        # Apply optimal threshold for prediction metrics
        val_preds = (val_probs >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_val, val_preds)
        
        # TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        precision = float(tp / (tp + fp))
        recall = float(tp / (tp + fn))
        
        # Log MLflow metrics
        mlflow.log_metric("validation_auc", val_auc)
        mlflow.log_metric("brier_score", brier_score)
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        mlflow.log_metric("f1_score", best_f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        metrics = {
            "dataset_used": dataset_name,
            "num_records": len(train_df),
            "validation_auc": float(val_auc),
            "brier_score": float(brier_score),
            "best_cv_auc": float(study.best_value),
            "optimal_threshold": optimal_threshold,
            "best_f1_score": best_f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "best_params": study.best_params
        }
        
        # Save training run details to SQLite Database
        if use_db and db_session is not None:
            try:
                save_training_run(db_session, run_id, metrics)
                print("Logged training run record to the relational database.")
            except Exception as e:
                print(f"Error logging training run metadata to database: {e}")
                
        # Also log to a general local training_metrics yaml
        metrics_path = os.path.join(config['model']['save_dir'], config['model']['metrics_filename'])
        os.makedirs(config['model']['save_dir'], exist_ok=True)
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f, default_flow_style=False)
            
    # Clean up DB session
    if db_session is not None:
        db_session.close()
        
    print("\n--- Training Evaluation ---")
    print(f"Validation ROC-AUC        : {val_auc:.4f}")
    print(f"Brier Score Loss          : {brier_score:.4f}")
    print(f"Optimal Decision Threshold: {optimal_threshold:.4f}")
    print(f"F1 Score at Threshold     : {best_f1:.4f}")
    print(f"Precision                 : {precision:.4f}")
    print(f"Recall                    : {recall:.4f}")
    
    # 8. Model Serialization
    save_dir = config['model']['save_dir']
    model_path = os.path.join(save_dir, config['model']['model_filename'])
    features_path = os.path.join(save_dir, config['model']['features_filename'])
    
    joblib.dump(final_pipeline, model_path)
    joblib.dump(feature_list, features_path)
        
    print(f"\nSaved models and metadata to '{save_dir}/':")
    print(f" - Pipeline: {model_path}")
    print(f" - Features: {features_path}")
    print(f" - MLflow metrics logged in SQLite backend.")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTT User Fatigue Train Script with Database & MLflow integrations")
    parser.add_argument("--use-scaled", action="store_true", help="Train model using the scaled/large CSV dataset")
    parser.add_argument("--use-db", action="store_true", help="Train model using records loaded from the Relational Database")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config yaml")
    args = parser.parse_args()
    
    run_training(use_scaled=args.use_scaled, use_db=args.use_db, config_path=args.config)
