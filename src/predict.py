import os
import sys
import yaml
import joblib
import pandas as pd
import numpy as np
import shap

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import engineer_features

class FatiguePredictor:
    """
    Inference class to handle loading the serialized model and running predictions
    with business archetypes and SHAP explanations.
    """
    def __init__(self, model_dir="models", config_path="config.yaml"):
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # Paths
        self.model_path = os.path.join(model_dir, "user_fatigue_pipeline.joblib")
        self.features_path = os.path.join(model_dir, "features_list.joblib")
        self.metrics_path = os.path.join(model_dir, "training_metrics.yaml")

        # Load models
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            self.feature_list = joblib.load(self.features_path)
            
            with open(self.metrics_path, "r") as f:
                self.metrics = yaml.safe_load(f)
            
            self.threshold = self.metrics.get("optimal_threshold", 0.45)
            
            # Initialize SHAP explainer
            self.xgb_model = self.pipeline.named_steps['model']
            self.capper = self.pipeline.named_steps['capper']
            self.imputer = self.pipeline.named_steps['imputer']
            self.explainer = shap.TreeExplainer(self.xgb_model)
        else:
            self.pipeline = None
            self.feature_list = []
            self.metrics = {}
            self.threshold = 0.45
            self.explainer = None

    def is_model_loaded(self):
        return self.pipeline is not None

    def get_archetype(self, row_dict, prob):
        """
        Determines the business archetype based on user metrics and fatigue risk.
        """
        if prob < self.threshold:
            return "Active & Engaged"
        
        rec_click = row_dict.get('recommendation_click_rate', 0.5)
        binges = row_dict.get('binge_sessions_last_30d', 0)
        recent_min = row_dict.get('avg_daily_minutes_last_7d', 0)
        
        # Determine archetype matching EDA notebook criteria
        if rec_click < 0.2:
            return "Frustrated Browser"
        elif binges >= 3:
            return "Binge-and-Leave"
        elif recent_min < 20:
            return "Waning Casual"
        else:
            return "General Churn Risk"

    def predict_single(self, raw_data_dict, db=None):
        """
        Runs prediction for a single user raw observation.
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model is not loaded. Train the model first.")

        # Convert single dict to DataFrame
        df_raw = pd.DataFrame([raw_data_dict])
        
        # Extract ID if present
        user_id = raw_data_dict.get('user_id', 'UNKNOWN')
        if 'user_id' in df_raw.columns:
            df_raw = df_raw.drop(columns=['user_id'])
            
        # Engineer features
        df_fe = engineer_features(df_raw)
        
        # Align features
        df_fe = df_fe.reindex(columns=self.feature_list, fill_value=0)
        
        # Predict probability
        prob = float(self.pipeline.predict_proba(df_fe)[0, 1])
        is_fatigued = prob >= self.threshold
        
        # Risk level categorization
        if prob < 0.25:
            risk_level = "Low Risk"
        elif prob < 0.55:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        archetype = self.get_archetype(raw_data_dict, prob)
        
        result = {
            "user_id": user_id,
            "fatigue_probability": prob,
            "is_fatigued": bool(is_fatigued),
            "risk_level": risk_level,
            "business_archetype": archetype
        }

        if db is not None:
            try:
                from src.db.crud import save_prediction
                save_prediction(db, result)
            except Exception as e:
                print(f"Error logging single prediction to database: {e}")

        return result

    def predict_batch(self, df_raw, db=None):
        """
        Runs batch prediction for a DataFrame of raw observations.
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model is not loaded. Train the model first.")

        df = df_raw.copy()
        user_ids = df['user_id'].values if 'user_id' in df.columns else [f"User_{i}" for i in range(len(df))]
        
        # Drop ID and target columns if present
        cols_to_drop = ['user_id', 'fatigue_label']
        df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # Feature engineering
        df_fe = engineer_features(df_clean)
        
        # Align features
        df_fe = df_fe.reindex(columns=self.feature_list, fill_value=0)
        
        # Batch predict
        probs = self.pipeline.predict_proba(df_fe)[:, 1]
        
        results = []
        for i in range(len(df)):
            prob = float(probs[i])
            is_fatigued = int(prob >= self.threshold)
            
            # Retrieve raw record as dict to inspect for archetype
            row_dict = df_clean.iloc[i].to_dict()
            
            # Map risk level
            if prob < 0.25:
                risk_level = "Low Risk"
            elif prob < 0.55:
                risk_level = "Medium Risk"
            else:
                risk_level = "High Risk"
                
            archetype = self.get_archetype(row_dict, prob)
            
            results.append({
                "user_id": user_ids[i],
                "predicted_fatigue_probability": prob,
                "fatigue_flag": is_fatigued,
                "risk_level": risk_level,
                "business_archetype": archetype
            })
            
        res_df = pd.DataFrame(results)

        if db is not None:
            try:
                from src.db.crud import save_batch_predictions
                save_batch_predictions(db, res_df)
            except Exception as e:
                print(f"Error logging batch predictions to database: {e}")

        return res_df

    def explain_single(self, raw_data_dict):
        """
        Computes SHAP explanations for a single raw user observation.
        """
        if not self.is_model_loaded() or self.explainer is None:
            raise RuntimeError("Model or explainer is not loaded.")

        df_raw = pd.DataFrame([raw_data_dict])
        if 'user_id' in df_raw.columns:
            df_raw = df_raw.drop(columns=['user_id'])
            
        # Feature engineering
        df_fe = engineer_features(df_raw)
        df_fe = df_fe.reindex(columns=self.feature_list, fill_value=0)
        
        # Preprocessing matching the pipeline: cap and impute
        X_capped = self.capper.transform(df_fe)
        X_prepared = self.imputer.transform(X_capped)
        X_prepared_df = pd.DataFrame(X_prepared, columns=self.feature_list)
        
        # Calculate SHAP
        shap_vals = self.explainer.shap_values(X_prepared_df)[0]
        
        # Package and sort by absolute contribution
        explanation = []
        for feat, val in zip(self.feature_list, shap_vals):
            explanation.append({
                "feature": feat,
                "shap_value": float(val),
                "abs_shap_value": float(abs(val)),
                "actual_value": float(X_prepared_df[feat].values[0])
            })
            
        explanation = sorted(explanation, key=lambda x: x['abs_shap_value'], reverse=True)
        return explanation
