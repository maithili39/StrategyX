import os
import sys
import yaml
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import FatiguePredictor
from src.data_generator import generate_synthetic_data
from src.train import run_training

# Database imports
from src.db.database import SessionLocal
from src.db.crud import init_db, get_users_dataframe, get_prediction_history, get_training_runs, bulk_insert_users

# Set page config
st.set_page_config(
    page_title="StrategyX Production OTT Control Center",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Dark Theme Accent Adjustments */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    .main-title {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    /* Metric Cards */
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #3b82f6;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        text-align: center;
    }
    .status-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_db()

# DB Session helper
def get_db_session():
    return SessionLocal()

# Instantiate predictor
predictor = FatiguePredictor()

# Load DB user statistics
def fetch_db_stats():
    db = get_db_session()
    try:
        df_users = get_users_dataframe(db)
        history = get_prediction_history(db, limit=10)
        runs = get_training_runs(db)
        return len(df_users), df_users, len(history), history, runs
    finally:
        db.close()

db_users_count, df_users_db, total_preds_logged, pred_history, training_runs = fetch_db_stats()

# Header Area
st.markdown("<h1 class='main-title'>StrategyX OTT Fatigue Control Center</h1>", unsafe_allow_html=True)
st.markdown("##### Production-grade ML system integrating SQLAlchemy Database Layer & MLflow Experiment Tracking.")

# Sidebar Status Panel
st.sidebar.markdown("### Production Infrastructure")
db_status = "🟢 Connected (SQLite)" if df_users_db is not None else "🔴 Offline"
st.sidebar.markdown(f"**Database**: {db_status}")
st.sidebar.markdown(f"**Subscribers in DB**: `{db_users_count:,}`")
st.sidebar.markdown(f"**API Inferences Logged**: `{total_preds_logged}`")

if predictor.is_model_loaded():
    st.sidebar.success("🟢 ML Pipeline: Active")
    st.sidebar.markdown(f"**Optimal Threshold**: `{predictor.threshold:.4f}`")
    st.sidebar.markdown(f"**Val ROC-AUC**: `{predictor.metrics.get('validation_auc', 0.0):.4f}`")
else:
    st.sidebar.warning("🔴 ML Pipeline: Untrained")
    st.sidebar.info("Please go to the **Data Scaling Sandbox** to generate a dataset and train the model.")

# App Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Dashboard",
    "👤 Subscriber Diagnostics",
    "⚙️ Database & MLflow Sandbox",
    "📥 Batch Predictor"
])

# =========================================================================
# TAB 1: Executive Dashboard
# =========================================================================
with tab1:
    st.markdown("### Executive Summary & DB Analytics")
    
    if db_users_count == 0:
        st.warning("⚠️ Database is empty. Please go to the **Database & MLflow Sandbox** tab to seed user interaction data.")
    elif not predictor.is_model_loaded():
        st.warning("⚠️ ML Pipeline is untrained. Please run the training pipeline in the Sandbox to log runs and run predictions.")
    else:
        # Load and run batch predictions on database records
        with st.spinner("Analyzing subscriber base from Database..."):
            predictions = predictor.predict_batch(df_users_db)
            
        total_users = len(predictions)
        fatigued_users = int(predictions['fatigue_flag'].sum())
        fatigue_pct = (fatigued_users / total_users) * 100 if total_users > 0 else 0.0
        avg_prob = predictions['predicted_fatigue_probability'].mean() * 100 if total_users > 0 else 0.0
        
        # Grid layout for KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Subscribers (Database)</div>
                <div class='metric-value'>{total_users:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>At-Risk (Fatigued)</div>
                <div class='metric-value' style='color:#ef4444;'>{fatigued_users:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>At-Risk Proportion</div>
                <div class='metric-value'>{fatigue_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Average Fatigue Score</div>
                <div class='metric-value' style='color:#f59e0b;'>{avg_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # Analytical Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Segment Distribution")
            archetype_counts = predictions['business_archetype'].value_counts()
            
            fig, ax = plt.subplots(figsize=(6, 3.8), facecolor='#111827')
            ax.set_facecolor('#111827')
            
            colors = ['#10b981', '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6']
            bars = ax.barh(archetype_counts.index, archetype_counts.values, color=colors[:len(archetype_counts)], edgecolor='black')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#4b5563')
            ax.spines['bottom'].set_color('#4b5563')
            ax.tick_params(colors='#9ca3af')
            ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#4b5563')
            ax.set_axisbelow(True)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + total_users * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:,}', 
                        va='center', color='#ffffff', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with c2:
            st.markdown("#### Probability Density Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.8), facecolor='#111827')
            ax.set_facecolor('#111827')
            
            sns.histplot(predictions['predicted_fatigue_probability'], bins=30, kde=True, color='#3b82f6', ax=ax, edgecolor='#1f2937')
            ax.axvline(predictor.threshold, color='#ef4444', linestyle='--', linewidth=2, label=f'Threshold ({predictor.threshold:.3f})')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#4b5563')
            ax.spines['bottom'].set_color('#4b5563')
            ax.tick_params(colors='#9ca3af')
            ax.set_xlabel("Probability", color='#9ca3af')
            ax.set_ylabel("Count", color='#9ca3af')
            ax.legend(facecolor='#111827', edgecolor='#1f2937', labelcolor='#ffffff')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        # Real-time inference logging logs
        st.markdown("---")
        st.markdown("#### Real-Time API Prediction Logging History (`predictions` table)")
        st.write("Tracks logs of single and batch model inferences written directly to the database.")
        
        db = get_db_session()
        try:
            recent_preds = get_prediction_history(db, limit=10)
            if not recent_preds:
                st.info("No API prediction logs recorded yet. Execute predictions in the 'Subscriber Diagnostics' tab or REST API to populate this log.")
            else:
                st.dataframe(pd.DataFrame(recent_preds), use_container_width=True)
        finally:
            db.close()

# =========================================================================
# TAB 2: Subscriber Diagnostics & Simulator
# =========================================================================
with tab2:
    st.markdown("### Subscriber Fatigue Profiler & Intervention Sandbox")
    
    if db_users_count == 0:
        st.warning("⚠️ Database contains no users. Please seed user records first.")
    else:
        # Selection Mode
        mode = st.radio("Select input mode:", ["Query subscriber from Database", "Input custom metrics manually"], horizontal=True)
        
        selected_user_data = None
        
        if mode == "Query subscriber from Database":
            # List users from DB
            user_list = df_users_db['user_id'].tolist()[:50]
            user_choice = st.selectbox("Select Subscriber ID (first 50 in DB shown):", user_list)
            
            # Fetch record
            raw_row = df_users_db[df_users_db['user_id'] == user_choice].iloc[0].to_dict()
            selected_user_data = raw_row
            st.success(f"Successfully loaded subscriber records from DB: {user_choice}")
            
        else:
            # Defaults
            selected_user_data = {
                "user_id": "U_MANUAL_1",
                "tenure_days": 120,
                "subscription_tier": "Standard",
                "avg_daily_minutes_last_7d": 12.0,
                "avg_daily_minutes_last_30d": 30.0,
                "sessions_last_7d": 1,
                "sessions_last_30d": 12,
                "avg_completion_rate": 0.25,
                "unique_genres_watched_30d": 3,
                "days_since_last_session": 6,
                "binge_sessions_last_30d": 1,
                "peak_hour_viewing_pct": 70.0,
                "original_content_pct": 45.0,
                "recommendation_click_rate": 0.12
            }
            
        st.markdown("---")
        
        col_inputs, col_results = st.columns([2, 3])
        
        with col_inputs:
            st.markdown("#### Churn What-If Simulator Settings")
            st.write("Sliders are initialized with current subscriber features. Modify values to simulate how actions change risk.")
            
            sim_user_id = selected_user_data.get('user_id', 'UNKNOWN')
            sim_tenure = st.slider("Tenure (Days)", 1, 1500, int(selected_user_data['tenure_days']))
            sim_tier = st.selectbox("Subscription Tier", ["Basic", "Standard", "Premium"], index=["Basic", "Standard", "Premium"].index(selected_user_data['subscription_tier']))
            
            sim_min_30 = st.slider("Avg Daily Minutes (Last 30 Days)", 1.0, 480.0, float(selected_user_data['avg_daily_minutes_last_30d']))
            sim_min_7 = st.slider("Avg Daily Minutes (Last 7 Days)", 0.0, float(sim_min_30 * 1.5), float(np.minimum(selected_user_data['avg_daily_minutes_last_7d'], sim_min_30 * 1.5)))
            
            sim_sess_30 = st.slider("Total Sessions (Last 30 Days)", 1, 120, int(selected_user_data['sessions_last_30d']))
            sim_sess_7 = st.slider("Total Sessions (Last 7 Days)", 0, int(sim_sess_30), int(np.minimum(selected_user_data['sessions_last_7d'], sim_sess_30)))
            
            sim_completion = st.slider("Average Content Completion Rate", 0.0, 1.0, float(selected_user_data['avg_completion_rate']))
            sim_genres = st.slider("Unique Genres Watched", 1, 15, int(selected_user_data['unique_genres_watched_30d']))
            sim_days_since = st.slider("Days Since Last Session", 0, 30, int(selected_user_data['days_since_last_session']))
            sim_binges = st.slider("Binge Sessions (Last 30 Days)", 0, int(sim_sess_30), int(np.minimum(selected_user_data['binge_sessions_last_30d'], sim_sess_30)))
            
            sim_peak = st.slider("Peak Hour Viewing %", 0.0, 100.0, float(selected_user_data['peak_hour_viewing_pct']))
            sim_originals = st.slider("Original Content Viewing %", 0.0, 100.0, float(selected_user_data['original_content_pct']))
            sim_click_rate = st.slider("Recommendation Click Rate", 0.0, 1.0, float(selected_user_data['recommendation_click_rate']))
            
            sim_payload = {
                "user_id": sim_user_id,
                "tenure_days": sim_tenure,
                "subscription_tier": sim_tier,
                "avg_daily_minutes_last_7d": sim_min_7,
                "avg_daily_minutes_last_30d": sim_min_30,
                "sessions_last_7d": sim_sess_7,
                "sessions_last_30d": sim_sess_30,
                "avg_completion_rate": sim_completion,
                "unique_genres_watched_30d": sim_genres,
                "days_since_last_session": sim_days_since,
                "binge_sessions_last_30d": sim_binges,
                "peak_hour_viewing_pct": sim_peak,
                "original_content_pct": sim_originals,
                "recommendation_click_rate": sim_click_rate
            }
            
        with col_results:
            if not predictor.is_model_loaded():
                st.warning("⚠️ ML Model pipeline is untrained.")
            else:
                st.markdown("#### Real-Time Diagnostics & Database Log Trigger")
                
                # Checkbox to choose whether to write simulation prediction to database
                log_to_db = st.checkbox("Log this inference to Database (`predictions` table)", value=True)
                
                with st.spinner("Calculating fatigue risk..."):
                    db = get_db_session() if log_to_db else None
                    try:
                        res = predictor.predict_single(sim_payload, db=db)
                    finally:
                        if db is not None:
                            db.close()
                            
                prob = res['fatigue_probability']
                risk = res['risk_level']
                archetype = res['business_archetype']
                is_fatigued = res['is_fatigued']
                
                status_class = "status-low" if risk == "Low Risk" else "status-medium" if risk == "Medium Risk" else "status-high"
                
                # Gauge visualization
                fig, ax = plt.subplots(figsize=(6, 1.4), facecolor='#111827')
                ax.set_facecolor('#111827')
                
                ax.barh([0], [1.0], color='#1f2937', height=0.4)
                gauge_color = '#10b981' if risk == "Low Risk" else '#f59e0b' if risk == "Medium Risk" else '#ef4444'
                ax.barh([0], [prob], color=gauge_color, height=0.4)
                ax.axvline(predictor.threshold, color='#ffffff', linestyle=':', linewidth=2)
                
                ax.set_xlim(0, 1.0)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.tick_params(colors='#9ca3af')
                
                plt.title(f"Fatigue Probability: {prob*100:.1f}%", color="#ffffff", fontweight='bold', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Diagnostic Info Box
                st.markdown(f"""
                <div style='background-color:#111827; padding:18px; border-radius:12px; border:1px solid #1f2937;'>
                    <table style='width:100%; color:#ffffff;'>
                        <tr>
                            <td><b>Subscriber ID:</b></td>
                            <td><code>{sim_user_id}</code></td>
                        </tr>
                        <tr>
                            <td><b>Risk Level:</b></td>
                            <td><span class='status-pill {status_class}'>{risk}</span></td>
                        </tr>
                        <tr>
                            <td><b>Intervention Action:</b></td>
                            <td>{'🚨 TRIGGER RETENTION ACTION' if is_fatigued else '✅ MONITOR ONLY'}</td>
                        </tr>
                        <tr>
                            <td><b>Business Archetype:</b></td>
                            <td><span style='color:#8b5cf6; font-weight:bold;'>{archetype}</span></td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                
                st.markdown("##### 🎯 Targeted Retention Blueprint")
                if archetype == "Frustrated Browser":
                    st.info("**Strategy — Frustrated Browsers:** High discovery failure. \n\n* **Actions**: A/B test simpler grid layout, trigger a personalized push notification detailing 'Top Picks for You' based on custom genre viewing history, offer a localized discount.")
                elif archetype == "Binge-and-Leave":
                    st.error("**Strategy — Binge-and-Leave:** High monthly binging, ready to drop. \n\n* **Actions**: Auto-queue similar shows immediately after binge, send post-binge email with recommendations for related titles within 24 hours, implement weekly release schedules for flagship original content.")
                elif archetype == "Waning Casual":
                    st.warning("**Strategy — Waning Casuals:** Gradual drop in weekly viewing. \n\n* **Actions**: Trigger 'We missed you' email, send short-form/low-commitment content recommendations to lower re-entry friction, offer subscription pause option instead of cancellation.")
                elif archetype == "Active & Engaged":
                    st.success("**Strategy — Engaged Users:** User displays stable, high-value signals. \n\n* **Actions**: Maintain current experience, pitch early access to premium features, cross-sell annual plans at a discount.")
                
                st.write("")
                
                # Model Explanation (SHAP)
                st.markdown("##### 🔬 SHAP Explainable AI drivers")
                with st.spinner("Generating SHAP explanations..."):
                    shap_exp = predictor.explain_single(sim_payload)
                    
                top_shaps = shap_exp[:7]
                labels = [e["feature"].replace('_', ' ').title() for e in top_shaps]
                vals = [e["shap_value"] for e in top_shaps]
                colors_shap = ['#ef4444' if v > 0 else '#10b981' for v in vals]
                
                fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#111827')
                ax.set_facecolor('#111827')
                
                y_pos = np.arange(len(labels))
                ax.barh(y_pos, vals, align='center', color=colors_shap, edgecolor='black', height=0.6)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#4b5563')
                ax.spines['bottom'].set_color('#4b5563')
                ax.tick_params(colors='#9ca3af')
                ax.axvline(0, color='#9ca3af', linestyle='-', linewidth=0.8)
                ax.set_xlabel("SHAP Impact", color='#9ca3af')
                
                plt.tight_layout()
                st.pyplot(fig)

# =========================================================================
# TAB 3: Database & MLflow Sandbox
# =========================================================================
with tab3:
    st.markdown("### Database & MLflow MLOps Control Center")
    st.write("Manage model runs, explore training history, and generate database records at scale.")
    
    col_gen, col_train = st.columns(2)
    
    with col_gen:
        st.markdown("#### 📁 Relational Data Seeder")
        st.write("Generates synthetic user interaction records and uploads them straight to the `users` table.")
        
        users_count = st.number_input("Number of subscribers to seed in DB:", min_value=1000, max_value=100000, value=15000, step=5000)
        
        if st.button("Generate & Seed Database"):
            with st.spinner("Synthesizing subscribers..."):
                df_seeded = generate_synthetic_data(num_users=users_count)
                
                db = get_db_session()
                try:
                    bulk_insert_users(db, df_seeded.to_dict(orient="records"))
                finally:
                    db.close()
                    
            st.success(f"Successfully populated database `users` table with {users_count} records!")
            # Force reload counts
            st.rerun()
            
    with col_train:
        st.markdown("#### ⚡ MLflow Experiment RETRAINING Pipeline")
        st.write("Runs Bayesian hyperparameter optimization and logs everything (parameters, metrics, and models) to MLflow.")
        
        dataset_mode = st.selectbox("Choose Data Source for Training:", [
            "Users Relational Database (Recommended)",
            "Original ott_train.csv (File Fallback)"
        ])
        
        trials_count = st.slider("Optuna Search Space Trials", 5, 40, 15)
        
        if st.button("Trigger MLflow Training Session"):
            db_flag = (dataset_mode == "Users Relational Database (Recommended)")
            
            if db_flag and db_users_count == 0:
                st.error("Error: Database has no records. Please run the seeder first.")
            else:
                # Dynamically write trial parameter to config yaml
                if os.path.exists("config.yaml"):
                    try:
                        with open("config.yaml", "r") as f:
                            cfg = yaml.safe_load(f)
                        cfg['training']['optuna_trials'] = trials_count
                        with open("config.yaml", "w") as f:
                            yaml.dump(cfg, f)
                    except Exception:
                        pass
                        
                status = st.empty()
                status.info("⏳ Fitting model... Running CV, Optuna hyperparameter searches, and exporting runs. Check console for MLflow logs.")
                
                try:
                    # Run training script logic
                    metrics_res = run_training(use_db=db_flag)
                    
                    status.empty()
                    st.success("🎉 MLflow training run complete! Model pipeline artifact registered.")
                    
                    # Refresh predictor instance
                    predictor = FatiguePredictor()
                    
                    st.json(metrics_res)
                    st.rerun()
                except Exception as e:
                    status.empty()
                    st.error(f"Training session encountered an error: {str(e)}")
                    
    # Training runs logs
    st.markdown("---")
    st.markdown("#### Relational Database Model Runs History (`training_runs` table)")
    st.write("This table captures model validation summaries logged in the database alongside MLflow runs.")
    
    if not training_runs:
        st.info("No database training logs found. Run the training pipeline above to register a run.")
    else:
        st.dataframe(pd.DataFrame(training_runs), use_container_width=True)

# =========================================================================
# TAB 4: Batch Predictor CSV Pipeline
# =========================================================================
with tab4:
    st.markdown("### Batch Classification CSV pipeline")
    st.write("Upload a CSV file of subscriber interactions. Predictions are calculated in batch, written to the database prediction logs, and returned for download.")
    
    uploaded_file = st.file_uploader("Upload Subscriber CSV:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.dataframe(df_upload.head(5))
            
            if not predictor.is_model_loaded():
                st.warning("⚠️ Model pipeline is not loaded.")
            else:
                required_cols = [
                    "tenure_days", "subscription_tier", "avg_daily_minutes_last_7d",
                    "avg_daily_minutes_last_30d", "sessions_last_7d", "sessions_last_30d",
                    "avg_completion_rate", "unique_genres_watched_30d", "days_since_last_session",
                    "binge_sessions_last_30d", "peak_hour_viewing_pct", "original_content_pct",
                    "recommendation_click_rate"
                ]
                
                missing = [col for col in required_cols if col not in df_upload.columns]
                if missing:
                    st.error(f"Uploaded CSV is missing columns: {missing}")
                else:
                    if st.button("Process Batch Predictions"):
                        with st.spinner("Processing..."):
                            db = get_db_session()
                            try:
                                batch_preds = predictor.predict_batch(df_upload, db=db)
                            finally:
                                db.close()
                                
                        st.success("Batch prediction logged & completed!")
                        st.dataframe(batch_preds.head(10))
                        
                        csv = batch_preds.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Classified CSV",
                            data=csv,
                            file_name="classified_subscribers.csv",
                            mime="text/csv"
                        )
                        st.rerun()
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")
