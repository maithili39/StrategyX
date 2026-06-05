# StrategyX: Production-Grade OTT Subscriber Fatigue Prediction System

StrategyX is an end-to-end machine learning system designed to predict subscriber fatigue (churn risk) for Over-The-Top (OTT) streaming platforms. It processes raw engagement indicators to identify behavior decay, classifies users into actionable business segments, and logs predictions for automated retention triggers.

This codebase has been scaled from a prototype Jupyter Notebook into a modular, production-ready microservice architecture.

---

## 🏗️ Production Architecture

1. **Relational Database Layer (`src/db/`)**:
   - Engineered with **SQLAlchemy ORM** to support robust database connections.
   - Connected to **SQLite** by default for zero-configuration local runs, with built-in configurations to seamlessly scale to **PostgreSQL**.
   - Tables log: raw interaction data (`users` table), real-time inferences history (`predictions` table), and training validation results (`training_runs` table).
2. **MLflow Experiment Tracking (MLOps)**:
   - Tracks training runs, log parameters (Optuna search trials), and saves metrics (ROC-AUC, F1, Brier Score).
   - Serializes and registers model pipeline binaries directly into the MLflow model store.
3. **REST API Service (`src/api.py` & `src/auth.py`)**:
   - Built with **FastAPI** to serve real-time predictions (`/predict` and `/predict/batch`) and model explanations (`/explain` via SHAP).
   - Secured with **OAuth2 Password Bearer flow and JWT access tokens** to prevent unauthorized access.
   - Includes **Pydantic V2 input validation** on numerical features to reject out-of-bounds/malformed telemetry.
   - Utilizes asynchronous **BackgroundTasks** to handle high-volume batch prediction requests non-blockingly without database locks or server timeouts.
   - Outputs structured **JSON logs** containing performance and audit attributes for aggregators.
4. **Streamlit Analytical Dashboard (`src/app.py`)**:
   - Displays real-time database KPIs, segment proportions, API prediction histories, and historical training sessions.
   - Includes a simulator sandbox to simulate user behaviors and view real-time changes in churn probability.
5. **Quality Assurance (`tests/`)**:
   - Automated unit and integration tests (`pytest`) covering feature engineering, database CRUD operations, and API token security checks.
   - **GitHub Actions** CI pipeline automatically validating code builds on commit.

---

## 📂 Project Structure

- `src/`: Core Python packages.
  - `db/`: Database configuration, ORM models, and CRUD operations.
  - `auth.py`: OAuth2 authentication handlers, password hashing, and token issuance.
  - `logger.py`: JSON structured logger setup.
  - `data_generator.py`: Synthetic database generator to scale test subscriber bases.
  - `features.py`: Feature engineering functions and capping transformers.
  - `train.py`: Model tuning and MLflow training pipeline.
  - `predict.py`: Inference wrapper with SHAP and business archetype mapping.
  - `api.py`: Secured FastAPI server.
  - `app.py`: Streamlit control center dashboard.
- `tests/`: Automated unit and API security tests folder.
- `Dockerfile`: Multi-stage Docker config for containerized execution.
- `.github/workflows/ci.yml`: GitHub Actions automated pytest checker.
- `config.yaml`: Centralized configuration variables.
- `.env.example`: Template for database and MLflow environment configurations.
- `requirements.txt`: Package dependencies.

---

## ⚡ Quick Start

### 1. Install & Configure
Clone the repository, create a virtual environment, and install dependencies:
```bash
pip install -r requirements.txt
```
Copy the environment template file:
```bash
copy .env.example .env
```

### 2. Seed Database
Simulate a database of 15,000 OTT subscribers and write them to the SQL database:
```bash
python src/data_generator.py --num-rows 15000 --to-db
```

### 3. Run Training (with DB & MLflow)
Load training data from the database, search parameters with Optuna, and log parameters/metrics/artifacts to MLflow:
```bash
python src/train.py --use-db
```

### 4. Start Dashboard
Launch the control center:
```bash
streamlit run src/app.py
```

### 5. Launch REST API
Start the FastAPI server:
```bash
uvicorn src.api:app --reload
```
View Swagger API documentation and authenticate at: `http://127.0.0.1:8000/docs`
- Use the **Authorize** padlock button.
- Default Admin credentials:
  - **Username**: `strategyx_admin`
  - **Password**: `strategyx_password`

### 6. Open MLflow Dashboard
Inspect training metrics and serialized model binaries:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open your browser to: `http://127.0.0.1:5000`

### 7. Run Unit Tests
Execute the pytest suite:
```bash
pytest tests/
```
