# RetentionIQ: Production-Grade OTT Subscriber Fatigue Prediction System

RetentionIQ is an end-to-end enterprise machine learning system designed to predict subscriber fatigue (churn risk) for Over-The-Top (OTT) streaming platforms. It processes raw engagement indicators to identify behavior decay, classifies users into actionable business segments, and logs predictions for automated retention triggers.

This codebase is structured as a modular, production-ready microservice architecture.

---

## Production Architecture

1. **Relational Database Layer (`src/db/`)**:
   - Engineered with SQLAlchemy ORM to support robust database connections.
   - Connected to SQLite by default for zero-configuration local runs, with built-in configurations to scale to PostgreSQL.
   - Tables log user interaction data (`users` table), real-time inferences history (`predictions` table), and training validation results (`training_runs` table).
2. **MLflow Experiment Tracking (MLOps)**:
   - Tracks training runs, log parameters (Optuna search trials), and saves metrics (ROC-AUC, F1, Brier Score).
   - Serializes and registers model pipeline binaries directly into the MLflow model store.
3. **REST API Service (`src/api.py` & `src/auth.py`)**:
   - Built with FastAPI to serve real-time predictions (`/predict` and `/predict/batch`), model explanations (`/explain` via SHAP), and subscriber lists (`/users/sample`).
   - Secured with OAuth2 Password Bearer flow and JWT access tokens to prevent unauthorized access.
   - Includes Pydantic V2 input validation on numerical features to reject out-of-bounds or malformed telemetry.
   - Utilizes asynchronous background tasks to handle high-volume batch prediction requests non-blockingly.
4. **Feast Feature Store (`src/feature_store/`)**:
   - Manages features in a centralized register to ensure training-serving consistency.
   - Definitions map subscriber engagement and usage aggregates.
5. **Real-time Streaming Ingestion (`src/streaming/`)**:
   - Listens to active user events from Kafka/Redpanda topic.
   - Automatically issues real-time inferences and broadcasts them back.
6. **React Dashboard Frontend (`frontend/`)**:
   - Modern dashboard built with React, Vite, and Tailwind CSS using a premium violet and pink light-mode color theme.
   - Features an interactive Subscriber Churn Simulator allowing users to load real subscriber statistics directly from the database, adjust telemetry parameters, simulate risk, view SHAP explainers, log feedback conversions, and track logs in real-time.
   - Features a live WebSocket telemetry feed to stream active user predictions.
7. **Quality Assurance (`tests/`)**:
   - Automated unit and integration tests (`pytest`) covering feature engineering, database CRUD operations, and API security.
   - GitHub Actions CI pipeline automatically validating code builds on commit.

---

## Project Structure

- `src/`: Core Python packages.
  - `db/`: Database configuration, ORM models, and CRUD operations.
  - `feature_store/`: Feast feature views and registry configuration.
  - `streaming/`: Kafka event consumers and WebSocket routers.
  - `auth.py`: OAuth2 authentication handlers, password hashing, and token issuance.
  - `logger.py`: JSON structured logger setup.
  - `data_generator.py`: Synthetic database generator to scale test subscriber bases.
  - `features.py`: Feature engineering functions and capping transformers.
  - `train.py`: Model tuning and MLflow training pipeline.
  - `predict.py`: Inference wrapper with SHAP and business archetype mapping.
  - `retrain.py`: Automated retraining execution script.
  - `api.py`: Secured FastAPI server.
- `frontend/`: React + Vite application.
  - `src/App.tsx`: Main dashboard application.
  - `src/index.css`: Core design system and tokens.
- `tests/`: Automated unit and API security tests folder.
- `Dockerfile`: Multi-stage Docker config for containerized execution.
- `docker-compose.yml`: Multi-container orchestrator configuration.
- `config.yaml`: Centralized configuration variables.
- `requirements.txt`: Python package dependencies.

---

## Quick Start

### 1. Install and Configure
Clone the repository, create a virtual environment, and install dependencies:
```bash
pip install -r requirements.txt
```
Copy the environment template file:
```bash
copy .env.example .env
```

### 2. Seed Database
Simulate a database of subscribers and write them to the SQL database:
```bash
python src/data_generator.py --num-rows 15000 --to-db
```

### 3. Run Training
Load training data from the database, search parameters with Optuna, and log parameters/metrics/artifacts to MLflow:
```bash
python src/train.py --use-db
```

### 4. Start REST API Backend
Start the FastAPI server:
```bash
uvicorn src.api:app --reload --port 8000
```
View Swagger API documentation and authenticate at: `http://127.0.0.1:8000/docs`
- Default credentials: `strategyx_admin` / `strategyx_password`

### 5. Start Dashboard Frontend
From the `frontend` folder, install packages and start the Vite development server:
```bash
cd frontend
npm install
npm run dev
```
Open your browser to: `http://localhost:5173`

### 6. Run Unit Tests
Execute the pytest suite:
```bash
pytest tests/
```
