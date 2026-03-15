# OTT User Fatigue Analysis

This project analyzes user engagement data from an OTT (Over-The-Top) streaming service to predict user fatigue, defined as a significant drop in engagement that puts a user at risk of churning.

The analysis involves:
1.  **Exploratory Data Analysis (EDA):** Visualizing distributions, correlations, and class balance.
2.  **Feature Engineering:** Creating 20 new features to capture complex user behaviors related to engagement decay, platform interaction, and content consumption patterns.
3.  **Model Development:**
    *   Establishing baseline models (Logistic Regression, untuned XGBoost).
    *   Hyperparameter tuning of an XGBoost classifier using Optuna with 5-fold cross-validation.
4.  **Model Evaluation:**
    *   Assessing performance using ROC AUC, calibration curves, and precision-recall analysis.
    *   Optimizing the decision threshold for flagging at-risk users.
5.  **Explainability (SHAP):**
    *   Using SHAP (SHapley Additive exPlanations) to understand the key drivers of the model's predictions.
    *   Identifying which features have the most impact on fatigue risk.
6.  **Business Archetypes:**
    *   Segmenting at-risk users into actionable archetypes based on their behavior (e.g., "Frustrated Browsers", "Binge-and-Leave").
    *   Proposing targeted retention strategies for each archetype.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the notebook:**
    Open and run the `SOZO_Analysis.ipynb` notebook in a Jupyter environment.

## Files

*   `ott_train.csv`: Training data with user features and `fatigue_label`.
*   `ott_test.csv`: Test data for final predictions.
*   `SOZO_Analysis.ipynb`: The main Jupyter Notebook containing all analysis and modeling code.
*   `SOZO_Predictions.csv`: The output file with predicted fatigue probabilities for the test set.
*   `requirements.txt`: A list of required Python packages.
