import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps numeric features at given percentile bounds.
    Fits thresholds on the training set to prevent leakage.
    """
    def __init__(self, lower_percentile=0.01, upper_percentile=0.99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.num_cols_ = None
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        # Identify numeric columns
        self.num_cols_ = X.select_dtypes(include=np.number).columns
        # Calculate bounds
        self.lower_bounds_ = X[self.num_cols_].quantile(self.lower_percentile)
        self.upper_bounds_ = X[self.num_cols_].quantile(self.upper_percentile)
        return self

    def transform(self, X):
        X_out = X.copy()
        if self.num_cols_ is not None:
            # Apply clipping
            X_out[self.num_cols_] = X_out[self.num_cols_].clip(
                self.lower_bounds_, self.upper_bounds_, axis=1
            )
        return X_out

def engineer_features(df):
    """
    Engineers 20 behavioral features indicating fatigue/churn risk from raw OTT data.
    """
    df = df.copy()

    # Ensure correct data types
    numeric_cols = [
        'tenure_days', 'avg_daily_minutes_last_7d', 'avg_daily_minutes_last_30d',
        'sessions_last_7d', 'sessions_last_30d', 'avg_completion_rate',
        'unique_genres_watched_30d', 'days_since_last_session',
        'binge_sessions_last_30d', 'peak_hour_viewing_pct',
        'original_content_pct', 'recommendation_click_rate'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Consumption velocity: ratio of 7d to 30d daily minutes (<1 = declining)
    df['consumption_velocity_ratio'] = (
        df['avg_daily_minutes_last_7d'] / (df['avg_daily_minutes_last_30d'] + 1e-5)
    )

    # 2. Session momentum: ratio of 7d to 30d session rate (<1 = slowing down)
    df['session_momentum'] = (
        df['sessions_last_7d'] / (df['sessions_last_30d'] + 1e-5)
    )

    # 3. Effective engagement: high minutes × high completion = truly engaged
    df['effective_engagement_score'] = (
        df['avg_daily_minutes_last_30d'] * df['avg_completion_rate']
    )

    # 4. UX friction: how often the algorithm fails the user
    df['discovery_friction'] = 1.0 - df['recommendation_click_rate']

    # 5. Catalog reliance risk: binging non-original content = replaceable
    df['catalog_reliance_risk'] = (
        (100.0 - df['original_content_pct']) * df['binge_sessions_last_30d']
    )

    # 6. Composite engagement index
    df['composite_engagement'] = (
        df['avg_daily_minutes_last_7d'] / 300.0 * 0.30 +
        df['avg_completion_rate']                * 0.30 +
        df['recommendation_click_rate']          * 0.20 +
        (1.0 - df['days_since_last_session'] / 30.0) * 0.20
    )

    # 7. Dormancy × dissatisfaction
    df['dormancy_x_dissatisfaction'] = (
        df['days_since_last_session'] * (1.0 - df['avg_completion_rate'])
    )

    # 8. Recency × low sessions
    df['recency_x_low_sessions'] = (
        df['days_since_last_session'] * (1.0 / (df['sessions_last_7d'] + 1.0))
    )

    # 9. Recency risk squared
    df['recency_risk_sq'] = df['days_since_last_session'] ** 2

    # 10. Platform stickiness
    df['platform_stickiness'] = (
        (df['original_content_pct'] / 100.0) * df['recommendation_click_rate']
    )

    # 11. Log days since last session
    df['log_days_since'] = np.log1p(df['days_since_last_session'])

    # 12. Low value signal
    df['low_value_signal'] = (
        (1.0 - df['original_content_pct'] / 100.0) * (1.0 - df['recommendation_click_rate'])
    )

    # 13. New user churn risk
    df['new_user_churn_risk'] = (
        (1.0 / (df['tenure_days'] + 1.0)) * df['days_since_last_session']
    )

    # 14. Minutes per session delta
    df['minutes_per_session_7d']  = (df['avg_daily_minutes_last_7d'] * 7.0
                                      / (df['sessions_last_7d'] + 1.0))
    df['minutes_per_session_30d'] = (df['avg_daily_minutes_last_30d'] * 30.0
                                      / (df['sessions_last_30d'] + 1.0))
    df['session_quality_change']  = (df['minutes_per_session_7d']
                                      - df['minutes_per_session_30d'])

    # 15. Daily session frequency
    df['session_freq_7d'] = df['sessions_last_7d'] / 7.0

    # 16. Catalog exhaustion
    df['catalog_exhaustion'] = df['unique_genres_watched_30d'] / 15.0

    # 17. Dormant flag
    df['is_dormant_7d'] = (df['sessions_last_7d'] == 0).astype(int)

    # 18. Binge rate
    df['binge_rate'] = df['binge_sessions_last_30d'] / (df['sessions_last_30d'] + 1.0)

    # Categorical encoding: subscription_tier
    # We implement dummy variable creation manually or with pandas, but explicitly map it to avoid missing cols during inference
    tiers = ['Basic', 'Standard', 'Premium']
    for tier in tiers[1:]:  # drop_first = True, so we drop the first category ('Basic')
        col_name = f'subscription_tier_{tier}'
        if 'subscription_tier' in df.columns:
            df[col_name] = (df['subscription_tier'] == tier).astype(int)
        else:
            df[col_name] = 0

    if 'subscription_tier' in df.columns:
        df = df.drop(columns=['subscription_tier'])

    return df
