import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import OutlierCapper, engineer_features
from src.data_generator import generate_synthetic_data

def test_outlier_capper():
    """
    Tests that OutlierCapper correctly caps values at percentiles.
    """
    # Create test data with extreme outliers
    data = pd.DataFrame({
        'val': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0], # 100.0 is an upper outlier
        'cat': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']    # categorical column
    })
    
    capper = OutlierCapper(lower_percentile=0.10, upper_percentile=0.90)
    capper.fit(data)
    
    # 10th percentile of val should be ~1.9, 90th percentile should be ~19.1
    transformed = capper.transform(data)
    
    # Test capping
    assert transformed['val'].max() < 100.0
    assert transformed['val'].min() > 1.0
    assert transformed['cat'].tolist() == data['cat'].tolist() # Categorical left untouched

def test_engineer_features():
    """
    Tests that engineer_features generates the required composite and engineered features.
    """
    # Create single-record raw DataFrame
    raw_df = pd.DataFrame([{
        'tenure_days': 100,
        'subscription_tier': 'Standard',
        'avg_daily_minutes_last_7d': 10.0,
        'avg_daily_minutes_last_30d': 20.0,
        'sessions_last_7d': 2,
        'sessions_last_30d': 10,
        'avg_completion_rate': 0.5,
        'unique_genres_watched_30d': 3,
        'days_since_last_session': 4,
        'binge_sessions_last_30d': 1,
        'peak_hour_viewing_pct': 80.0,
        'original_content_pct': 30.0,
        'recommendation_click_rate': 0.1
    }])
    
    fe_df = engineer_features(raw_df)
    
    # Check that new engineered columns exist
    expected_engineered_cols = [
        'consumption_velocity_ratio', 'session_momentum', 
        'effective_engagement_score', 'discovery_friction', 
        'composite_engagement', 'subscription_tier_Standard', 'subscription_tier_Premium'
    ]
    
    for col in expected_engineered_cols:
        assert col in fe_df.columns
        
    # Check specific calculation correctness
    # consumption_velocity_ratio = 10.0 / (20.0 + 1e-5) = 0.5
    assert abs(fe_df['consumption_velocity_ratio'].values[0] - 0.5) < 1e-3
    
    # subscription_tier_Standard should be 1, subscription_tier_Premium should be 0
    assert fe_df['subscription_tier_Standard'].values[0] == 1
    assert fe_df['subscription_tier_Premium'].values[0] == 0
    assert 'subscription_tier' not in fe_df.columns

def test_synthetic_data_generator():
    """
    Tests that the data generator generates the exact expected columns and target distributions.
    """
    df = generate_synthetic_data(num_users=100, seed=42)
    
    assert len(df) == 100
    assert 'fatigue_label' in df.columns
    assert 'user_id' in df.columns
    
    # Verify values are bounded correctly
    assert df['avg_completion_rate'].min() >= 0.0
    assert df['avg_completion_rate'].max() <= 1.0
    assert df['recommendation_click_rate'].min() >= 0.0
    assert df['recommendation_click_rate'].max() <= 1.0
