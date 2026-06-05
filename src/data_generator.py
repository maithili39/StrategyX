import argparse
import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_users=10000, seed=42):
    """
    Generates high-fidelity synthetic OTT user interaction data matching the original dataset's columns
    and statistical properties, while modeling fatigue churn signals.
    """
    np.random.seed(seed)
    
    # 1. User IDs
    user_ids = [f"U{str(i).zfill(6)}" for i in range(1, num_users + 1)]
    
    # 2. Tenure days (Exponential/Log-normal-like distribution)
    tenure_days = np.random.exponential(scale=200, size=num_users) + 5
    tenure_days = np.clip(tenure_days, 1, 1500).astype(int)
    
    # 3. Subscription tier
    tiers = ['Basic', 'Standard', 'Premium']
    subscription_tier = np.random.choice(tiers, size=num_users, p=[0.45, 0.40, 0.15])
    
    # 4. Latent fatigue probability (to construct correlated behaviors)
    # Higher base risk for newer users, lower for long-tenured loyal users
    base_risk = 0.3 * np.exp(-tenure_days / 365) + 0.1
    # Add random individual variance to risk profile
    latent_fatigue = np.clip(np.random.normal(loc=base_risk, scale=0.15, size=num_users), 0.01, 0.99)
    
    # 5. Days since last session: Fatigue is heavily correlated with recency gap
    # Engaged users: mostly 0-3 days. Fatigued: mostly 4-30 days
    days_since_last_session = np.zeros(num_users, dtype=int)
    for i in range(num_users):
        if latent_fatigue[i] > 0.4:
            days_since_last_session[i] = np.random.geometric(p=0.15) - 1
        else:
            days_since_last_session[i] = np.random.geometric(p=0.6) - 1
    days_since_last_session = np.clip(days_since_last_session, 0, 30)
    
    # 6. Avg daily minutes last 30d
    # Log-normal distribution of watch times
    avg_daily_minutes_last_30d = np.random.lognormal(mean=3.2, sigma=0.8, size=num_users)
    avg_daily_minutes_last_30d = np.clip(avg_daily_minutes_last_30d, 1.0, 480.0)
    
    # 7. Avg daily minutes last 7d
    # Fatigued users have a steep decline in recent minutes compared to monthly
    avg_daily_minutes_last_7d = np.zeros(num_users)
    for i in range(num_users):
        if latent_fatigue[i] > 0.45:
            # Drop in engagement
            multiplier = np.random.beta(a=1, b=4)
        else:
            # Active/growing engagement
            multiplier = np.random.normal(loc=1.0, scale=0.25)
        avg_daily_minutes_last_7d[i] = avg_daily_minutes_last_30d[i] * multiplier
    avg_daily_minutes_last_7d = np.clip(avg_daily_minutes_last_7d, 0.0, 480.0)
    
    # 8. Sessions last 30d
    # Correlated with minutes. Average session is ~30-45 minutes.
    sessions_last_30d = (avg_daily_minutes_last_30d * 30 / np.random.normal(35, 10, size=num_users)).astype(int)
    sessions_last_30d = np.clip(sessions_last_30d, 1, 120)
    
    # 9. Sessions last 7d
    # Similar to minutes multiplier
    sessions_last_7d = np.zeros(num_users, dtype=int)
    for i in range(num_users):
        ratio = avg_daily_minutes_last_7d[i] / (avg_daily_minutes_last_30d[i] + 1e-5)
        sessions_last_7d[i] = int(sessions_last_30d[i] * (ratio / 4.0 + np.random.normal(0.0, 0.05)))
    sessions_last_7d = np.clip(sessions_last_7d, 0, 30)
    
    # Adjust days_since_last_session to be consistent with sessions
    for i in range(num_users):
        if sessions_last_7d[i] > 0 and days_since_last_session[i] > 6:
            days_since_last_session[i] = np.random.choice([0, 1, 2])
            
    # 10. Avg completion rate (0.0 to 1.0)
    # At-risk users leave videos early (frustrated browsing or low quality)
    avg_completion_rate = np.zeros(num_users)
    for i in range(num_users):
        if latent_fatigue[i] > 0.4:
            avg_completion_rate[i] = np.random.beta(a=1.5, b=6)
        else:
            avg_completion_rate[i] = np.random.beta(a=5, b=2.5)
    avg_completion_rate = np.clip(avg_completion_rate, 0.01, 1.0)
    
    # 11. Unique genres watched last 30d (1 to 15)
    # Active users watch a variety, fatigued users watch very few or are exhausted
    unique_genres_watched_30d = np.zeros(num_users, dtype=int)
    for i in range(num_users):
        if latent_fatigue[i] > 0.5:
            unique_genres_watched_30d[i] = np.random.binomial(n=15, p=0.2) + 1
        else:
            unique_genres_watched_30d[i] = np.random.binomial(n=15, p=0.45) + 1
    unique_genres_watched_30d = np.clip(unique_genres_watched_30d, 1, 15)
    
    # 12. Binge sessions last 30d
    # Correlated with total sessions and minutes.
    binge_sessions_last_30d = np.zeros(num_users, dtype=int)
    for i in range(num_users):
        p_binge = 0.25 if latent_fatigue[i] > 0.4 else 0.45
        binge_sessions_last_30d[i] = np.random.binomial(n=sessions_last_30d[i], p=p_binge)
    
    # 13. Peak hour viewing percentage (0 to 100)
    peak_hour_viewing_pct = np.random.normal(loc=72.0, scale=12.0, size=num_users)
    peak_hour_viewing_pct = np.clip(peak_hour_viewing_pct, 10.0, 100.0)
    
    # 14. Original content percentage (0 to 100)
    # Users watching originals are more sticky (less fatigued)
    original_content_pct = np.zeros(num_users)
    for i in range(num_users):
        if latent_fatigue[i] > 0.45:
            original_content_pct[i] = np.random.normal(loc=35.0, scale=15.0)
        else:
            original_content_pct[i] = np.random.normal(loc=55.0, scale=15.0)
    original_content_pct = np.clip(original_content_pct, 0.0, 100.0)
    
    # 15. Recommendation click rate (0 to 1.0)
    # Low click rate means discovery friction -> fatigue
    recommendation_click_rate = np.zeros(num_users)
    for i in range(num_users):
        if latent_fatigue[i] > 0.4:
            recommendation_click_rate[i] = np.random.beta(a=1.2, b=8.0)
        else:
            recommendation_click_rate[i] = np.random.beta(a=3.5, b=5.0)
    recommendation_click_rate = np.clip(recommendation_click_rate, 0.0, 1.0)
    
    # 16. Determine fatigue label
    # Score metrics that indicate fatigue
    score = (
        0.35 * (days_since_last_session / 30.0)
        + 0.20 * (1.0 - avg_completion_rate)
        + 0.15 * (1.0 - recommendation_click_rate)
        + 0.20 * (1.0 - np.minimum(1.0, avg_daily_minutes_last_7d / (avg_daily_minutes_last_30d + 1e-5)))
        + 0.10 * np.exp(-tenure_days / 180.0)
    )
    
    # Probabilistic label assignment based on score threshold
    prob = 1.0 / (1.0 + np.exp(-15.0 * (score - 0.45)))
    fatigue_label = np.random.binomial(n=1, p=prob)
    
    # Form dataframe
    df = pd.DataFrame({
        'user_id': user_ids,
        'tenure_days': tenure_days,
        'subscription_tier': subscription_tier,
        'avg_daily_minutes_last_7d': np.round(avg_daily_minutes_last_7d, 2),
        'avg_daily_minutes_last_30d': np.round(avg_daily_minutes_last_30d, 2),
        'sessions_last_7d': sessions_last_7d,
        'sessions_last_30d': sessions_last_30d,
        'avg_completion_rate': np.round(avg_completion_rate, 3),
        'unique_genres_watched_30d': unique_genres_watched_30d,
        'days_since_last_session': days_since_last_session,
        'binge_sessions_last_30d': binge_sessions_last_30d,
        'peak_hour_viewing_pct': np.round(peak_hour_viewing_pct, 1),
        'original_content_pct': np.round(original_content_pct, 1),
        'recommendation_click_rate': np.round(recommendation_click_rate, 3),
        'fatigue_label': fatigue_label
    })
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic OTT Fatigue Data Generator")
    parser.add_argument("--num-rows", type=int, default=50000, help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save generated CSVs")
    parser.add_argument("--to-db", action="store_true", help="Write generated records to database")
    args = parser.parse_args()
    
    print(f"Generating {args.num_rows} synthetic user fatigue records...")
    df = generate_synthetic_data(num_users=args.num_rows, seed=args.seed)
    
    # Train/Test Split (80/20)
    train_size = int(args.num_rows * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:].copy().drop(columns=['fatigue_label'])
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_file = os.path.join(args.output_dir, "ott_train_large.csv")
    test_file = os.path.join(args.output_dir, "ott_test_large.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Generation successful!")
    print(f" - Train set saved to: {train_file} (shape: {train_df.shape})")
    print(f" - Test set saved to: {test_file} (shape: {test_df.shape})")
    
    if args.to_db:
        print("Writing generated users to database...")
        import sys
        # Ensure parent directory is in path for module imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.db.database import SessionLocal
        from src.db.crud import init_db, bulk_insert_users
        
        init_db()
        db = SessionLocal()
        try:
            bulk_insert_users(db, df.to_dict(orient="records"))
            print(f"Successfully seeded database with {len(df)} users.")
        finally:
            db.close()
