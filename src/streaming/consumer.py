import json
import os
import time
import pandas as pd
from datetime import datetime
from kafka import KafkaConsumer
from feast import FeatureStore

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME = "subscriber_events"

print(f"Starting simulated OTT Event Consumer connecting to broker: {KAFKA_BROKER}...")

# Initialize Feast Feature Store
# Assumes feature_store.yaml is in 'src/feature_store/'
try:
    store = FeatureStore(repo_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "feature_store"))
    print("Feast Feature Store initialized successfully!")
except Exception as e:
    print(f"Failed to initialize Feast Feature Store: {e}")
    # Workaround if repo_path is not found or config fails
    store = None

# Retry connection to broker to tolerate startup lag in containers
consumer = None
for attempt in range(10):
    try:
        consumer = KafkaConsumer(
            TOPIC_NAME,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="strategyx_consumer_group"
        )
        print("Successfully connected to message broker!")
        break
    except Exception as e:
        print(f"Broker not ready yet (attempt {attempt+1}/10): {e}")
        time.sleep(3)

if not consumer:
    print("Could not connect to broker. Exiting consumer.")
    exit(1)

print("Listening for subscriber events...")

try:
    for message in consumer:
        event = message.value
        user_id = event["user_id"]
        print(f"Received event for subscriber: {user_id} | Type: {event['event_type']}")
        
        if store:
            # Map event parameters to Feast schema features
            # Real-world processors calculate rolling statistics; here we simulate/update Feast online records
            feature_data = {
                "user_id": [user_id],
                "timestamp": [pd.to_datetime(event["timestamp"])],
                "tenure_days": [event.get("tenure_days", 100)],
                "subscription_tier": [event.get("subscription_tier", "Standard")],
                "avg_daily_minutes_last_7d": [round(event.get("minutes_viewed", 10.0) * 0.15, 2)],
                "avg_daily_minutes_last_30d": [round(event.get("minutes_viewed", 10.0) * 0.3, 2)],
                "sessions_last_7d": [1],
                "sessions_last_30d": [10],
                "avg_completion_rate": [event.get("avg_completion_rate", 0.5)],
                "unique_genres_watched_30d": [3],
                "days_since_last_session": [0], # Just watched
                "binge_sessions_last_30d": [1],
                "peak_hour_viewing_pct": [65.0],
                "original_content_pct": [40.0],
                "recommendation_click_rate": [0.1],
            }
            
            df = pd.DataFrame(feature_data)
            
            try:
                # Push the real-time features to Feast online store (PostgreSQL)
                store.push("subscriber_activity", df)
                print(f"Pushed real-time features to Feast online store for user: {user_id}")
            except Exception as fe:
                print(f"Feast Push error for user {user_id}: {fe}")
        else:
            print("Feast Feature Store unavailable. Event logged only.")

except KeyboardInterrupt:
    print("Consumer stopped by user.")
except Exception as e:
    print(f"Consumer error: {e}")
