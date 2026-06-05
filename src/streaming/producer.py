import json
import time
import random
import os
from datetime import datetime
from kafka import KafkaProducer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME = "subscriber_events"

print(f"Starting simulated OTT Event Producer connecting to broker: {KAFKA_BROKER}...")

# Retry connection to broker to tolerate startup lag in containers
producer = None
for attempt in range(10):
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        print("Successfully connected to message broker!")
        break
    except Exception as e:
        print(f"Broker not ready yet (attempt {attempt+1}/10): {e}")
        time.sleep(3)

if not producer:
    print("Could not connect to broker. Exiting simulator.")
    exit(1)

EVENT_TYPES = ["playback_started", "playback_stopped", "recommendation_clicked", "page_viewed"]
TIERS = ["Basic", "Standard", "Premium"]

try:
    while True:
        # Generate event for random subscriber
        user_id = f"U{random.randint(100000, 999999)}"
        event_type = random.choice(EVENT_TYPES)
        
        event = {
            "user_id": user_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "subscription_tier": random.choice(TIERS),
            "tenure_days": random.randint(30, 720),
            "avg_completion_rate": round(random.uniform(0.1, 0.95), 2),
            "minutes_viewed": round(random.uniform(1.0, 120.0), 1)
        }
        
        print(f"Sending event: {event['user_id']} | {event['event_type']}")
        
        producer.send(TOPIC_NAME, event)
        producer.flush()
        
        # Sleep randomly to simulate streaming cadence
        time.sleep(random.uniform(0.5, 3.0))

except KeyboardInterrupt:
    print("Simulated Producer stopped by user.")
except Exception as e:
    print(f"Producer error: {e}")
