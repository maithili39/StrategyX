from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    PushSource,
    ValueType,
)
from feast.infra.offline_stores.postgres_source import PostgreSQLSource
from feast.types import Float32, Int64, String

# 1. Define Entity
subscriber = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    description="Unique user identifier",
)

# 2. Define PostgreSQL Offline Source (reads historical features)
subscriber_activity_source = PostgreSQLSource(
    name="subscriber_activity_source",
    connection_string="postgresql://postgres:postgres_password@db:5432/strategyx",
    table="users",
    timestamp_field="timestamp",  # Since SQLite database doesn't have timestamp in 'users', we can add it or mock it if needed.
    created_timestamp_column="created_timestamp",
)

# 3. Define Push Source (reads real-time events streamed from Kafka/Redpanda consumer)
subscriber_activity_push_source = PushSource(
    name="subscriber_activity_push_source",
    batch_source=subscriber_activity_source,
)

# 4. Define Feature View
subscriber_activity_fv = FeatureView(
    name="subscriber_activity",
    entities=[subscriber],
    ttl=timedelta(days=365),
    schema=[
        Field(name="tenure_days", dtype=Int64),
        Field(name="subscription_tier", dtype=String),
        Field(name="avg_daily_minutes_last_7d", dtype=Float32),
        Field(name="avg_daily_minutes_last_30d", dtype=Float32),
        Field(name="sessions_last_7d", dtype=Int64),
        Field(name="sessions_last_30d", dtype=Int64),
        Field(name="avg_completion_rate", dtype=Float32),
        Field(name="unique_genres_watched_30d", dtype=Int64),
        Field(name="days_since_last_session", dtype=Int64),
        Field(name="binge_sessions_last_30d", dtype=Int64),
        Field(name="peak_hour_viewing_pct", dtype=Float32),
        Field(name="original_content_pct", dtype=Float32),
        Field(name="recommendation_click_rate", dtype=Float32),
    ],
    online=True,
    source=subscriber_activity_push_source,
)
