-- ClickHouse Schema for Experiment 03: LLM Query Oneshot
-- Web Analytics style: events + users

CREATE TABLE events
(
    event_id       UInt64,
    timestamp      DateTime64(3, 'UTC'),
    user_id        UInt64,
    event_type     LowCardinality(String),
    page           String,
    duration_ms    Nullable(UInt32),
    properties     Map(String, String),
    tags           Array(String),
    country        LowCardinality(String),
    device         LowCardinality(String),
    revenue_cents  Nullable(Int64)
)
ENGINE = MergeTree()
ORDER BY (timestamp, user_id)
PARTITION BY toYYYYMM(timestamp);

CREATE TABLE users
(
    user_id    UInt64,
    created_at DateTime64(3, 'UTC'),
    plan       LowCardinality(String),
    email      String,
    age        Nullable(UInt8),
    country    LowCardinality(String),
    metadata   Map(String, String)
)
ENGINE = MergeTree()
ORDER BY user_id;
