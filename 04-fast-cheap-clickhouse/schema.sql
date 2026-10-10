CREATE DATABASE IF NOT EXISTS exploration;

CREATE TABLE IF NOT EXISTS exploration.taxi_zones
(
    location_id UInt16,
    borough LowCardinality(String),
    zone LowCardinality(String),
    service_zone LowCardinality(String)
)
ENGINE = MergeTree ORDER BY location_id;

CREATE TABLE IF NOT EXISTS exploration.trips
(
    source_month Date COMMENT 'Month of the official source file, not an event timestamp',
    pickup_datetime DateTime64(6, 'UTC') COMMENT 'TLC NYC local wall-clock value stored without timezone conversion; not a UTC event instant',
    dropoff_datetime DateTime64(6, 'UTC') COMMENT 'Same local wall-clock convention as pickup_datetime',
    pickup_location_id UInt16,
    dropoff_location_id UInt16,
    pickup_borough LowCardinality(String),
    pickup_zone LowCardinality(String),
    dropoff_borough LowCardinality(String),
    dropoff_zone LowCardinality(String),
    passenger_count Nullable(Float64),
    trip_distance Float64 COMMENT 'Miles; raw records may contain invalid values',
    fare_amount Float64 COMMENT 'USD, base fare',
    tip_amount Float64 COMMENT 'USD; recorded card tips only, cash tips are not represented',
    tolls_amount Float64 COMMENT 'USD',
    total_amount Float64 COMMENT 'USD, recorded total; cash tips are excluded',
    payment_type UInt8 COMMENT '1 card, 2 cash, 3 no charge, 4 dispute, 5 unknown, 6 voided',
    vendor_id UInt8,
    rate_code_id Nullable(Float64)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(source_month)
ORDER BY (pickup_location_id, pickup_datetime)
SETTINGS index_granularity = 8192;
