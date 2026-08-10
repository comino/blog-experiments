-- Experiment 02: Setup
-- Database
CREATE DATABASE IF NOT EXISTS exp02_projections;

-- Base table (no projections yet)
CREATE TABLE IF NOT EXISTS exp02_projections.web_analytics_base
(
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String)
)
ENGINE = MergeTree()
ORDER BY (page, timestamp);

-- Table with Projection A (re-sort by country, timestamp)
CREATE TABLE IF NOT EXISTS exp02_projections.web_analytics_proj
(
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String),
    PROJECTION proj_country_time (
        SELECT * ORDER BY (country, timestamp)
    ),
    PROJECTION proj_hourly_stats (
        SELECT
            page,
            toStartOfHour(timestamp) AS hour,
            count() AS hits,
            avg(duration_ms) AS avg_duration,
            sum(duration_ms) AS sum_duration
        GROUP BY page, hour
    )
)
ENGINE = MergeTree()
ORDER BY (page, timestamp);

-- MV target table (AggregatingMergeTree)
CREATE TABLE IF NOT EXISTS exp02_projections.hourly_stats_mv_target
(
    page LowCardinality(String),
    hour DateTime,
    hits AggregateFunction(count, UInt64),
    avg_duration AggregateFunction(avg, UInt32),
    sum_duration AggregateFunction(sum, UInt32)
)
ENGINE = AggregatingMergeTree()
ORDER BY (page, hour);

-- MV (will populate from web_analytics_mv_source)
-- We need a separate source table for MV so we can measure ingest independently
CREATE TABLE IF NOT EXISTS exp02_projections.web_analytics_mv_source
(
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String)
)
ENGINE = MergeTree()
ORDER BY (page, timestamp);

CREATE MATERIALIZED VIEW IF NOT EXISTS exp02_projections.hourly_stats_mv
TO exp02_projections.hourly_stats_mv_target
AS
SELECT
    page,
    toStartOfHour(timestamp) AS hour,
    countState() AS hits,
    avgState(duration_ms) AS avg_duration,
    sumState(duration_ms) AS sum_duration
FROM exp02_projections.web_analytics_mv_source
GROUP BY page, hour;

-- Ingest test tables (created later, clean slate each time)
-- For ingest impact: base_only, base_proj, base_mv
