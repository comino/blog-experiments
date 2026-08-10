-- =============================================================
-- Exp02 Data Generator: 200M Web Analytics Events
-- =============================================================
-- Environment: ClickHouse 25.11.3.54, single node, 32 GB RAM
-- Reproduces the dataset used in the projections vs MV benchmark
--
-- Schema: 6 columns simulating web analytics pageview events
--   - timestamp: ~1 year span (2024), events distributed across time
--   - user_id: random UInt32 (simulating unique users)
--   - page: ~500 distinct pages (/page/0 .. /page/499)
--   - duration_ms: random 0-10000ms page view duration
--   - country: ~50 distinct countries
--   - device_type: 3 values (desktop, mobile, tablet)
--
-- Total: 200,000,000 rows

CREATE DATABASE IF NOT EXISTS exp02_projections;

-- Base table (no projections)
CREATE TABLE IF NOT EXISTS exp02_projections.web_analytics_base
(
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY (page, timestamp);

-- Generate 200M rows
INSERT INTO exp02_projections.web_analytics_base
SELECT
    toDateTime('2024-01-01 00:00:00') + rand(number) % (365 * 86400) AS timestamp,
    rand(number + 1) AS user_id,
    concat('/page/', toString(rand(number + 2) % 500)) AS page,
    rand(number + 3) % 10000 AS duration_ms,
    ['US','UK','DE','FR','JP','CN','BR','IN','CA','AU','MX','KR','IT','ES','NL',
     'SE','NO','DK','FI','PL','CZ','AT','CH','BE','PT','IE','RU','TR','ZA','EG',
     'NG','KE','AR','CL','CO','PE','TH','VN','PH','MY','SG','ID','TW','HK','NZ',
     'IL','SA','AE','QA','UA'][1 + rand(number + 4) % 50] AS country,
    ['desktop', 'mobile', 'tablet'][1 + rand(number + 5) % 3] AS device_type
FROM numbers(200000000);

OPTIMIZE TABLE exp02_projections.web_analytics_base FINAL;

-- Projection table (2 projections: re-sort + aggregating)
CREATE TABLE IF NOT EXISTS exp02_projections.web_analytics_proj
(
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String),
    PROJECTION proj_country_time
    (
        SELECT * ORDER BY country, timestamp
    ),
    PROJECTION proj_hourly_stats
    (
        SELECT
            page,
            toStartOfHour(timestamp) AS hour,
            count() AS hits,
            avg(duration_ms) AS avg_duration,
            sum(duration_ms) AS sum_duration
        GROUP BY page, hour
    )
)
ENGINE = MergeTree
ORDER BY (page, timestamp);

INSERT INTO exp02_projections.web_analytics_proj 
SELECT * FROM exp02_projections.web_analytics_base;
OPTIMIZE TABLE exp02_projections.web_analytics_proj FINAL;

-- MV source (identical schema to base)
CREATE TABLE IF NOT EXISTS exp02_projections.web_analytics_mv_source
(
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY (page, timestamp);

-- MV target (pre-aggregated hourly stats)
CREATE TABLE IF NOT EXISTS exp02_projections.hourly_stats_mv_target
(
    page LowCardinality(String),
    hour DateTime,
    hits AggregateFunction(count, UInt64),
    avg_duration AggregateFunction(avg, UInt32),
    sum_duration AggregateFunction(sum, UInt32)
)
ENGINE = AggregatingMergeTree
ORDER BY (page, hour);

-- Materialized View definition
CREATE MATERIALIZED VIEW IF NOT EXISTS exp02_projections.hourly_stats_mv 
TO exp02_projections.hourly_stats_mv_target AS
SELECT
    page,
    toStartOfHour(timestamp) AS hour,
    countState() AS hits,
    avgState(duration_ms) AS avg_duration,
    sumState(duration_ms) AS sum_duration
FROM exp02_projections.web_analytics_mv_source
GROUP BY page, hour;

-- Populate MV source (triggers MV)
INSERT INTO exp02_projections.web_analytics_mv_source 
SELECT * FROM exp02_projections.web_analytics_base;
OPTIMIZE TABLE exp02_projections.web_analytics_mv_source FINAL;
OPTIMIZE TABLE exp02_projections.hourly_stats_mv_target FINAL;
