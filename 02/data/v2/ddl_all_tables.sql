CREATE TABLE exp02_projections.web_analytics_base
(
    `timestamp` DateTime,
    `user_id` UInt32,
    `page` LowCardinality(String),
    `duration_ms` UInt32,
    `country` LowCardinality(String),
    `device_type` LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY (page, timestamp)
SETTINGS index_granularity = 8192;

CREATE TABLE exp02_projections.web_analytics_proj
(
    `timestamp` DateTime,
    `user_id` UInt32,
    `page` LowCardinality(String),
    `duration_ms` UInt32,
    `country` LowCardinality(String),
    `device_type` LowCardinality(String),
    PROJECTION proj_country_time
    (
        SELECT *
        ORDER BY country, timestamp
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
ORDER BY (page, timestamp)
SETTINGS index_granularity = 8192;

CREATE TABLE exp02_projections.web_analytics_mv_source
(
    `timestamp` DateTime,
    `user_id` UInt32,
    `page` LowCardinality(String),
    `duration_ms` UInt32,
    `country` LowCardinality(String),
    `device_type` LowCardinality(String)
)
ENGINE = MergeTree
ORDER BY (page, timestamp)
SETTINGS index_granularity = 8192;

CREATE TABLE exp02_projections.hourly_stats_mv_target
(
    `page` LowCardinality(String),
    `hour` DateTime,
    `hits` AggregateFunction(count, UInt64),
    `avg_duration` AggregateFunction(avg, UInt32),
    `sum_duration` AggregateFunction(sum, UInt32)
)
ENGINE = AggregatingMergeTree
ORDER BY (page, hour)
SETTINGS index_granularity = 8192;

CREATE MATERIALIZED VIEW exp02_projections.hourly_stats_mv
TO exp02_projections.hourly_stats_mv_target AS
SELECT
    page,
    toStartOfHour(timestamp) AS hour,
    countState() AS hits,
    avgState(duration_ms) AS avg_duration,
    sumState(duration_ms) AS sum_duration
FROM exp02_projections.web_analytics_mv_source
GROUP BY page, hour;
