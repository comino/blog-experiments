# Fix 11: MV SELECT Queries with -Merge() Functions

## MV Target DDL
```sql
CREATE TABLE hourly_stats_mv_target
(
    page LowCardinality(String),
    hour DateTime,
    hits AggregateFunction(count, UInt64),
    avg_duration AggregateFunction(avg, UInt32),
    sum_duration AggregateFunction(sum, UInt32)
)
ENGINE = AggregatingMergeTree
ORDER BY (page, hour);
```

## MV Definition
```sql
CREATE MATERIALIZED VIEW hourly_stats_mv TO hourly_stats_mv_target AS
SELECT
    page,
    toStartOfHour(timestamp) AS hour,
    countState() AS hits,
    avgState(duration_ms) AS avg_duration,
    sumState(duration_ms) AS sum_duration
FROM web_analytics_mv_source
GROUP BY page, hour;
```

## Q3 equivalent (Hourly stats for a specific page)
```sql
-- Projection version (reads from base table, optimizer picks proj_hourly_stats):
SELECT page, toStartOfHour(timestamp) AS hour, 
       count() AS hits, avg(duration_ms) AS avg_duration
FROM web_analytics_proj 
WHERE page = '/page/0' 
GROUP BY page, hour ORDER BY hour;

-- MV version (reads from pre-aggregated target, uses -Merge() combinators):
SELECT page, hour, 
       countMerge(hits) AS hits, 
       avgMerge(avg_duration) AS avg_duration
FROM hourly_stats_mv_target 
WHERE page = '/page/0' 
GROUP BY page, hour ORDER BY hour;
```

## Q5 equivalent (Top pages by total duration)
```sql
-- Projection version:
SELECT page, toStartOfHour(timestamp) AS hour, 
       sum(duration_ms) AS total_duration
FROM web_analytics_proj 
GROUP BY page, hour 
ORDER BY total_duration DESC LIMIT 10;

-- MV version:
SELECT page, hour, 
       sumMerge(sum_duration) AS total_duration
FROM hourly_stats_mv_target 
GROUP BY page, hour 
ORDER BY total_duration DESC LIMIT 10;
```

## Key Differences
1. **Projection**: Uses standard SQL functions (`count()`, `avg()`, `sum()`). ClickHouse optimizer transparently reads from the projection's pre-aggregated data.
2. **MV**: Requires `-Merge()` combinator functions (`countMerge()`, `avgMerge()`, `sumMerge()`). The user must know that the target stores `AggregateFunction` states, not final values.
3. **Semantic equivalence**: Both produce identical results. The MV approach requires explicit knowledge of the intermediate state format.

## Note on `AggregateFunction(count, UInt64)`
The DDL declares `AggregateFunction(count, UInt64)` while `countState()` takes no arguments.
ClickHouse accepts this — the `UInt64` type argument is effectively ignored for `count`.
The canonical form would be `AggregateFunction(count)`, but both work identically.
