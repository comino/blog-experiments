-- Experiment 01: The 3 benchmark queries
-- These are executed by scripts/benchmark.sh with unique query_ids
-- Each runs 5× cold (after SYSTEM DROP FILESYSTEM CACHE) and 5× warm

-- Q1: Range scan + aggregation (timestamp filter + avg)
SELECT toStartOfHour(timestamp) h, avg(value)
FROM exp01_compression.{variant}
WHERE timestamp BETWEEN '2024-01-15' AND '2024-02-15'
GROUP BY h
FORMAT Null;

-- Q2: Top-K aggregation (full table scan, GROUP BY host)
SELECT host, sum(counter)
FROM exp01_compression.{variant}
GROUP BY host
ORDER BY 2 DESC
LIMIT 10
FORMAT Null;

-- Q3: Wide scan with mixed aggregates (filter on metric_name)
SELECT count(), avg(value), max(counter)
FROM exp01_compression.{variant}
WHERE metric_name = 'cpu_usage'
FORMAT Null;
