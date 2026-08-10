-- Extended Query Patterns for Experiment 02

-- Q1: Dashboard Rollup (hourly, 1 day)
-- Base
SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur
FROM web_analytics_base WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02'
GROUP BY hour, page ORDER BY hour;
-- Proj
SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur
FROM web_analytics_proj WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02'
GROUP BY hour, page ORDER BY hour;
-- MV
SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_dur
FROM hourly_stats_mv_target WHERE page='/page/42' AND hour >= '2024-06-01' AND hour < '2024-06-02'
GROUP BY page, hour ORDER BY hour;

-- Q2: Dashboard Rollup (hourly, 30 days)
SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur
FROM web_analytics_base WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-07-01'
GROUP BY hour, page ORDER BY hour;

-- Q3: Country Filter (exact match, full range)
SELECT count(), avg(duration_ms) FROM web_analytics_base WHERE country='DE';
SELECT count(), avg(duration_ms) FROM web_analytics_proj WHERE country='DE';

-- Q4: Country + Time Range (compound filter)
SELECT count(), avg(duration_ms) FROM web_analytics_base WHERE country='DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01';
SELECT count(), avg(duration_ms) FROM web_analytics_proj WHERE country='DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01';

-- Q5: Top-K Pages by avg duration (full table agg)
SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_base GROUP BY page ORDER BY avg_dur DESC LIMIT 10;
SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_proj GROUP BY page ORDER BY avg_dur DESC LIMIT 10;
SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM hourly_stats_mv_target GROUP BY page ORDER BY avg_dur DESC LIMIT 10;

-- Q6: Top-K with HAVING (min events > 100)
SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_base GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10;
SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_proj GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10;
SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM hourly_stats_mv_target GROUP BY page HAVING countMerge(hits) > 100 ORDER BY avg_dur DESC LIMIT 10;

-- Q7: Cardinality (uniqExact per country)
SELECT country, uniqExact(user_id) AS unique_users FROM web_analytics_base GROUP BY country ORDER BY unique_users DESC;
SELECT country, uniqExact(user_id) AS unique_users FROM web_analytics_proj GROUP BY country ORDER BY unique_users DESC;

-- Q8: Multi-Dimension GROUP BY (country × device × hour)
SELECT country, device_type, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur
FROM web_analytics_base GROUP BY country, device_type, hour ORDER BY hits DESC LIMIT 100;
SELECT country, device_type, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur
FROM web_analytics_proj GROUP BY country, device_type, hour ORDER BY hits DESC LIMIT 100;
