#!/bin/bash
# Extension 5: EXPLAIN Analysis for all 8 query patterns
set -euo pipefail

OUTDIR="/root/.openclaw/workspace/blog/experiments/results/02/data/explain_outputs"
mkdir -p "$OUTDIR"

run_explain() {
  local KEY="$1"
  local QUERY="$2"
  echo "EXPLAIN $KEY..."
  ssh thesis-clickhouse "clickhouse-client -d exp02_projections -q \"EXPLAIN header=1, actions=1 $QUERY\"" > "$OUTDIR/${KEY}_explain.txt" 2>&1 || true
  echo "---" >> "$OUTDIR/${KEY}_explain.txt"
  ssh thesis-clickhouse "clickhouse-client -d exp02_projections -q \"EXPLAIN PLAN header=1 $QUERY\"" >> "$OUTDIR/${KEY}_explain.txt" 2>&1 || true
}

run_explain "Q1_base" "SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur FROM web_analytics_base WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' GROUP BY hour, page ORDER BY hour"

run_explain "Q1_proj" "SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur FROM web_analytics_proj WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' GROUP BY hour, page ORDER BY hour"

run_explain "Q1_mv" "SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_dur FROM hourly_stats_mv_target WHERE page='/page/42' AND hour >= '2024-06-01' AND hour < '2024-06-02' GROUP BY page, hour ORDER BY hour"

run_explain "Q2_base" "SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur FROM web_analytics_base WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-07-01' GROUP BY hour, page ORDER BY hour"

run_explain "Q2_proj" "SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur FROM web_analytics_proj WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-07-01' GROUP BY hour, page ORDER BY hour"

run_explain "Q2_mv" "SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_dur FROM hourly_stats_mv_target WHERE page='/page/42' AND hour >= '2024-06-01' AND hour < '2024-07-01' GROUP BY page, hour ORDER BY hour"

run_explain "Q3_base" "SELECT count(), avg(duration_ms) FROM web_analytics_base WHERE country='DE'"
run_explain "Q3_proj" "SELECT count(), avg(duration_ms) FROM web_analytics_proj WHERE country='DE'"

run_explain "Q4_base" "SELECT count(), avg(duration_ms) FROM web_analytics_base WHERE country='DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'"
run_explain "Q4_proj" "SELECT count(), avg(duration_ms) FROM web_analytics_proj WHERE country='DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'"

run_explain "Q5_base" "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_base GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
run_explain "Q5_proj" "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_proj GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
run_explain "Q5_mv" "SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM hourly_stats_mv_target GROUP BY page ORDER BY avg_dur DESC LIMIT 10"

run_explain "Q6_base" "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_base GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10"
run_explain "Q6_proj" "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_proj GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10"
run_explain "Q6_mv" "SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM hourly_stats_mv_target GROUP BY page HAVING countMerge(hits) > 100 ORDER BY avg_dur DESC LIMIT 10"

run_explain "Q7_base" "SELECT country, uniqExact(user_id) AS unique_users FROM web_analytics_base GROUP BY country ORDER BY unique_users DESC"
run_explain "Q7_proj" "SELECT country, uniqExact(user_id) AS unique_users FROM web_analytics_proj GROUP BY country ORDER BY unique_users DESC"

run_explain "Q8_base" "SELECT country, device_type, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur FROM web_analytics_base GROUP BY country, device_type, hour ORDER BY hits DESC LIMIT 100"
run_explain "Q8_proj" "SELECT country, device_type, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur FROM web_analytics_proj GROUP BY country, device_type, hour ORDER BY hits DESC LIMIT 100"

echo "=== EXPLAIN analysis complete ==="
