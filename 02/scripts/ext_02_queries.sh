#!/bin/bash
# Extension 2: Extended Query Patterns (8 queries) on 200M data
set -euo pipefail

OUTDIR="/root/.openclaw/workspace/blog/experiments/results/02/data"
echo "query,variant,run,cache,elapsed_s" > "$OUTDIR/extended_queries.csv"

ch() {
  ssh thesis-clickhouse "clickhouse-client -d exp02_projections -q \"$1\"" 2>/dev/null
}

run_timed() {
  local QUERY="$1" LABEL="$2" VARIANT="$3" RUN="$4" CACHE="$5"
  ch "SYSTEM DROP FILESYSTEM CACHE"
  # --time outputs elapsed seconds to stderr
  ELAPSED=$(ssh thesis-clickhouse "clickhouse-client -d exp02_projections --time -q \"$QUERY FORMAT Null\"" 2>&1)
  echo "$LABEL,$VARIANT,$RUN,$CACHE,$ELAPSED" >> "$OUTDIR/extended_queries.csv"
}

ch "SYSTEM STOP MERGES"

for RUN in 1 2 3 4 5; do
  echo "=== Run $RUN ==="
  
  # Q1: Dashboard Rollup (hourly, 1 day)
  for V in base proj; do
    TBL="web_analytics_$V"; [[ "$V" == "base" ]] && TBL="web_analytics_base"
    [[ "$V" == "proj" ]] && TBL="web_analytics_proj"
    run_timed "SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur FROM $TBL WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' GROUP BY hour, page ORDER BY hour" "Q1" "$V" "$RUN" "cold"
  done
  run_timed "SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_dur FROM hourly_stats_mv_target WHERE page='/page/42' AND hour >= '2024-06-01' AND hour < '2024-06-02' GROUP BY page, hour ORDER BY hour" "Q1" "mv" "$RUN" "cold"

  # Q2: Dashboard Rollup (hourly, 30 days)
  for V in base proj; do
    TBL="web_analytics_$V"
    run_timed "SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur FROM $TBL WHERE page='/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-07-01' GROUP BY hour, page ORDER BY hour" "Q2" "$V" "$RUN" "cold"
  done
  run_timed "SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_dur FROM hourly_stats_mv_target WHERE page='/page/42' AND hour >= '2024-06-01' AND hour < '2024-07-01' GROUP BY page, hour ORDER BY hour" "Q2" "mv" "$RUN" "cold"

  # Q3: Country Filter (exact match, full range)
  run_timed "SELECT count(), avg(duration_ms) FROM web_analytics_base WHERE country='DE'" "Q3" "base" "$RUN" "cold"
  run_timed "SELECT count(), avg(duration_ms) FROM web_analytics_proj WHERE country='DE'" "Q3" "proj" "$RUN" "cold"

  # Q4: Country + Time Range
  run_timed "SELECT count(), avg(duration_ms) FROM web_analytics_base WHERE country='DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'" "Q4" "base" "$RUN" "cold"
  run_timed "SELECT count(), avg(duration_ms) FROM web_analytics_proj WHERE country='DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'" "Q4" "proj" "$RUN" "cold"

  # Q5: Top-K Pages by avg duration
  run_timed "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_base GROUP BY page ORDER BY avg_dur DESC LIMIT 10" "Q5" "base" "$RUN" "cold"
  run_timed "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_proj GROUP BY page ORDER BY avg_dur DESC LIMIT 10" "Q5" "proj" "$RUN" "cold"
  run_timed "SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM hourly_stats_mv_target GROUP BY page ORDER BY avg_dur DESC LIMIT 10" "Q5" "mv" "$RUN" "cold"

  # Q6: Top-K with HAVING
  run_timed "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_base GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10" "Q6" "base" "$RUN" "cold"
  run_timed "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_proj GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10" "Q6" "proj" "$RUN" "cold"
  run_timed "SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM hourly_stats_mv_target GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10" "Q6" "mv" "$RUN" "cold"

  # Q7: Cardinality (uniqExact per country)
  run_timed "SELECT country, uniqExact(user_id) AS unique_users FROM web_analytics_base GROUP BY country ORDER BY unique_users DESC" "Q7" "base" "$RUN" "cold"
  run_timed "SELECT country, uniqExact(user_id) AS unique_users FROM web_analytics_proj GROUP BY country ORDER BY unique_users DESC" "Q7" "proj" "$RUN" "cold"

  # Q8: Multi-Dimension GROUP BY
  run_timed "SELECT country, device_type, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur FROM web_analytics_base GROUP BY country, device_type, hour ORDER BY hits DESC LIMIT 100" "Q8" "base" "$RUN" "cold"
  run_timed "SELECT country, device_type, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur FROM web_analytics_proj GROUP BY country, device_type, hour ORDER BY hits DESC LIMIT 100" "Q8" "proj" "$RUN" "cold"
done

ch "SYSTEM START MERGES"
echo "=== Extended queries complete ==="
