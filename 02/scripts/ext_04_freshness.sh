#!/bin/bash
# Extension 4: MV Freshness
set -euo pipefail

OUTDIR="/root/.openclaw/workspace/blog/experiments/results/02/data"
echo "insert_num,variant,query_result,elapsed_s" > "$OUTDIR/mv_freshness.csv"

ch() {
  ssh thesis-clickhouse "clickhouse-client -d exp02_projections -q \"$1\"" 2>/dev/null
}

ch "DROP TABLE IF EXISTS fresh_proj" || true
ch "DROP TABLE IF EXISTS fresh_mv_target" || true
ch "DROP VIEW IF EXISTS fresh_mv" || true
ch "DROP TABLE IF EXISTS fresh_mv_source" || true

ch "CREATE TABLE fresh_proj (timestamp DateTime, user_id UInt32, page LowCardinality(String), duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String), PROJECTION proj_hourly (SELECT page, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur GROUP BY page, hour)) ENGINE = MergeTree() ORDER BY (page, timestamp)"

ch "CREATE TABLE fresh_mv_source (timestamp DateTime, user_id UInt32, page LowCardinality(String), duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)) ENGINE = MergeTree() ORDER BY (page, timestamp)"

ch "CREATE TABLE fresh_mv_target (page LowCardinality(String), hour DateTime, hits AggregateFunction(count, UInt64), avg_dur AggregateFunction(avg, UInt32)) ENGINE = AggregatingMergeTree() ORDER BY (page, hour)"

ch "CREATE MATERIALIZED VIEW fresh_mv TO fresh_mv_target AS SELECT page, toStartOfHour(timestamp) AS hour, countState() AS hits, avgState(duration_ms) AS avg_dur FROM fresh_mv_source GROUP BY page, hour"

for I in $(seq 1 10); do
  echo "=== Batch $I ==="
  OFFSET=$(( (I-1) * 100000 ))
  END_NUM=$(( OFFSET + 100000 ))
  
  # Insert into projection table
  ch "INSERT INTO fresh_proj SELECT toDateTime('2024-06-15 10:00:00') + toIntervalSecond(number % 3600), rand()%1000+1, '/page/freshtest', 100+rand()%900, 'DE', 'desktop' FROM numbers($OFFSET, 100000)"
  
  # Insert into MV source
  ch "INSERT INTO fresh_mv_source SELECT toDateTime('2024-06-15 10:00:00') + toIntervalSecond(number % 3600), rand()%1000+1, '/page/freshtest', 100+rand()%900, 'DE', 'desktop' FROM numbers($OFFSET, 100000)"
  
  # Query projection table immediately
  ELAPSED=$(ssh thesis-clickhouse "clickhouse-client -d exp02_projections --time --format TSV -q \"SELECT count() FROM fresh_proj WHERE page='/page/freshtest'\"" 2>&1)
  PROJ_COUNT=$(echo "$ELAPSED" | head -1)
  PROJ_TIME=$(echo "$ELAPSED" | tail -1)
  echo "$I,proj,$PROJ_COUNT,$PROJ_TIME" >> "$OUTDIR/mv_freshness.csv"
  
  # Query MV immediately
  ELAPSED=$(ssh thesis-clickhouse "clickhouse-client -d exp02_projections --time --format TSV -q \"SELECT countMerge(hits) FROM fresh_mv_target WHERE page='/page/freshtest'\"" 2>&1)
  MV_COUNT=$(echo "$ELAPSED" | head -1)
  MV_TIME=$(echo "$ELAPSED" | tail -1)
  echo "$I,mv,$MV_COUNT,$MV_TIME" >> "$OUTDIR/mv_freshness.csv"
  
  echo "  proj=$PROJ_COUNT (${PROJ_TIME}s) mv=$MV_COUNT (${MV_TIME}s) expected=$((I * 100000))"
done

echo "=== MV Freshness test complete ==="
