#!/bin/bash
# Extension 1: Scaling Analysis — 1M, 10M, 50M, 200M
set -euo pipefail

OUTDIR="/root/.openclaw/workspace/blog/experiments/results/02/data"
echo "query,variant,size,run,elapsed_s" > "$OUTDIR/scaling.csv"
echo "size,table,disk_bytes,rows" > "$OUTDIR/scaling_storage.csv"

ch() {
  ssh thesis-clickhouse "clickhouse-client -d exp02_projections -q \"$1\"" 2>/dev/null
}

ch_time() {
  ssh thesis-clickhouse "clickhouse-client -d exp02_projections --time -q \"$1 FORMAT Null\"" 2>&1
}

SIZES=(1000000 10000000 50000000)
SIZE_LABELS=("1M" "10M" "50M")

INSERT_BASE="SELECT toDateTime('2024-01-01') + toIntervalSecond(rand() % (365*86400)), rand()%1000000+1, concat('/page/',toString(rand()%1000)), 50+rand()%9950, arrayElement(['US','DE','UK','FR','JP','BR','IN','CA','AU','MX','IT','ES','KR','NL','SE','CH','AT','BE','PL','CZ','DK','NO','FI','PT','IE','RO','HU','GR','BG','HR','SK','SI','LT','LV','EE','IL','TR','ZA','NG','EG','KE','AR','CL','CO','PE','TH','VN','MY','PH','ID'],(rand()%50)+1), arrayElement(['desktop','mobile','tablet','smart_tv','wearable'],(rand()%5)+1) FROM numbers"

for i in "${!SIZES[@]}"; do
  N=${SIZES[$i]}
  LABEL=${SIZE_LABELS[$i]}
  echo "=== SIZE: $LABEL ==="

  # Create tables
  for TBL in "scale_base_${LABEL}" "scale_mv_source_${LABEL}"; do
    ch "DROP TABLE IF EXISTS $TBL" || true
    ch "CREATE TABLE $TBL (timestamp DateTime, user_id UInt32, page LowCardinality(String), duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)) ENGINE = MergeTree() ORDER BY (page, timestamp)"
  done

  ch "DROP TABLE IF EXISTS scale_proj_${LABEL}" || true
  ch "CREATE TABLE scale_proj_${LABEL} (timestamp DateTime, user_id UInt32, page LowCardinality(String), duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String), PROJECTION proj_country_time (SELECT * ORDER BY (country, timestamp)), PROJECTION proj_hourly_stats (SELECT page, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_duration, sum(duration_ms) AS sum_duration GROUP BY page, hour)) ENGINE = MergeTree() ORDER BY (page, timestamp)"

  ch "DROP TABLE IF EXISTS scale_mv_target_${LABEL}" || true
  ch "CREATE TABLE scale_mv_target_${LABEL} (page LowCardinality(String), hour DateTime, hits AggregateFunction(count, UInt64), avg_duration AggregateFunction(avg, UInt32), sum_duration AggregateFunction(sum, UInt32)) ENGINE = AggregatingMergeTree() ORDER BY (page, hour)"

  ch "DROP VIEW IF EXISTS scale_mv_${LABEL}" || true
  ch "CREATE MATERIALIZED VIEW scale_mv_${LABEL} TO scale_mv_target_${LABEL} AS SELECT page, toStartOfHour(timestamp) AS hour, countState() AS hits, avgState(duration_ms) AS avg_duration, sumState(duration_ms) AS sum_duration FROM scale_mv_source_${LABEL} GROUP BY page, hour"

  echo "  Inserting into base..."
  ch "INSERT INTO scale_base_${LABEL} $INSERT_BASE($N)"
  echo "  Inserting into proj..."
  ch "INSERT INTO scale_proj_${LABEL} SELECT * FROM scale_base_${LABEL}"
  echo "  Inserting into mv_source..."
  ch "INSERT INTO scale_mv_source_${LABEL} SELECT * FROM scale_base_${LABEL}"

  echo "  OPTIMIZE..."
  ch "OPTIMIZE TABLE scale_base_${LABEL} FINAL"
  ch "OPTIMIZE TABLE scale_proj_${LABEL} FINAL"
  ch "OPTIMIZE TABLE scale_mv_source_${LABEL} FINAL"
  ch "OPTIMIZE TABLE scale_mv_target_${LABEL} FINAL"

  # Storage
  for TBL in "scale_base_${LABEL}" "scale_proj_${LABEL}" "scale_mv_target_${LABEL}"; do
    ROW=$(ssh thesis-clickhouse "clickhouse-client -d exp02_projections --format TSV -q \"SELECT sum(bytes_on_disk), sum(rows) FROM system.parts WHERE database='exp02_projections' AND table='$TBL' AND active\"" 2>/dev/null)
    echo "$LABEL,$TBL,$ROW" >> "$OUTDIR/scaling_storage.csv"
  done
done

# Add 200M storage
for TBL in "web_analytics_base" "web_analytics_proj" "hourly_stats_mv_target"; do
  ROW=$(ssh thesis-clickhouse "clickhouse-client -d exp02_projections --format TSV -q \"SELECT sum(bytes_on_disk), sum(rows) FROM system.parts WHERE database='exp02_projections' AND table='$TBL' AND active\"" 2>/dev/null)
  echo "200M,$TBL,$ROW" >> "$OUTDIR/scaling_storage.csv"
done

echo "=== Running benchmarks ==="
ch "SYSTEM STOP MERGES"

# All sizes including 200M
ALL_LABELS=("1M" "10M" "50M" "200M")
declare -A BASE_TBLS=([1M]="scale_base_1M" [10M]="scale_base_10M" [50M]="scale_base_50M" [200M]="web_analytics_base")
declare -A PROJ_TBLS=([1M]="scale_proj_1M" [10M]="scale_proj_10M" [50M]="scale_proj_50M" [200M]="web_analytics_proj")
declare -A MV_TBLS=([1M]="scale_mv_target_1M" [10M]="scale_mv_target_10M" [50M]="scale_mv_target_50M" [200M]="hourly_stats_mv_target")

for LABEL in "${ALL_LABELS[@]}"; do
  BASE=${BASE_TBLS[$LABEL]}
  PROJ=${PROJ_TBLS[$LABEL]}
  MV=${MV_TBLS[$LABEL]}
  
  for RUN in 1 2 3 4 5; do
    echo "  $LABEL run $RUN..."
    
    # Q2: Country filter
    ch "SYSTEM DROP FILESYSTEM CACHE"
    E=$(ch_time "SELECT count(), avg(duration_ms) FROM $BASE WHERE country='DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'")
    echo "Q2,base,$LABEL,$RUN,$E" >> "$OUTDIR/scaling.csv"
    
    ch "SYSTEM DROP FILESYSTEM CACHE"
    E=$(ch_time "SELECT count(), avg(duration_ms) FROM $PROJ WHERE country='DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'")
    echo "Q2,proj,$LABEL,$RUN,$E" >> "$OUTDIR/scaling.csv"
    
    # Q3: Top-K full agg
    ch "SYSTEM DROP FILESYSTEM CACHE"
    E=$(ch_time "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM $BASE GROUP BY page ORDER BY avg_dur DESC LIMIT 10")
    echo "Q3,base,$LABEL,$RUN,$E" >> "$OUTDIR/scaling.csv"
    
    ch "SYSTEM DROP FILESYSTEM CACHE"
    E=$(ch_time "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM $PROJ GROUP BY page ORDER BY avg_dur DESC LIMIT 10")
    echo "Q3,proj,$LABEL,$RUN,$E" >> "$OUTDIR/scaling.csv"
    
    ch "SYSTEM DROP FILESYSTEM CACHE"
    E=$(ch_time "SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM $MV GROUP BY page ORDER BY avg_dur DESC LIMIT 10")
    echo "Q3,mv,$LABEL,$RUN,$E" >> "$OUTDIR/scaling.csv"
  done
done

ch "SYSTEM START MERGES"
echo "=== Scaling analysis complete ==="
