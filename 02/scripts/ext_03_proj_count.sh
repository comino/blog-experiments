#!/bin/bash
# Extension 3: Projection Count Scaling — 0, 1, 3, 5 projections
set -euo pipefail

OUTDIR="/root/.openclaw/workspace/blog/experiments/results/02/data"
echo "proj_count,metric,value" > "$OUTDIR/projection_count.csv"

ch() {
  ssh thesis-clickhouse "clickhouse-client -d exp02_projections -q \"$1\"" 2>/dev/null
}

echo "=== Creating tables ==="

# 0 projections (baseline)
ch "DROP TABLE IF EXISTS proj_count_0"
ch "CREATE TABLE proj_count_0 (timestamp DateTime, user_id UInt32, page LowCardinality(String), duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)) ENGINE = MergeTree() ORDER BY (page, timestamp)"

# 1 projection
ch "DROP TABLE IF EXISTS proj_count_1"
ch "CREATE TABLE proj_count_1 (timestamp DateTime, user_id UInt32, page LowCardinality(String), duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String), PROJECTION p1_country (SELECT * ORDER BY (country, timestamp))) ENGINE = MergeTree() ORDER BY (page, timestamp)"

# 3 projections
ch "DROP TABLE IF EXISTS proj_count_3"
ch "CREATE TABLE proj_count_3 (timestamp DateTime, user_id UInt32, page LowCardinality(String), duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String), PROJECTION p1_country (SELECT * ORDER BY (country, timestamp)), PROJECTION p2_device (SELECT * ORDER BY (device_type, timestamp)), PROJECTION p3_hourly (SELECT page, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_duration, sum(duration_ms) AS sum_duration GROUP BY page, hour)) ENGINE = MergeTree() ORDER BY (page, timestamp)"

# 5 projections
ch "DROP TABLE IF EXISTS proj_count_5"
ch "CREATE TABLE proj_count_5 (timestamp DateTime, user_id UInt32, page LowCardinality(String), duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String), PROJECTION p1_country (SELECT * ORDER BY (country, timestamp)), PROJECTION p2_device (SELECT * ORDER BY (device_type, timestamp)), PROJECTION p3_hourly (SELECT page, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_duration, sum(duration_ms) AS sum_duration GROUP BY page, hour), PROJECTION p4_user (SELECT * ORDER BY (user_id, timestamp)), PROJECTION p5_country_device (SELECT country, device_type, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_duration GROUP BY country, device_type, hour)) ENGINE = MergeTree() ORDER BY (page, timestamp)"

echo "=== Inserting 10M rows each ==="
INSERT_SQL="SELECT toDateTime('2024-01-01') + toIntervalSecond(rand() % (365*86400)), rand()%1000000+1, concat('/page/',toString(rand()%1000)), 50+rand()%9950, arrayElement(['US','DE','UK','FR','JP','BR','IN','CA','AU','MX','IT','ES','KR','NL','SE','CH','AT','BE','PL','CZ','DK','NO','FI','PT','IE','RO','HU','GR','BG','HR','SK','SI','LT','LV','EE','IL','TR','ZA','NG','EG','KE','AR','CL','CO','PE','TH','VN','MY','PH','ID'],(rand()%50)+1), arrayElement(['desktop','mobile','tablet','smart_tv','wearable'],(rand()%5)+1) FROM numbers(10000000)"

for COUNT in 0 1 3 5; do
  TBL="proj_count_${COUNT}"
  echo "  $TBL..."
  ELAPSED=$(ssh thesis-clickhouse "clickhouse-client -d exp02_projections --time -q \"INSERT INTO $TBL $INSERT_SQL\"" 2>&1 | tail -1)
  echo "  Insert took ${ELAPSED}s"
  
  ch "OPTIMIZE TABLE $TBL FINAL"
  
  STORAGE=$(ch "SELECT sum(data_compressed_bytes) FROM system.parts WHERE database='exp02_projections' AND table='$TBL' AND active")
  
  # Convert elapsed to ms
  INGEST_MS=$(echo "$ELAPSED * 1000" | bc | cut -d. -f1)
  RPS=$(echo "10000000 / $ELAPSED" | bc | cut -d. -f1)
  
  echo "$COUNT,ingest_ms,$INGEST_MS" >> "$OUTDIR/projection_count.csv"
  echo "$COUNT,storage_bytes,$STORAGE" >> "$OUTDIR/projection_count.csv"
  echo "$COUNT,rows_per_sec,$RPS" >> "$OUTDIR/projection_count.csv"
  echo "  Storage: $STORAGE bytes, RPS: $RPS"
done

echo "=== Projection count scaling complete ==="
