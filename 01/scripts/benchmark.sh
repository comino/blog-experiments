#!/bin/bash
# Experiment 01: Full benchmark script
# Run ON the ClickHouse server: bash benchmark.sh
# Executes: query benchmark (3Q × 5V × 5runs × cold/warm, randomized) + ingest benchmark
set -e

CH="clickhouse-client --database exp01_compression"
VARIANTS="v1_default v2_zstd v3_percolumn v4_percolumn_zstd v5_aggressive"
RUNS=5
OUTDIR="/tmp/exp01_results"
mkdir -p $OUTDIR

# ── Benchmark Controls ──
# Stop merges to prevent background interference
$CH --query "SYSTEM STOP MERGES exp01_compression"

echo "query,variant,run,temp,duration_ms,read_rows,read_bytes,cpu_us" > $OUTDIR/queries.csv

# ── Generate randomized run order ──
PLAN=""
for run in $(seq 1 $RUNS); do
  for v in $VARIANTS; do
    for q in Q1 Q2 Q3; do
      PLAN="$PLAN $q:$v:$run"
    done
  done
done
PLAN=$(echo $PLAN | tr ' ' '\n' | shuf | tr '\n' ' ')

# ── Query Benchmark ──
for entry in $PLAN; do
  q=$(echo $entry | cut -d: -f1)
  v=$(echo $entry | cut -d: -f2)
  run=$(echo $entry | cut -d: -f3)

  case $q in
    Q1) SQL="SELECT toStartOfHour(timestamp) h, avg(value) FROM $v WHERE timestamp BETWEEN '2024-01-15' AND '2024-02-15' GROUP BY h FORMAT Null" ;;
    Q2) SQL="SELECT host, sum(counter) FROM $v GROUP BY host ORDER BY 2 DESC LIMIT 10 FORMAT Null" ;;
    Q3) SQL="SELECT count(), avg(value), max(counter) FROM $v WHERE metric_name = 'cpu_usage' FORMAT Null" ;;
  esac

  # Cold run: drop filesystem cache first
  $CH --query "SYSTEM DROP FILESYSTEM CACHE"
  QID="exp01_cold_${q}_${v}_${run}_$(date +%s%N)"
  $CH --query_id "$QID" --query "$SQL" 2>/dev/null
  sleep 0.5
  ROW=$($CH --query "SELECT query_duration_ms, read_rows, read_bytes, ProfileEvents['OSCPUVirtualTimeMicroseconds'] FROM system.query_log WHERE query_id='$QID' AND type='QueryFinish' ORDER BY event_time DESC LIMIT 1 FORMAT TSV" 2>/dev/null)
  if [ -n "$ROW" ]; then
    echo "$q,$v,$run,cold,$(echo $ROW | tr '\t' ',')" >> $OUTDIR/queries.csv
  fi

  # Warm run: same query, data now cached
  QID="exp01_warm_${q}_${v}_${run}_$(date +%s%N)"
  $CH --query_id "$QID" --query "$SQL" 2>/dev/null
  sleep 0.5
  ROW=$($CH --query "SELECT query_duration_ms, read_rows, read_bytes, ProfileEvents['OSCPUVirtualTimeMicroseconds'] FROM system.query_log WHERE query_id='$QID' AND type='QueryFinish' ORDER BY event_time DESC LIMIT 1 FORMAT TSV" 2>/dev/null)
  if [ -n "$ROW" ]; then
    echo "$q,$v,$run,warm,$(echo $ROW | tr '\t' ',')" >> $OUTDIR/queries.csv
  fi

  echo "Done: $q $v run$run"
done

echo "=== QUERY BENCHMARK DONE ==="

# ── Ingest Benchmark ──
echo "variant,run,rows,duration_s,rows_per_s" > $OUTDIR/ingest.csv

for v in $VARIANTS; do
  INGEST_TABLE="${v}_ingest_tmp"
  $CH --query "DROP TABLE IF EXISTS $INGEST_TABLE"
  $CH --query "CREATE TABLE $INGEST_TABLE AS $v"

  for run in $(seq 1 3); do
    $CH --query "TRUNCATE TABLE $INGEST_TABLE"
    QID="exp01_ingest_${v}_${run}_$(date +%s%N)"
    START=$(date +%s%N)
    $CH --query_id "$QID" --query "INSERT INTO $INGEST_TABLE SELECT * FROM source LIMIT 10000000"
    END=$(date +%s%N)
    DUR=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
    RPS=$(echo "scale=0; 10000000 / $DUR" | bc)
    echo "$v,$run,10000000,$DUR,$RPS" >> $OUTDIR/ingest.csv
    echo "Ingest: $v run$run ${DUR}s"
  done

  $CH --query "DROP TABLE $INGEST_TABLE"
done

echo "=== INGEST BENCHMARK DONE ==="

# Restart merges
$CH --query "SYSTEM START MERGES exp01_compression"

echo "=== ALL DONE ==="
echo "Results in $OUTDIR/"
