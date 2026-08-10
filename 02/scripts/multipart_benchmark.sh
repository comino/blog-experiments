#!/bin/bash
# Task 1: Multi-part benchmark — 200M rows in 20 batches, no OPTIMIZE FINAL
set -euo pipefail

DB="exp02_projections"
OUTDIR="/root/exp02_v5"
mkdir -p "$OUTDIR"

echo "=== STEP 1: Create multipart tables ==="

clickhouse-client -q "
DROP TABLE IF EXISTS ${DB}.mp_mv;
DROP TABLE IF EXISTS ${DB}.mp_mv_target;
DROP TABLE IF EXISTS ${DB}.mp_mv_source;
DROP TABLE IF EXISTS ${DB}.mp_proj;
DROP TABLE IF EXISTS ${DB}.mp_base;
"

# Base table
clickhouse-client -q "
CREATE TABLE ${DB}.mp_base (
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String)
) ENGINE = MergeTree()
ORDER BY (page, timestamp)
SETTINGS parts_to_delay_insert = 500, parts_to_throw_insert = 1000;
"

# Projection table
clickhouse-client -q "
CREATE TABLE ${DB}.mp_proj (
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String),
    PROJECTION proj_hourly_stats (
        SELECT page, toStartOfHour(timestamp) AS hour,
               count() AS hits, avg(duration_ms) AS avg_duration, sum(duration_ms) AS sum_duration
        GROUP BY page, hour
    )
) ENGINE = MergeTree()
ORDER BY (page, timestamp)
SETTINGS parts_to_delay_insert = 500, parts_to_throw_insert = 1000;
"

# MV source
clickhouse-client -q "
CREATE TABLE ${DB}.mp_mv_source (
    timestamp DateTime,
    user_id UInt32,
    page LowCardinality(String),
    duration_ms UInt32,
    country LowCardinality(String),
    device_type LowCardinality(String)
) ENGINE = MergeTree()
ORDER BY (page, timestamp)
SETTINGS parts_to_delay_insert = 500, parts_to_throw_insert = 1000;
"

# MV target
clickhouse-client -q "
CREATE TABLE ${DB}.mp_mv_target (
    page LowCardinality(String),
    hour DateTime,
    hits AggregateFunction(count, UInt64),
    avg_duration AggregateFunction(avg, UInt32),
    sum_duration AggregateFunction(sum, UInt32)
) ENGINE = AggregatingMergeTree()
ORDER BY (page, hour)
SETTINGS parts_to_delay_insert = 500, parts_to_throw_insert = 1000;
"

# MV
clickhouse-client -q "
CREATE MATERIALIZED VIEW ${DB}.mp_mv TO ${DB}.mp_mv_target AS
SELECT page, toStartOfHour(timestamp) AS hour,
       countState() AS hits, avgState(duration_ms) AS avg_duration, sumState(duration_ms) AS sum_duration
FROM ${DB}.mp_mv_source GROUP BY page, hour;
"

echo "=== STEP 2: Stop background merges ==="
for t in mp_base mp_proj mp_mv_source mp_mv_target; do
    clickhouse-client -q "SYSTEM STOP MERGES ${DB}.${t}"
done

echo "=== STEP 3: Insert 200M rows in 20 batches of 10M ==="
PAGES=1000
for i in $(seq 1 20); do
    echo "  Batch $i/20..."
    SEED=$((42 + i))
    
    # Insert into all 3 source tables
    for tbl in mp_base mp_proj mp_mv_source; do
        clickhouse-client -q "
        INSERT INTO ${DB}.${tbl}
        SELECT
            toDateTime('2024-01-01') + toIntervalSecond(rand(${SEED}) % (365*86400)),
            rand(${SEED}+1) % 100000,
            concat('/page/', toString(rand(${SEED}+2) % ${PAGES})),
            rand(${SEED}+3) % 10000,
            ['US','DE','GB','FR','JP','AU','BR','IN','CA','NL'][1 + rand(${SEED}+4) % 10],
            ['desktop','mobile','tablet'][1 + rand(${SEED}+5) % 3]
        FROM numbers(10000000)
        SETTINGS max_block_size=1000000, max_insert_block_size=1000000;
        "
    done
done

echo "=== STEP 4: Record system.parts state (BEFORE optimize) ==="
clickhouse-client -q "
SELECT table, count() as parts, sum(rows) as total_rows, formatReadableSize(sum(bytes_on_disk)) as disk
FROM system.parts
WHERE database='${DB}' AND table LIKE 'mp_%' AND active=1
GROUP BY table ORDER BY table
FORMAT TSVWithNames
" > "$OUTDIR/parts_before_optimize.tsv"
cat "$OUTDIR/parts_before_optimize.tsv"

echo ""
echo "=== STEP 4b: Detailed parts snapshot ==="
clickhouse-client -q "
SELECT table, partition, count() as parts, min(rows) as min_rows, max(rows) as max_rows,
  sum(rows) as total_rows, formatReadableSize(sum(bytes_on_disk)) as disk
FROM system.parts
WHERE database='${DB}' AND table LIKE 'mp_%' AND active=1
GROUP BY table, partition ORDER BY table
FORMAT TSVWithNames
" > "$OUTDIR/parts_detailed_before.tsv"

echo "=== STEP 5: Benchmark Q3 (full aggregation) — 10 cold + 10 warm ==="

Q3_BASE="SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM ${DB}.mp_base GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
Q3_PROJ="SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM ${DB}.mp_proj GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
Q3_MV="SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM ${DB}.mp_mv_target GROUP BY page ORDER BY avg_dur DESC LIMIT 10"

echo "query,variant,run,cache,elapsed_ms,rows_read,bytes_read" > "$OUTDIR/multipart_benchmark.csv"

run_query() {
    local qname=$1 variant=$2 sql=$3 cache=$4 run=$5
    if [ "$cache" = "cold" ]; then
        clickhouse-client -q "SYSTEM DROP FILESYSTEM CACHE" 2>/dev/null || true
        clickhouse-client -q "SYSTEM DROP MARK CACHE"
        clickhouse-client -q "SYSTEM DROP UNCOMPRESSED CACHE"
        sync; echo 3 > /proc/sys/vm/drop_caches
        sleep 0.5
    fi
    
    local qid="exp02_mp_${variant}_${cache}_${run}_$(date +%s%N)"
    clickhouse-client --database="$DB" --query="$sql" --query_id="$qid" --format=Null 2>/dev/null
    sleep 0.3
    clickhouse-client -q "SYSTEM FLUSH LOGS"
    sleep 0.2
    
    local metrics
    metrics=$(clickhouse-client -q "
        SELECT query_duration_ms, read_rows, read_bytes
        FROM system.query_log
        WHERE query_id = '${qid}' AND type = 'QueryFinish'
        ORDER BY event_time DESC LIMIT 1
        FORMAT TSV
    ")
    
    if [ -z "$metrics" ]; then
        echo "  WARNING: no metrics for $qid" >&2
        return
    fi
    
    local elapsed rows bytes
    elapsed=$(echo "$metrics" | cut -f1)
    rows=$(echo "$metrics" | cut -f2)
    bytes=$(echo "$metrics" | cut -f3)
    echo "  ${qname}/${variant}/${cache}/run${run}: ${elapsed}ms, ${rows} rows"
    echo "${qname},${variant},${run},${cache},${elapsed},${rows},${bytes}" >> "$OUTDIR/multipart_benchmark.csv"
}

# Cold runs
echo "--- Cold runs ---"
for run in $(seq 1 10); do
    echo "Cold run $run/10..."
    run_query Q3 base "$Q3_BASE" cold $run
    run_query Q3 proj "$Q3_PROJ" cold $run
    run_query Q3 mv "$Q3_MV" cold $run
done

# Warm runs
echo "--- Warm runs ---"
# Warm up first
clickhouse-client --database="$DB" -q "$Q3_BASE" --format=Null 2>/dev/null
clickhouse-client --database="$DB" -q "$Q3_PROJ" --format=Null 2>/dev/null
clickhouse-client --database="$DB" -q "$Q3_MV" --format=Null 2>/dev/null

for run in $(seq 1 10); do
    echo "Warm run $run/10..."
    run_query Q3 base "$Q3_BASE" warm $run
    run_query Q3 proj "$Q3_PROJ" warm $run
    run_query Q3 mv "$Q3_MV" warm $run
done

echo "=== STEP 6: OPTIMIZE FINAL ==="
for t in mp_base mp_proj mp_mv_source mp_mv_target; do
    clickhouse-client -q "SYSTEM START MERGES ${DB}.${t}"
done

for t in mp_base mp_proj mp_mv_source mp_mv_target; do
    echo "Optimizing ${t}..."
    clickhouse-client -q "OPTIMIZE TABLE ${DB}.${t} FINAL" --send_timeout=600 --receive_timeout=600
done

echo "=== STEP 7: Record system.parts state (AFTER optimize) ==="
clickhouse-client -q "
SELECT table, count() as parts, sum(rows) as total_rows, formatReadableSize(sum(bytes_on_disk)) as disk
FROM system.parts
WHERE database='${DB}' AND table LIKE 'mp_%' AND active=1
GROUP BY table ORDER BY table
FORMAT TSVWithNames
" > "$OUTDIR/parts_after_optimize.tsv"
cat "$OUTDIR/parts_after_optimize.tsv"

clickhouse-client -q "
SELECT table, partition, count() as parts, min(rows) as min_rows, max(rows) as max_rows,
  sum(rows) as total_rows, formatReadableSize(sum(bytes_on_disk)) as disk
FROM system.parts
WHERE database='${DB}' AND table LIKE 'mp_%' AND active=1
GROUP BY table, partition ORDER BY table
FORMAT TSVWithNames
" > "$OUTDIR/parts_detailed_after.tsv"

echo "=== STEP 8: Post-optimize benchmark (10 cold + 10 warm) ==="
echo "query,variant,run,cache,elapsed_ms,rows_read,bytes_read" > "$OUTDIR/singlepart_benchmark.csv"

run_query_sp() {
    local qname=$1 variant=$2 sql=$3 cache=$4 run=$5
    if [ "$cache" = "cold" ]; then
        clickhouse-client -q "SYSTEM DROP FILESYSTEM CACHE" 2>/dev/null || true
        clickhouse-client -q "SYSTEM DROP MARK CACHE"
        clickhouse-client -q "SYSTEM DROP UNCOMPRESSED CACHE"
        sync; echo 3 > /proc/sys/vm/drop_caches
        sleep 0.5
    fi
    
    local qid="exp02_sp_${variant}_${cache}_${run}_$(date +%s%N)"
    clickhouse-client --database="$DB" --query="$sql" --query_id="$qid" --format=Null 2>/dev/null
    sleep 0.3
    clickhouse-client -q "SYSTEM FLUSH LOGS"
    sleep 0.2
    
    local metrics
    metrics=$(clickhouse-client -q "
        SELECT query_duration_ms, read_rows, read_bytes
        FROM system.query_log
        WHERE query_id = '${qid}' AND type = 'QueryFinish'
        ORDER BY event_time DESC LIMIT 1
        FORMAT TSV
    ")
    
    if [ -z "$metrics" ]; then
        echo "  WARNING: no metrics for $qid" >&2
        return
    fi
    
    local elapsed rows bytes
    elapsed=$(echo "$metrics" | cut -f1)
    rows=$(echo "$metrics" | cut -f2)
    bytes=$(echo "$metrics" | cut -f3)
    echo "  ${qname}/${variant}/${cache}/run${run}: ${elapsed}ms, ${rows} rows"
    echo "${qname},${variant},${run},${cache},${elapsed},${rows},${bytes}" >> "$OUTDIR/singlepart_benchmark.csv"
}

# Stop merges again for clean benchmark
for t in mp_base mp_proj mp_mv_source mp_mv_target; do
    clickhouse-client -q "SYSTEM STOP MERGES ${DB}.${t}"
done

for run in $(seq 1 10); do
    echo "Post-opt cold run $run/10..."
    run_query_sp Q3 base "$Q3_BASE" cold $run
    run_query_sp Q3 proj "$Q3_PROJ" cold $run
    run_query_sp Q3 mv "$Q3_MV" cold $run
done

clickhouse-client --database="$DB" -q "$Q3_BASE" --format=Null 2>/dev/null
clickhouse-client --database="$DB" -q "$Q3_PROJ" --format=Null 2>/dev/null
clickhouse-client --database="$DB" -q "$Q3_MV" --format=Null 2>/dev/null

for run in $(seq 1 10); do
    echo "Post-opt warm run $run/10..."
    run_query_sp Q3 base "$Q3_BASE" warm $run
    run_query_sp Q3 proj "$Q3_PROJ" warm $run
    run_query_sp Q3 mv "$Q3_MV" warm $run
done

echo "=== DONE ==="
echo "Results in $OUTDIR/"
ls -la "$OUTDIR/"
