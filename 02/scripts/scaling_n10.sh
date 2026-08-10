#!/bin/bash
# Task 2: Scaling with n=10 + IQR at 10M and 50M
set -euo pipefail

DB="exp02_projections"
OUTDIR="/root/exp02_v5"
mkdir -p "$OUTDIR"

echo "=== Check if 50M tables exist ==="
TABLES_50M=$(clickhouse-client -q "SELECT count() FROM system.tables WHERE database='${DB}' AND name LIKE 'scale_50m%'")
echo "50M tables found: $TABLES_50M"

if [ "$TABLES_50M" -eq 0 ]; then
    echo "Creating 50M tables..."
    
    clickhouse-client -q "
    CREATE TABLE ${DB}.scale_50m_base (
        timestamp DateTime, user_id UInt32, page LowCardinality(String),
        duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)
    ) ENGINE = MergeTree() ORDER BY (page, timestamp);
    "
    
    clickhouse-client -q "
    CREATE TABLE ${DB}.scale_50m_proj (
        timestamp DateTime, user_id UInt32, page LowCardinality(String),
        duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String),
        PROJECTION proj_hourly_stats (
            SELECT page, toStartOfHour(timestamp) AS hour,
                   count() AS hits, avg(duration_ms) AS avg_duration, sum(duration_ms) AS sum_duration
            GROUP BY page, hour
        )
    ) ENGINE = MergeTree() ORDER BY (page, timestamp);
    "
    
    clickhouse-client -q "
    CREATE TABLE ${DB}.scale_50m_mv_source (
        timestamp DateTime, user_id UInt32, page LowCardinality(String),
        duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)
    ) ENGINE = MergeTree() ORDER BY (page, timestamp);
    "
    
    clickhouse-client -q "
    CREATE TABLE ${DB}.scale_50m_mv_target (
        page LowCardinality(String), hour DateTime,
        hits AggregateFunction(count, UInt64),
        avg_duration AggregateFunction(avg, UInt32),
        sum_duration AggregateFunction(sum, UInt32)
    ) ENGINE = AggregatingMergeTree() ORDER BY (page, hour);
    "
    
    clickhouse-client -q "
    CREATE MATERIALIZED VIEW ${DB}.scale_50m_mv TO ${DB}.scale_50m_mv_target AS
    SELECT page, toStartOfHour(timestamp) AS hour,
           countState() AS hits, avgState(duration_ms) AS avg_duration, sumState(duration_ms) AS sum_duration
    FROM ${DB}.scale_50m_mv_source GROUP BY page, hour;
    "
    
    echo "Inserting 50M rows..."
    for tbl in scale_50m_base scale_50m_proj scale_50m_mv_source; do
        clickhouse-client -q "
        INSERT INTO ${DB}.${tbl}
        SELECT
            toDateTime('2024-01-01') + toIntervalSecond(rand(42) % (365*86400)),
            rand(43) % 100000,
            concat('/page/', toString(rand(44) % 1000)),
            rand(45) % 10000,
            ['US','DE','GB','FR','JP','AU','BR','IN','CA','NL'][1 + rand(46) % 10],
            ['desktop','mobile','tablet'][1 + rand(47) % 3]
        FROM numbers(50000000);
        "
    done
    
    echo "Optimizing 50M tables..."
    for tbl in scale_50m_base scale_50m_proj scale_50m_mv_source scale_50m_mv_target; do
        clickhouse-client -q "OPTIMIZE TABLE ${DB}.${tbl} FINAL"
    done
fi

echo "=== Run Q3 n=10 at 10M ==="
echo "query,variant,size,run,cache,elapsed_ms,rows_read,bytes_read" > "$OUTDIR/scaling_n10.csv"

run_bench() {
    local size=$1 variant=$2 sql=$3 cache=$4 run=$5
    if [ "$cache" = "cold" ]; then
        clickhouse-client -q "SYSTEM DROP FILESYSTEM CACHE" 2>/dev/null || true
        clickhouse-client -q "SYSTEM DROP MARK CACHE"
        clickhouse-client -q "SYSTEM DROP UNCOMPRESSED CACHE"
        sync; echo 3 > /proc/sys/vm/drop_caches
        sleep 0.5
    fi
    
    local qid="exp02_scale_${size}_${variant}_${cache}_${run}_$(date +%s%N)"
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
        echo "  WARNING: no metrics" >&2
        return
    fi
    
    local elapsed rows bytes
    elapsed=$(echo "$metrics" | cut -f1)
    rows=$(echo "$metrics" | cut -f2)
    bytes=$(echo "$metrics" | cut -f3)
    echo "  Q3/${variant}/${size}/${cache}/run${run}: ${elapsed}ms"
    echo "Q3,${variant},${size},${run},${cache},${elapsed},${rows},${bytes}" >> "$OUTDIR/scaling_n10.csv"
}

# 10M queries
Q3_10M_BASE="SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM ${DB}.scale_10m_base GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
Q3_10M_PROJ="SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM ${DB}.scale_10m_proj GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
Q3_10M_MV="SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM ${DB}.scale_10m_mv_target GROUP BY page ORDER BY avg_dur DESC LIMIT 10"

# 50M queries
Q3_50M_BASE="SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM ${DB}.scale_50m_base GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
Q3_50M_PROJ="SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM ${DB}.scale_50m_proj GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
Q3_50M_MV="SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM ${DB}.scale_50m_mv_target GROUP BY page ORDER BY avg_dur DESC LIMIT 10"

echo "--- 10M cold ---"
for run in $(seq 1 10); do
    run_bench 10M base "$Q3_10M_BASE" cold $run
    run_bench 10M proj "$Q3_10M_PROJ" cold $run
    run_bench 10M mv "$Q3_10M_MV" cold $run
done

echo "--- 10M warm ---"
clickhouse-client --database="$DB" -q "$Q3_10M_BASE" --format=Null 2>/dev/null
clickhouse-client --database="$DB" -q "$Q3_10M_PROJ" --format=Null 2>/dev/null
clickhouse-client --database="$DB" -q "$Q3_10M_MV" --format=Null 2>/dev/null
for run in $(seq 1 10); do
    run_bench 10M base "$Q3_10M_BASE" warm $run
    run_bench 10M proj "$Q3_10M_PROJ" warm $run
    run_bench 10M mv "$Q3_10M_MV" warm $run
done

echo "--- 50M cold ---"
for run in $(seq 1 10); do
    run_bench 50M base "$Q3_50M_BASE" cold $run
    run_bench 50M proj "$Q3_50M_PROJ" cold $run
    run_bench 50M mv "$Q3_50M_MV" cold $run
done

echo "--- 50M warm ---"
clickhouse-client --database="$DB" -q "$Q3_50M_BASE" --format=Null 2>/dev/null
clickhouse-client --database="$DB" -q "$Q3_50M_PROJ" --format=Null 2>/dev/null
clickhouse-client --database="$DB" -q "$Q3_50M_MV" --format=Null 2>/dev/null
for run in $(seq 1 10); do
    run_bench 50M base "$Q3_50M_BASE" warm $run
    run_bench 50M proj "$Q3_50M_PROJ" warm $run
    run_bench 50M mv "$Q3_50M_MV" warm $run
done

echo "=== DONE ==="
