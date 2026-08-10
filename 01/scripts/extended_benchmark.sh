#!/bin/bash
# Experiment 01 Extended: Full benchmark suite
# Run ON the ClickHouse server: bash extended_benchmark.sh
# Duration: ~2-4 hours
set -e

CH="clickhouse-client --database exp01_compression"
OUTDIR="/tmp/exp01_extended"
mkdir -p $OUTDIR

VARIANTS="v1_default v2_zstd v3_percolumn v4_percolumn_zstd v5_aggressive"

# ═══════════════════════════════════════════════════
# PART 1: SCALING ANALYSIS (1M, 10M, 100M, 500M)
# ═══════════════════════════════════════════════════
echo "=== PART 1: SCALING ANALYSIS ==="

create_variant_table() {
    local name=$1
    local variant=$2
    case $variant in
        v1_default)
            $CH --query "CREATE TABLE IF NOT EXISTS $name (
                timestamp DateTime, metric_name LowCardinality(String), value Float64,
                host LowCardinality(String), region LowCardinality(String), counter UInt64
            ) ENGINE = MergeTree() ORDER BY (metric_name, host, timestamp) SETTINGS index_granularity = 8192" ;;
        v2_zstd)
            $CH --query "CREATE TABLE IF NOT EXISTS $name (
                timestamp DateTime CODEC(ZSTD(3)), metric_name LowCardinality(String) CODEC(ZSTD(3)),
                value Float64 CODEC(ZSTD(3)), host LowCardinality(String) CODEC(ZSTD(3)),
                region LowCardinality(String) CODEC(ZSTD(3)), counter UInt64 CODEC(ZSTD(3))
            ) ENGINE = MergeTree() ORDER BY (metric_name, host, timestamp) SETTINGS index_granularity = 8192" ;;
        v3_percolumn)
            $CH --query "CREATE TABLE IF NOT EXISTS $name (
                timestamp DateTime CODEC(DoubleDelta, LZ4), metric_name LowCardinality(String) CODEC(LZ4),
                value Float64 CODEC(Gorilla, LZ4), host LowCardinality(String) CODEC(LZ4),
                region LowCardinality(String) CODEC(LZ4), counter UInt64 CODEC(Delta, ZSTD(1))
            ) ENGINE = MergeTree() ORDER BY (metric_name, host, timestamp) SETTINGS index_granularity = 8192" ;;
        v4_percolumn_zstd)
            $CH --query "CREATE TABLE IF NOT EXISTS $name (
                timestamp DateTime CODEC(DoubleDelta, ZSTD(3)), metric_name LowCardinality(String) CODEC(ZSTD(3)),
                value Float64 CODEC(Gorilla, ZSTD(3)), host LowCardinality(String) CODEC(ZSTD(3)),
                region LowCardinality(String) CODEC(ZSTD(3)), counter UInt64 CODEC(Delta, ZSTD(3))
            ) ENGINE = MergeTree() ORDER BY (metric_name, host, timestamp) SETTINGS index_granularity = 8192" ;;
        v5_aggressive)
            $CH --query "CREATE TABLE IF NOT EXISTS $name (
                timestamp DateTime CODEC(DoubleDelta, ZSTD(9)), metric_name LowCardinality(String) CODEC(ZSTD(9)),
                value Float64 CODEC(Gorilla, ZSTD(3)), host LowCardinality(String) CODEC(ZSTD(9)),
                region LowCardinality(String) CODEC(ZSTD(9)), counter UInt64 CODEC(Delta, ZSTD(9))
            ) ENGINE = MergeTree() ORDER BY (metric_name, host, timestamp) SETTINGS index_granularity = 8192" ;;
    esac
}

# Create scaling variant tables and load data
for size in 1m 10m 500m; do
    src="source_${size}"
    for v in $VARIANTS; do
        tbl="scale_${size}_${v}"
        echo "Creating + loading $tbl..."
        $CH --query "DROP TABLE IF EXISTS $tbl"
        create_variant_table "$tbl" "$v"
        $CH --query "INSERT INTO $tbl SELECT * FROM $src"
        $CH --query "OPTIMIZE TABLE $tbl FINAL"
        echo "$tbl done"
    done
done

# 100M tables already exist from original experiment (v1_default..v5_aggressive)

echo "Collecting scaling storage data..."
echo "size,variant,column,compressed_bytes,uncompressed_bytes,ratio" > $OUTDIR/scaling_storage.csv

# 100M from original tables
for v in $VARIANTS; do
    $CH --query "SELECT '100m','$v', name, data_compressed_bytes, data_uncompressed_bytes,
        round(data_uncompressed_bytes / data_compressed_bytes, 2)
        FROM system.columns WHERE database='exp01_compression' AND table='$v'
        FORMAT CSV" >> $OUTDIR/scaling_storage.csv
done

# Other sizes from scale_ tables
for size in 1m 10m 500m; do
    for v in $VARIANTS; do
        $CH --query "SELECT '$size','$v', name, data_compressed_bytes, data_uncompressed_bytes,
            round(data_uncompressed_bytes / data_compressed_bytes, 2)
            FROM system.columns WHERE database='exp01_compression' AND table='scale_${size}_${v}'
            FORMAT CSV" >> $OUTDIR/scaling_storage.csv
    done
done

echo "=== SCALING STORAGE DONE ==="

# ── Scaling Query Benchmark ──
echo "query,size,variant,run,temp,duration_ms,read_rows,read_bytes,cpu_us" > $OUTDIR/scaling_queries.csv

$CH --query "SYSTEM STOP MERGES exp01_compression"

for size in 1m 10m 100m 500m; do
    for v in $VARIANTS; do
        if [ "$size" = "100m" ]; then
            tbl="$v"
        else
            tbl="scale_${size}_${v}"
        fi

        for run in 1 2 3; do
            for q in Q1 Q2 Q3 Q4 Q5 Q6; do
                case $q in
                    Q1) SQL="SELECT toStartOfHour(timestamp) h, avg(value) FROM $tbl WHERE timestamp BETWEEN '2024-01-15 00:00:00' AND '2024-01-15 01:00:00' GROUP BY h FORMAT Null" ;;
                    Q2) SQL="SELECT toStartOfHour(timestamp) h, avg(value) FROM $tbl WHERE timestamp BETWEEN '2024-01-15' AND '2024-01-22' GROUP BY h FORMAT Null" ;;
                    Q3) SQL="SELECT host, sum(counter) FROM $tbl GROUP BY host ORDER BY 2 DESC LIMIT 10 FORMAT Null" ;;
                    Q4) SQL="SELECT timestamp, value, counter FROM $tbl WHERE host = 'host-7' AND timestamp BETWEEN '2024-01-20 12:00:00' AND '2024-01-20 12:05:00' FORMAT Null" ;;
                    Q5) SQL="SELECT count() FROM $tbl WHERE metric_name IN ('cpu_usage','mem_free') AND value > 50 AND region = 'eu-central' FORMAT Null" ;;
                    Q6) SQL="SELECT host, region, metric_name, count(), avg(value), max(counter), min(timestamp), max(timestamp) FROM $tbl GROUP BY host, region, metric_name FORMAT Null" ;;
                esac

                # Cold
                $CH --query "SYSTEM DROP FILESYSTEM CACHE" 2>/dev/null
                QID="ext_cold_${q}_${size}_${v}_${run}_$(date +%s%N)"
                $CH --query_id "$QID" --query "$SQL" 2>/dev/null
                sleep 0.3

                # Warm
                QID2="ext_warm_${q}_${size}_${v}_${run}_$(date +%s%N)"
                $CH --query_id "$QID2" --query "$SQL" 2>/dev/null
                sleep 0.3
            done
            echo "  Done: $size $v run$run"
        done
    done
    echo "=== Size $size queries done ==="
done

# Collect query results from query_log
$CH --query "
SELECT
    extractAllGroupsVertical(query_id, 'ext_(cold|warm)_(Q[0-9])_([0-9]+m)_(v[0-9a-z_]+)_([0-9]+)_')[1][2] AS query,
    extractAllGroupsVertical(query_id, 'ext_(cold|warm)_(Q[0-9])_([0-9]+m)_(v[0-9a-z_]+)_([0-9]+)_')[1][3] AS size,
    extractAllGroupsVertical(query_id, 'ext_(cold|warm)_(Q[0-9])_([0-9]+m)_(v[0-9a-z_]+)_([0-9]+)_')[1][4] AS variant,
    extractAllGroupsVertical(query_id, 'ext_(cold|warm)_(Q[0-9])_([0-9]+m)_(v[0-9a-z_]+)_([0-9]+)_')[1][5] AS run,
    extractAllGroupsVertical(query_id, 'ext_(cold|warm)_(Q[0-9])_([0-9]+m)_(v[0-9a-z_]+)_([0-9]+)_')[1][1] AS temp,
    query_duration_ms,
    read_rows,
    read_bytes,
    ProfileEvents['OSCPUVirtualTimeMicroseconds'] AS cpu_us
FROM system.query_log
WHERE query_id LIKE 'ext_%' AND type = 'QueryFinish'
    AND extractAllGroupsVertical(query_id, 'ext_(cold|warm)_(Q[0-9])_([0-9]+m)_(v[0-9a-z_]+)_([0-9]+)_')[1][1] != ''
ORDER BY query, size, variant, run, temp
FORMAT CSVWithNames
" > $OUTDIR/scaling_queries.csv

echo "=== SCALING QUERIES COLLECTED ==="

# ── Scaling Ingest Benchmark ──
echo "size,variant,run,rows,duration_s,rows_per_s" > $OUTDIR/scaling_ingest.csv

for size in 1m 10m 100m 500m; do
    case $size in
        1m) ROWS=1000000; src="source_1m" ;;
        10m) ROWS=10000000; src="source_10m" ;;
        100m) ROWS=100000000; src="source" ;;
        500m) ROWS=500000000; src="source_500m" ;;
    esac

    for v in $VARIANTS; do
        INGEST_TBL="ingest_tmp_${v}"
        $CH --query "DROP TABLE IF EXISTS $INGEST_TBL"
        create_variant_table "$INGEST_TBL" "$v"

        for run in 1 2 3; do
            $CH --query "TRUNCATE TABLE $INGEST_TBL"
            START=$(date +%s%N)
            $CH --query "INSERT INTO $INGEST_TBL SELECT * FROM $src LIMIT $ROWS"
            END=$(date +%s%N)
            DUR=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
            RPS=$(echo "scale=0; $ROWS / $DUR" | bc)
            echo "$size,$v,$run,$ROWS,$DUR,$RPS" >> $OUTDIR/scaling_ingest.csv
            echo "  Ingest: $size $v run$run ${DUR}s (${RPS} rows/s)"
        done

        $CH --query "DROP TABLE $INGEST_TBL"
    done
    echo "=== Ingest $size done ==="
done

$CH --query "SYSTEM START MERGES exp01_compression"

echo "=== PART 1 COMPLETE ==="
