#!/bin/bash
# Experiment 01 Extended: Distribution Analysis
# Tests 4 data distributions × 5 codec variants
# Run ON the ClickHouse server: bash distribution_benchmark.sh
set -e

CH="clickhouse-client --database exp01_compression"
OUTDIR="/tmp/exp01_extended"
mkdir -p $OUTDIR

DISTRIBUTIONS="monotone sinus spiky random"
VARIANTS="v1 v2 v3 v4 v5"

echo "=== PART 2: DISTRIBUTION ANALYSIS ==="

# Create 20 distribution variant tables
for dist in $DISTRIBUTIONS; do
    src="dist_${dist}"
    for v in $VARIANTS; do
        tbl="dist_${dist}_${v}"
        echo "Creating $tbl..."
        $CH --query "DROP TABLE IF EXISTS $tbl"

        case $v in
            v1) $CH --query "CREATE TABLE $tbl (
                    timestamp DateTime, value Float64, counter UInt64, tag LowCardinality(String)
                ) ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192" ;;
            v2) $CH --query "CREATE TABLE $tbl (
                    timestamp DateTime CODEC(ZSTD(3)), value Float64 CODEC(ZSTD(3)),
                    counter UInt64 CODEC(ZSTD(3)), tag LowCardinality(String) CODEC(ZSTD(3))
                ) ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192" ;;
            v3) $CH --query "CREATE TABLE $tbl (
                    timestamp DateTime CODEC(DoubleDelta, LZ4), value Float64 CODEC(Gorilla, LZ4),
                    counter UInt64 CODEC(Delta, ZSTD(1)), tag LowCardinality(String) CODEC(LZ4)
                ) ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192" ;;
            v4) $CH --query "CREATE TABLE $tbl (
                    timestamp DateTime CODEC(DoubleDelta, ZSTD(3)), value Float64 CODEC(Gorilla, ZSTD(3)),
                    counter UInt64 CODEC(Delta, ZSTD(3)), tag LowCardinality(String) CODEC(ZSTD(3))
                ) ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192" ;;
            v5) $CH --query "CREATE TABLE $tbl (
                    timestamp DateTime CODEC(DoubleDelta, ZSTD(9)), value Float64 CODEC(Gorilla, ZSTD(3)),
                    counter UInt64 CODEC(Delta, ZSTD(9)), tag LowCardinality(String) CODEC(ZSTD(9))
                ) ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192" ;;
        esac

        $CH --query "INSERT INTO $tbl SELECT * FROM $src"
        $CH --query "OPTIMIZE TABLE $tbl FINAL"
        echo "$tbl loaded + optimized"
    done
done

# Collect storage
echo "distribution,variant,column,compressed_bytes,uncompressed_bytes,ratio" > $OUTDIR/distributions_storage.csv
for dist in $DISTRIBUTIONS; do
    for v in $VARIANTS; do
        $CH --query "SELECT '$dist','$v', name, data_compressed_bytes, data_uncompressed_bytes,
            round(data_uncompressed_bytes / if(data_compressed_bytes=0,1,data_compressed_bytes), 2)
            FROM system.columns WHERE database='exp01_compression' AND table='dist_${dist}_${v}'
            FORMAT CSV" >> $OUTDIR/distributions_storage.csv
    done
done
echo "=== DISTRIBUTION STORAGE DONE ==="

# Query benchmark on distributions
$CH --query "SYSTEM STOP MERGES exp01_compression"

for dist in $DISTRIBUTIONS; do
    for v in $VARIANTS; do
        tbl="dist_${dist}_${v}"
        for run in 1 2 3; do
            # DQ1: Range aggregation
            $CH --query "SYSTEM DROP FILESYSTEM CACHE" 2>/dev/null
            QID="dist_cold_DQ1_${dist}_${v}_${run}_$(date +%s%N)"
            $CH --query_id "$QID" --query "SELECT toStartOfHour(timestamp) h, avg(value) FROM $tbl WHERE timestamp BETWEEN '2024-02-01' AND '2024-02-08' GROUP BY h FORMAT Null" 2>/dev/null
            sleep 0.2
            QID="dist_warm_DQ1_${dist}_${v}_${run}_$(date +%s%N)"
            $CH --query_id "$QID" --query "SELECT toStartOfHour(timestamp) h, avg(value) FROM $tbl WHERE timestamp BETWEEN '2024-02-01' AND '2024-02-08' GROUP BY h FORMAT Null" 2>/dev/null
            sleep 0.2

            # DQ2: Full scan aggregation
            $CH --query "SYSTEM DROP FILESYSTEM CACHE" 2>/dev/null
            QID="dist_cold_DQ2_${dist}_${v}_${run}_$(date +%s%N)"
            $CH --query_id "$QID" --query "SELECT count(), avg(value), max(counter) FROM $tbl FORMAT Null" 2>/dev/null
            sleep 0.2
            QID="dist_warm_DQ2_${dist}_${v}_${run}_$(date +%s%N)"
            $CH --query_id "$QID" --query "SELECT count(), avg(value), max(counter) FROM $tbl FORMAT Null" 2>/dev/null
            sleep 0.2
        done
        echo "  Done: $dist $v"
    done
done

# Collect distribution query results
sleep 1
$CH --query "
SELECT
    extractAllGroupsVertical(query_id, 'dist_(cold|warm)_(DQ[0-9])_([a-z]+)_(v[0-9])_([0-9]+)_')[1][2] AS query,
    extractAllGroupsVertical(query_id, 'dist_(cold|warm)_(DQ[0-9])_([a-z]+)_(v[0-9])_([0-9]+)_')[1][3] AS distribution,
    extractAllGroupsVertical(query_id, 'dist_(cold|warm)_(DQ[0-9])_([a-z]+)_(v[0-9])_([0-9]+)_')[1][4] AS variant,
    extractAllGroupsVertical(query_id, 'dist_(cold|warm)_(DQ[0-9])_([a-z]+)_(v[0-9])_([0-9]+)_')[1][5] AS run,
    extractAllGroupsVertical(query_id, 'dist_(cold|warm)_(DQ[0-9])_([a-z]+)_(v[0-9])_([0-9]+)_')[1][1] AS temp,
    query_duration_ms,
    read_rows,
    read_bytes,
    ProfileEvents['OSCPUVirtualTimeMicroseconds'] AS cpu_us
FROM system.query_log
WHERE query_id LIKE 'dist_%' AND type = 'QueryFinish'
    AND extractAllGroupsVertical(query_id, 'dist_(cold|warm)_(DQ[0-9])_([a-z]+)_(v[0-9])_([0-9]+)_')[1][1] != ''
ORDER BY query, distribution, variant, run, temp
FORMAT CSVWithNames
" > $OUTDIR/distributions_queries.csv

$CH --query "SYSTEM START MERGES exp01_compression"

echo "=== PART 2 COMPLETE ==="
