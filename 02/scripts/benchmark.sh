#!/bin/bash
# Experiment 02: Benchmark Script
# Runs Q1/Q2/Q3 on base/proj/mv with 5 repetitions each (cold + warm)

set -euo pipefail

DB="exp02_projections"
REPS=5
OUTDIR="/root/exp02_results"
mkdir -p "$OUTDIR"

CH="clickhouse-client --database=$DB"

# Stop merges during benchmarks
$CH --query "SYSTEM STOP MERGES $DB.web_analytics_base"
$CH --query "SYSTEM STOP MERGES $DB.web_analytics_proj"
$CH --query "SYSTEM STOP MERGES $DB.web_analytics_mv_source"
$CH --query "SYSTEM STOP MERGES $DB.hourly_stats_mv_target"

echo "query,variant,run,cache,elapsed_ms,rows_read,bytes_read" > "$OUTDIR/benchmark.csv"

run_query() {
    local name="$1" variant="$2" sql="$3" cache="$4" run="$5"
    
    if [ "$cache" = "cold" ]; then
        $CH --query "SYSTEM DROP MARK CACHE"
        $CH --query "SYSTEM DROP UNCOMPRESSED CACHE"
        $CH --query "SYSTEM DROP COMPILED EXPRESSION CACHE"
        sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    fi
    
    result=$($CH --query "$sql" --format Null 2>&1 | grep -oP '(?<=Elapsed: )\S+|(?<=Read )\S+ rows|(?<=\()\S+ rows' || echo "")
    
    # Use query_log instead
    query_id="exp02_${name}_${variant}_${cache}_${run}_$(date +%s%N)"
    $CH --query_id="$query_id" --query "$sql" --format Null 2>/dev/null
    sleep 0.5
    
    $CH --query "SYSTEM FLUSH LOGS"
    metrics=$($CH --query "
        SELECT 
            round(query_duration_ms),
            read_rows,
            read_bytes
        FROM system.query_log 
        WHERE query_id = '$query_id' AND type = 'QueryFinish'
        ORDER BY event_time DESC LIMIT 1
        FORMAT TSV
    ")
    
    elapsed=$(echo "$metrics" | cut -f1)
    rows=$(echo "$metrics" | cut -f2)
    bytes=$(echo "$metrics" | cut -f3)
    
    echo "$name,$variant,$run,$cache,$elapsed,$rows,$bytes" >> "$OUTDIR/benchmark.csv"
    echo "  $name/$variant/$cache/run$run: ${elapsed}ms, ${rows} rows read"
}

# Query definitions
Q1_BASE="SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_duration FROM web_analytics_base WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' GROUP BY hour, page ORDER BY hour"
Q1_PROJ="SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_duration FROM web_analytics_proj WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' GROUP BY hour, page ORDER BY hour"
Q1_MV="SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_duration FROM hourly_stats_mv_target WHERE page = '/page/42' AND hour >= '2024-06-01' AND hour < '2024-06-02' GROUP BY page, hour ORDER BY hour"

Q2_BASE="SELECT count(), avg(duration_ms) FROM web_analytics_base WHERE country = 'DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'"
Q2_PROJ="SELECT count(), avg(duration_ms) FROM web_analytics_proj WHERE country = 'DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'"

Q3_BASE="SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_base GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
Q3_PROJ="SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_proj GROUP BY page ORDER BY avg_dur DESC LIMIT 10"
Q3_MV="SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM hourly_stats_mv_target GROUP BY page ORDER BY avg_dur DESC LIMIT 10"

echo "=== Starting Benchmark ==="

# Randomized order for fairness
for cache in cold warm; do
    for run in $(seq 1 $REPS); do
        echo "--- $cache run $run ---"
        
        # Shuffle query order
        for combo in $(shuf <<< "Q1:base:$Q1_BASE
Q1:proj:$Q1_PROJ
Q1:mv:$Q1_MV
Q2:base:$Q2_BASE
Q2:proj:$Q2_PROJ
Q3:base:$Q3_BASE
Q3:proj:$Q3_PROJ
Q3:mv:$Q3_MV"); do
            qname=$(echo "$combo" | cut -d: -f1)
            variant=$(echo "$combo" | cut -d: -f2)
            sql=$(echo "$combo" | cut -d: -f3-)
            run_query "$qname" "$variant" "$sql" "$cache" "$run"
        done
    done
done

echo "=== Benchmark Complete ==="

# Resume merges
$CH --query "SYSTEM START MERGES $DB.web_analytics_base"
$CH --query "SYSTEM START MERGES $DB.web_analytics_proj"
$CH --query "SYSTEM START MERGES $DB.web_analytics_mv_source"
$CH --query "SYSTEM START MERGES $DB.hourly_stats_mv_target"

echo "Results in $OUTDIR/benchmark.csv"
