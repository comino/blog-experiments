#!/bin/bash
# Task 6: Performance re-run with n=10 for 5 interesting queries
set -euo pipefail

DB="exp03_llm"
OUTDIR="/root/exp03_v3"
mkdir -p "$OUTDIR"

echo "question_id,variant,run,elapsed_ms,read_rows,read_bytes" > "$OUTDIR/performance_n10.csv"

run_query() {
    local qid=$1 variant=$2 sql=$3 run=$4
    
    # Drop caches for cold measurement
    clickhouse-client -q "SYSTEM DROP FILESYSTEM CACHE" 2>/dev/null || true
    clickhouse-client -q "SYSTEM DROP MARK CACHE"
    clickhouse-client -q "SYSTEM DROP UNCOMPRESSED CACHE"
    sync; echo 3 > /proc/sys/vm/drop_caches
    sleep 0.3
    
    local uid="exp03_perf_${qid}_${variant}_${run}_$(date +%s%N)"
    clickhouse-client --database="$DB" --query="$sql" --query_id="$uid" --format=Null 2>/dev/null
    sleep 0.3
    clickhouse-client -q "SYSTEM FLUSH LOGS"
    sleep 0.2
    
    local metrics
    metrics=$(clickhouse-client -q "
        SELECT query_duration_ms, read_rows, read_bytes
        FROM system.query_log
        WHERE query_id = '${uid}' AND type = 'QueryFinish'
        ORDER BY event_time DESC LIMIT 1
        FORMAT TSV
    ")
    
    if [ -z "$metrics" ]; then
        echo "  WARNING: no metrics for $uid" >&2
        return
    fi
    
    local elapsed rows bytes
    elapsed=$(echo "$metrics" | cut -f1)
    rows=$(echo "$metrics" | cut -f2)
    bytes=$(echo "$metrics" | cut -f3)
    echo "  ${qid}/${variant}/run${run}: ${elapsed}ms, ${rows} rows"
    echo "${qid},${variant},${run},${elapsed},${rows},${bytes}" >> "$OUTDIR/performance_n10.csv"
}

echo "=== t4_03: Users with no events ==="
REF_t4_03="SELECT u.user_id, u.email FROM users AS u LEFT JOIN events AS e ON u.user_id = e.user_id WHERE e.user_id IS NULL"
LLM_t4_03="SELECT u.user_id FROM users AS u LEFT JOIN (SELECT DISTINCT user_id FROM events) AS e USING (user_id) WHERE e.user_id IS NULL"
for run in $(seq 1 10); do
    run_query t4_03 ref "$REF_t4_03" $run
    run_query t4_03 llm "$LLM_t4_03" $run
done

echo "=== t4_07: Avg events per plan ==="
REF_t4_07="WITH user_counts AS (SELECT e.user_id, u.plan, count() AS cnt FROM events AS e INNER JOIN users AS u ON e.user_id = u.user_id GROUP BY e.user_id, u.plan) SELECT plan, avg(cnt) AS avg_events_per_user FROM user_counts GROUP BY plan ORDER BY avg_events_per_user DESC"
LLM_t4_07="SELECT u.plan, avg(e.events_per_user) AS avg_events_per_user FROM (SELECT user_id, count() AS events_per_user FROM events GROUP BY user_id) AS e INNER JOIN users AS u ON e.user_id = u.user_id GROUP BY u.plan ORDER BY avg_events_per_user DESC"
for run in $(seq 1 10); do
    run_query t4_07 ref "$REF_t4_07" $run
    run_query t4_07 llm "$LLM_t4_07" $run
done

echo "=== t5_09: Hourly counts with fill ==="
REF_t5_09="SELECT toStartOfHour(timestamp) AS hour, count() AS cnt FROM events WHERE toDate(timestamp) = '2024-06-15' GROUP BY hour ORDER BY hour WITH FILL FROM toDateTime('2024-06-15 00:00:00') TO toDateTime('2024-06-16 00:00:00') STEP INTERVAL 1 HOUR"
LLM_t5_09="SELECT toStartOfHour(timestamp) AS hour, count() AS cnt FROM events WHERE timestamp >= '2024-06-15 00:00:00' AND timestamp < '2024-06-16 00:00:00' GROUP BY hour ORDER BY hour WITH FILL FROM toDateTime('2024-06-15 00:00:00') TO toDateTime('2024-06-16 00:00:00') STEP INTERVAL 1 HOUR"
for run in $(seq 1 10); do
    run_query t5_09 ref "$REF_t5_09" $run
    run_query t5_09 llm "$LLM_t5_09" $run
done

echo "=== t2_06: Avg pageview duration ==="
REF_t2_06="SELECT avgIf(duration_ms, event_type = 'pageview') AS avg_pageview_duration FROM events"
LLM_t2_06="SELECT avg(duration_ms) FROM events WHERE event_type = 'pageview'"
for run in $(seq 1 10); do
    run_query t2_06 ref "$REF_t2_06" $run
    run_query t2_06 llm "$LLM_t2_06" $run
done

echo "=== t2_08: Distinct countries per event type ==="
REF_t2_08="SELECT event_type, uniq(country) AS country_count FROM events GROUP BY event_type ORDER BY country_count DESC"
LLM_t2_08="SELECT event_type, count(DISTINCT country) AS distinct_countries FROM events GROUP BY event_type ORDER BY distinct_countries DESC"
for run in $(seq 1 10); do
    run_query t2_08 ref "$REF_t2_08" $run
    run_query t2_08 llm "$LLM_t2_08" $run
done

echo "=== DONE ==="
