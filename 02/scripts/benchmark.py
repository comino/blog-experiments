#!/usr/bin/env python3
"""Experiment 02: Benchmark - Projections vs Materialized Views"""

import subprocess
import csv
import random
import time
import sys
import os

DB = "exp02_projections"
REPS = 5
OUTDIR = "/root/exp02_results"
os.makedirs(OUTDIR, exist_ok=True)

def ch(sql, **kwargs):
    cmd = ["clickhouse-client", f"--database={DB}", "--query", sql]
    for k, v in kwargs.items():
        cmd.extend([f"--{k.replace('_','-')}", str(v)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return r.stdout.strip()

# Queries
QUERIES = [
    ("Q1", "base", "SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_duration FROM web_analytics_base WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' GROUP BY hour, page ORDER BY hour"),
    ("Q1", "proj", "SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_duration FROM web_analytics_proj WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' GROUP BY hour, page ORDER BY hour"),
    ("Q1", "mv", "SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_duration FROM hourly_stats_mv_target WHERE page = '/page/42' AND hour >= '2024-06-01' AND hour < '2024-06-02' GROUP BY page, hour ORDER BY hour"),
    ("Q2", "base", "SELECT count(), avg(duration_ms) FROM web_analytics_base WHERE country = 'DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'"),
    ("Q2", "proj", "SELECT count(), avg(duration_ms) FROM web_analytics_proj WHERE country = 'DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'"),
    ("Q3", "base", "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_base GROUP BY page ORDER BY avg_dur DESC LIMIT 10"),
    ("Q3", "proj", "SELECT page, avg(duration_ms) AS avg_dur, count() AS hits FROM web_analytics_proj GROUP BY page ORDER BY avg_dur DESC LIMIT 10"),
    ("Q3", "mv", "SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits FROM hourly_stats_mv_target GROUP BY page ORDER BY avg_dur DESC LIMIT 10"),
]

def drop_caches():
    ch("SYSTEM DROP MARK CACHE")
    ch("SYSTEM DROP UNCOMPRESSED CACHE")
    ch("SYSTEM DROP COMPILED EXPRESSION CACHE")
    try:
        subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, timeout=10)
    except:
        pass

def run_query(qname, variant, sql, cache, run):
    if cache == "cold":
        drop_caches()
    
    qid = f"exp02_{qname}_{variant}_{cache}_{run}_{int(time.time()*1000)}"
    ch(sql, query_id=qid, format="Null")
    time.sleep(0.3)
    ch("SYSTEM FLUSH LOGS")
    time.sleep(0.2)
    
    metrics = ch(f"""
        SELECT query_duration_ms, read_rows, read_bytes
        FROM system.query_log
        WHERE query_id = '{qid}' AND type = 'QueryFinish'
        ORDER BY event_time DESC LIMIT 1
        FORMAT TSV
    """)
    
    if not metrics:
        print(f"  WARNING: No metrics for {qid}", file=sys.stderr)
        return None
    
    parts = metrics.split('\t')
    elapsed, rows, bytes_ = int(parts[0]), int(parts[1]), int(parts[2])
    print(f"  {qname}/{variant}/{cache}/run{run}: {elapsed}ms, {rows:,} rows", flush=True)
    return {"query": qname, "variant": variant, "run": run, "cache": cache,
            "elapsed_ms": elapsed, "rows_read": rows, "bytes_read": bytes_}

# Stop merges
print("Stopping merges...")
for t in ["web_analytics_base", "web_analytics_proj", "web_analytics_mv_source", "hourly_stats_mv_target"]:
    ch(f"SYSTEM STOP MERGES {DB}.{t}")

results = []

for cache in ["cold", "warm"]:
    for run in range(1, REPS + 1):
        print(f"\n--- {cache} run {run}/{REPS} ---", flush=True)
        order = list(QUERIES)
        random.shuffle(order)
        for qname, variant, sql in order:
            r = run_query(qname, variant, sql, cache, run)
            if r:
                results.append(r)

# Write CSV
outpath = f"{OUTDIR}/benchmark.csv"
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=["query", "variant", "run", "cache", "elapsed_ms", "rows_read", "bytes_read"])
    w.writeheader()
    w.writerows(results)

print(f"\nResults written to {outpath}")

# Resume merges
for t in ["web_analytics_base", "web_analytics_proj", "web_analytics_mv_source", "hourly_stats_mv_target"]:
    ch(f"SYSTEM START MERGES {DB}.{t}")

print("Merges resumed. Done!")
