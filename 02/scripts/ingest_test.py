#!/usr/bin/env python3
"""Experiment 02: Ingest Impact Test"""

import subprocess
import time
import os
import csv

DB = "exp02_projections"
ROWS = 10_000_000
REPS = 3
OUTDIR = "/root/exp02_results"

def ch(sql, timeout=300):
    r = subprocess.run(["clickhouse-client", f"--database={DB}", "--query", sql],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"ERROR: {r.stderr}")
    return r.stdout.strip()

INSERT_SQL = f"""INSERT INTO {{table}}
SELECT
    toDateTime('2025-01-01') + toIntervalSecond(rand() % (30 * 86400)) AS timestamp,
    rand() % 1000000 + 1 AS user_id,
    concat('/page/', toString(rand() % 1000)) AS page,
    50 + rand() % 9950 AS duration_ms,
    arrayElement(['US','DE','UK','FR','JP','BR','IN','CA','AU','MX','IT','ES','KR','NL','SE','CH','AT','BE','PL','CZ','DK','NO','FI','PT','IE','RO','HU','GR','BG','HR','SK','SI','LT','LV','EE','IL','TR','ZA','NG','EG','KE','AR','CL','CO','PE','TH','VN','MY','PH','ID'], (rand() % 50) + 1) AS country,
    arrayElement(['desktop','mobile','tablet','smart_tv','wearable'], (rand() % 5) + 1) AS device_type
FROM numbers({ROWS})"""

# Scenario 1: Base only (no projections, no MV)
# Scenario 2: Base + Projections
# Scenario 3: Base + MV

SCENARIOS = {
    "base_only": {
        "setup": [
            "DROP TABLE IF EXISTS ingest_base_only",
            f"""CREATE TABLE ingest_base_only (
                timestamp DateTime, user_id UInt32, page LowCardinality(String),
                duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)
            ) ENGINE = MergeTree() ORDER BY (page, timestamp)"""
        ],
        "table": "ingest_base_only",
        "cleanup": ["DROP TABLE IF EXISTS ingest_base_only"]
    },
    "base_proj": {
        "setup": [
            "DROP TABLE IF EXISTS ingest_base_proj",
            f"""CREATE TABLE ingest_base_proj (
                timestamp DateTime, user_id UInt32, page LowCardinality(String),
                duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String),
                PROJECTION proj_country (SELECT * ORDER BY (country, timestamp)),
                PROJECTION proj_agg (SELECT page, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur, sum(duration_ms) AS sum_dur GROUP BY page, hour)
            ) ENGINE = MergeTree() ORDER BY (page, timestamp)"""
        ],
        "table": "ingest_base_proj",
        "cleanup": ["DROP TABLE IF EXISTS ingest_base_proj"]
    },
    "base_mv": {
        "setup": [
            "DROP TABLE IF EXISTS ingest_mv_view",
            "DROP TABLE IF EXISTS ingest_mv_target",
            "DROP TABLE IF EXISTS ingest_base_mv",
            f"""CREATE TABLE ingest_base_mv (
                timestamp DateTime, user_id UInt32, page LowCardinality(String),
                duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)
            ) ENGINE = MergeTree() ORDER BY (page, timestamp)""",
            f"""CREATE TABLE ingest_mv_target (
                page LowCardinality(String), hour DateTime,
                hits AggregateFunction(count, UInt64),
                avg_duration AggregateFunction(avg, UInt32),
                sum_duration AggregateFunction(sum, UInt32)
            ) ENGINE = AggregatingMergeTree() ORDER BY (page, hour)""",
            f"""CREATE MATERIALIZED VIEW ingest_mv_view TO ingest_mv_target AS
            SELECT page, toStartOfHour(timestamp) AS hour,
                countState() AS hits, avgState(duration_ms) AS avg_duration, sumState(duration_ms) AS sum_duration
            FROM ingest_base_mv GROUP BY page, hour"""
        ],
        "table": "ingest_base_mv",
        "cleanup": [
            "DROP TABLE IF EXISTS ingest_mv_view",
            "DROP TABLE IF EXISTS ingest_mv_target", 
            "DROP TABLE IF EXISTS ingest_base_mv"
        ]
    }
}

results = []

for scenario_name, cfg in SCENARIOS.items():
    for rep in range(1, REPS + 1):
        print(f"\n{scenario_name} rep {rep}/{REPS}")
        
        # Setup
        for sql in cfg["setup"]:
            ch(sql)
        
        # Insert
        sql = INSERT_SQL.format(table=cfg["table"])
        t0 = time.time()
        ch(sql)
        elapsed = time.time() - t0
        
        # OPTIMIZE FINAL
        ch(f"OPTIMIZE TABLE {cfg['table']} FINAL")
        
        rows_per_sec = ROWS / elapsed
        print(f"  {elapsed:.2f}s, {rows_per_sec:,.0f} rows/s")
        
        results.append({
            "scenario": scenario_name,
            "rep": rep,
            "rows": ROWS,
            "elapsed_s": round(elapsed, 3),
            "rows_per_sec": round(rows_per_sec)
        })
        
        # Cleanup
        for sql in cfg["cleanup"]:
            ch(sql)

# Write CSV
outpath = f"{OUTDIR}/ingest_impact.csv"
with open(outpath, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=["scenario", "rep", "rows", "elapsed_s", "rows_per_sec"])
    w.writeheader()
    w.writerows(results)

print(f"\nResults: {outpath}")
