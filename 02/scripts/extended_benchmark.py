#!/usr/bin/env python3
"""Experiment 02 Extended: Comprehensive Projections vs MVs benchmark.

Phases:
  1. Scaling analysis: 1M, 10M, 50M, 200M rows × queries × variants
  2. Projection count scaling: 1, 3, 5 projections
  3. MV freshness test
  4. EXPLAIN analysis for every query × variant
"""

import subprocess
import csv
import random
import time
import sys
import os
import json

DB = "exp02_projections"
OUTDIR = "/root/exp02_results"
REPS = 5
os.makedirs(f"{OUTDIR}/explain_outputs", exist_ok=True)

def ch(sql, timeout=600, **kwargs):
    cmd = ["clickhouse-client", f"--database={DB}", "--query", sql]
    for k, v in kwargs.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and 'ALREADY_EXISTS' not in r.stderr:
        print(f"ERROR: {r.stderr[:200]}", file=sys.stderr)
    return r.stdout.strip()

def ch_multi(sql, timeout=600):
    cmd = ["clickhouse-client", f"--database={DB}", "--multiquery"]
    r = subprocess.run(cmd, input=sql, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"ERROR: {r.stderr[:300]}", file=sys.stderr)
    return r.stdout.strip()

def drop_caches():
    ch("SYSTEM DROP MARK CACHE")
    ch("SYSTEM DROP UNCOMPRESSED CACHE")
    ch("SYSTEM DROP COMPILED EXPRESSION CACHE")
    try:
        subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True, timeout=10)
    except:
        pass

# ============================================================
# DATA GENERATION (for different sizes)
# ============================================================

GEN_SQL = """INSERT INTO {table}
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(rand() % (365 * 86400)) AS timestamp,
    rand() % 1000000 + 1 AS user_id,
    concat('/page/', toString(rand() % 1000)) AS page,
    50 + rand() % 9950 AS duration_ms,
    arrayElement(['US','DE','UK','FR','JP','BR','IN','CA','AU','MX','IT','ES','KR','NL','SE','CH','AT','BE','PL','CZ','DK','NO','FI','PT','IE','RO','HU','GR','BG','HR','SK','SI','LT','LV','EE','IL','TR','ZA','NG','EG','KE','AR','CL','CO','PE','TH','VN','MY','PH','ID'], (rand() % 50) + 1) AS country,
    arrayElement(['desktop','mobile','tablet','smart_tv','wearable'], (rand() % 5) + 1) AS device_type
FROM numbers({rows})"""

def create_base_table(name):
    ch(f"""CREATE TABLE IF NOT EXISTS {name} (
        timestamp DateTime, user_id UInt32, page LowCardinality(String),
        duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)
    ) ENGINE = MergeTree() ORDER BY (page, timestamp)""")

def create_proj_table(name):
    ch(f"""CREATE TABLE IF NOT EXISTS {name} (
        timestamp DateTime, user_id UInt32, page LowCardinality(String),
        duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String),
        PROJECTION proj_country_time (SELECT * ORDER BY (country, timestamp)),
        PROJECTION proj_hourly_stats (
            SELECT page, toStartOfHour(timestamp) AS hour,
                count() AS hits, avg(duration_ms) AS avg_duration, sum(duration_ms) AS sum_duration
            GROUP BY page, hour
        )
    ) ENGINE = MergeTree() ORDER BY (page, timestamp)""")

def create_mv_setup(base_name, mv_name, target_name):
    create_base_table(base_name)
    ch(f"""CREATE TABLE IF NOT EXISTS {target_name} (
        page LowCardinality(String), hour DateTime,
        hits AggregateFunction(count, UInt64),
        avg_duration AggregateFunction(avg, UInt32),
        sum_duration AggregateFunction(sum, UInt32)
    ) ENGINE = AggregatingMergeTree() ORDER BY (page, hour)""")
    ch(f"""CREATE MATERIALIZED VIEW IF NOT EXISTS {mv_name} TO {target_name} AS
        SELECT page, toStartOfHour(timestamp) AS hour,
            countState() AS hits, avgState(duration_ms) AS avg_duration,
            sumState(duration_ms) AS sum_duration
        FROM {base_name} GROUP BY page, hour""")

# ============================================================
# QUERIES (8 patterns)
# ============================================================

def get_queries(base_t, proj_t, mv_target_t):
    """Return list of (query_name, variant, sql) tuples."""
    queries = []

    # Q1: Dashboard Rollup (single day)
    for vname, tbl in [("base", base_t), ("proj", proj_t)]:
        queries.append(("Q1_day", vname,
            f"SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur "
            f"FROM {tbl} WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' "
            f"GROUP BY hour, page ORDER BY hour"))
    if mv_target_t:
        queries.append(("Q1_day", "mv",
            f"SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_dur "
            f"FROM {mv_target_t} WHERE page = '/page/42' AND hour >= '2024-06-01' AND hour < '2024-06-02' "
            f"GROUP BY page, hour ORDER BY hour"))

    # Q2: Dashboard Rollup (30 days)
    for vname, tbl in [("base", base_t), ("proj", proj_t)]:
        queries.append(("Q2_month", vname,
            f"SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits, avg(duration_ms) AS avg_dur "
            f"FROM {tbl} WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-07-01' "
            f"GROUP BY hour, page ORDER BY hour"))
    if mv_target_t:
        queries.append(("Q2_month", "mv",
            f"SELECT page, hour, countMerge(hits) AS hits, avgMerge(avg_duration) AS avg_dur "
            f"FROM {mv_target_t} WHERE page = '/page/42' AND hour >= '2024-06-01' AND hour < '2024-07-01' "
            f"GROUP BY page, hour ORDER BY hour"))

    # Q3: Country Filter (exact)
    for vname, tbl in [("base", base_t), ("proj", proj_t)]:
        queries.append(("Q3_country", vname,
            f"SELECT count(), avg(duration_ms) FROM {tbl} WHERE country = 'DE'"))

    # Q4: Country + Time Range (compound)
    for vname, tbl in [("base", base_t), ("proj", proj_t)]:
        queries.append(("Q4_country_time", vname,
            f"SELECT count(), avg(duration_ms) FROM {tbl} "
            f"WHERE country = 'DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'"))

    # Q5: Top-K Pages by avg duration
    for vname, tbl in [("base", base_t), ("proj", proj_t)]:
        queries.append(("Q5_topk", vname,
            f"SELECT page, avg(duration_ms) AS avg_dur, count() AS hits "
            f"FROM {tbl} GROUP BY page ORDER BY avg_dur DESC LIMIT 10"))
    if mv_target_t:
        queries.append(("Q5_topk", "mv",
            f"SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits "
            f"FROM {mv_target_t} GROUP BY page ORDER BY avg_dur DESC LIMIT 10"))

    # Q6: Top-K with HAVING
    for vname, tbl in [("base", base_t), ("proj", proj_t)]:
        queries.append(("Q6_topk_having", vname,
            f"SELECT page, avg(duration_ms) AS avg_dur, count() AS hits "
            f"FROM {tbl} GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10"))
    if mv_target_t:
        queries.append(("Q6_topk_having", "mv",
            f"SELECT page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits "
            f"FROM {mv_target_t} GROUP BY page HAVING hits > 100 ORDER BY avg_dur DESC LIMIT 10"))

    # Q7: Cardinality per country
    for vname, tbl in [("base", base_t), ("proj", proj_t)]:
        queries.append(("Q7_cardinality", vname,
            f"SELECT country, uniqExact(user_id) AS unique_users FROM {tbl} GROUP BY country ORDER BY unique_users DESC LIMIT 10"))

    # Q8: Multi-dimension GROUP BY
    for vname, tbl in [("base", base_t), ("proj", proj_t)]:
        queries.append(("Q8_multidim", vname,
            f"SELECT country, device_type, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur "
            f"FROM {tbl} WHERE timestamp >= '2024-06-01' AND timestamp < '2024-06-08' "
            f"GROUP BY country, device_type, hour ORDER BY hits DESC LIMIT 50"))

    return queries


# ============================================================
# PHASE 1: SCALING ANALYSIS
# ============================================================

def run_scaling_analysis():
    print("\n" + "="*60)
    print("PHASE 1: SCALING ANALYSIS")
    print("="*60)

    SIZES = [1_000_000, 10_000_000, 50_000_000, 200_000_000]
    results = []

    for size in SIZES:
        label = f"{size//1_000_000}M"
        print(f"\n--- Size: {label} ({size:,} rows) ---")

        base_t = f"scale_{label}_base"
        proj_t = f"scale_{label}_proj"
        mv_src = f"scale_{label}_mv_src"
        mv_tgt = f"scale_{label}_mv_tgt"
        mv_view = f"scale_{label}_mv_view"

        # Create tables
        print(f"  Creating tables...", flush=True)
        ch(f"DROP TABLE IF EXISTS {mv_view}")
        ch(f"DROP TABLE IF EXISTS {mv_tgt}")
        for t in [base_t, proj_t, mv_src]:
            ch(f"DROP TABLE IF EXISTS {t}")

        create_base_table(base_t)
        create_proj_table(proj_t)
        create_mv_setup(mv_src, mv_view, mv_tgt)

        # Generate data
        print(f"  Generating {label} rows into base...", flush=True)
        t0 = time.time()
        ch(GEN_SQL.format(table=base_t, rows=size), timeout=3600)
        gen_base_time = time.time() - t0
        print(f"    Base: {gen_base_time:.1f}s ({size/gen_base_time:,.0f} rows/s)")

        print(f"  Copying to proj table...", flush=True)
        t0 = time.time()
        ch(f"INSERT INTO {proj_t} SELECT * FROM {base_t}", timeout=3600)
        gen_proj_time = time.time() - t0
        print(f"    Proj: {gen_proj_time:.1f}s ({size/gen_proj_time:,.0f} rows/s)")

        print(f"  Copying to MV source (populates MV)...", flush=True)
        t0 = time.time()
        ch(f"INSERT INTO {mv_src} SELECT * FROM {base_t}", timeout=3600)
        gen_mv_time = time.time() - t0
        print(f"    MV: {gen_mv_time:.1f}s ({size/gen_mv_time:,.0f} rows/s)")

        # Record ingest speeds
        results.append({"size": size, "query": "_ingest", "variant": "base", "cache": "na",
                        "metric": "rows_per_sec", "value": round(size/gen_base_time), "rows_read": 0})
        results.append({"size": size, "query": "_ingest", "variant": "proj", "cache": "na",
                        "metric": "rows_per_sec", "value": round(size/gen_proj_time), "rows_read": 0})
        results.append({"size": size, "query": "_ingest", "variant": "mv", "cache": "na",
                        "metric": "rows_per_sec", "value": round(size/gen_mv_time), "rows_read": 0})

        # OPTIMIZE FINAL
        print(f"  OPTIMIZE FINAL...", flush=True)
        for t in [base_t, proj_t, mv_src, mv_tgt]:
            ch(f"OPTIMIZE TABLE {t} FINAL", timeout=3600)

        # Storage
        print(f"  Collecting storage...", flush=True)
        for t in [base_t, proj_t, mv_src, mv_tgt]:
            storage = ch(f"SELECT sum(bytes_on_disk), sum(data_compressed_bytes), sum(rows) "
                        f"FROM system.parts WHERE database='{DB}' AND table='{t}' AND active FORMAT TSV")
            if storage:
                parts = storage.split('\t')
                variant = "base" if "base" in t else ("proj" if "proj" in t else ("mv_target" if "tgt" in t else "mv_source"))
                results.append({"size": size, "query": "_storage_disk", "variant": variant,
                               "cache": "na", "metric": "bytes", "value": int(parts[0]), "rows_read": int(parts[2])})
                results.append({"size": size, "query": "_storage_compressed", "variant": variant,
                               "cache": "na", "metric": "bytes", "value": int(parts[1]), "rows_read": int(parts[2])})

        # Stop merges for benchmarks
        for t in [base_t, proj_t, mv_src, mv_tgt]:
            ch(f"SYSTEM STOP MERGES {DB}.{t}")

        # Benchmark queries
        queries = get_queries(base_t, proj_t, mv_tgt)
        print(f"  Benchmarking {len(queries)} queries × {REPS} reps × cold+warm...", flush=True)

        for cache in ["cold", "warm"]:
            for run in range(1, REPS + 1):
                order = list(queries)
                random.shuffle(order)
                for qname, variant, sql in order:
                    if cache == "cold":
                        drop_caches()

                    qid = f"s_{label}_{qname}_{variant}_{cache}_{run}_{int(time.time()*1000)}"
                    ch(sql, query_id=qid, format="Null")
                    time.sleep(0.2)

                ch("SYSTEM FLUSH LOGS")
                time.sleep(0.3)

                for qname, variant, sql in order:
                    qid_prefix = f"s_{label}_{qname}_{variant}_{cache}_{run}_"
                    metrics = ch(f"""SELECT query_duration_ms, read_rows, read_bytes
                        FROM system.query_log
                        WHERE query_id LIKE '{qid_prefix}%' AND type='QueryFinish'
                        ORDER BY event_time DESC LIMIT 1 FORMAT TSV""")
                    if metrics:
                        p = metrics.split('\t')
                        results.append({"size": size, "query": qname, "variant": variant,
                                       "cache": cache, "metric": "elapsed_ms", "value": int(p[0]),
                                       "rows_read": int(p[1])})

        # Resume merges
        for t in [base_t, proj_t, mv_src, mv_tgt]:
            ch(f"SYSTEM START MERGES {DB}.{t}")

        # Cleanup smaller sizes to save disk (keep 200M for later phases)
        if size < 200_000_000:
            print(f"  Cleaning up {label} tables...", flush=True)
            ch(f"DROP TABLE IF EXISTS {mv_view}")
            ch(f"DROP TABLE IF EXISTS {mv_tgt}")
            for t in [base_t, proj_t, mv_src]:
                ch(f"DROP TABLE IF EXISTS {t}")

    # Write results
    outpath = f"{OUTDIR}/scaling.csv"
    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["size", "query", "variant", "cache", "metric", "value", "rows_read"])
        w.writeheader()
        w.writerows(results)
    print(f"\nScaling results: {outpath} ({len(results)} rows)")


# ============================================================
# PHASE 2: PROJECTION COUNT SCALING
# ============================================================

def run_projection_count_scaling():
    print("\n" + "="*60)
    print("PHASE 2: PROJECTION COUNT SCALING")
    print("="*60)

    SIZE = 50_000_000  # 50M rows — large enough to matter, fast enough to iterate
    results = []

    PROJ_DEFS = [
        ("proj_country", "SELECT * ORDER BY (country, timestamp)"),
        ("proj_device", "SELECT * ORDER BY (device_type, timestamp)"),
        ("proj_user", "SELECT * ORDER BY (user_id, timestamp)"),
        ("proj_hourly", "SELECT page, toStartOfHour(timestamp) AS hour, count() AS hits, avg(duration_ms) AS avg_dur GROUP BY page, hour"),
        ("proj_daily", "SELECT country, toDate(timestamp) AS day, count() AS hits, avg(duration_ms) AS avg_dur GROUP BY country, day"),
    ]

    for num_proj in [0, 1, 3, 5]:
        label = f"pc_{num_proj}"
        tbl = f"projcount_{num_proj}"
        print(f"\n--- {num_proj} projections ---", flush=True)

        ch(f"DROP TABLE IF EXISTS {tbl}")

        # Build CREATE TABLE with N projections
        proj_clause = ""
        if num_proj > 0:
            projs = [f"PROJECTION {name} ({defn})" for name, defn in PROJ_DEFS[:num_proj]]
            proj_clause = ", " + ", ".join(projs)

        ch(f"""CREATE TABLE {tbl} (
            timestamp DateTime, user_id UInt32, page LowCardinality(String),
            duration_ms UInt32, country LowCardinality(String), device_type LowCardinality(String)
            {proj_clause}
        ) ENGINE = MergeTree() ORDER BY (page, timestamp)""")

        # Insert
        print(f"  Inserting {SIZE:,} rows...", flush=True)
        t0 = time.time()
        ch(GEN_SQL.format(table=tbl, rows=SIZE), timeout=3600)
        ingest_time = time.time() - t0
        ingest_rps = SIZE / ingest_time
        print(f"    {ingest_time:.1f}s ({ingest_rps:,.0f} rows/s)")

        ch(f"OPTIMIZE TABLE {tbl} FINAL", timeout=3600)

        # Storage
        storage = ch(f"SELECT sum(bytes_on_disk), sum(data_compressed_bytes) "
                     f"FROM system.parts WHERE database='{DB}' AND table='{tbl}' AND active FORMAT TSV")
        disk_bytes, comp_bytes = 0, 0
        if storage:
            parts = storage.split('\t')
            disk_bytes, comp_bytes = int(parts[0]), int(parts[1])

        results.append({
            "num_projections": num_proj,
            "ingest_rows_per_sec": round(ingest_rps),
            "ingest_time_s": round(ingest_time, 2),
            "disk_bytes": disk_bytes,
            "compressed_bytes": comp_bytes,
            "rows": SIZE,
        })

        print(f"    Disk: {disk_bytes/1024/1024:.1f} MiB, Ingest: {ingest_rps:,.0f} rows/s")

        # Cleanup
        ch(f"DROP TABLE IF EXISTS {tbl}")

    outpath = f"{OUTDIR}/projection_count.csv"
    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["num_projections", "ingest_rows_per_sec", "ingest_time_s",
                                          "disk_bytes", "compressed_bytes", "rows"])
        w.writeheader()
        w.writerows(results)
    print(f"\nProjection count results: {outpath}")


# ============================================================
# PHASE 3: MV FRESHNESS TEST
# ============================================================

def run_mv_freshness():
    print("\n" + "="*60)
    print("PHASE 3: MV FRESHNESS TEST")
    print("="*60)

    # Create fresh tables
    proj_t = "fresh_proj"
    mv_src = "fresh_mv_src"
    mv_tgt = "fresh_mv_tgt"
    mv_view = "fresh_mv_view"

    for t in [mv_view, mv_tgt, mv_src, proj_t]:
        ch(f"DROP TABLE IF EXISTS {t}")

    create_proj_table(proj_t)
    create_mv_setup(mv_src, mv_view, mv_tgt)

    results = []
    BATCH = 100_000
    NUM_BATCHES = 10

    # Use a specific page we'll query
    INSERT_BATCH = f"""INSERT INTO {{table}}
    SELECT
        toDateTime('2024-06-15 12:00:00') + toIntervalSecond(rand() % 3600) AS timestamp,
        rand() % 1000000 + 1 AS user_id,
        '/page/freshtest' AS page,
        50 + rand() % 9950 AS duration_ms,
        'DE' AS country,
        'desktop' AS device_type
    FROM numbers({BATCH})"""

    for batch_num in range(1, NUM_BATCHES + 1):
        expected_rows = batch_num * BATCH
        print(f"\n  Batch {batch_num}/{NUM_BATCHES} ({expected_rows:,} total rows)", flush=True)

        # Insert into both tables
        ch(INSERT_BATCH.format(table=proj_t))
        ch(INSERT_BATCH.format(table=mv_src))

        # Immediately query projection table
        t0 = time.time()
        proj_result = ch(f"SELECT count(), avg(duration_ms) FROM {proj_t} WHERE page = '/page/freshtest' FORMAT TSV")
        proj_latency = (time.time() - t0) * 1000

        # Immediately query MV
        t0 = time.time()
        mv_result = ch(f"SELECT countMerge(hits), avgMerge(avg_duration) FROM {mv_tgt} WHERE page = '/page/freshtest' FORMAT TSV")
        mv_latency = (time.time() - t0) * 1000

        # Parse counts
        proj_count = int(proj_result.split('\t')[0]) if proj_result else 0
        mv_count = int(mv_result.split('\t')[0]) if mv_result else 0

        print(f"    Proj: {proj_count:,} rows ({proj_latency:.1f}ms) | MV: {mv_count:,} rows ({mv_latency:.1f}ms)")

        results.append({
            "batch": batch_num,
            "expected_rows": expected_rows,
            "proj_count": proj_count,
            "proj_latency_ms": round(proj_latency, 1),
            "proj_correct": proj_count == expected_rows,
            "mv_count": mv_count,
            "mv_latency_ms": round(mv_latency, 1),
            "mv_correct": mv_count == expected_rows,
        })

    # Cleanup
    for t in [mv_view, mv_tgt, mv_src, proj_t]:
        ch(f"DROP TABLE IF EXISTS {t}")

    outpath = f"{OUTDIR}/mv_freshness.csv"
    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["batch", "expected_rows", "proj_count", "proj_latency_ms",
                                          "proj_correct", "mv_count", "mv_latency_ms", "mv_correct"])
        w.writeheader()
        w.writerows(results)
    print(f"\nFreshness results: {outpath}")


# ============================================================
# PHASE 4: EXPLAIN ANALYSIS
# ============================================================

def run_explain_analysis():
    print("\n" + "="*60)
    print("PHASE 4: EXPLAIN ANALYSIS")
    print("="*60)

    # Use existing 200M tables
    base_t = "web_analytics_base"
    proj_t = "web_analytics_proj"
    mv_tgt = "hourly_stats_mv_target"

    queries = get_queries(base_t, proj_t, mv_tgt)
    explain_results = []

    for qname, variant, sql in queries:
        print(f"  EXPLAIN {qname}/{variant}...", flush=True)

        # Simple EXPLAIN
        explain = ch(f"EXPLAIN {sql}")
        # EXPLAIN with actions to see projection usage
        explain_actions = ch(f"EXPLAIN actions=1 {sql}")

        # Detect projection usage
        uses_projection = "proj_" in explain_actions or "Projection" in explain
        projection_name = ""
        for line in explain_actions.split('\n'):
            if "ReadFromMergeTree" in line and "proj_" in line:
                # Extract projection name
                start = line.find("(") + 1
                end = line.find(")")
                if start > 0 and end > start:
                    projection_name = line[start:end]

        explain_results.append({
            "query": qname,
            "variant": variant,
            "uses_projection": uses_projection,
            "projection_name": projection_name,
        })

        # Save full EXPLAIN output
        fname = f"{OUTDIR}/explain_outputs/{qname}_{variant}.txt"
        with open(fname, 'w') as f:
            f.write(f"-- Query: {qname} / Variant: {variant}\n")
            f.write(f"-- SQL: {sql}\n\n")
            f.write("=== EXPLAIN ===\n")
            f.write(explain + "\n\n")
            f.write("=== EXPLAIN actions=1 ===\n")
            f.write(explain_actions + "\n")

        print(f"    → Projection: {projection_name if uses_projection else 'NOT USED'}")

    # Also test force_optimize_projection_name
    print(f"\n  Testing force_optimize_projection_name...", flush=True)
    # Try forcing Q1 to use proj_hourly_stats
    forced = ch(f"EXPLAIN actions=1 SELECT toStartOfHour(timestamp) AS hour, page, count() AS hits "
                f"FROM {proj_t} WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02' "
                f"GROUP BY hour, page ORDER BY hour "
                f"SETTINGS force_optimize_projection_name='proj_hourly_stats'")
    forced_uses = "proj_hourly_stats" in forced
    print(f"    Q1 forced to proj_hourly_stats: {'YES' if forced_uses else 'NO (optimizer refused)'}")

    with open(f"{OUTDIR}/explain_outputs/Q1_forced_projection.txt", 'w') as f:
        f.write(f"-- Forced projection: proj_hourly_stats on Q1\n\n{forced}\n")

    outpath = f"{OUTDIR}/explain_summary.csv"
    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["query", "variant", "uses_projection", "projection_name"])
        w.writeheader()
        w.writerows(explain_results)
    print(f"\nEXPLAIN summary: {outpath}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (1-4), 0=all")
    args = parser.parse_args()

    print(f"ClickHouse version: {ch('SELECT version()')}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    phases = {
        1: ("Scaling Analysis", run_scaling_analysis),
        2: ("Projection Count", run_projection_count_scaling),
        3: ("MV Freshness", run_mv_freshness),
        4: ("EXPLAIN Analysis", run_explain_analysis),
    }

    if args.phase == 0:
        for num, (name, func) in phases.items():
            func()
    elif args.phase in phases:
        phases[args.phase][1]()
    else:
        print(f"Unknown phase: {args.phase}")
        sys.exit(1)

    print(f"\nAll done! {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
