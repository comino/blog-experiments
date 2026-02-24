#!/usr/bin/env python3
"""Benchmark LLM-generated queries vs reference queries on ClickHouse."""

import csv
import subprocess
import time
import uuid
import sys
import os
import tempfile

DATA_DIR = "/root/.openclaw/workspace/blog/experiments/results/03/data"
OUTPUT_CSV = f"{DATA_DIR}/query_performance.csv"
SSH_HOST = "thesis-clickhouse"
DB = "exp03_llm"
RUNS = 3

def run_query_on_ch(sql, query_id):
    """Execute query by writing to temp file and piping via stdin."""
    clean_sql = sql.replace('\n', ' ').replace('\r', ' ').strip().rstrip(';')
    full_sql = f"{clean_sql} FORMAT Null"
    # Pipe SQL via stdin using echo through ssh
    proc = subprocess.run(
        ['ssh', SSH_HOST, 'clickhouse-client', f'--database={DB}', f'--query_id={query_id}'],
        input=full_sql, capture_output=True, text=True, timeout=120
    )
    return proc.returncode == 0

def get_query_metrics(query_id):
    sql = f"SELECT query_duration_ms, read_rows, read_bytes, result_rows FROM system.query_log WHERE query_id = '{query_id}' AND type = 'QueryFinish' ORDER BY event_time DESC LIMIT 1 FORMAT TSV"
    result = subprocess.run(
        ['ssh', SSH_HOST, 'clickhouse-client'],
        input=sql, capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split('\t')
        if len(parts) == 4:
            return {
                'elapsed_ms': float(parts[0]),
                'read_rows': int(parts[1]),
                'read_bytes': int(parts[2]),
                'result_rows': int(parts[3])
            }
    return None

def flush_query_log():
    subprocess.run(
        ['ssh', SSH_HOST, 'clickhouse-client'],
        input='SYSTEM FLUSH LOGS', capture_output=True, text=True, timeout=15
    )

def main():
    # Load all score=3, run=1 pairs
    pairs = []
    seen = set()
    with open(f"{DATA_DIR}/scores.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['score'] == '3' and row['run'] == '1':
                key = (row['model'], row['question_id'])
                if key not in seen:
                    seen.add(key)
                    pairs.append({
                        'model': row['model'],
                        'question_id': row['question_id'],
                        'tier': row['tier'],
                        'llm_sql': row['llm_sql'],
                        'reference_sql': row['reference_sql']
                    })

    # Check what's already done
    completed = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row['question_id'], row['model'], row['run']))

    remaining = []
    for pair in pairs:
        for run in range(1, RUNS + 1):
            key = (pair['question_id'], pair['model'], str(run))
            if key not in completed:
                remaining.append((pair, run))

    print(f"Total pairs: {len(pairs)}, already done: {len(completed)}, remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    mode = 'a' if completed else 'w'
    with open(OUTPUT_CSV, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'question_id', 'tier', 'model', 'run',
            'llm_elapsed_ms', 'ref_elapsed_ms',
            'llm_read_rows', 'ref_read_rows',
            'llm_read_bytes', 'ref_read_bytes',
            'llm_result_rows', 'ref_result_rows',
            'speedup_ratio'
        ])
        if not completed:
            writer.writeheader()

        errors = 0
        success = 0
        batch_count = 0
        for pair, run in remaining:
            uid = uuid.uuid4().hex[:6]
            llm_qid = f"bl_{uid}"
            ref_qid = f"br_{uid}"

            llm_ok = run_query_on_ch(pair['llm_sql'], llm_qid)
            ref_ok = run_query_on_ch(pair['reference_sql'], ref_qid)

            batch_count += 1
            # Flush every 10 queries
            if batch_count % 5 == 0:
                flush_query_log()
                time.sleep(0.05)
            else:
                time.sleep(0.02)
                flush_query_log()
                time.sleep(0.05)

            llm_metrics = get_query_metrics(llm_qid) if llm_ok else None
            ref_metrics = get_query_metrics(ref_qid) if ref_ok else None

            if llm_metrics and ref_metrics:
                speedup = llm_metrics['elapsed_ms'] / ref_metrics['elapsed_ms'] if ref_metrics['elapsed_ms'] > 0 else None
                writer.writerow({
                    'question_id': pair['question_id'],
                    'tier': pair['tier'],
                    'model': pair['model'],
                    'run': run,
                    'llm_elapsed_ms': llm_metrics['elapsed_ms'],
                    'ref_elapsed_ms': ref_metrics['elapsed_ms'],
                    'llm_read_rows': llm_metrics['read_rows'],
                    'ref_read_rows': ref_metrics['read_rows'],
                    'llm_read_bytes': llm_metrics['read_bytes'],
                    'ref_read_bytes': ref_metrics['read_bytes'],
                    'llm_result_rows': llm_metrics['result_rows'],
                    'ref_result_rows': ref_metrics['result_rows'],
                    'speedup_ratio': round(speedup, 4) if speedup else None
                })
                f.flush()
                success += 1
            else:
                errors += 1
                if errors <= 10:
                    print(f"  SKIP {pair['model']}/{pair['question_id']}/r{run}: llm_ok={llm_ok} ref_ok={ref_ok}", file=sys.stderr)

            if (success + errors) % 50 == 0:
                print(f"  Progress: {success+errors}/{len(remaining)} | ok={success} err={errors}")

    print(f"Done! success={success} errors={errors}. Total rows in CSV: {len(completed) + success}")

if __name__ == "__main__":
    main()
