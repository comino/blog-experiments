#!/usr/bin/env python3
"""Collect database evidence and checksum artifacts after the full run stops."""
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import numpy
import requests

BASE = Path(__file__).resolve().parent


def query(sql):
    r = requests.post("http://127.0.0.1:18123/", data=sql.encode(), timeout=60)
    r.raise_for_status()
    return r.text


def main():
    audit = json.loads((BASE / "results/audit.json").read_text())
    assert audit["passed"]
    cpu = subprocess.check_output(["lscpu", "-J"], text=True)
    environment = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "requests": requests.__version__, "numpy": numpy.__version__,
        "lscpu": json.loads(cpu), "meminfo": Path("/proc/meminfo").read_text(),
        "lsblk": json.loads(subprocess.check_output(["lsblk", "-J", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS"], text=True)),
        "clickhouse_version": query("SELECT version()").strip(),
        "client_and_database_location": "Hetzner fsn1, same CCX13 instance; database on loopback",
    }
    reader_settings = requests.post(
        "http://127.0.0.1:18123/", headers={"X-ClickHouse-User": "budget_reader"},
        data=("SELECT name, value, changed FROM system.settings WHERE name IN "
              "('readonly','max_threads','max_memory_usage','max_execution_time','max_result_rows',"
              "'max_result_bytes','result_overflow_mode','use_query_cache',"
              "'output_format_json_quote_64bit_integers','output_format_json_quote_64bit_floats',"
              "'date_time_output_format') ORDER BY name FORMAT JSONEachRow").encode(), timeout=30,
    )
    reader_settings.raise_for_status()
    environment["benchmark_user_settings"] = [json.loads(line) for line in reader_settings.text.splitlines()]
    (BASE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    fingerprint_sql = "SELECT 'events' AS tbl, count(), sum(cityHash64(*)) AS checksum FROM exp04_budget.events UNION ALL SELECT 'users', count(), sum(cityHash64(*)) FROM exp04_budget.users FORMAT JSONEachRow"
    after_text = query(fingerprint_sql)
    (BASE / "data-fingerprints-after.jsonl").write_text(after_text)
    before = [json.loads(line) for line in (BASE / "data-fingerprints.jsonl").read_text().splitlines()]
    after = [json.loads(line) for line in after_text.splitlines()]
    assert sorted(before, key=lambda r: r["tbl"]) == sorted(after, key=lambda r: r["tbl"]), "Dataset changed during benchmark"
    (BASE / "dataset-verification.json").write_text(json.dumps({"unchanged": True, "before": before, "after": after}, indent=2) + "\n")
    query("SYSTEM FLUSH LOGS")
    query_log = query("SELECT query_id, type, event_time_microseconds, query_duration_ms, read_rows, read_bytes, result_rows, memory_usage, exception_code, exception, ProfileEvents FROM system.query_log WHERE startsWith(query_id, 'budget_') AND type != 'QueryStart' ORDER BY event_time_microseconds FORMAT JSONEachRow")
    (BASE / "results/query-log.jsonl").write_text(query_log)
    artifacts = {}
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pid" or path.name == "artifacts-sha256.json":
            continue
        artifacts[str(path.relative_to(BASE))] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    (BASE / "artifacts-sha256.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    print(f"Dataset unchanged. Collected {len(query_log.splitlines())} query-log outcomes and checksummed {len(artifacts)} artifacts.")


if __name__ == "__main__":
    main()
