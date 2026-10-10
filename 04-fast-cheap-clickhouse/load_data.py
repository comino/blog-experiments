#!/usr/bin/env python3
"""Load the exact manifested NYC source files directly into ClickHouse."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / 'evidence'
PRIVATE = json.loads((BASE / 'runtime/credentials.json').read_text())
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.auth = ('loader', PRIVATE['loader'])


def query(sql, query_id=None, timeout=300):
    params = {'max_threads': 4, 'max_insert_threads': 2, 'max_memory_usage': 9000000000}
    if query_id:
        params['query_id'] = query_id
    response = SESSION.post('http://127.0.0.1:18123/', params=params, data=sql.encode(), timeout=timeout)
    if not response.ok:
        raise RuntimeError(response.text[:5000])
    return response.text


def main():
    manifest = json.loads((EVIDENCE / 'source-manifest.json').read_text())
    for sql in (BASE / 'schema.sql').read_text().split(';\n'):
        if sql.strip():
            query(sql)
    zones = int(query('SELECT count() FROM exploration.taxi_zones'))
    if not zones:
        query("INSERT INTO exploration.taxi_zones SELECT LocationID, Borough, Zone, service_zone FROM url('https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv', 'CSVWithNames')", 'exploration_load_zones')
    completed = {r['month']: r for r in map(json.loads, (EVIDENCE / 'load.jsonl').read_text().splitlines())} if (EVIDENCE / 'load.jsonl').exists() else {}
    for file in manifest['files']:
        month = file['month']
        existing = int(query(f"SELECT count() FROM exploration.trips WHERE source_month = toDate('{month}-01')"))
        if month in completed:
            assert existing == completed[month]['loaded_rows'] == file['rows']
            continue
        assert existing == 0, f'Unreconciled partial load for {month}; inspect before retrying'
        head = requests.head(file['url'], timeout=30)
        head.raise_for_status()
        assert head.headers.get('ETag') == file['etag'] and int(head.headers['Content-Length']) == file['bytes'], 'Source changed after manifest capture'
        sql = f"""INSERT INTO exploration.trips
SELECT toDate('{month}-01'), t.tpep_pickup_datetime, t.tpep_dropoff_datetime,
    ifNull(t.PULocationID, 0), ifNull(t.DOLocationID, 0),
    pu.borough, pu.zone, do.borough, do.zone,
    t.passenger_count, ifNull(t.trip_distance, 0), ifNull(t.fare_amount, 0),
    ifNull(t.tip_amount, 0), ifNull(t.tolls_amount, 0), ifNull(t.total_amount, 0),
    ifNull(t.payment_type, 0), ifNull(t.VendorID, 0), t.RatecodeID
FROM url('{file['url']}', 'Parquet') AS t
LEFT ANY JOIN exploration.taxi_zones AS pu ON t.PULocationID = pu.location_id
LEFT ANY JOIN exploration.taxi_zones AS do ON t.DOLocationID = do.location_id
"""
        started = datetime.now(timezone.utc).isoformat()
        clock = time.perf_counter()
        query_id = 'exploration_load_' + month.replace('-', '')
        query(sql, query_id)
        count = int(query(f"SELECT count() FROM exploration.trips WHERE source_month = toDate('{month}-01')"))
        assert count == file['rows'], (month, count, file['rows'])
        record = {'month': month, 'started_at': started, 'ended_at': datetime.now(timezone.utc).isoformat(), 'elapsed_s': time.perf_counter() - clock, 'loaded_rows': count, 'query_id': query_id, 'sql': sql, 'source_etag_verified': True}
        with (EVIDENCE / 'load.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(month, count, f"{record['elapsed_s']:.2f}s", flush=True)
    total = int(query('SELECT count() FROM exploration.trips'))
    assert total == manifest['source_rows']
    query('SYSTEM STOP MERGES exploration.trips')
    query('SYSTEM FLUSH LOGS')
    stats = json.loads(query("SELECT sum(rows) AS rows, sum(bytes_on_disk) AS bytes_on_disk, sum(data_uncompressed_bytes) AS uncompressed_bytes, count() AS parts FROM system.parts WHERE active AND database='exploration' AND table='trips' FORMAT JSON"))['data'][0]
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'source_rows': manifest['source_rows'], 'loaded_rows': total, 'source_files': len(manifest['files']), 'zone_rows': int(query('SELECT count() FROM exploration.taxi_zones')), 'clickhouse_version': query('SELECT version()').strip(), 'storage': stats, 'schema_sha256': hashlib.sha256((BASE / 'schema.sql').read_bytes()).hexdigest(), 'null_handling': 'Nullable passengers/rate codes retained; null monetary/distance/vendor/location/payment values become zero; taxi-zone names are joined once during ingestion.', 'timestamp_convention': 'TLC local wall-clock fields preserved numerically in UTC-typed DateTime64 columns; no timezone conversion or real-time feed.', 'background_merges_stopped_after_load': True, 'query_result_cache': False}
    (EVIDENCE / 'dataset.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
