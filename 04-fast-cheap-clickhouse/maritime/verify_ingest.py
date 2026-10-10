#!/usr/bin/env python3
"""Capture completed-import evidence from source manifests and ClickHouse logs."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

from load import BASE, manifest, query, rows, utcnow, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-resume', action='store_true', help='Rerun the completed importer and verify the ledger and row counts remain unchanged.')
    args = parser.parse_args()
    resume = None
    if args.check_resume:
        before_log = hashlib.sha256((BASE / 'load.jsonl').read_bytes()).hexdigest()
        before_rows = [int(query('SELECT count() FROM exploration.' + table)) for table in ('ais_positions', 'storm_tracks')]
        subprocess.run([sys.executable, str(BASE / 'load.py'), '--workers', '2'], check=True)
        after_log = hashlib.sha256((BASE / 'load.jsonl').read_bytes()).hexdigest()
        after_rows = [int(query('SELECT count() FROM exploration.' + table)) for table in ('ais_positions', 'storm_tracks')]
        assert before_log == after_log and before_rows == after_rows, 'Resume changed completed data or successful-import ledger'
        resume = {'passed': True, 'rows_before': before_rows, 'rows_after': after_rows,
                  'successful_import_ledger_sha256_before_and_after': before_log}
    source = manifest()
    dataset = json.loads((BASE / 'dataset.json').read_text())
    assert dataset['complete'], 'Run full load and validation before final ingestion audit'
    events = [json.loads(line) for line in (BASE / 'load.jsonl').read_text().splitlines()]
    successes = {event['file']: event for event in events}
    expected = {entry['date'] for entry in source['ais']['files']}
    assert len(expected) == 31
    assert len(events) == len(successes), 'Repeated successful source imports found in loader log'
    assert expected <= set(successes)
    assert expected == set(dataset['coverage']['completed_dates'])
    assert sum(successes[day]['rows'] for day in expected) == dataset['position_count']
    assert successes['storms']['rows'] == dataset['storm_points']
    active = rows("SELECT query_id FROM system.processes WHERE startsWith(query_id,'maritime_import_')")
    assert not active, 'An import query is still running'
    finished = rows("SELECT query_id,event_time,read_rows,written_rows,query_duration_ms,exception_code FROM system.query_log WHERE type='QueryFinish' AND startsWith(query_id,'maritime_import_') ORDER BY query_id,event_time")
    matched = {}
    for record in finished:
        key = record['query_id']
        assert key not in matched, 'Duplicate finished import query identifier: ' + key
        matched[key] = record
    for day in sorted(expected):
        query_id = 'maritime_import_ais_' + day.replace('-', '')
        assert query_id in matched, 'Successful ClickHouse log entry missing: ' + query_id
        assert int(matched[query_id]['written_rows']) == successes[day]['rows'], day
        assert int(matched[query_id]['exception_code']) == 0, day
    assert int(matched['maritime_import_storms_2024']['written_rows']) == successes['storms']['rows']
    storage = rows("SELECT table,sum(rows) AS rows,sum(bytes_on_disk) AS bytes_on_disk,sum(data_compressed_bytes) AS data_compressed_bytes,sum(data_uncompressed_bytes) AS data_uncompressed_bytes FROM system.parts WHERE active AND database='exploration' AND table IN ('ais_positions','storm_tracks') GROUP BY table ORDER BY table")
    report = {'checked_at': utcnow(), 'passed': True,
              'checks': {'exact_31_day_coverage': True, 'all_import_rows_match_table_counts': True,
                         'all_import_rows_match_clickhouse_query_log': True, 'no_repeated_successful_imports': True,
                         'no_running_maritime_insert': True},
              'first_date': min(expected), 'last_date': max(expected),
              'ais_source_bytes': sum(entry['bytes'] for entry in source['ais']['files']),
              'ais_positions': dataset['position_count'], 'vessel_ids': dataset['vessel_count'],
              'storm_points': dataset['storm_points'], 'storm_ids': dataset['storm_count'],
              'storage': storage, 'host_free_bytes_after_import': shutil.disk_usage(BASE.parent).free,
              'completed_imports': finished}
    if resume is not None:
        report['idempotent_resume'] = resume
        report['checks']['idempotent_resume'] = True
    write_json(BASE / 'ingest-validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'completed_imports'}), flush=True)


if __name__ == '__main__':
    main()
