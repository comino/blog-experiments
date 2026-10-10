#!/usr/bin/env python3
"""Execute reproducible bounded query checks; these are not model benchmarks."""
import hashlib
import json
import time

from load import BASE, query, utcnow


def main():
    examples = json.loads((BASE / 'example-queries.json').read_text())
    results = []
    for example in examples:
        started = time.monotonic()
        payload = json.loads(query(example['sql'] + ' FORMAT JSON', 'maritime_check_' + example['id']))
        result = {'id': example['id'], 'purpose': example['purpose'],
                  'rows': len(payload['data']), 'passed': len(payload['data']) >= example['min_rows'],
                  'elapsed_seconds': time.monotonic()-started, 'statistics': payload['statistics'],
                  'columns': payload['meta'], 'sample': payload['data'][:3],
                  'sql_sha256': hashlib.sha256(example['sql'].encode()).hexdigest()}
        results.append(result)
        print(json.dumps({key: result[key] for key in ['id', 'rows', 'passed', 'elapsed_seconds']}), flush=True)
    report = {'checked_at': utcnow(), 'purpose': 'Query execution and semantic smoke checks against imported records, not LLM or production latency benchmarks.',
              'passed': all(r['passed'] for r in results), 'queries': results}
    (BASE / 'query-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['passed'], 'One or more query checks returned too few rows'


if __name__ == '__main__':
    main()
