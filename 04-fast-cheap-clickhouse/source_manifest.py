#!/usr/bin/env python3
"""Read official Parquet metadata without downloading trip columns."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import requests

BASE = Path(__file__).resolve().parent


def inspect(year_month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year_month}.parquet'
    response = requests.head(url, timeout=30)
    response.raise_for_status()
    fs = fsspec.filesystem('http')
    with fs.open(url, 'rb', block_size=65536) as handle:
        metadata = pq.read_metadata(handle)
    return {
        'month': year_month, 'url': url, 'bytes': int(response.headers['Content-Length']),
        'etag': response.headers.get('ETag'), 'last_modified': response.headers.get('Last-Modified'),
        'rows': metadata.num_rows, 'row_groups': metadata.num_row_groups,
        'columns': metadata.schema.names, 'arrow_schema': str(metadata.schema.to_arrow_schema()),
    }


def main():
    months = [f'{year}-{month:02d}' for year in (2023, 2024, 2025) for month in range(1, 13)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(inspect, months))
    report = {
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'source': 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page',
        'dataset': 'NYC TLC yellow taxi, 2023-2025 monthly files',
        'files': files, 'source_rows': sum(f['rows'] for f in files),
        'parquet_bytes': sum(f['bytes'] for f in files),
    }
    (BASE / 'evidence').mkdir(exist_ok=True)
    (BASE / 'evidence/source-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
    for year in (2023, 2024, 2025):
        selected = [f for f in files if f['month'].startswith(str(year))]
        print(year, 'rows', sum(f['rows'] for f in selected), 'bytes', sum(f['bytes'] for f in selected))


if __name__ == '__main__':
    main()
