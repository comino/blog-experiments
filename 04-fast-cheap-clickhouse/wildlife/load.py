#!/usr/bin/env python3
"""Reproducible full-snapshot iNaturalist metadata import. Run on the demo host."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import requests

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from load_data import query

BASE.mkdir(exist_ok=True)
S3 = 'https://inaturalist-open-data.s3.amazonaws.com/'
S = requests.Session()
S.trust_env = False
FILES = {
    'taxa': ('wildlife_taxa', 'taxon_id UInt32, ancestry String, rank_level Float32, rank LowCardinality(String), name String, active Bool', 'taxon_id'),
    'observers': ('wildlife_observers', 'observer_id UInt32, login String, name String', 'observer_id'),
    'observations': ('wildlife', 'observation_uuid UUID, observer_id UInt32, latitude Nullable(Float64), longitude Nullable(Float64), positional_accuracy Nullable(UInt32), taxon_id UInt32, quality_grade LowCardinality(String), observed_on Nullable(Date32), observed_on_original String, anomaly_score Nullable(Float32), scientific_name LowCardinality(String), group_name LowCardinality(String)', '(group_name, taxon_id, ifNull(observed_on, toDate32(\'1900-01-01\')), observation_uuid)'),
    'photos': ('wildlife_photos', 'photo_id UInt64, observation_uuid UUID, observer_id UInt32, extension LowCardinality(String), license LowCardinality(String), position UInt16', '(observation_uuid, position, photo_id)'),
}
RAW_OBS = 'observation_uuid UUID, observer_id UInt32, latitude Nullable(Float64), longitude Nullable(Float64), positional_accuracy Nullable(UInt32), taxon_id Nullable(UInt32), quality_grade String, observed_on String, anomaly_score Nullable(Float32)'
RAW_PHOTOS = 'photo_uuid UUID, photo_id UInt64, observation_uuid UUID, observer_id UInt32, extension String, license String, width UInt32, height UInt32, position UInt16'

def main():
    manifest_path = BASE / 'source-manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {'captured_at': datetime.now(timezone.utc).isoformat(), 'source': 'https://registry.opendata.aws/inaturalist-open-data/', 'files': {}}
        for name in FILES:
            r = S.head(S3 + name + '.csv.gz', timeout=30)
            r.raise_for_status()
            manifest['files'][name] = {'url': r.url, 'bytes': int(r.headers['Content-Length']), 'etag': r.headers['ETag'], 'last_modified': r.headers['Last-Modified']}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    log = BASE / 'load.jsonl'
    done = {r['file']: r for r in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
    for name, (table, schema, order) in FILES.items():
        query(f'CREATE TABLE IF NOT EXISTS exploration.{table} ({schema}) ENGINE=MergeTree ORDER BY {order}')
        existing = int(query(f'SELECT count() FROM exploration.{table}'))
        if name in done:
            assert existing == done[name]['rows'], 'Previously completed import changed'
            continue
        assert existing == 0, f'Partial import in {table}; inspect before retrying'
        source = manifest['files'][name]
        head = S.head(source['url'], timeout=30)
        head.raise_for_status()
        assert head.headers['ETag'] == source['etag'] and int(head.headers['Content-Length']) == source['bytes'], 'Monthly source changed'
        if name == 'observations':
            sql = f"""INSERT INTO exploration.wildlife
SELECT o.observation_uuid, o.observer_id, o.latitude, o.longitude, o.positional_accuracy,
 ifNull(o.taxon_id,0), o.quality_grade, if(o.observed_on >= '1900-01-01' AND o.observed_on <= '2299-12-31',toDate32OrNull(o.observed_on),NULL), o.observed_on, o.anomaly_score, t.name,
 multiIf(has(splitByChar('/',t.ancestry),'3') OR t.taxon_id=3,'Birds',
 has(splitByChar('/',t.ancestry),'40151') OR t.taxon_id=40151,'Mammals',
 has(splitByChar('/',t.ancestry),'47158') OR t.taxon_id=47158,'Insects',
 has(splitByChar('/',t.ancestry),'47126') OR t.taxon_id=47126,'Plants',
 has(splitByChar('/',t.ancestry),'47170') OR t.taxon_id=47170,'Fungi',
 has(splitByChar('/',t.ancestry),'26036') OR t.taxon_id=26036,'Reptiles',
 has(splitByChar('/',t.ancestry),'20978') OR t.taxon_id=20978,'Amphibians',
 has(splitByChar('/',t.ancestry),'47178') OR t.taxon_id=47178,'Ray-finned fishes','Other')
FROM url('{source['url']}','TabSeparatedRawWithNames','{RAW_OBS}') o
LEFT ANY JOIN exploration.wildlife_taxa t ON ifNull(o.taxon_id,0)=t.taxon_id
SETTINGS input_format_tsv_empty_as_default=1, input_format_parallel_parsing=0, max_insert_block_size=500000, max_bytes_before_external_sort=1000000000, max_bytes_before_external_group_by=1000000000"""
        elif name == 'photos':
            sql = f"""INSERT INTO exploration.wildlife_photos
SELECT photo_id,observation_uuid,observer_id,extension,license,position
FROM url('{source['url']}','TabSeparatedRawWithNames','{RAW_PHOTOS}')
WHERE license IN ('CC0','CC-BY')
SETTINGS input_format_tsv_empty_as_default=1, input_format_parallel_parsing=0, max_insert_block_size=500000, max_bytes_before_external_sort=1000000000"""
        else:
            sql = f"INSERT INTO exploration.{table} SELECT * FROM url('{source['url']}','TabSeparatedRawWithNames','{schema}') SETTINGS input_format_tsv_empty_as_default=1"
        print('Starting', name, source['bytes'], 'compressed bytes', flush=True)
        started = time.perf_counter()
        query(sql, 'wildlife_import_' + name, timeout=7200)
        rows = int(query(f'SELECT count() FROM exploration.{table}'))
        record = {'file': name, 'table': table, 'rows': rows, 'seconds': time.perf_counter()-started, 'completed_at': datetime.now(timezone.utc).isoformat(), 'sql_sha256': hashlib.sha256(sql.encode()).hexdigest()}
        with log.open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    stats = json.loads(query("SELECT table,sum(rows) AS rows,sum(bytes_on_disk) AS bytes_on_disk FROM system.parts WHERE active AND database='exploration' AND table LIKE 'wildlife%' GROUP BY table FORMAT JSON"))['data']
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'tables': stats, 'photo_selection': 'Only CC0 and CC-BY photo metadata; observations are not filtered by photo license.', 'coordinates': 'Publicly exported coordinates only; may be obscured. Counts measure reported observations, not animal abundance.', 'schema': 'wildlife/load.py', 'source_manifest': 'wildlife/source-manifest.json'}
    (BASE/'dataset.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
