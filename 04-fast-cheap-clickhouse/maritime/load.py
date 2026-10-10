#!/usr/bin/env python3
"""Load manifested public NOAA AIS and IBTrACS sources into NEW maritime tables.

Run on the existing demo host from /opt/ch-exploration. Uses its private loader
credential file; never logs secrets. The import does not alter any other tables.
Two bounded workers load different dates. It resumes verified completed dates and
refuses ambiguous partial inserts, rather than silently appending duplicates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date, datetime, timedelta, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import uuid

import requests

BASE = Path(__file__).resolve().parent
AIS_BASE = 'https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/'
STORMS_URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv'
BBOX = {'west': -98, 'south': 18, 'east': -75, 'north': 33}
FIRST = date(2024, 9, 15)
LAST = date(2024, 10, 15)
LOCK = threading.Lock()
CATEGORY = {-5: 'Unknown', -4: 'Post-tropical', -3: 'Disturbance', -2: 'Subtropical', -1: 'Tropical depression', 0: 'Tropical storm', **{i: f'Category {i}' for i in range(1, 6)}}
IMAGERY = {
    'source': 'NASA Global Imagery Browse Services',
    'layer': 'MODIS_Aqua_CorrectedReflectance_TrueColor',
    'documentation': 'https://nasa-gibs.github.io/gibs-api-docs/access-basics/',
    'usage_policy': 'https://www.nasa.gov/nasa-brand-center/images-and-media/',
    'attribution': 'NASA GIBS / Aqua MODIS',
    'usage': 'NASA permits factual educational/informational use under its media guidelines; acknowledge NASA and do not imply endorsement.',
    'tile_template': 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Aqua_CorrectedReflectance_TrueColor/default/{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg',
    'time_resolution': 'Daily imagery for the selected UTC date; not an hourly satellite frame and not a live feed.',
    'storage': 'Optional browser-requested public WMTS tiles; imagery is not ingested into ClickHouse.'}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name('.' + path.stem + '.' + uuid.uuid4().hex + '.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def public_session():
    session = requests.Session()
    session.trust_env = False
    return session


def query(sql, query_id=None, timeout=900):
    # A fresh session per worker prevents sharing credentials/cookies across hosts.
    credentials = json.loads((BASE.parent / 'runtime/credentials.json').read_text())
    with public_session() as session:
        session.auth = ('loader', credentials['loader'])
        params = {'max_threads': 2, 'max_insert_threads': 1, 'max_memory_usage': 4000000000,
                  'max_bytes_before_external_group_by': 1000000000,
                  'max_bytes_before_external_sort': 1000000000}
        if query_id:
            params['query_id'] = query_id
        response = session.post('http://127.0.0.1:18123/', params=params, data=sql.encode(), timeout=timeout)
        if not response.ok:
            raise RuntimeError(response.text[:3500])
        return response.text


def rows(sql):
    return json.loads(query(sql + ' FORMAT JSON'))['data']


def head(url):
    with public_session() as session:
        response = session.head(url, timeout=45)
        response.raise_for_status()
        return {'url': url, 'bytes': int(response.headers['Content-Length']),
                'etag': response.headers['ETag'], 'last_modified': response.headers.get('Last-Modified'),
                'source_content_md5_base64': response.headers.get('Content-MD5')}


def check_source(source):
    current = head(source['url'])
    if current['etag'] != source['etag'] or current['bytes'] != source['bytes']:
        raise RuntimeError('Manifested source changed: ' + source['url'])


def manifest():
    path = BASE / 'source-manifest.json'
    if path.exists():
        result = json.loads(path.read_text())
        assert result['ais']['bbox'] == BBOX and result['ais']['first_date'] == str(FIRST) and result['ais']['last_date'] == str(LAST)
        expected = {str(FIRST + timedelta(days=i)) for i in range((LAST - FIRST).days + 1)}
        files = result['ais']['files']
        assert len(files) == len(expected) and {entry['date'] for entry in files} == expected, 'Source manifest is missing or repeating an AIS day'
        assert all(entry['url'] == AIS_BASE + 'ais-' + entry['date'] + '.parquet' for entry in files), 'Unexpected AIS source URL'
        assert result['storms']['file']['url'] == STORMS_URL, 'Unexpected storm source URL'
        return result
    dates = [FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        sources = list(pool.map(lambda day: dict(head(AIS_BASE + f'ais-{day}.parquet'), date=str(day)), dates))
    result = {
        'captured_at': utcnow(), 'timezone': 'UTC',
        'ais': {'source': 'https://github.com/ocm-marinecadastre/ais-vessel-traffic',
                'documentation': 'https://github.com/ocm-marinecadastre/ais-vessel-traffic/blob/main/data/ais-broadcast-points-2024-readme.md',
                'license': 'CC0-1.0', 'license_url': 'https://github.com/ocm-marinecadastre/ais-vessel-traffic/blob/main/LICENSE.md',
                'citation': 'Martin, Daniel R., Jesse Brass, Matthew Dornback, and Jeremy Fontenault. 2025. Nationwide Automatic Identification System 2024. NOAA Office for Coastal Management. U.S. Coast Guard Navigation Center.',
                'first_date': str(FIRST), 'last_date': str(LAST), 'bbox': BBOX,
                'subset': 'All valid-position AIS rows within bbox and date window. No vessel sampling and no additional temporal downsampling.',
                'source_cadence': 'NOAA product is downsampled to approximately one-minute broadcasts; timestamps retain original seconds. Transmission/receiver gaps remain.',
                'files': sources},
        'storms': {'source': 'https://www.ncei.noaa.gov/products/international-best-track-archive',
                   'documentation': 'https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf',
                   'doi': 'https://doi.org/10.25921/82ty-9e16',
                   'citation': 'Gahtan, J., et al., 2024. International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01. North Atlantic 2024 MAIN tracks. NOAA NCEI. doi:10.25921/82ty-9e16. Accessed ' + datetime.now(timezone.utc).date().isoformat(),
                   'paper': 'Knapp, K. R., et al., 2010. The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data. BAMS, 91, 363-376. doi:10.1175/2009BAMS2755.1.',
                   'access': 'Public NOAA archive; retain NOAA/IBTrACS attribution and dataset citation.',
                   'subset': 'SEASON=2024 AND BASIN=NA AND TRACK_TYPE=MAIN; all source times and full tracks retained, including locations outside the AIS bbox.',
                   'wind': 'USA_WIND, knots, 1-minute mean; pressure USA_PRES in hPa; no mixing with other agency averaging periods.',
                   'radius': 'USA_R34_NE/SE/SW/NW converted from nautical miles to km (x1.852); quadrant maximum 34-knot wind extents, not a forecast cone.',
                   'interpolation': 'No new interpolated points; source IBTrACS may interpolate values. First IFLAG character retained as usa_interpolation.',
                   'file': head(STORMS_URL)}}
    result['imagery'] = IMAGERY
    write_json(path, result)
    return result


def record(event):
    with LOCK:
        with (BASE / 'load.jsonl').open('a') as handle:
            handle.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)


def nullable_number(value, *, minimum=None, maximum=None):
    value = (value or '').strip()
    if not value:
        return None
    number = float(value)
    if minimum is not None and number < minimum or maximum is not None and number > maximum:
        return None
    return number


def parse_storms(text):
    selected = []
    for row in csv.DictReader(io.StringIO(text)):
        if row['SEASON'].strip() != '2024' or row['BASIN'].strip() != 'NA' or row['TRACK_TYPE'].strip().upper() != 'MAIN':
            continue
        lat = nullable_number(row['LAT'], minimum=-90, maximum=90)
        lon = nullable_number(row['LON'], minimum=-180, maximum=180)
        if lat is None or lon is None:
            continue
        code = nullable_number(row['USA_SSHS'], minimum=-5, maximum=5)
        code = int(code) if code is not None else None
        item = {'storm_id': row['SID'].strip(), 'storm_name': row['NAME'].strip(),
                'observed_at': row['ISO_TIME'].strip(), 'latitude': lat, 'longitude': lon,
                'wind_knots': nullable_number(row['USA_WIND'], minimum=0, maximum=300),
                'pressure_hpa': nullable_number(row['USA_PRES'], minimum=850, maximum=1100),
                'category': CATEGORY.get(code, 'Unknown'), 'category_code': code,
                'nature': row['NATURE'].strip(), 'basin': 'NA', 'season': 2024,
                'wind_source': row['USA_AGENCY'].strip(), 'usa_interpolation': row['IFLAG'].strip()[:1],
                'usa_record': row['USA_RECORD'].strip()}
        for quadrant in ('ne', 'se', 'sw', 'nw'):
            radius = nullable_number(row['USA_R34_' + quadrant.upper()], minimum=0)
            item['radius_34_' + quadrant + '_km'] = radius * 1.852 if radius is not None else None
        selected.append(item)
    assert selected and any(r['storm_name'] == 'MILTON' for r in selected), 'No Milton data found'
    assert len({(r['storm_id'], r['observed_at']) for r in selected}) == len(selected), 'Duplicate storm/time pairs'
    return selected


def load_storms(source, completed):
    existing = int(query('SELECT count() FROM exploration.storm_tracks'))
    if 'storms' in completed:
        assert existing == completed['storms']['rows'], 'Storm table changed after completed import'
        return
    assert not existing, 'Unreconciled partial storm import; inspect before retrying'
    check_source(source)
    with public_session() as session:
        response = session.get(source['url'], timeout=120)
        response.raise_for_status()
    assert len(response.content) == source['bytes'], 'Storm download truncated'
    assert response.headers.get('ETag') == source['etag'], 'Storm source changed during download'
    selected = parse_storms(response.text)
    content = '\n'.join(json.dumps(row) for row in selected)
    started = time.monotonic()
    query('INSERT INTO exploration.storm_tracks FORMAT JSONEachRow\n' + content, 'maritime_import_storms_2024')
    count = int(query('SELECT count() FROM exploration.storm_tracks'))
    assert count == len(selected)
    record({'file': 'storms', 'rows': count, 'source_sha256': hashlib.sha256(response.content).hexdigest(),
            'selected_sha256': hashlib.sha256(content.encode()).hexdigest(), 'seconds': time.monotonic() - started,
            'completed_at': utcnow()})


def ais_insert_sql(source):
    day = source['date']
    assert source['url'] == AIS_BASE + f'ais-{day}.parquet'
    date.fromisoformat(day)
    return f"""INSERT INTO exploration.ais_positions
SELECT toDate('{day}'), toString(assumeNotNull(mmsi)), toDateTime(assumeNotNull(base_date_time), 'UTC'),
 tupleElement(geometry, 2), tupleElement(geometry, 1),
 if(sog >= 0 AND sog < 102.3, sog, NULL),
 if(cog >= 0 AND cog < 360, cog, NULL),
 if(heading >= 0 AND heading < 360, toFloat32(heading), NULL),
 ifNull(vessel_name,''),
 multiIf(vessel_type=30,'Fishing',vessel_type IN (31,32),'Towing',vessel_type=33,'Dredging',
  vessel_type=34,'Diving',vessel_type=35,'Military',vessel_type=36,'Sailing',vessel_type=37,'Pleasure craft',
  vessel_type BETWEEN 40 AND 49,'High-speed craft',vessel_type=50,'Pilot',vessel_type=51,'Search and rescue',
  vessel_type=52,'Tug',vessel_type=53,'Port tender',vessel_type=54,'Anti-pollution',vessel_type=55,'Law enforcement',
  vessel_type=58,'Medical transport',vessel_type=59,'Noncombatant',vessel_type BETWEEN 60 AND 69,'Passenger',
  vessel_type BETWEEN 70 AND 79,'Cargo',vessel_type BETWEEN 80 AND 89,'Tanker',vessel_type BETWEEN 90 AND 99,'Other',
  'Unknown'),
 if(vessel_type BETWEEN 1 AND 99,toUInt16(vessel_type),NULL),
 if(status BETWEEN 0 AND 14,toUInt8(status),NULL),
 if(length > 0,length,NULL),if(draft > 0,draft,NULL)
FROM url('{source['url']}', 'Parquet')
WHERE geometry.1 BETWEEN {BBOX['west']} AND {BBOX['east']}
 AND geometry.2 BETWEEN {BBOX['south']} AND {BBOX['north']}
 AND mmsi BETWEEN 100000000 AND 999999999
 AND base_date_time >= toDateTime('{day} 00:00:00','UTC')
 AND base_date_time < toDateTime('{day} 00:00:00','UTC') + INTERVAL 1 DAY
SETTINGS input_format_parallel_parsing=0, max_insert_block_size=250000"""


def load_ais(source, completed):
    day = source['date']
    count_sql = f"SELECT count() FROM exploration.ais_positions WHERE source_date=toDate('{day}')"
    existing = int(query(count_sql))
    if day in completed:
        assert existing == completed[day]['rows'], 'Completed AIS date changed: ' + day
        return
    assert not existing, 'Unreconciled partial AIS import; inspect before retrying: ' + day
    check_source(source)
    sql = ais_insert_sql(source)
    print('Loading AIS ' + day + ' (' + str(source['bytes']) + ' source bytes)', flush=True)
    started = time.monotonic()
    query(sql, 'maritime_import_ais_' + day.replace('-', ''), timeout=900)
    count = int(query(count_sql))
    assert count > 0, 'No rows retained for source date ' + day
    check_source(source)
    record({'file': day, 'rows': count, 'seconds': time.monotonic() - started, 'completed_at': utcnow(),
            'sql_sha256': hashlib.sha256(sql.encode()).hexdigest(), 'source_etag_verified_before_and_after': True})


def validate(data, complete=True):
    log = BASE / 'load.jsonl'
    completed = {r['file']: r for r in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
    daily = rows('SELECT source_date, count() AS positions, uniqExact(vessel_id) AS vessels, min(observed_at) AS first_observed_at, max(observed_at) AS last_observed_at FROM exploration.ais_positions GROUP BY source_date ORDER BY source_date')
    for day in daily:
        prior = completed.get(day['source_date'])
        day['import_complete'] = bool(prior and int(day['positions']) == prior['rows'])
    bounds = rows('SELECT count() AS positions, uniqExact(vessel_id) AS vessels, min(observed_at) AS first_observed_at, max(observed_at) AS last_observed_at, min(longitude) AS west, max(longitude) AS east, min(latitude) AS south, max(latitude) AS north, countIf(speed_knots IS NULL) AS missing_speed FROM exploration.ais_positions')[0]
    storms = rows('SELECT storm_name, storm_id, count() AS positions, min(observed_at) AS first_observed_at, max(observed_at) AS last_observed_at, max(wind_knots) AS max_wind_knots, min(pressure_hpa) AS min_pressure_hpa FROM exploration.storm_tracks GROUP BY storm_name, storm_id ORDER BY first_observed_at')
    invalid = rows("SELECT countIf(toDate(observed_at) != source_date) AS wrong_day, countIf(longitude < -98 OR longitude > -75 OR latitude < 18 OR latitude > 33) AS outside_bbox, countIf(length(vessel_id) != 9) AS invalid_vessel_id FROM exploration.ais_positions")[0]
    expected_dates = {source['date'] for source in data['ais']['files']}
    got_dates = {day['source_date'] for day in daily if day['import_complete']}
    checks = {'date_coverage': got_dates == expected_dates,
              'nonempty_days': bool(daily) and all(int(day['positions']) > 0 for day in daily),
              'consistent_source_dates': int(invalid['wrong_day']) == 0,
              'only_selected_bbox': int(invalid['outside_bbox']) == 0,
              'valid_9_digit_vessel_id': int(invalid['invalid_vessel_id']) == 0,
              'milton_present': any(storm['storm_name'] == 'MILTON' and float(storm['max_wind_knots']) >= 137 for storm in storms)}
    result = {'name': 'Ships & storms', 'validated_at': utcnow(), 'complete': complete and all(checks.values()),
              'position_count': int(bounds['positions']), 'vessel_count': int(bounds['vessels']),
              'storm_count': len(storms), 'storm_points': sum(int(storm['positions']) for storm in storms),
              'coverage': {'first_date': str(FIRST), 'last_date': str(LAST), 'bbox': BBOX,
                           'completed_dates': sorted(got_dates),
                           'in_progress_dates': [day['source_date'] for day in daily if not day['import_complete']],
                           'description': 'Historical reported AIS positions in Gulf/Florida/nearby Atlantic; not all vessels, not global coverage, not a live feed.'},
              'ais': bounds, 'daily_coverage': daily, 'storms': storms, 'checks': checks,
              'missing_dates': sorted(expected_dates - got_dates), 'invalid': invalid,
              'storage': rows("SELECT table,sum(rows) AS rows,sum(bytes_on_disk) AS bytes_on_disk FROM system.parts WHERE active AND database='exploration' AND table IN ('ais_positions','storm_tracks') GROUP BY table ORDER BY table"),
              'clickhouse_version': query('SELECT version()').strip(),
              'schema_sha256': hashlib.sha256((BASE / 'schema.sql').read_bytes()).hexdigest(),
              'source_manifest': 'maritime/source-manifest.json',
              'caveats': ['AIS counts measure received broadcasts, not vessel abundance or complete port arrivals.',
                          'Storm/coastal receiver outages can reduce AIS counts; traffic changes alone do not establish storm causation.',
                          'No positions are fabricated or interpolated by this loader. IBTrACS source interpolation flags are retained.',
                          'Sparse tracks and long gaps must not be animated as continuous known journeys.',
                          'Stationary reports alone do not establish anchoring, waiting, or port closure.']}
    write_json(BASE / 'dataset.json', result)
    print(json.dumps({'validation_complete': result['complete'], 'checks': checks, 'ais': bounds}), flush=True)
    if complete:
        assert all(checks.values()), checks
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--only-date', action='append', help='Load just these manifested dates for a partial smoke test; does not claim complete coverage.')
    args = parser.parse_args()
    # Keep two invocations from both observing an empty day and inserting it twice.
    import_lock = None
    if not args.validate_only:
        import_lock = (BASE / '.import.lock').open('a')
        try:
            fcntl.flock(import_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another maritime import is running; wait for it to finish.') from error
    data = manifest()
    if args.validate_only:
        validate(data)
        return
    for sql in (BASE / 'schema.sql').read_text().split(';'):
        if sql.strip():
            query(sql)
    log = BASE / 'load.jsonl'
    completed = {r['file']: r for r in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
    load_storms(data['storms']['file'], completed)
    sources = data['ais']['files']
    if args.only_date:
        assert set(args.only_date) <= {s['date'] for s in sources}, 'Requested date not in manifested source range'
        sources = [s for s in sources if s['date'] in args.only_date]
    # Landfall window becomes available first while the preceding baseline loads.
    sources = sorted(sources, key=lambda s: abs((date.fromisoformat(s['date']) - date(2024, 10, 9)).days))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(load_ais, source, completed) for source in sources]):
            future.result()
    validate(data, complete=not args.only_date)


if __name__ == '__main__':
    main()
