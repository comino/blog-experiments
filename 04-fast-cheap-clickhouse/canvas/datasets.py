"""Dataset boundaries and bounded playback SQL, independent of model output."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from selection import source, literal


DATASETS = {
    'wildlife': {
        'id': 'wildlife', 'label': 'Wildlife worldwide',
        'description': 'Species, seasons, observation density and licensed wildlife photographs.',
        'tables': {'wildlife', 'wildlife_gallery'},
        'charts': ['map', 'density', 'seasons', 'gallery', 'line', 'bar', 'heatmap', 'surface', 'scatter', 'table'],
        'coverage': 'Worldwide · iNaturalist snapshot, 27 September 2026',
        'examples': ['Show monarch observation density in 2025', 'Compare bird and mammal observations by month in 2025',
                     'Show seasonal observations for birds, insects and plants in 2025'],
    },
    'maritime': {
        'id': 'maritime', 'label': 'Ships & storms',
        'description': 'Vessel movements around the Gulf and Florida, with Atlantic hurricane tracks.',
        'tables': {'ais_positions', 'storm_tracks'},
        'charts': ['tracks', 'density', 'map', 'line', 'bar', 'heatmap', 'surface', 'scatter', 'table'],
        'coverage': 'Gulf & Florida · September–October 2024',
        'examples': ['Replay 12 moving cargo ships in the Gulf on October 9, 2024, with average speed above 5 knots',
                     'Compare hourly vessel activity by vessel type during Hurricane Milton',
                     'Show the Milton hurricane track and wind speed'],
    },
}
MAX_PLAYBACK_VESSELS = 24
MAX_PLAYBACK_STORMS = 8
MAX_PLAYBACK_DAYS = 14
MAX_PLAYBACK_POINTS = 12000
MAX_STORM_POINTS = 1000
MAX_PLAYBACK_BYTES = 4_000_000


def dataset_id(value=None):
    value = 'wildlife' if value is None else value
    if not isinstance(value, str) or value not in DATASETS:
        raise ValueError('Choose Wildlife worldwide or Ships & storms.')
    return value


def catalogue():
    items = []
    for key, description in DATASETS.items():
        item = {k: v for k, v in description.items() if k != 'tables'}
        path = Path('/dataset/dataset.json') if key == 'wildlife' else Path('/maritime/dataset.json')
        item['available'] = False
        if path.is_file():
            try:
                metadata = json.loads(path.read_text())
                # Only publish source/coverage measurements, never paths or runtime credentials.
                item['metadata'] = {k: metadata[k] for k in (
                    'name', 'rows', 'observations', 'observation_count', 'position_count', 'vessel_count',
                    'storm_count', 'storm_points', 'coverage', 'start', 'end', 'date_from', 'date_to',
                    'bbox', 'sources', 'license', 'licenses', 'status', 'loaded_at', 'counts', 'complete',
                    'ais', 'daily_coverage', 'missing_dates', 'validated_at', 'caveats',
                ) if k in metadata}
                item['available'] = metadata.get('status') not in {'loading', 'failed', 'pending'} and metadata.get('complete', True)
                if key == 'wildlife':
                    item['row_count'] = next((t['rows'] for t in metadata.get('tables', []) if t.get('table') == 'wildlife'), None)
                else:
                    item['row_count'] = metadata.get('ais', {}).get('positions')
                    item['vessel_count'] = metadata.get('ais', {}).get('vessels')
                item['status'] = 'ready' if item['available'] else 'loading'
            except (OSError, ValueError, TypeError):
                pass
        items.append(item)
    return items


def validate_sql(sql, active_dataset):
    """Only SELECTs over this dataset's real tables; CTE aliases are resolved by scope.

    The ClickHouse account is separately read-only and resource-limited. This guard
    also prevents an innocent model error from crossing datasets or reading an
    external table function. Never append model-provided SETTINGS.
    """
    active_dataset = dataset_id(active_dataset)
    if not isinstance(sql, str) or not 1 <= len(sql.strip()) <= 160000:
        raise ValueError('Invalid SQL')
    try:
        statements = sqlglot.parse(sql, read='clickhouse')
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError('The query could not be read safely. Please simplify it.') from exc
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise ValueError('Only a single read-only SELECT is allowed.')
    tree = statements[0]
    for node in tree.walk():
        if node.args.get('settings') or node.args.get('into') or node.args.get('locks'):
            raise ValueError('Query settings and output destinations are not allowed.')
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower().replace('_', '')
            if name.startswith(('dictget', 'dicthas', 'dictisin', 'sleep')) or name in {
                'url', 's3', 's3cluster', 'file', 'hdfs', 'remote', 'remotesecure', 'cluster', 'clusterallreplicas',
                'mysql', 'postgresql', 'sqlite', 'executable', 'getmacro', 'getsetting', 'getserverport',
            }:
                raise ValueError('External sources and server functions are not available to canvas queries.')
    allowed = DATASETS[active_dataset]['tables']
    try:
        for scope in traverse_scope(tree):
            for _, (_, resolved) in scope.selected_sources.items():
                if isinstance(resolved, exp.Table):
                    if not isinstance(resolved.this, exp.Identifier) or resolved.catalog or resolved.db != 'exploration' or resolved.name not in allowed:
                        raise ValueError('This query leaves the current dataset. Start a chat in the other dataset first.')
            # Table functions can appear as UDTFs rather than resolved table sources.
            for table in scope.tables:
                if not isinstance(table.this, exp.Identifier):
                    raise ValueError('Table functions are not allowed in canvas queries.')
        for table in tree.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                raise ValueError('Table functions are not allowed in canvas queries.')
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError('The query has ambiguous sources. Please simplify it.') from exc
    return sql


def playback_scope(card):
    """Recover a single source and exact saved predicates; never accept request SQL."""
    if card.get('chart') != 'tracks':
        raise ValueError('Choose a vessel or storm track view for playback.')
    validate_sql(card.get('sql'), 'maritime')
    tree = sqlglot.parse_one(card['sql'], read='clickhouse')
    if not isinstance(tree, exp.Select):
        raise ValueError('Replay a track view based on one source table.')
    base, predicates, _ = source(tree)
    if base not in {'ais_positions', 'storm_tracks'}:
        raise ValueError('Replay a vessel or storm track view.')
    # HAVING/QUALIFY decide which groups are selected. The retained result IDs
    # carry that restriction; WHERE conditions carry the source-row restriction.
    key = 'vessel_id' if base == 'ais_positions' else 'storm_id'
    columns = card['data']['columns']
    if key not in columns:
        raise ValueError('The track query must return ' + key + '.')
    identifiers = list(dict.fromkeys(row[columns.index(key)] for row in card['data']['rows']))
    maximum = MAX_PLAYBACK_VESSELS if base == 'ais_positions' else MAX_PLAYBACK_STORMS
    if not identifiers:
        return {'table': base, 'key': key, 'ids': [], 'predicate': '0'}
    if len(identifiers) > maximum:
        raise ValueError(f'Replay at most {maximum} tracks. Limit the track query to a smaller selection.')
    if any(type(value) not in (str, int) or not 1 <= len(str(value)) <= 64 for value in identifiers):
        raise ValueError('The track query returned an invalid identifier.')
    predicates += [exp.In(this=exp.column(key), expressions=[literal(value) for value in identifiers])]
    return {'table': base, 'key': key, 'ids': identifiers, 'predicate': exp.and_(*predicates).sql(dialect='clickhouse')}


def range_sql(scope):
    return ("SELECT min(observed_at) AS start,max(observed_at) AS end,count() AS positions,"
            "min(longitude) AS west,max(longitude) AS east,min(latitude) AS south,max(latitude) AS north "
            f"FROM exploration.{scope['table']} WHERE ({scope['predicate']}) "
            "AND longitude BETWEEN -180 AND 180 AND latitude BETWEEN -90 AND 90")


def utc(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('The track query returned an invalid UTC time.') from exc
    else:
        raise ValueError('The track query returned an invalid UTC time.')
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def playback_range(data):
    if not data.get('rows'):
        return None
    values = dict(zip(data['columns'], data['rows'][0]))
    if not values.get('positions'):
        return None
    start, end = utc(values['start']), utc(values['end'])
    span = (end - start).total_seconds()
    if span < 0 or span > MAX_PLAYBACK_DAYS * 86400:
        raise ValueError(f'Replay a window of at most {MAX_PLAYBACK_DAYS} days. Ask for a shorter date range.')
    box = {key: float(values[key]) for key in ('west', 'east', 'south', 'north')}
    if any(not math.isfinite(value) for value in box.values()):
        raise ValueError('The track query returned invalid coordinates.')
    # At most 500 buckets per vessel, including both partial boundary buckets.
    interval = max(60, math.ceil(max(1, span) / 498 / 60) * 60)
    return {'start': start.isoformat().replace('+00:00', 'Z'), 'end': end.isoformat().replace('+00:00', 'Z'),
            'interval_seconds': interval, 'positions': int(values['positions']), 'bounds': box}


def vessel_sql(scope, window, identifiers):
    chosen = exp.In(this=exp.column('vessel_id'), expressions=[literal(v) for v in identifiers]).sql(dialect='clickhouse')
    interval = window['interval_seconds']
    # argMax keeps a real observed position/time from each bucket, not an invented
    # average that could put a ship on land. Keep heading/course/speed consistent.
    return f"""SELECT vessel_id,point.1 AS observed_at,point.2 AS longitude,point.3 AS latitude,
point.4 AS speed_knots,point.5 AS course_deg,point.6 AS heading_deg,name_label AS vessel_name,type_label AS vessel_type
FROM (SELECT vessel_id,argMax(tuple(observed_at,longitude,latitude,speed_knots,course_deg,heading_deg),tuple(observed_at,longitude,latitude)) AS point,
any(vessel_name) AS name_label,any(vessel_type) AS type_label
FROM exploration.ais_positions WHERE ({scope['predicate']}) AND ({chosen})
AND longitude BETWEEN -180 AND 180 AND latitude BETWEEN -90 AND 90
GROUP BY vessel_id,toStartOfInterval(observed_at,INTERVAL {interval} SECOND))
ORDER BY vessel_id,observed_at LIMIT 6001"""


def storm_sql(scope, window):
    if scope['table'] == 'storm_tracks':
        predicate = scope['predicate']
    else:
        # The Gulf subset is regional. Show nearby Atlantic storms in the same
        # time window and expose their names; do not imply a causal relationship.
        box = window['bounds']
        predicate = (f"observed_at >= parseDateTimeBestEffort({literal(window['start']).sql(dialect='clickhouse')}) "
                     f"AND observed_at <= parseDateTimeBestEffort({literal(window['end']).sql(dialect='clickhouse')}) "
                     f"AND longitude BETWEEN {max(-180, box['west']-8)} AND {min(180, box['east']+8)} "
                     f"AND latitude BETWEEN {max(-90, box['south']-8)} AND {min(90, box['north']+8)}")
    return ("SELECT storm_id,storm_name,observed_at,longitude,latitude,wind_knots,pressure_hpa,category,category_code,usa_interpolation "
            f"FROM exploration.storm_tracks WHERE ({predicate}) "
            "AND longitude BETWEEN -180 AND 180 AND latitude BETWEEN -90 AND 90 "
            f"ORDER BY storm_id,observed_at LIMIT {MAX_STORM_POINTS + 1}")
