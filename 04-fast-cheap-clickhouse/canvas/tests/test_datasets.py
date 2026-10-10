import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets
from datasets import validate_sql, playback_scope, playback_range, vessel_sql, storm_sql
from domain import blank, validate_state, validate_view, validate_plan
from selection import build_scope, apply_scope


def track_card(ids=('123456789',)):
    return {'id': 'tracks', 'revision': 'r1', 'chart': 'tracks', 'color': 'teal',
            'x': 'lon_center', 'y': 'lat_center', 'z': '',
            'sql': "SELECT vessel_id,avg(longitude) AS lon_center,avg(latitude) AS lat_center "
                   "FROM exploration.ais_positions WHERE lower(vessel_type)='cargo' "
                   "AND observed_at >= '2024-10-08' AND observed_at < '2024-10-12' "
                   "GROUP BY vessel_id ORDER BY count() DESC LIMIT 24",
            'data': {'columns': ['vessel_id', 'lon_center', 'lat_center'],
                     'rows': [[ident, -84, 27] for ident in ids]}}


def extent(end='2024-10-11 23:59:00'):
    return {'columns': ['start', 'end', 'positions', 'west', 'east', 'south', 'north'],
            'rows': [['2024-10-08 00:00:00', end, 100000, -85, -82, 25, 29]]}


class DatasetSql(unittest.TestCase):
    def test_same_dataset_joins_ctes_and_unions_work(self):
        for query in [
            'SELECT count() FROM exploration.wildlife',
            'WITH cohort AS (SELECT * FROM exploration.wildlife) SELECT count() FROM cohort',
            'SELECT w.taxon_id FROM exploration.wildlife w JOIN exploration.wildlife_gallery g USING(observation_uuid)',
            'SELECT count() FROM exploration.wildlife UNION ALL SELECT count() FROM exploration.wildlife_gallery',
        ]:
            with self.subTest(query=query): self.assertEqual(validate_sql(query, 'wildlife'), query)
        validate_sql('SELECT p.vessel_id FROM exploration.ais_positions p JOIN exploration.storm_tracks s ON p.observed_at=s.observed_at', 'maritime')

    def test_every_source_stays_in_its_dataset(self):
        for query in [
            'SELECT * FROM exploration.ais_positions',
            'WITH x AS (SELECT * FROM exploration.storm_tracks) SELECT * FROM x',
            'SELECT * FROM exploration.wildlife UNION ALL SELECT * FROM exploration.ais_positions',
            'SELECT * FROM exploration.wildlife WHERE taxon_id IN (SELECT vessel_id FROM exploration.ais_positions)',
            # An unrelated nested CTE name must not authorize an outer real table.
            'SELECT * FROM ais_positions WHERE 1 IN (WITH ais_positions AS (SELECT 1) SELECT 1)',
            'SELECT * FROM system.users', 'SELECT * FROM wildlife',
        ]:
            with self.subTest(query=query), self.assertRaises(ValueError): validate_sql(query, 'wildlife')
        with self.assertRaises(ValueError): validate_sql('SELECT * FROM exploration.wildlife', 'maritime')

    def test_writes_external_functions_and_settings_are_rejected(self):
        for query in [
            'DROP TABLE exploration.wildlife', 'SELECT 1; SELECT 2',
            "SELECT * FROM url('https://example.org/secret')", "SELECT * FROM file('/etc/passwd')",
            "SELECT * FROM remote('example.org', exploration.wildlife)",
            "SELECT * FROM s3('https://example.org/data')", 'SELECT * FROM numbers(100)',
            "SELECT dictGet('private','key',1)", 'SELECT sleep(10)',
            'SELECT * FROM exploration.wildlife SETTINGS max_threads=900',
            "SELECT * FROM exploration.wildlife INTO OUTFILE '/tmp/data'",
        ]:
            with self.subTest(query=query), self.assertRaises(ValueError): validate_sql(query, 'wildlife')

    def test_new_chart_contracts_and_wrong_dataset_are_validated(self):
        for chart, columns, axes in [
            ('density', ['lon', 'lat', 'n'], ['lon', 'lat', 'n']),
            ('seasons', ['month', 'species', 'n'], ['month', 'species', 'n']),
            ('seasons', ['month', 'n'], ['month', 'n', '']),
        ]:
            card = {'chart': chart, 'x': axes[0], 'y': axes[1], 'z': axes[2],
                    'data': {'columns': columns, 'rows': []}}
            validate_view(card)
        validate_view(track_card())
        with self.assertRaises(ValueError): validate_view(track_card() | {'data': {'columns': ['lon_center','lat_center'], 'rows': []}})
        op = {'action': 'add', 'title': 'Wrong dataset', 'sql': 'SELECT 1', 'chart': 'gallery'}
        with self.assertRaises(ValueError): validate_plan({'operations': [op]}, blank('maritime'), None)
        with self.assertRaises(ValueError): blank('unknown')

    def test_catalogue_reports_partial_data_without_claiming_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            wildlife = Path(temp) / 'wildlife.json'
            maritime = Path(temp) / 'maritime.json'
            wildlife.write_text(json.dumps({'tables': [{'table': 'wildlife', 'rows': 282933221}]}))
            maritime.write_text(json.dumps({'complete': False, 'ais': {'positions': 123, 'vessels': 4},
                                           'missing_dates': ['2024-10-10'], 'secret': 'must not leave file'}))
            def path(value): return wildlife if str(value).startswith('/dataset/') else maritime
            with patch.object(datasets, 'Path', side_effect=path): items = datasets.catalogue()
        self.assertTrue(items[0]['available'])
        self.assertFalse(items[1]['available'])
        self.assertEqual(items[1]['row_count'], 123)
        self.assertEqual(items[1]['status'], 'loading')
        self.assertNotIn('secret', json.dumps(items))


class PlaybackBounds(unittest.TestCase):
    def test_saved_ids_and_exact_filters_survive_playback_sql(self):
        scope = playback_scope(track_card(('123456789', '234567890')))
        self.assertIn("lower(vessel_type) = 'cargo'", scope['predicate'])
        self.assertIn("observed_at < '2024-10-12'", scope['predicate'])
        self.assertIn("vessel_id IN ('123456789', '234567890')", scope['predicate'])
        window = playback_range(extent())
        self.assertTrue(window['start'].endswith('Z'))
        self.assertGreater(window['interval_seconds'], 60)
        sql = vessel_sql(scope, window, scope['ids'])
        self.assertIn(scope['predicate'], sql)
        self.assertIn('argMax(tuple(', sql)
        self.assertIn('LIMIT 6001', sql)
        validate_sql(sql, 'maritime')
        storm = storm_sql(scope, window)
        self.assertIn('longitude BETWEEN -93.0 AND -74.0', storm)
        self.assertIn('usa_interpolation', storm)
        validate_sql(storm, 'maritime')

    def test_too_many_tracks_long_windows_and_complex_sources_fail(self):
        with self.assertRaises(ValueError): playback_scope(track_card(tuple(str(i) for i in range(25))))
        with self.assertRaises(ValueError): playback_range(extent('2024-10-25 00:00:00'))
        with self.assertRaises(ValueError): playback_range(extent('2024-10-01 00:00:00'))
        with self.assertRaises(ValueError): playback_range(extent('not a timestamp'))
        for sql in [
            'SELECT p.vessel_id FROM exploration.ais_positions p JOIN exploration.storm_tracks s ON p.observed_at=s.observed_at',
            'WITH p AS (SELECT * FROM exploration.ais_positions) SELECT * FROM p',
            'SELECT * FROM exploration.wildlife',
        ]:
            with self.subTest(sql=sql), self.assertRaises(ValueError): playback_scope(track_card() | {'sql': sql})

    def test_empty_selection_and_empty_source_are_safe(self):
        self.assertEqual(playback_scope(track_card(()))['ids'], [])
        empty = extent(); empty['rows'][0][2] = 0
        self.assertIsNone(playback_range(empty))
        self.assertIsNone(playback_range({'rows': [], 'columns': []}))

    def test_prewhere_is_preserved_and_sampling_cannot_broaden(self):
        card = track_card()
        card['sql'] = card['sql'].replace("WHERE lower(vessel_type)='cargo' AND", "PREWHERE lower(vessel_type)='cargo' WHERE")
        self.assertIn("lower(vessel_type) = 'cargo'", playback_scope(card)['predicate'])
        card['sql'] = card['sql'].replace('FROM exploration.ais_positions', 'FROM exploration.ais_positions SAMPLE 0.1')
        with self.assertRaises(ValueError): playback_scope(card)

    def test_qualified_source_filter_is_not_confused_with_aggregate_alias(self):
        card = track_card()
        card['sql'] = "SELECT p.vessel_id,any(p.vessel_type) AS vessel_type,avg(p.longitude) AS lon_center,avg(p.latitude) AS lat_center FROM exploration.ais_positions p WHERE p.vessel_type='Cargo' AND p.observed_at >= '2024-10-08' AND p.observed_at < '2024-10-12' GROUP BY p.vessel_id"
        predicate = playback_scope(card)['predicate']
        self.assertIn("vessel_type = 'Cargo'", predicate)
        self.assertNotIn('any(', predicate.lower())

    def test_correlated_subquery_filter_cannot_lose_its_scope_on_replay(self):
        card = track_card()
        card['sql'] = "SELECT p.vessel_id,avg(p.longitude) AS lon_center,avg(p.latitude) AS lat_center FROM exploration.ais_positions p WHERE p.observed_at >= '2024-10-08' AND p.observed_at < '2024-10-12' AND EXISTS(SELECT 1 FROM exploration.storm_tracks s WHERE s.observed_at=p.observed_at) GROUP BY p.vessel_id"
        validate_sql(card['sql'], 'maritime')
        with self.assertRaisesRegex(ValueError, 'subquery filter'): playback_scope(card)

    def test_storm_replay_preserves_name_and_time_filter(self):
        card = track_card() | {'sql': "SELECT storm_id,avg(longitude) AS lon_center,avg(latitude) AS lat_center FROM exploration.storm_tracks WHERE storm_name='MILTON' AND observed_at >= '2024-10-08' AND observed_at < '2024-10-12' GROUP BY storm_id",
                              'data': {'columns': ['storm_id', 'lon_center', 'lat_center'], 'rows': [['milton', -85, 27]]}}
        scope = playback_scope(card)
        self.assertIn("storm_name = 'MILTON'", storm_sql(scope, playback_range(extent())))
        self.assertIn("storm_id IN ('milton')", storm_sql(scope, playback_range(extent())))

    def test_marked_ships_cannot_broaden_into_storms(self):
        card = track_card()
        mark = {'kind': 'points', 'card_id': card['id'], 'revision': card['revision'], 'row_indices': [0]}
        scope = build_scope(mark, {'cards': [card]}, card['id'])
        with self.assertRaises(ValueError): apply_scope('SELECT count() FROM exploration.storm_tracks', scope)
        sql = apply_scope('SELECT toHour(observed_at) AS hour,count() FROM exploration.ais_positions GROUP BY hour', scope)
        self.assertIn('123456789', sql)
        self.assertIn("lower(vessel_type) = 'cargo'", sql)

    def test_maritime_region_preserves_original_time_and_prewhere(self):
        card = track_card()
        card['sql'] = card['sql'].replace("WHERE lower(vessel_type)='cargo' AND", "PREWHERE lower(vessel_type)='cargo' WHERE")
        mark = {'kind': 'region', 'card_id': card['id'], 'revision': card['revision'],
                'bounds': {'west': -86, 'east': -81, 'south': 24, 'north': 29}}
        for chart in ('map', 'density', 'tracks'):
            with self.subTest(chart=chart):
                card['chart'] = chart
                scope = build_scope(mark, {'cards': [card]}, card['id'])
                sql = apply_scope('SELECT vessel_type,count() FROM exploration.ais_positions GROUP BY vessel_type', scope)
                for expected in ("lower(vessel_type) = 'cargo'", "observed_at < '2024-10-12'", 'longitude BETWEEN -86 AND -81', 'latitude BETWEEN 24 AND 29'):
                    self.assertIn(expected, sql)
                if chart == 'tracks': self.assertIn("vessel_id IN ('123456789')", sql)
                with self.assertRaises(ValueError): apply_scope('SELECT * FROM exploration.storm_tracks', scope)
        card['sql'] = "SELECT storm_id,longitude,latitude FROM exploration.storm_tracks WHERE storm_name='MILTON'"
        card['data'] = {'columns': ['storm_id', 'longitude', 'latitude'], 'rows': [['milton', -85, 27]]}
        scope = build_scope(mark, {'cards': [card]}, card['id'])
        sql = apply_scope('SELECT count() FROM exploration.storm_tracks', scope)
        self.assertIn("storm_name = 'MILTON'", sql)
        self.assertIn('longitude BETWEEN -86 AND -81', sql)
        self.assertIn("storm_id IN ('milton')", sql)
        with self.assertRaises(ValueError): apply_scope('SELECT count() FROM exploration.ais_positions', scope)


if __name__ == '__main__': unittest.main()
