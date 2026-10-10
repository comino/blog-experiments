"""Check source normalization that can otherwise quietly corrupt maritime science."""
import csv
import io
import unittest

from load import parse_storms


class StormParsingTest(unittest.TestCase):
    def source(self, **changes):
        row = {'SID': '2024279N21265', 'SEASON': '2024', 'BASIN': 'NA', 'TRACK_TYPE': 'main',
               'NAME': 'MILTON', 'ISO_TIME': '2024-10-09 06:00:00', 'LAT': '23.5', 'LON': '-85.5',
               'USA_WIND': '140', 'USA_PRES': '907', 'USA_SSHS': '5', 'NATURE': 'TS',
               'USA_AGENCY': 'hurdat_atl', 'IFLAG': 'P______________', 'USA_RECORD': 'L',
               'USA_R34_NE': '100', 'USA_R34_SE': '0', 'USA_R34_SW': ' ', 'USA_R34_NW': '50'}
        row.update(changes)
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({name: 'unit' for name in row})  # IBTrACS units row is not data.
        writer.writerow(row)
        return handle.getvalue()

    def test_real_lowercase_main_and_units(self):
        result = parse_storms(self.source())[0]
        self.assertEqual(result['category'], 'Category 5')
        self.assertAlmostEqual(result['radius_34_ne_km'], 185.2)
        self.assertEqual(result['radius_34_se_km'], 0)
        self.assertIsNone(result['radius_34_sw_km'])
        self.assertEqual(result['usa_interpolation'], 'P')
        self.assertEqual(result['usa_record'], 'L')
        self.assertEqual(result['observed_at'], '2024-10-09 06:00:00')

    def test_missing_and_sentinel_values_stay_unknown(self):
        result = parse_storms(self.source(USA_WIND=' ', USA_PRES='-9999', USA_SSHS='-4'))[0]
        self.assertIsNone(result['wind_knots'])
        self.assertIsNone(result['pressure_hpa'])
        self.assertEqual(result['category'], 'Post-tropical')

    def test_provisional_spur_and_wrong_season_are_rejected(self):
        for field, value in [('TRACK_TYPE', 'spur'), ('TRACK_TYPE', 'PROVISIONAL'), ('SEASON', '2025'), ('BASIN', 'EP')]:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(AssertionError, 'No Milton'):
                    parse_storms(self.source(**{field: value}))

    def test_duplicate_points_are_rejected(self):
        source = self.source()
        source += source.splitlines()[-1] + '\n'
        with self.assertRaisesRegex(AssertionError, 'Duplicate'):
            parse_storms(source)


if __name__ == '__main__':
    unittest.main()
