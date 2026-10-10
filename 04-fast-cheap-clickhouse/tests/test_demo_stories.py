"""Offline checks against committed rehearsal evidence; no live services needed."""

import copy
import gzip
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_demo_stories", ROOT / "verify_demo_stories.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
EVIDENCE = ROOT / "demos" / "rehearsal-evidence.json.gz"


class DemoReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = verifier.read(EVIDENCE)
        cls.stories = {story["id"]: story for story in cls.report["stories"]}
        cls.references, cls.supplemental, _ = verifier.load_references(EVIDENCE)

    def card(self, story_id, step_id):
        story = self.stories[story_id]
        return copy.deepcopy(next(step["card"] for step in story["steps"] if step["id"] == step_id))

    def semantic(self, card, reference_id):
        return {
            name: passed
            for name, passed, _ in verifier.semantic_checks(card, self.references[reference_id], reference_id)
        }

    def test_all_four_successful_stories(self):
        result = verifier.verify_report(EVIDENCE)
        self.assertEqual(set(self.stories), set(verifier.MAPPINGS))
        self.assertTrue(result["passed"], json.dumps(result, indent=2))

    def test_aggregate_change_rejected(self):
        card = self.card("monarch-seasonal-footprint", "map")
        card["data"]["rows"][0][card["data"]["columns"].index(card["z"])] += 1
        self.assertFalse(verifier.table_check(card, "monarch_density_5deg", self.references["monarch_density_5deg"])[0])

    def test_harmless_alias_and_row_column_order(self):
        card = self.card("monarch-seasonal-footprint", "map")
        original = card["z"]
        card["sql"] = re.sub(r"\b" + re.escape(original) + r"\b", "renamed_count", card["sql"])
        card["z"] = "renamed_count"
        card["data"]["columns"] = ["renamed_count" if name == original else name for name in card["data"]["columns"]][::-1]
        card["data"]["rows"] = [row[::-1] for row in card["data"]["rows"][::-1]]
        self.assertTrue(verifier.table_check(card, "monarch_density_5deg", self.references["monarch_density_5deg"])[0])

    def test_copied_good_data_wrong_year_rejected(self):
        card = self.card("monarch-seasonal-footprint", "map")
        self.assertIn("2025-01-01", card["sql"])
        card["sql"] = card["sql"].replace("2025-01-01", "2024-01-01")
        self.assertFalse(self.semantic(card, "monarch_density_5deg")["source_cohort_scope"])

    def test_gallery_wrong_photo_rejected(self):
        story_id = "fungi-opposite-seasons"
        card = self.card(story_id, "photos")
        card["data"]["rows"][0][card["data"]["columns"].index("photo_id")] = 1
        self.assertFalse(verifier.gallery_check(
            card, story_id, self.references["fungi_sydney_may_gallery_diverse"],
            self.supplemental[story_id + "_gallery_cohort"],
        )[0])

    def test_elsa_missing_day_rejected_even_one_row(self):
        if "elsa-and-milton" in self.stories:
            card = self.card("elsa-and-milton", "replay")
            self.assertEqual(len(card["data"]["rows"]), 1)
        else:
            card = {"sql": self.references["elsa_replay"]["sql"]}
        self.assertIn("2024-10-13", card["sql"])
        card["sql"] = card["sql"].replace("2024-10-13", "2024-10-12")
        self.assertFalse(self.semantic(card, "elsa_replay")["source_cohort_scope"])

    def test_speed_grouping_by_aggregate_rejected(self):
        card = self.card("mississippi-motion-and-stillness", "speeds")
        card["sql"], replacements = re.subn(
            r"GROUP BY\s+hour\s*,\s*vessel_id", "GROUP BY hour, name_label", card["sql"], flags=re.I,
        )
        self.assertEqual(replacements, 1)
        self.assertFalse(self.semantic(card, "river_speeds")["hour_and_vessel_id_group_keys"])

    def test_gap_over_ten_minutes_rejected(self):
        card = self.card("mississippi-motion-and-stillness", "slow")
        card["sql"], replacements = re.subn(r">\s*600\b", "> 900", card["sql"])
        self.assertEqual(replacements, 1)
        self.assertFalse(self.semantic(card, "river_slow_spells")["breaks_fast_unknown_and_over_10_minute_gaps"])

    def test_fast_interruptions_must_split(self):
        card = self.card("mississippi-motion-and-stillness", "slow")
        card["sql"], replacements = re.subn(r"speed_knots\s*>=\s*1\s+OR\s+", "", card["sql"], flags=re.I)
        self.assertEqual(replacements, 1)
        self.assertFalse(self.semantic(card, "river_slow_spells")["breaks_fast_unknown_and_over_10_minute_gaps"])

    def test_rounded_six_hour_threshold_rejected(self):
        card = self.card("mississippi-motion-and-stillness", "slow")
        card["sql"], replacements = re.subn(
            r"HAVING\s+dateDiff\('second',\s*min\(observed_at\),\s*max\(observed_at\)\)\s*>=\s*21600",
            "HAVING duration_hours >= 6", card["sql"], flags=re.I,
        )
        self.assertEqual(replacements, 1)
        self.assertFalse(self.semantic(card, "river_slow_spells")["six_hour_threshold_uses_unrounded_seconds"])

    def test_wrong_raw_threshold_rejected(self):
        card = self.card("mississippi-motion-and-stillness", "slow")
        card["sql"], replacements = re.subn(r">=\s*21600\b", ">= 2160", card["sql"])
        self.assertEqual(replacements, 1)
        self.assertFalse(self.semantic(card, "river_slow_spells")["six_hour_threshold_uses_unrounded_seconds"])

    def test_fake_playback_position_rejected(self):
        story_id = "mississippi-motion-and-stillness"
        playback = copy.deepcopy(self.stories[story_id]["playback_responses"][0])
        playback["rows"][0][playback["columns"].index("longitude")] += 0.1
        self.assertFalse(verifier.playback_check(playback, story_id, self.supplemental)[0])

    def test_foreign_storm_rejected(self):
        story_id = "mississippi-motion-and-stillness"
        playback = copy.deepcopy(self.stories[story_id]["playback_responses"][0])
        playback["storm"]["rows"][0][playback["storm"]["columns"].index("storm_name")] = "LESLIE"
        self.assertFalse(verifier.playback_check(playback, story_id, self.supplemental)[0])

    def test_plain_and_gzip_json_read_the_same(self):
        value = {"stories": [{"id": "portable"}], "note": "UTC • observations"}
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "evidence.json"
            compressed = Path(directory) / "evidence.json.gz"
            plain.write_text(json.dumps(value), encoding="utf-8")
            with gzip.open(compressed, "wt", encoding="utf-8") as output:
                json.dump(value, output)
            self.assertEqual(verifier.read(plain), value)
            self.assertEqual(verifier.read(compressed), value)

    def test_story_filter_checks_only_requested_and_rejects_missing(self):
        requested = ["monarch-seasonal-footprint", "fungi-opposite-seasons"]
        result = verifier.verify_report(EVIDENCE, stories=",".join(requested))
        self.assertTrue(result["passed"])
        self.assertEqual({story["id"] for story in result["stories"]}, set(requested))
        subset = {"stories": [self.stories[requested[0]]], "passed": True}
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "subset.json"
            evidence.write_text(json.dumps(subset), encoding="utf-8")
            result = verifier.verify_report(evidence, stories=requested)
        self.assertFalse(result["passed"])
        missing = next(story for story in result["stories"] if story["id"] == requested[1])
        self.assertEqual(missing["checks"][0]["check"], "requested_story_present")
        with self.assertRaises(ValueError):
            verifier.verify_report(EVIDENCE, stories=["unknown-story"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
