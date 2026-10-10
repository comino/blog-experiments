import unittest

from evaluation import cell_equal, compare, extract_sql


def result(rows):
    width = len(rows[0]) if rows else 1
    return {"meta": [{"name": str(i)} for i in range(width)], "data": rows}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.contract = {"ordered": True, "float_rtol": 1e-6, "float_atol": 1e-6}

    def test_different_fourth_column_fails(self):
        self.assertFalse(compare(result([[1, 2, 3, 4]]), result([[1, 2, 3, 9]]), self.contract)[0])

    def test_wrong_order_fails_when_requested(self):
        self.assertFalse(compare(result([[1], [2]]), result([[2], [1]]), self.contract)[0])

    def test_unordered_rows_preserve_duplicates(self):
        self.contract["ordered"] = False
        self.assertTrue(compare(result([[1], [2], [2]]), result([[2], [1], [2]]), self.contract)[0])
        self.assertFalse(compare(result([[1], [2], [2]]), result([[1], [1], [2]]), self.contract)[0])

    def test_counts_are_exact_at_large_scale(self):
        self.assertFalse(cell_equal(1_000_000, 1_000_001))
        self.assertFalse(cell_equal(10_000, 10_001.0))
        self.assertTrue(cell_equal(10_000, 10_000.0))

    def test_float_roundoff_and_nested_values(self):
        self.assertTrue(cell_equal([1.25, {"tags": ["a", "b"]}], [1.25000001, {"tags": ["a", "b"]}]))
        self.assertFalse(cell_equal({"tags": ["a", "b"]}, {"tags": ["b", "a"]}))
        self.assertFalse(cell_equal(None, 0))
        self.assertFalse(cell_equal("42", 42))

    def test_extra_column_fails(self):
        self.assertFalse(compare(result([[1, 2]]), result([[1, 2, 3]]), self.contract)[0])

    def test_fenced_sql(self):
        self.assertEqual(extract_sql("```sql\nSELECT 1;\n```"), "SELECT 1")


if __name__ == "__main__":
    unittest.main()
