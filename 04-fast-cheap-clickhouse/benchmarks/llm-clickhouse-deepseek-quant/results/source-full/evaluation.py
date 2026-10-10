"""Compare every result cell with explicit ordering and exact integer semantics."""
import json
import math
import re


def extract_sql(text):
    text = (text or "").strip()
    fenced = re.search(r"```(?:sql|clickhouse)?\s*\n(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    return text.rstrip().rstrip(";").strip()


def cell_equal(expected, actual, rtol=1e-6, atol=1e-6):
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int):
            return expected == actual  # No relative tolerance for counts or integer money.
        if not math.isfinite(expected) or not math.isfinite(actual):
            return expected == actual
        return math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(cell_equal(a, b, rtol, atol) for a, b in zip(expected, actual))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(cell_equal(v, actual[k], rtol, atol) for k, v in expected.items())
    return type(expected) is type(actual) and expected == actual


def compare(reference, result, contract):
    expected, actual = reference["data"], result["data"]
    if len(reference["meta"]) != len(result["meta"]):
        return False, "column_count"
    if len(expected) != len(actual):
        return False, "row_count"
    if any(len(row) != len(reference["meta"]) for row in actual):
        return False, "ragged_row"
    if not contract["ordered"]:
        key = lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))
        expected, actual = sorted(expected, key=key), sorted(actual, key=key)
    for index, (left, right) in enumerate(zip(expected, actual)):
        if not cell_equal(left, right, contract["float_rtol"], contract["float_atol"]):
            return False, f"row_{index}_values_or_order"
    return True, "match"


def query_is_select(sql):
    # The database user supplies the permission boundary. This only rejects
    # plainly out-of-contract statements before sending them to the database.
    clean = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.S).lstrip()
    return bool(re.match(r"(?:SELECT|WITH)\b", clean, re.I))
