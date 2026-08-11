#!/usr/bin/env python3
"""Exp03: Output-based classification of score-2 results (format/column/logic).

Encodes the February manual rubric as executable rules:
  - row-order / LIMIT / value-formatting differences -> format_mismatch
  - same logic, extra or missing output columns      -> column_mismatch
  - anything else                                    -> logic_error

Runs ON the ClickHouse host. Reads full SQL from the responses JSONs,
runs LLM and reference queries, compares outputs structurally.

Usage: classify_score2.py <scores_csv> <out_csv>
"""

import csv, json, os, re, subprocess, sys
from pathlib import Path

WORK = Path("/tmp/exp03")
DATA = WORK / "data"

def run_query(sql, timeout=30):
    try:
        result = subprocess.run(
            ["clickhouse-client", "-d", "exp03_llm",
             "-u", os.environ.get("CH_USER", "default"),
             "--password", os.environ.get("CH_PASSWORD", ""),
             f"--max_execution_time={timeout}", "-q", sql],
            capture_output=True, text=True, timeout=timeout + 5)
        return (True, result.stdout) if result.returncode == 0 else (False, result.stderr)
    except Exception as e:
        return False, str(e)

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})([ T]00:00:00(\.0+)?)?$")

def norm_cell(c):
    c = c.strip()
    m = DATE_RE.match(c)
    if m:
        return m.group(1)
    try:
        f = float(c)
        return round(f, 4)
    except ValueError:
        return c.lower()

def parse(out):
    rows = [line.split("\t") for line in out.strip().split("\n") if line.strip()]
    return [[norm_cell(c) for c in r] for r in rows]

def columns(rows):
    """Column-wise multisets (only meaningful when rows are rectangular)."""
    if not rows:
        return []
    width = min(len(r) for r in rows)
    return [sorted(map(str, (r[i] for r in rows))) for i in range(width)]

def flat_values(rows):
    return sorted(str(c) for r in rows for c in r)

def col_overlap(a, b):
    """Multiset overlap of column a within column b, as fraction of a."""
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    inter = sum(min(ca[k], cb[k]) for k in ca)
    return inter / max(1, sum(ca.values()))

def classify(ref_rows, llm_rows):
    if not llm_rows:
        return "logic_error", "empty result"
    sr, sl = sorted(map(tuple, ref_rows)), sorted(map(tuple, llm_rows))
    if sr == sl:
        return "format_mismatch", "identical after normalization/sorting"

    # equal-width row-subset: LIMIT or ordering difference
    ref_w = min(len(r) for r in ref_rows)
    llm_w = min(len(r) for r in llm_rows)
    set_r, set_l = set(map(tuple, sr)), set(map(tuple, sl))
    if ref_w == llm_w and (set_r <= set_l or set_l <= set_r):
        return "format_mismatch", "row subset (LIMIT/ordering difference)"

    ref_cols, llm_cols = columns(ref_rows), columns(llm_rows)

    # column projection: match each reference column to its best LLM column
    # by multiset overlap (tolerates LIMIT differences and ORDER BY ties)
    if ref_cols and llm_cols:
        small, large = (ref_cols, llm_cols) if len(ref_cols) <= len(llm_cols) else (llm_cols, ref_cols)
        matched = sum(1 for c in small if max(col_overlap(c, d) for d in large) >= 0.8)
        if matched == len(small):
            if len(ref_cols) != len(llm_cols):
                return "column_mismatch", "extra/missing columns, shared columns overlap >=80%"
            return "format_mismatch", "same columns after overlap matching, formatting/order differs"
        if matched >= max(1, len(small) - 1) and len(ref_cols) != len(llm_cols):
            return "column_mismatch", f"{matched}/{len(small)} columns overlap >=80%, column count differs"

    # scalar tolerance: approx vs exact aggregates
    if len(sr) == 1 == len(sl) and len(sr[0]) == len(sl[0]):
        try:
            pairs = [(float(a), float(b)) for a, b in zip(sr[0], sl[0])]
            if all(abs(a - b) <= max(0.05 * abs(a), 0.5) for a, b in pairs):
                return "format_mismatch", "scalar within 5% (approx vs exact aggregate)"
        except (ValueError, TypeError):
            pass

    # shape transposition: same flattened values (GROUP BY rows vs countIf columns)
    if flat_values(ref_rows) == flat_values(llm_rows):
        return "format_mismatch", "same values, transposed shape"

    return "logic_error", "results differ structurally"

def main():
    scores_csv, out_csv = sys.argv[1], sys.argv[2]

    with open(DATA / "questions.json") as f:
        questions = {q["id"]: q for q in json.load(f)}

    full_sql = {}
    for rf in (DATA / "responses").glob("*.json"):
        for r in json.load(open(rf)):
            full_sql[(r["model"], r["question_id"], r["run"])] = r.get("extracted_sql", "")

    ref_cache = {}
    def ref_result(qid):
        if qid not in ref_cache:
            ok, out = run_query(questions[qid]["reference_sql"])
            ref_cache[qid] = parse(out) if ok else None
        return ref_cache[qid]

    rows_out = []
    todo = [r for r in csv.DictReader(open(scores_csv)) if r["score"] == "2"]
    print(f"{len(todo)} score-2 rows", flush=True)
    for i, r in enumerate(todo):
        key = (r["model"], r["question_id"], int(r["run"]))
        sql = full_sql.get(key) or r["llm_sql"]
        ok, out = run_query(sql)
        if not ok:
            label, why = "logic_error", "no longer executes"
        else:
            ref = ref_result(r["question_id"])
            if ref is None:
                label, why = "unknown", "reference failed"
            else:
                label, why = classify(ref, parse(out))
        rows_out.append({"model": r["model"], "question_id": r["question_id"],
                         "run": r["run"], "failure_type": label, "rule": why})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out_csv}", flush=True)

if __name__ == "__main__":
    main()
