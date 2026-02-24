#!/usr/bin/env python3
"""Exp03: Evaluate LLM ClickHouse SQL queries. Runs ON thesis-clickhouse directly."""

import json, csv, subprocess, os, sys, re
from pathlib import Path
from collections import defaultdict

# Paths - will be set up via scp
WORK = Path("/tmp/exp03")
DATA = WORK / "data"

MODEL_PRICING = {
    "claude-sonnet-4.5": {"in": 3.0, "out": 15.0},
    "claude-opus-4.6": {"in": 15.0, "out": 75.0},
    "claude-opus-4": {"in": 15.0, "out": 75.0},
    "gpt-5.2": {"in": 2.0, "out": 8.0},
    "deepseek-v3.2": {"in": 0.27, "out": 1.10},
    "gemini-3-flash": {"in": 0.15, "out": 0.60},
    "kimi-k2.5": {"in": 0.60, "out": 2.40},
    "minimax-m2.5": {"in": 0.50, "out": 2.00},
}

def run_query(sql, timeout=30):
    """Run SQL via local clickhouse-client."""
    try:
        result = subprocess.run(
            ["clickhouse-client", "-d", "exp03_llm",
             f"--max_execution_time={timeout}", "-q", sql],
            capture_output=True, text=True, timeout=timeout + 5
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def normalize_result(text):
    if not text:
        return ""
    lines = ["\t".join(cell.strip() for cell in line.split("\t"))
             for line in text.strip().split("\n") if line.strip()]
    return "\n".join(lines)

def results_match(ref_result, llm_result, result_type):
    ref_norm = normalize_result(ref_result)
    llm_norm = normalize_result(llm_result)
    
    if ref_norm == llm_norm:
        return True
    
    # Numeric comparison for scalars
    if result_type == "scalar":
        try:
            ref_vals = [float(x) for x in ref_norm.replace("\t", " ").split()]
            llm_vals = [float(x) for x in llm_norm.replace("\t", " ").split()]
            if len(ref_vals) == len(llm_vals):
                return all(abs(r - l) < max(0.01, abs(r) * 0.001) for r, l in zip(ref_vals, llm_vals))
        except (ValueError, ZeroDivisionError):
            pass
    
    # Sorted comparison (unordered results)
    if sorted(ref_norm.split("\n")) == sorted(llm_norm.split("\n")):
        return True
    
    # Key columns match for table results
    ref_rows = ref_norm.split("\n")
    llm_rows = llm_norm.split("\n")
    if len(ref_rows) == len(llm_rows) and len(ref_rows) > 0:
        ref_keys = ["\t".join(r.split("\t")[:3]) for r in ref_rows]
        llm_keys = ["\t".join(r.split("\t")[:3]) for r in llm_rows]
        if ref_keys == llm_keys:
            return True
    
    return False

def classify_error(sql, error_msg, reference_sql):
    sql_upper = (sql or "").upper()
    error_lower = (error_msg or "").lower()
    
    pg_funcs = ["DATE_TRUNC", "PERCENTILE_CONT", "MEDIAN(", "STRING_AGG", 
                "GENERATE_SERIES", "ARRAY_AGG", "UNNEST("]
    for pf in pg_funcs:
        if pf in sql_upper:
            return "PostgreSQL-itis"
    
    if "::int" in (sql or "") or "::text" in (sql or "") or "::timestamp" in (sql or ""):
        return "PostgreSQL-itis"
    
    if "unknown function" in error_lower:
        return "Wrong Function"
    
    if "syntax error" in error_lower or "expected" in error_lower:
        return "Syntax Error"
    
    if "missing columns" in error_lower or "unknown identifier" in error_lower:
        return "Hallucination"
    if "doesn't exist" in error_lower:
        return "Hallucination"
    
    if error_msg == "WRONG_RESULT":
        return "Logic Error"
    
    if "TIMEOUT" in (error_msg or ""):
        return "Timeout"
    
    return "Other Error"

def main():
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    
    with open(DATA / "questions.json") as f:
        questions = {q["id"]: q for q in json.load(f)}
    print(f"Loaded {len(questions)} questions", flush=True)
    
    # Run reference queries
    print("\n=== Running reference queries ===", flush=True)
    ref_results = {}
    for qid, q in sorted(questions.items()):
        success, result = run_query(q["reference_sql"])
        ref_results[qid] = result if success else None
        sym = "✓" if success else "✗"
        print(f"  {sym} {qid}", flush=True)
    
    ok = sum(1 for v in ref_results.values() if v is not None)
    print(f"\nReference: {ok}/{len(ref_results)} succeeded", flush=True)
    
    # Load responses
    print("\n=== Loading LLM responses ===", flush=True)
    responses_dir = DATA / "responses"
    all_responses = []
    for resp_file in sorted(responses_dir.glob("*.json")):
        with open(resp_file) as f:
            responses = json.load(f)
        print(f"  {resp_file.stem}: {len(responses)}", flush=True)
        all_responses.extend(responses)
    print(f"Total: {len(all_responses)}", flush=True)
    
    # Score
    print("\n=== Scoring ===", flush=True)
    scores = []
    
    for i, resp in enumerate(all_responses):
        qid = resp["question_id"]
        model = resp["model"]
        run_num = resp["run"]
        sql = resp.get("extracted_sql") or resp.get("sql") or ""
        tier = resp.get("tier", int(qid.split("_")[0][1:]))
        
        if not sql or not sql.strip():
            score, error_type, error_msg = 0, "No SQL", "Empty"
        else:
            success, result = run_query(sql)
            if not success:
                if any(x in result.lower() for x in ["syntax error", "expected"]):
                    score = 0
                else:
                    score = 1
                error_type = classify_error(sql, result, questions[qid]["reference_sql"])
                error_msg = result[:200]
            else:
                ref = ref_results.get(qid)
                if ref is not None and results_match(ref, result, questions[qid]["expected_result_type"]):
                    score, error_type, error_msg = 3, None, None
                else:
                    score = 2
                    error_type = classify_error(sql, "WRONG_RESULT", questions[qid]["reference_sql"])
                    error_msg = f"Got: {result[:100]}"
        
        tokens_in = resp.get("tokens_in", 0)
        tokens_out = resp.get("tokens_out", 0)
        pricing = MODEL_PRICING.get(model, {"in": 1.0, "out": 4.0})
        cost = (tokens_in * pricing["in"] + tokens_out * pricing["out"]) / 1_000_000
        
        scores.append({
            "model": model, "question_id": qid, "tier": tier, "run": run_num,
            "score": score, "error_type": error_type, "error_msg": error_msg,
            "llm_sql": sql[:500] if sql else "", "reference_sql": questions[qid]["reference_sql"],
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "cost_usd": cost, "latency_ms": resp.get("latency_ms", 0),
        })
        
        if (i + 1) % 100 == 0:
            correct_so_far = sum(1 for s in scores if s["score"] == 3)
            print(f"  {i+1}/{len(all_responses)} scored ({correct_so_far} correct)", flush=True)
    
    # Save scores CSV
    with open(DATA / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scores[0].keys())
        writer.writeheader()
        writer.writerows(scores)
    
    # Summary
    model_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0, "scores": [], "costs": []}))
    for s in scores:
        ms = model_stats[s["model"]][s["tier"]]
        ms["total"] += 1
        ms["scores"].append(s["score"])
        ms["costs"].append(s["cost_usd"])
        if s["score"] == 3:
            ms["correct"] += 1
    
    summary_rows = []
    for model in sorted(model_stats.keys()):
        for tier in sorted(model_stats[model].keys()):
            st = model_stats[model][tier]
            acc = st["correct"] / st["total"] if st["total"] > 0 else 0
            total_cost = sum(st["costs"])
            cpc = total_cost / st["correct"] if st["correct"] > 0 else float("inf")
            summary_rows.append({
                "model": model, "tier": tier, "total": st["total"],
                "correct": st["correct"], "accuracy": round(acc, 4),
                "avg_score": round(sum(st["scores"]) / len(st["scores"]), 2),
                "total_cost_usd": round(total_cost, 6),
                "cost_per_correct_usd": round(cpc, 6) if cpc != float("inf") else "inf",
            })
    
    with open(DATA / "accuracy_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    
    # Print summary
    print(f"\n{'Model':<22} {'T1':>5} {'T2':>5} {'T3':>5} {'T4':>5} {'T5':>5} {'Tot':>7} {'Acc':>6}", flush=True)
    print("-" * 70, flush=True)
    for model in sorted(model_stats.keys()):
        parts = []
        tc = tt = 0
        for tier in range(1, 6):
            s = model_stats[model][tier]
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            parts.append(f"{acc:.0%}")
            tc += s["correct"]; tt += s["total"]
        ov = tc / tt if tt > 0 else 0
        print(f"{model:<22} {parts[0]:>5} {parts[1]:>5} {parts[2]:>5} {parts[3]:>5} {parts[4]:>5} {tc:>3}/{tt:<3} {ov:.0%}", flush=True)
    
    # Error taxonomy
    print("\n=== Error Taxonomy ===", flush=True)
    error_counts = defaultdict(int)
    error_examples = defaultdict(list)
    for s in scores:
        if s["error_type"]:
            error_counts[s["error_type"]] += 1
            if len(error_examples[s["error_type"]]) < 5:
                error_examples[s["error_type"]].append({
                    "model": s["model"], "qid": s["question_id"],
                    "sql": s["llm_sql"][:200], "error": (s["error_msg"] or "")[:150]
                })
    for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {err}: {cnt}", flush=True)
    
    # Save full data
    with open(DATA / "eval_data.json", "w") as f:
        json.dump({
            "scores": scores, "summary": summary_rows,
            "error_counts": dict(error_counts),
            "error_examples": {k: v for k, v in error_examples.items()},
            "model_stats": {m: {str(t): {"correct": d["correct"], "total": d["total"]}
                               for t, d in tiers.items()}
                          for m, tiers in model_stats.items()},
        }, f, indent=2, default=str)
    
    print(f"\n✓ All done. Results in {DATA}", flush=True)

if __name__ == "__main__":
    main()
