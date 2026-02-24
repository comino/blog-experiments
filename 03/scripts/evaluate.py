#!/usr/bin/env python3
"""Exp03: Evaluate LLM-generated ClickHouse SQL queries against reference results."""

import json
import csv
import subprocess
import os
import sys
import re
from pathlib import Path
from collections import defaultdict

BASE = Path("/root/.openclaw/workspace/blog/experiments/results/03")
DATA = BASE / "data"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

# Model pricing ($ per 1M tokens input/output)
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
    """Run SQL on thesis-clickhouse. Returns (success, result_or_error)."""
    try:
        result = subprocess.run(
            ["ssh", "thesis-clickhouse",
             f"clickhouse-client -d exp03_llm --max_execution_time={timeout} -q {repr(sql)}"],
            capture_output=True, text=True, timeout=timeout + 10
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def run_query_batch(queries, timeout=30):
    """Run multiple queries via a single SSH session for efficiency."""
    # Build a script that runs each query separated by markers
    script_lines = []
    for i, sql in enumerate(queries):
        # Escape for shell
        safe_sql = sql.replace("'", "'\\''")
        script_lines.append(
            f"echo '===MARKER_{i}_START==='; "
            f"clickhouse-client -d exp03_llm --max_execution_time={timeout} "
            f"-q '{safe_sql}' 2>&1; echo \"===MARKER_{i}_RC=$?===\""
        )
    
    full_script = "; ".join(script_lines)
    try:
        result = subprocess.run(
            ["ssh", "thesis-clickhouse", "bash", "-c", repr(full_script)],
            capture_output=True, text=True, timeout=len(queries) * (timeout + 5)
        )
        output = result.stdout + result.stderr
        
        results = {}
        for i in range(len(queries)):
            start_marker = f"===MARKER_{i}_START==="
            rc_pattern = f"===MARKER_{i}_RC=(\\d+)==="
            
            start_idx = output.find(start_marker)
            rc_match = re.search(rc_pattern, output)
            
            if start_idx >= 0 and rc_match:
                content_start = start_idx + len(start_marker) + 1
                content_end = rc_match.start()
                content = output[content_start:content_end].strip()
                rc = int(rc_match.group(1))
                results[i] = (rc == 0, content)
            else:
                results[i] = (False, "PARSE_ERROR")
        
        return results
    except subprocess.TimeoutExpired:
        return {i: (False, "TIMEOUT") for i in range(len(queries))}
    except Exception as e:
        return {i: (False, str(e)) for i in range(len(queries))}


def normalize_result(text):
    """Normalize query result for comparison."""
    if not text:
        return ""
    # Sort lines for unordered results, normalize whitespace
    lines = ["\t".join(cell.strip() for cell in line.split("\t")) for line in text.strip().split("\n") if line.strip()]
    return "\n".join(lines)


def results_match(ref_result, llm_result, result_type):
    """Compare two query results."""
    ref_norm = normalize_result(ref_result)
    llm_norm = normalize_result(llm_result)
    
    if ref_norm == llm_norm:
        return True
    
    # For scalar results, try numeric comparison
    if result_type == "scalar":
        try:
            ref_vals = [float(x) for x in ref_norm.replace("\t", " ").split()]
            llm_vals = [float(x) for x in llm_norm.replace("\t", " ").split()]
            if len(ref_vals) == len(llm_vals):
                return all(abs(r - l) < max(0.01, abs(r) * 0.001) for r, l in zip(ref_vals, llm_vals))
        except (ValueError, ZeroDivisionError):
            pass
    
    # For table results, try comparing sorted rows (for unordered results)
    ref_lines = sorted(ref_norm.split("\n"))
    llm_lines = sorted(llm_norm.split("\n"))
    if ref_lines == llm_lines:
        return True
    
    # Check if row counts and first few values match (for LIMIT queries with SELECT *)
    ref_rows = ref_norm.split("\n")
    llm_rows = llm_norm.split("\n")
    if len(ref_rows) == len(llm_rows) and len(ref_rows) > 0:
        # Compare key columns (first 2-3) for table results
        ref_keys = ["\t".join(r.split("\t")[:3]) for r in ref_rows]
        llm_keys = ["\t".join(r.split("\t")[:3]) for r in llm_rows]
        if ref_keys == llm_keys:
            return True
    
    return False


def classify_error(sql, error_msg, reference_sql):
    """Classify the type of error."""
    sql_lower = (sql or "").lower()
    error_lower = (error_msg or "").lower()
    
    # PostgreSQL-itis: using PG-specific syntax
    pg_patterns = [
        ("COUNT(*)", "count_star"),  # count(*) works in CH but count() is idiomatic
        ("ILIKE", "pg_ilike"),
        ("::int", "pg_cast"),
        ("::text", "pg_cast"),
        ("::timestamp", "pg_cast"),
        ("SERIAL", "pg_serial"),
        ("NOW()", "pg_now"),
        ("EXTRACT(", "pg_extract"),
        ("DATE_TRUNC", "pg_date_trunc"),
        ("STRING_AGG", "pg_string_agg"),
        ("GENERATE_SERIES", "pg_generate_series"),
        ("PERCENTILE_CONT", "pg_percentile"),
        ("MEDIAN(", "pg_median"),
        ("BOOL", "pg_bool"),
    ]
    
    for pattern, category in pg_patterns:
        if pattern.lower() in sql_lower and "Unknown function" in error_msg:
            return "PostgreSQL-itis"
    
    if "DATE_TRUNC" in (sql or "").upper() or "EXTRACT(" in (sql or "").upper():
        return "PostgreSQL-itis"
    
    if "PERCENTILE_CONT" in (sql or "").upper() or "MEDIAN(" in (sql or "").upper():
        return "PostgreSQL-itis"
    
    # Wrong function
    if "unknown function" in error_lower or "no such function" in error_lower:
        return "Wrong Function"
    
    # Syntax error
    if "syntax error" in error_lower or "expected" in error_lower:
        return "Syntax Error"
    
    # Hallucination - referencing non-existent columns/tables
    if "missing columns" in error_lower or "unknown identifier" in error_lower:
        return "Hallucination"
    if "doesn't exist" in error_lower and ("column" in error_lower or "table" in error_lower):
        return "Hallucination"
    
    # Logic error (runs but wrong result)
    if error_msg == "WRONG_RESULT":
        return "Logic Error"
    
    if "TIMEOUT" in (error_msg or ""):
        return "Timeout"
    
    return "Other Error"


def main():
    # Load questions
    with open(DATA / "questions.json") as f:
        questions = {q["id"]: q for q in json.load(f)}
    
    print(f"Loaded {len(questions)} questions")
    
    # Step 1: Run all reference queries
    print("\n=== Running reference queries ===")
    ref_results = {}
    ref_sqls = [(qid, q["reference_sql"]) for qid, q in sorted(questions.items())]
    
    for qid, sql in ref_sqls:
        success, result = run_query(sql)
        if success:
            ref_results[qid] = result
            print(f"  ✓ {qid}: {result[:60]}...")
        else:
            print(f"  ✗ {qid} FAILED: {result[:100]}")
            ref_results[qid] = None
    
    print(f"\nReference queries: {sum(1 for v in ref_results.values() if v is not None)}/{len(ref_results)} succeeded")
    
    # Step 2: Load all LLM responses
    print("\n=== Loading LLM responses ===")
    responses_dir = DATA / "responses"
    all_responses = []
    
    for resp_file in sorted(responses_dir.glob("*.json")):
        with open(resp_file) as f:
            responses = json.load(f)
        model_name = resp_file.stem
        print(f"  {model_name}: {len(responses)} responses")
        for r in responses:
            r["_model_file"] = model_name
        all_responses.extend(responses)
    
    print(f"Total responses: {len(all_responses)}")
    
    # Step 3: Score all responses
    print("\n=== Scoring LLM responses ===")
    scores = []
    
    for i, resp in enumerate(all_responses):
        qid = resp["question_id"]
        model = resp["model"]
        run = resp["run"]
        sql = resp.get("extracted_sql") or resp.get("sql") or ""
        tier = resp.get("tier", int(qid.split("_")[0][1:]))
        
        if not sql or not sql.strip():
            score = 0
            error_type = "No SQL"
            error_msg = "Empty response"
        else:
            # Run the LLM's SQL
            success, result = run_query(sql)
            
            if not success:
                # Check if it's a syntax error vs runtime error
                if "syntax error" in result.lower() or "expected" in result.lower():
                    score = 0
                    error_type = classify_error(sql, result, questions[qid]["reference_sql"])
                    error_msg = result[:200]
                else:
                    score = 1
                    error_type = classify_error(sql, result, questions[qid]["reference_sql"])
                    error_msg = result[:200]
            else:
                # Query ran - check result
                ref = ref_results.get(qid)
                if ref is not None and results_match(ref, result, questions[qid]["expected_result_type"]):
                    score = 3
                    error_type = None
                    error_msg = None
                else:
                    score = 2
                    error_type = classify_error(sql, "WRONG_RESULT", questions[qid]["reference_sql"])
                    error_msg = f"Expected: {(ref or '')[:80]}... Got: {result[:80]}..."
        
        # Calculate cost
        tokens_in = resp.get("tokens_in", 0)
        tokens_out = resp.get("tokens_out", 0)
        pricing = MODEL_PRICING.get(model, {"in": 1.0, "out": 4.0})
        cost = (tokens_in * pricing["in"] + tokens_out * pricing["out"]) / 1_000_000
        
        scores.append({
            "model": model,
            "question_id": qid,
            "tier": tier,
            "run": run,
            "score": score,
            "error_type": error_type,
            "error_msg": error_msg,
            "llm_sql": sql[:500] if sql else "",
            "reference_sql": questions[qid]["reference_sql"],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost,
            "latency_ms": resp.get("latency_ms", 0),
        })
        
        status = ["✗ SYNTAX", "⚠ RUNTIME", "~ WRONG", "✓ CORRECT"][score]
        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(all_responses)}...")
    
    print(f"\nScoring complete: {len(scores)} total")
    
    # Save scores
    with open(DATA / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scores[0].keys())
        writer.writeheader()
        writer.writerows(scores)
    print(f"Saved scores to {DATA / 'scores.csv'}")
    
    # Step 4: Generate accuracy summary
    print("\n=== Generating accuracy summary ===")
    
    # Group by model
    model_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0, "scores": [], "costs": []}))
    
    for s in scores:
        model = s["model"]
        tier = s["tier"]
        model_stats[model][tier]["total"] += 1
        model_stats[model][tier]["scores"].append(s["score"])
        model_stats[model][tier]["costs"].append(s["cost_usd"])
        if s["score"] == 3:
            model_stats[model][tier]["correct"] += 1
    
    # Build summary
    summary_rows = []
    for model in sorted(model_stats.keys()):
        for tier in sorted(model_stats[model].keys()):
            stats = model_stats[model][tier]
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            total_cost = sum(stats["costs"])
            cost_per_correct = total_cost / stats["correct"] if stats["correct"] > 0 else float("inf")
            summary_rows.append({
                "model": model,
                "tier": tier,
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(accuracy, 4),
                "avg_score": round(avg_score, 2),
                "total_cost_usd": round(total_cost, 6),
                "cost_per_correct_usd": round(cost_per_correct, 6) if cost_per_correct != float("inf") else "inf",
            })
    
    with open(DATA / "accuracy_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    
    # Print summary table
    print(f"\n{'Model':<22} {'T1':>5} {'T2':>5} {'T3':>5} {'T4':>5} {'T5':>5} {'Total':>6} {'Acc':>6}")
    print("-" * 70)
    for model in sorted(model_stats.keys()):
        tier_accs = []
        total_correct = 0
        total_total = 0
        for tier in range(1, 6):
            s = model_stats[model][tier]
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0
            tier_accs.append(f"{acc:.0%}")
            total_correct += s["correct"]
            total_total += s["total"]
        overall = total_correct / total_total if total_total > 0 else 0
        print(f"{model:<22} {tier_accs[0]:>5} {tier_accs[1]:>5} {tier_accs[2]:>5} {tier_accs[3]:>5} {tier_accs[4]:>5} {total_correct:>3}/{total_total:<3} {overall:.0%}")
    
    # Error taxonomy
    print("\n=== Error Taxonomy ===")
    error_counts = defaultdict(int)
    for s in scores:
        if s["error_type"]:
            error_counts[s["error_type"]] += 1
    for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {err}: {cnt}")
    
    # Save data for charts
    with open(DATA / "eval_data.json", "w") as f:
        json.dump({
            "scores": scores,
            "summary": summary_rows,
            "error_counts": dict(error_counts),
            "model_stats": {m: {str(t): {"correct": d["correct"], "total": d["total"]} 
                               for t, d in tiers.items()} 
                          for m, tiers in model_stats.items()},
        }, f, indent=2, default=str)
    
    print(f"\nAll data saved. Run generate_charts.py next.")


if __name__ == "__main__":
    main()
