#!/usr/bin/env python3
"""
Experiment 03: Call 7 LLMs via OpenRouter for 50 ClickHouse questions × 3 runs.
Raw HTTP calls with `requests` — no frameworks.

Usage:
    python3 call_llms.py                          # All models, zero-shot
    python3 call_llms.py --model claude-opus-4.6
    python3 call_llms.py --prompt few-shot
    python3 call_llms.py --prompt all             # Both zero-shot + few-shot
    python3 call_llms.py --summary-only           # Regenerate summary.csv
"""

import json, time, os, sys, csv, argparse
from pathlib import Path
from datetime import datetime, timezone

import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESP_DIR = DATA_DIR / "responses"
SQL_DIR  = BASE_DIR / "sql"
PROGRESS = BASE_DIR / "progress.md"

RESP_DIR.mkdir(parents=True, exist_ok=True)

# ── OpenRouter Config ──────────────────────────────────────────────────────────
def _load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        cfg = json.load(open("/root/.openclaw/openclaw.json"))
        return cfg["models"]["providers"]["openrouter"]["apiKey"]
    except Exception:
        raise RuntimeError("Cannot find OpenRouter API key")

API_KEY = _load_api_key()
API_URL = "https://openrouter.ai/api/v1/chat/completions"
RUNS    = 3

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = {
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "claude-opus-4.6":   "anthropic/claude-opus-4.6",
    "gpt-5.2":           "openai/gpt-5.2",
    "minimax-m2.5":      "minimax/minimax-m2.5",
    "kimi-k2.5":         "moonshotai/kimi-k2.5",
    "gemini-3-flash":    "google/gemini-3-flash-preview",
    "deepseek-v3.2":     "deepseek/deepseek-v3.2",
}

# ── Load schema + questions ────────────────────────────────────────────────────
DDL = (SQL_DIR / "schema.sql").read_text()

with open(DATA_DIR / "questions.json") as f:
    QUESTIONS = json.load(f)

# ── Prompt pieces ──────────────────────────────────────────────────────────────
DATA_CONTEXT = """\
~1M events, ~100K users. Data spans 2024-01-01 to 2024-12-31.
Event types: pageview, click, purchase, signup, logout.
Devices: desktop, mobile, tablet. Countries: ISO 2-letter (US, DE, GB, FR, JP …).
Plans: free, pro, enterprise. Tags examples: promo, vip, premium, beta, internal."""

SYSTEM = "You are a ClickHouse SQL expert. Generate a single ClickHouse SQL query to answer the question."


def build_prompt(question_text):
    base = f"## Schema\n{DDL}\n\n## Data Context\n{DATA_CONTEXT}\n"
    return SYSTEM, f"{base}\n## Question\n{question_text}\n\nRespond with ONLY the SQL query, no explanation."


# ── SQL extraction ─────────────────────────────────────────────────────────────
def extract_sql(raw):
    raw = raw.strip()
    if "```" in raw:
        lines = raw.split("\n")
        sql_lines, inside = [], False
        for line in lines:
            if line.strip().startswith("```") and not inside:
                inside = True; continue
            elif line.strip().startswith("```") and inside:
                break
            elif inside:
                sql_lines.append(line)
        if sql_lines:
            raw = "\n".join(sql_lines).strip()
    return raw


# ── API call ───────────────────────────────────────────────────────────────────
def call_model(model_id, system_prompt, user_prompt, max_retries=5):
    request_body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "seed": 42,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openclaw.ai",
        "X-Title": "Experiment 03 - LLM ClickHouse Query Oneshot",
    }

    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = requests.post(API_URL, headers=headers, json=request_body, timeout=120)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)

            if resp.status_code in (429, 502, 503) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  ⏳ retry {attempt+1}/{max_retries} in {wait}s (HTTP {resp.status_code})")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                return _err(f"HTTP {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            usage = data.get("usage", {})
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return {
                "raw_response": content,
                "raw_api_response": data,
                "tokens_in":  usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "cost_usd":   usage.get("cost"),
                "latency_ms": latency_ms,
                "model_id_returned": data.get("model", ""),
                "error": None,
            }
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  ⏳ timeout, retry in {wait}s")
                time.sleep(wait)
                continue
            return _err("Timeout after 120s")
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  ⏳ {e}, retry in {wait}s")
                time.sleep(wait)
                continue
            return _err(str(e))

    return _err("Max retries exceeded")


def _err(msg):
    return {"raw_response": "", "raw_api_response": None, "tokens_in": 0,
            "tokens_out": 0, "cost_usd": None, "latency_ms": 0,
            "model_id_returned": "", "error": msg}


# ── Helpers ────────────────────────────────────────────────────────────────────
def log_progress(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS, "a") as f:
        f.write(f"\n- [{ts}] {msg}")
    print(msg)


def load_responses(path):
    return json.load(open(path)) if path.exists() else []


def save_responses(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def done_keys(responses):
    return {(r["question_id"], r["run"]) for r in responses if not r.get("error")}


# ── Main loop ──────────────────────────────────────────────────────────────────
def run(model_filter=None):
    total = len(MODELS) * len(QUESTIONS) * RUNS
    log_progress(f"Starting: {len(MODELS)} models × {len(QUESTIONS)} Qs × {RUNS} runs = {total}")

    for model_name, model_id in MODELS.items():
        if model_filter and model_name != model_filter:
            continue

        path = RESP_DIR / f"{model_name}.json"
        responses = load_responses(path)
        done = done_keys(responses)
        log_progress(f"{model_name}: {len(done)} already done")

        for q in QUESTIONS:
            for run_nr in range(1, RUNS + 1):
                if (q["id"], run_nr) in done:
                    continue

                system, user = build_prompt(q["natural_language"])
                print(f"  [{model_name}] q={q['id']} run={run_nr} …", end=" ", flush=True)

                result = call_model(model_id, system, user)
                sql = extract_sql(result["raw_response"]) if result["raw_response"] else ""

                entry = {
                    "model": model_name, "model_id": model_id,
                    "question_id": q["id"], "run": run_nr, "tier": q["tier"],
                    "extracted_sql": sql,
                    "raw_response": result["raw_response"],
                    "tokens_in": result["tokens_in"], "tokens_out": result["tokens_out"],
                    "cost_usd": result["cost_usd"], "latency_ms": result["latency_ms"],
                    "model_id_returned": result["model_id_returned"],
                    "error": result["error"],
                    "raw_api_response": result["raw_api_response"],
                }
                responses.append(entry)

                status = "OK" if not result["error"] else f"ERR: {result['error'][:80]}"
                cost = f"${result['cost_usd']:.4f}" if result["cost_usd"] else "n/a"
                print(f"{status} | {result['latency_ms']}ms | {result['tokens_out']}tok | {cost}")

                save_responses(path, responses)
                time.sleep(0.5)

        log_progress(f"{model_name} done: {len(responses)} responses")

    log_progress("All complete!")


# ── Summary CSV ────────────────────────────────────────────────────────────────
def generate_summary():
    rows = []
    for root, _, files in os.walk(RESP_DIR):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            with open(Path(root) / fname) as f:
                data = json.load(f)
            for r in data:
                rows.append({
                    "model": r.get("model", ""),
                    "question_id": r.get("question_id", ""),
                    "tier": r.get("tier", ""),
                    "run": r.get("run", ""),
                    "tokens_in": r.get("tokens_in", 0),
                    "tokens_out": r.get("tokens_out", 0),
                    "cost_usd": r.get("cost_usd", ""),
                    "latency_ms": r.get("latency_ms", 0),
                    "error": r.get("error", ""),
                    "sql_length": len(r.get("extracted_sql", "")),
                })
    if rows:
        with open(DATA_DIR / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Summary: {len(rows)} rows → summary.csv")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Run only this model")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    if args.summary_only:
        generate_summary()
        return

    run(args.model)
    generate_summary()


if __name__ == "__main__":
    main()
