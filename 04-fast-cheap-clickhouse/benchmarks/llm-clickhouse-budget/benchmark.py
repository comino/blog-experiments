#!/usr/bin/env python3
"""Sequential OpenRouter benchmark with durable attempts, billed cost, and SQL execution."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import time
import uuid

import requests

from evaluation import compare, extract_sql, query_is_select

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
API = "https://openrouter.ai/api/v1"
DB_URL = "http://127.0.0.1:18123/"
CAP_USD = 9.50
MAX_TOKENS = 4096
SYSTEM = "You are a ClickHouse SQL expert. Generate one SELECT query answering the question. Follow the output contract. Return only SQL, with no explanation and no FORMAT clause."
CONTEXT = "ClickHouse 26.8.19.9, UTC. The current database contains 1,000,000 events and 10,000 users. Events span 2024-01-01 through 2024-12-31. Some users have no events. Event types: pageview, click, purchase, signup, logout. Devices: desktop, mobile, tablet. Countries: US, DE, GB, FR, JP, BR, IN, CA, AU, NL. Plans: free, pro, enterprise. Tags include promo, vip_member, premium_user, mobile, new. Column aliases are not graded; column order and values are graded."


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Deadline(Exception):
    pass


def expired(signum, frame):
    raise Deadline("Model request exceeded the 60 second wall-clock limit")


class Database:
    def __init__(self):
        self.session = requests.Session()
        self.session.trust_env = False

    def execute(self, sql, query_id):
        start = time.perf_counter()
        if not query_is_select(sql):
            return {"ok": False, "error": "Not a SELECT/WITH query", "elapsed_ms": 0, "query_id": query_id}
        try:
            response = self.session.post(DB_URL, params={"database": "exp04_budget", "query_id": query_id}, headers={"X-ClickHouse-User": "budget_reader"}, data=(sql + "\nFORMAT JSONCompact").encode(), timeout=(3, 15))
            elapsed = (time.perf_counter() - start) * 1000
            if response.status_code != 200:
                return {"ok": False, "error": response.text[:8000], "http_status": response.status_code, "elapsed_ms": elapsed, "query_id": query_id}
            result = response.json()
            if "exception" in result:
                return {"ok": False, "error": result["exception"], "elapsed_ms": elapsed, "query_id": query_id}
            return {"ok": True, "result": result, "elapsed_ms": (time.perf_counter() - start) * 1000, "query_id": query_id, "summary": response.headers.get("X-ClickHouse-Summary")}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "elapsed_ms": (time.perf_counter() - start) * 1000, "query_id": query_id}


def configure():
    metadata = {m["id"]: m for m in load(BASE / "run-model-metadata.json")}
    endpoints = {m["model_id"]: m["response"]["data"]["endpoints"] for m in load(BASE / "openrouter-endpoints-2026-10-07.json")}
    choices = {
        "mistralai/mistral-large-4-0": ["mistral"],
        "openai/gpt-6-luna": ["openai", "openai/fast", "azure/eu"],
        "upstage/solar-mini4": ["upstage/zdr"],
        "deepseek/deepseek-v4.1-flash": ["coreweave/fp8", "inference-net/fp8", "baseten/fp8", "parasail/fp8"],
        "inception/mercury-2.5": ["inception"],
        "qwen/qwen3.8-flash": ["alibaba"],
        "ibm-granite/granite-4.2-8b": ["coreweave/bf16", "deepinfra/bf16"],
    }
    candidates = []
    for model, tags in choices.items():
        info = metadata[model]
        for tag in tags:
            matches = [e for e in endpoints[model] if e["tag"] == tag]
            assert matches, (model, tag)
            endpoint = matches[0]
            supported = set(endpoint["supported_parameters"])
            params = {"max_tokens": MAX_TOKENS}
            if "temperature" in supported:
                params["temperature"] = 0
            if "seed" in supported:
                params["seed"] = 42
            reasoning = info.get("reasoning", {})
            if reasoning:
                assert not reasoning.get("mandatory"), model
                params["reasoning"] = {"effort": "none"} if "none" in reasoning.get("supported_efforts", []) else {"enabled": False}
            price = endpoint["pricing"]
            candidates.append({
                "route_id": model + "@" + tag, "model_id": model, "name": info["name"],
                "provider_tag": tag, "provider_name": endpoint["provider_name"],
                "endpoint_records_matching_tag": len(matches), "quantization": endpoint["quantization"],
                "unknown_precision_accepted_by_user": endpoint["quantization"] == "unknown",
                "service_tier": "priority" if tag.endswith("/fast") else "default",
                "parameters": params, "pricing_per_token": price,
            })
    save(BASE / "candidate-routes.json", candidates)
    shortlist = load(BASE / "model-shortlist.json")
    policy = shortlist["provider_selection"]
    policy["status"] = "user_authorized_execution_provider_screening_pending"
    policy["quantization_interpretation"] = "Highest disclosed bit-width / least compression; unknown-precision exceptions accepted by the user on 2026-10-07."
    policy["unknown_precision_policy"] = "User accepted including undisclosed precision, with that limitation recorded."
    for model in shortlist["models"]:
        if model["provider_review"]["highest_disclosed_quantization"] is None:
            model["provider_review"]["status"] = "undisclosed_precision_exception_accepted"
    save(BASE / "model-shortlist.json", shortlist)
    print(f"Configured {len(candidates)} candidate routes for {len(choices)} models; no inference yet.")


def validate():
    database = Database()
    references = {}
    checks = []
    for question in load(BASE / "questions.json"):
        qid = question["id"]
        first = database.execute(question["reference_sql"], "budget_ref_" + qid + "_a")
        second = database.execute(question["reference_sql"], "budget_ref_" + qid + "_b")
        if not first["ok"] or not second["ok"]:
            print(qid, first.get("error"), second.get("error"), flush=True)
            raise RuntimeError("Reference query failed; no model calls permitted")
        same, reason = compare(first["result"], second["result"], question["contract"])
        if not same:
            raise RuntimeError(f"Unstable reference {qid}: {reason}")
        references[qid] = first["result"]
        checks.append({"question_id": qid, "rows": len(first["result"]["data"]), "stable_twice": same})
    assert references["t1_01"]["data"] == [[1_000_000]]
    assert len(references["t4_03"]["data"]) == 100
    assert references["t6_03"]["data"][0][0] > 0
    save(BASE / "references.json", references)
    save(BASE / "reference-checks.json", {"checked_at": now(), "questions_sha256": sha(BASE / "questions.json"), "checks": checks})
    print("All 60 reference queries executed twice with stable results; fixture assertions passed.", flush=True)


def prompt(question):
    return f"## Schema\n{(BASE / 'schema.sql').read_text()}\n## Data context\n{CONTEXT}\n## Question\n{question['natural_language']}"


class Runner:
    def __init__(self):
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.session = requests.Session()
        self.session.headers.update({"Authorization": "Bearer " + self.key, "Content-Type": "application/json", "X-OpenRouter-Title": "ClickHouse fast and cheap benchmark"})
        self.database = Database()
        self.references = load(BASE / "references.json")
        self.accounted = sum(r["reserved_or_billed_cost_usd"] for path in RESULTS.glob("*.jsonl") if path.name != "started.jsonl" for r in read_rows(path))

    def generation_cost(self, generation_id):
        if not generation_id:
            return None
        try:
            response = self.session.get(API + "/generation", params={"id": generation_id}, timeout=10)
            if response.status_code == 200:
                return response.json().get("data", {}).get("total_cost")
        except requests.RequestException:
            pass
        return None

    def one(self, route, question, repetition, phase, task_id):
        user_prompt = prompt(question)
        price = route["pricing_per_token"]
        reserved = (len(user_prompt.encode()) + len(SYSTEM.encode()) + 256) * float(price["prompt"]) + MAX_TOKENS * float(price["completion"])
        if self.accounted + reserved > CAP_USD:
            raise RuntimeError(f"Spending cap would be exceeded: accounted ${self.accounted:.4f}, request reserve ${reserved:.4f}")
        request = {
            "model": route["model_id"], "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}],
            **route["parameters"], "stream": True, "stream_options": {"include_usage": True},
            "service_tier": route["service_tier"],
            "provider": {"only": [route["provider_tag"]], "quantizations": [route["quantization"]], "allow_fallbacks": False, "require_parameters": True},
        }
        record = {
            "task_id": task_id, "phase": phase, "route_id": route["route_id"], "model_id": route["model_id"],
            "provider_tag_requested": route["provider_tag"], "quantization_reported": route["quantization"],
            "question_id": question["id"], "tier": question["tier"], "repetition": repetition,
            "started_at": now(), "request": request, "api_events": [], "error": None, "correct": False,
            "sql_ready_ms": None, "result_ready_ms": None, "first_content_ms": None, "first_event_ms": None,
        }
        append(RESULTS / "started.jsonl", {"task_id": task_id, "started_at": record["started_at"], "reserved_cost_usd": reserved})
        text, usage, first_content, last_content = "", {}, None, None
        start = time.perf_counter()
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 60)
        try:
            with self.session.post(API + "/chat/completions", json=request, stream=True, timeout=(10, 60)) as response:
                record["http_status"] = response.status_code
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:8000]}")
                for line in response.iter_lines(chunk_size=1):
                    if not line or not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    event = json.loads(payload)
                    elapsed = (time.perf_counter() - start) * 1000
                    if record["first_event_ms"] is None:
                        record["first_event_ms"] = elapsed
                    record["api_events"].append(event)
                    if event.get("error"):
                        raise RuntimeError(json.dumps(event["error"]))
                    for key in ("id", "model", "provider", "service_tier"):
                        if key in event:
                            record[key + "_returned"] = event[key]
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            text += content
                            if first_content is None:
                                first_content = elapsed
                            last_content = elapsed
                        if choice.get("finish_reason"):
                            record["finish_reason"] = choice["finish_reason"]
            record["api_complete_ms"] = (time.perf_counter() - start) * 1000
            record["raw_response"] = text
            if not record.get("finish_reason"):
                raise RuntimeError("Stream ended without a completion finish reason")
            sql = extract_sql(text)
            record["sql"] = sql
            if not sql:
                raise RuntimeError("Empty visible SQL response")
            if record.get("finish_reason") == "length":
                raise RuntimeError("Completion token limit reached")
            record["sql_ready_ms"] = (time.perf_counter() - start) * 1000
        except (Exception, KeyboardInterrupt) as exc:
            record["error"] = str(exc).replace(self.key, "<REDACTED>")
            record["api_complete_ms"] = (time.perf_counter() - start) * 1000
            record["raw_response"] = text
            record["interrupted"] = isinstance(exc, KeyboardInterrupt)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        # SQL execution is immediately after generation; scoring and cost lookup are outside this clock.
        if record["sql_ready_ms"] is not None:
            db = self.database.execute(record["sql"], "budget_" + uuid.uuid4().hex)
            record["database"] = db
            record["result_attempt_ms"] = (time.perf_counter() - start) * 1000
            if db["ok"]:
                record["result_ready_ms"] = record["result_attempt_ms"]
                record["correct"], record["comparison"] = compare(self.references[question["id"]], db["result"], question["contract"])
            else:
                record["comparison"] = "sql_error"
        record["usage"] = usage
        cost = usage.get("cost")
        if cost is None:
            cost = self.generation_cost(record.get("id_returned"))
        record["cost_usd"] = cost
        record["reserved_or_billed_cost_usd"] = float(cost) if cost is not None else reserved
        record["first_content_ms"] = first_content
        record["last_content_ms"] = last_content
        completion = usage.get("completion_tokens")
        reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        visible = completion - reasoning if completion is not None else None
        record["visible_output_tokens_estimated"] = visible
        record["visible_generation_tps"] = (visible - 1) * 1000 / (last_content - first_content) if visible and visible > 1 and first_content is not None and last_content > first_content else None
        record["ended_at"] = now()
        self.accounted += record["reserved_or_billed_cost_usd"]
        return record


def preflight(routes, block):
    snapshot = []
    for model in sorted({r["model_id"] for r in routes}):
        response = requests.get(API + "/models/" + model + "/endpoints", timeout=20)
        response.raise_for_status()
        body = response.json()
        snapshot.append({"model_id": model, "retrieved_at": now(), "response": body})
        for route in [r for r in routes if r["model_id"] == model]:
            endpoints = [e for e in body["data"]["endpoints"] if e["tag"] == route["provider_tag"]]
            if len(endpoints) != route["endpoint_records_matching_tag"]:
                raise RuntimeError("Endpoint scope changed: " + route["route_id"])
            for endpoint in endpoints:
                if endpoint["quantization"] != route["quantization"]:
                    raise RuntimeError("Precision changed: " + route["route_id"])
                for rate in ("prompt", "completion", "input_cache_read"):
                    if endpoint["pricing"].get(rate) != route["pricing_per_token"].get(rate):
                        raise RuntimeError("Price changed: " + route["route_id"])
    save(RESULTS / f"preflight-{block}.json", snapshot)


def run(phase):
    assert load(BASE / "reference-checks.json")["questions_sha256"] == sha(BASE / "questions.json")
    questions = {q["id"]: q for q in load(BASE / "questions.json")}
    routes = load(BASE / ("candidate-routes.json" if phase == "screen" else "selected-routes.json"))
    qids = sorted(questions)
    if phase == "screen":
        qids = ["t1_02", "t2_03", "t3_07", "t4_04", "t6_01"]
    elif phase == "pilot":
        qids = [f"t{tier}_{index:02d}" for tier in range(1, 7) for index in (1, 7)]
    repetitions = 5 if phase == "full" else 1
    schedule = []
    for repetition in range(1, repetitions + 1):
        rng = random.Random(20261007 + repetition)
        ordered_qids = qids.copy()
        rng.shuffle(ordered_qids)
        for qid in ordered_qids:
            ordered_routes = routes.copy()
            rng.shuffle(ordered_routes)
            for route in ordered_routes:
                task_id = f"{phase}:{repetition}:{qid}:{route['route_id']}"
                schedule.append({"task_id": task_id, "question_id": qid, "route": route, "repetition": repetition})
    manifest = {"phase": phase, "questions_sha256": sha(BASE / "questions.json"), "references_sha256": sha(BASE / "references.json"), "source_sha256": {f: sha(BASE / f) for f in ("benchmark.py", "evaluation.py", "schema.sql")}, "schedule": schedule, "concurrency": 1, "spending_cap_usd_all_phases": CAP_USD, "wall_clock_request_timeout_seconds": 60, "automatic_retries": 0}
    manifest_path = RESULTS / f"manifest-{phase}.json"
    if manifest_path.exists() and load(manifest_path) != manifest:
        raise RuntimeError("Manifest changed; cannot mix measurements on resume")
    save(manifest_path, manifest)
    output = RESULTS / f"{phase}.jsonl"
    done = {r["task_id"] for r in read_rows(output)}
    started = {r["task_id"] for r in read_rows(RESULTS / "started.jsonl")}
    pending = started - {r["task_id"] for file in RESULTS.glob("*.jsonl") if file.name != "started.jsonl" for r in read_rows(file)}
    if pending:
        raise RuntimeError("Interrupted requests need reconciliation before resume: " + repr(pending))
    runner = Runner()
    block = None
    for index, task in enumerate(schedule, 1):
        if task["task_id"] in done:
            continue
        if task["repetition"] != block:
            block = task["repetition"]
            preflight(routes, f"{phase}-{block}")
        result = runner.one(task["route"], questions[task["question_id"]], task["repetition"], phase, task["task_id"])
        append(output, result)
        state = "correct" if result["correct"] else result.get("comparison") or "api_error"
        seconds = (result.get("result_ready_ms") or result.get("api_complete_ms") or 0) / 1000
        print(f"{phase} {index}/{len(schedule)} {task['route']['route_id']} {task['question_id']} {state} {seconds:.3f}s billed={result['cost_usd']} accounted=${runner.accounted:.4f}", flush=True)
        if result["error"]:
            print("  " + result["error"][:500], flush=True)
        if result.get("interrupted"):
            raise KeyboardInterrupt
        if result.get("http_status") in (401, 402, 403):
            raise RuntimeError("Authentication, credit, or access failure; stopping without retry")
    print(f"COMPLETE {phase}: {len(schedule)} scheduled requests.", flush=True)


def select(allow_blocked=False):
    rows = read_rows(RESULTS / "screen.jsonl")
    routes = {r["route_id"]: r for r in load(BASE / "candidate-routes.json")}
    groups = defaultdict(list)
    for route_id, route in routes.items():
        attempts = [r for r in rows if r["route_id"] == route_id]
        completed = [r for r in attempts if r["sql_ready_ms"] is not None]
        if len(attempts) != 5:
            raise RuntimeError("Incomplete provider screening: " + route_id)
        costs = [r["cost_usd"] for r in attempts]
        groups[route["model_id"]].append({"route_id": route_id, "completed": len(completed), "median_sql_ready_ms": statistics.median(r["sql_ready_ms"] for r in completed) if completed else None, "mean_cost_usd": statistics.mean(costs) if all(c is not None for c in costs) else None})
    selected, decisions, blocked = [], [], []
    for model, stats in groups.items():
        most_completed = max(s["completed"] for s in stats)
        if most_completed < 4:
            blocked.append({"model_id": model, "candidates": stats})
            if allow_blocked:
                print("BLOCKED", model, flush=True)
                continue
            raise RuntimeError("No viable provider route for " + model)
        eligible = [s for s in stats if s["completed"] == most_completed and s["mean_cost_usd"] is not None]
        fastest = min(s["median_sql_ready_ms"] for s in eligible)
        close = [s for s in eligible if s["median_sql_ready_ms"] <= fastest * 1.20]
        winner = min(close, key=lambda s: (s["mean_cost_usd"], s["median_sql_ready_ms"]))
        selected.append(routes[winner["route_id"]])
        decisions.append({"model_id": model, "selected": winner["route_id"], "candidates": stats})
        print(winner["route_id"], f"{winner['median_sql_ready_ms']/1000:.3f}s", f"${winner['mean_cost_usd']:.6f}/request")
    save(BASE / "selected-routes.json", selected)
    save(RESULTS / "blocked-models.json", blocked)
    save(RESULTS / "route-selection.json", {"rule": "Prefer the highest screening completion count (at least 4/5); among routes within 20% of the fastest median SQL-ready delay, choose the lowest observed mean billed cost. SQL correctness is not used for route selection. Five screening prompts are not a definitive provider ranking.", "decisions": decisions})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["configure", "validate", "screen", "select", "pilot", "full"])
    parser.add_argument("--allow-blocked", action="store_true", help="Prepare eligible routes while blocked models remain unresolved")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.command == "configure":
        configure()
    elif args.command == "validate":
        validate()
    elif args.command == "select":
        select(args.allow_blocked)
    else:
        run(args.command)
