#!/usr/bin/env python3
"""Audit the complete run against its frozen schedule, prompts, and result scorer."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from evaluation import compare

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    manifest = load(RESULTS / "manifest-full.json")
    measured = rows(RESULTS / "full.jsonl")
    schedule = manifest["schedule"]
    assert len(measured) == len(schedule), (len(measured), len(schedule))
    assert len({r["task_id"] for r in measured}) == len(measured)
    journal = [r for r in rows(RESULTS / "started.jsonl") if r["task_id"].startswith("full:")]
    assert Counter(r["task_id"] for r in journal) == Counter(r["task_id"] for r in measured), "Unreconciled or repeated request start"
    questions = {q["id"]: q for q in load(BASE / "questions.json")}
    references = load(BASE / "references.json")
    assert hashlib.sha256((BASE / "questions.json").read_bytes()).hexdigest() == manifest["questions_sha256"]
    assert hashlib.sha256((BASE / "references.json").read_bytes()).hexdigest() == manifest["references_sha256"]
    for filename, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((RESULTS / "source-full" / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
    initial = {m["model_id"]: m["response"]["data"]["endpoints"] for m in load(BASE / "openrouter-endpoints-2026-10-07.json")}
    selected = {t["route"]["route_id"]: t["route"] for t in schedule}
    for repetition in range(1, 6):
        snapshot = {m["model_id"]: m["response"]["data"]["endpoints"] for m in load(RESULTS / f"preflight-full-{repetition}.json")}
        for route in selected.values():
            before = [e for e in initial[route["model_id"]] if e["tag"] == route["provider_tag"]]
            during = [e for e in snapshot[route["model_id"]] if e["tag"] == route["provider_tag"]]
            assert len(before) == len(during) == route["endpoint_records_matching_tag"]
            assert {e.get("model_id") for e in before} == {e.get("model_id") for e in during}
            assert {e["quantization"] for e in during} == {route["quantization"]}
            for endpoint in during:
                for rate in ("prompt", "completion", "input_cache_read"):
                    assert endpoint["pricing"].get(rate) == route["pricing_per_token"].get(rate)
    prompts, counts, providers = {}, Counter(), Counter()
    missing_tiers = 0
    missing_costs = 0
    generated_ids = []
    query_ids = []
    for record, task in zip(measured, schedule, strict=True):
        assert record["task_id"] == task["task_id"]
        route = task["route"]
        request = record["request"]
        assert request["model"] == route["model_id"] == record["model_id"]
        assert request["provider"]["only"] == [route["provider_tag"]]
        assert request["provider"]["quantizations"] == [route["quantization"]]
        assert request["provider"]["allow_fallbacks"] is False
        assert request["provider"]["require_parameters"] is True
        assert request["service_tier"] == route["service_tier"]
        for key, value in route["parameters"].items():
            assert request[key] == value, (record["task_id"], key)
        qid = record["question_id"]
        prompt = json.dumps(request["messages"], sort_keys=True)
        if qid in prompts:
            assert prompts[qid] == prompt, "Prompt varied across model/repetition"
        prompts[qid] = prompt
        counts[(record["model_id"], qid)] += 1
        assert datetime.fromisoformat(record["ended_at"]) >= datetime.fromisoformat(record["started_at"])
        assert record["api_complete_ms"] > 0
        if record.get("id_returned"):
            generated_ids.append(record["id_returned"])
        if record.get("model_returned"):
            assert record["model_returned"] == record["model_id"]
        if record.get("provider_returned"):
            assert record["provider_returned"].lower() == route["provider_name"].lower()
            providers[(record["model_id"], record["provider_returned"])] += 1
        if record.get("service_tier_returned"):
            reported = {"standard": "default", "fast": "priority"}.get(record["service_tier_returned"], record["service_tier_returned"])
            assert reported == route["service_tier"]
        elif record["sql_ready_ms"] is not None:
            missing_tiers += 1
        if record["sql_ready_ms"] is not None:
            assert record["sql_ready_ms"] >= record["api_complete_ms"]
            if record["first_content_ms"] is not None:
                assert 0 <= record["first_content_ms"] <= record["sql_ready_ms"]
        if record.get("database"):
            query_ids.append(record["database"]["query_id"])
        if record["result_ready_ms"] is not None:
            assert record["database"]["ok"]
            assert record["result_ready_ms"] >= record["sql_ready_ms"]
            correct, reason = compare(references[qid], record["database"]["result"], questions[qid]["contract"])
            assert correct == record["correct"] and reason == record["comparison"]
        else:
            assert record["correct"] is False
        if record["cost_usd"] is None:
            missing_costs += 1
        else:
            assert record["cost_usd"] >= 0
    assert set(counts.values()) == {5}
    assert len(counts) == 60 * len({r["model_id"] for r in measured})
    assert len(set(generated_ids)) == len(generated_ids), "Duplicate generation ID"
    assert len(set(query_ids)) == len(query_ids), "Duplicate database query ID"
    report = {
        "passed": True, "scheduled_and_recorded_attempts": len(measured),
        "distinct_models": len({r["model_id"] for r in measured}), "distinct_questions": len(prompts),
        "repetitions_per_model_question": 5, "unique_generation_ids": len(generated_ids),
        "unique_database_query_ids": len(query_ids), "recomputed_comparisons_match": True,
        "source_hashes_match_frozen_copy": True, "identical_prompts_across_models_and_repetitions": True,
        "all_started_requests_reconciled_once": True,
        "provider_model_precision_scope_and_rates_stable_across_five_blocks": True,
        "returned_models_and_providers_match_when_exposed": True,
        "completed_responses_without_returned_service_tier": missing_tiers,
        "attempts_without_observed_billing": missing_costs,
    }
    (RESULTS / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
