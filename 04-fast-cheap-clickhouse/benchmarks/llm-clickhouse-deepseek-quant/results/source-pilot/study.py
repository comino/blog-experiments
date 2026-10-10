"""Choose explicit model/format routes without conflating their scored results."""
import json
import statistics

V4 = "deepseek/deepseek-v4-flash"
V41 = "deepseek/deepseek-v4.1-flash"
BASELINE = V41 + "@inference-net/fp8"
PAIR_TAGS = ("deepinfra/fp8", "streamlake/fp8", "parasail/fp8")


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def configure_routes(base):
    snapshots = {m["model_id"]: m["response"]["data"]["endpoints"] for m in read(base / "openrouter-endpoints-2026-10-07.json")}
    models = {m["id"]: m for m in read(base / "run-model-metadata.json")}
    choices = {
        V4: ["relace/fp4", "open-inference/fp4", "atlas-cloud/fp4", *PAIR_TAGS],
        V41: ["sail-research/fp4", "decart/fp4", "inference-net/fp8", *PAIR_TAGS],
    }
    routes = []
    for model, tags in choices.items():
        assert not models[model]["reasoning"].get("mandatory")
        for tag in tags:
            endpoints = [e for e in snapshots[model] if e["tag"] == tag]
            assert len(endpoints) == 1, (model, tag, len(endpoints))
            e = endpoints[0]
            assert e["quantization"] in ("fp4", "fp8")
            supported = set(e["supported_parameters"])
            assert {"reasoning", "temperature", "max_tokens"} <= supported
            parameters = {"max_tokens": 4096, "temperature": 0, "reasoning": {"enabled": False}}
            if "seed" in supported:
                parameters["seed"] = 42
            version = "V4" if model == V4 else "V4.1"
            routes.append({
                "route_id": model + "@" + tag, "model_id": model,
                "name": models[model]["name"], "label": f"{version} · {e['quantization'].upper()} · {e['provider_name']}",
                "provider_tag": tag, "provider_name": e["provider_name"],
                "endpoint_records_matching_tag": 1, "endpoint_model_ids": [e.get("model_id")],
                "quantization": e["quantization"], "unknown_precision_accepted_by_user": False,
                "service_tier": "default", "parameters": parameters, "pricing_per_token": e["pricing"],
            })
    save(base / "candidate-routes.json", routes)
    save(base / "study-plan.json", {
        "requested_models": [V4, V41], "requested_formats": ["fp4", "fp8"],
        "model_name_interpretation": "Flash 4 is interpreted as DeepSeek V4 Flash 0423; optional clarification was requested before execution.",
        "release_window_exception": "V4 Flash was released 2026-04-24; included as the specifically requested older baseline. V4.1 Flash was released 2026-09-10.",
        "release_sources": ["https://openrouter.ai/deepseek/deepseek-v4-flash", "https://openrouter.ai/deepseek/deepseek-v4.1-flash"],
        "comparison_unit": "Pinned model, provider, reported precision, and inference settings",
        "quantization_causality_limit": "No provider in this snapshot offers both FP4 and FP8 for the same version. Precision is confounded with provider, hardware, and serving implementation.",
        "planned_main_configurations": 5, "questions_per_configuration": 60, "repetitions": 5,
        "screening_requests": len(routes) * 5, "pilot_requests": 60, "main_requests": 1500,
        "configuration_selection": "One viable FP4 route per version, one shared viable FP8 provider for both versions, plus the pinned previous InferenceNet FP8 winner.",
        "sql_scores_used_for_provider_selection": False,
        "inference_spend_cap_usd": 2.0,
    })
    print(f"Configured {len(routes)} routes for screening.")


def select_routes(base):
    rows = [json.loads(line) for line in (base / "results/screen.jsonl").read_text().splitlines()]
    routes = {r["route_id"]: r for r in read(base / "candidate-routes.json")}
    stats = {}
    for rid, route in routes.items():
        attempts = [r for r in rows if r["route_id"] == rid]
        assert len(attempts) == 5, rid
        complete = [r for r in attempts if r["sql_ready_ms"] is not None]
        costs = [r["cost_usd"] for r in attempts]
        reasoning = sum((r.get("usage", {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0 for r in attempts)
        stats[rid] = {
            "route_id": rid, "completed": len(complete), "reasoning_tokens": reasoning,
            "median_sql_ready_ms": statistics.median(r["sql_ready_ms"] for r in complete) if complete else None,
            "mean_billed_usd": statistics.mean(costs) if all(v is not None for v in costs) else None,
            "viable": len(complete) >= 4 and all(v is not None for v in costs) and reasoning == 0,
        }
    def select_one(candidates):
        candidates = [s for s in candidates if s["viable"]]
        assert candidates, "No viable route; do not silently omit a required configuration"
        most = max(s["completed"] for s in candidates)
        candidates = [s for s in candidates if s["completed"] == most]
        fastest = min(s["median_sql_ready_ms"] for s in candidates)
        candidates = [s for s in candidates if s["median_sql_ready_ms"] <= fastest * 1.2]
        return min(candidates, key=lambda s: (s["mean_billed_usd"], s["median_sql_ready_ms"]))
    assert stats[BASELINE]["viable"], "Previous winner is unavailable; retain the failure and reassess the control"
    chosen = [BASELINE]
    for model in (V4, V41):
        candidates = [stats[rid] for rid, route in routes.items() if route["model_id"] == model and route["quantization"] == "fp4"]
        chosen.append(select_one(candidates)["route_id"])
    pairs = []
    for tag in PAIR_TAGS:
        a, b = stats[V4 + "@" + tag], stats[V41 + "@" + tag]
        pairs.append({
            "route_id": tag, "completed": min(a["completed"], b["completed"]),
            "viable": a["viable"] and b["viable"],
            "median_sql_ready_ms": (a["median_sql_ready_ms"] + b["median_sql_ready_ms"]) / 2 if a["median_sql_ready_ms"] is not None and b["median_sql_ready_ms"] is not None else None,
            "mean_billed_usd": (a["mean_billed_usd"] + b["mean_billed_usd"]) / 2 if a["mean_billed_usd"] is not None and b["mean_billed_usd"] is not None else None,
        })
    pair = select_one(pairs)["route_id"]
    chosen.extend([V4 + "@" + pair, V41 + "@" + pair])
    assert len(set(chosen)) == 5
    save(base / "selected-routes.json", [routes[rid] for rid in chosen])
    save(base / "results/route-selection.json", {
        "rule": "Require at least 4/5 completed responses, all request billing known, and no observed reasoning tokens. Favor the highest completion count; among routes within 20% of the fastest median SQL-ready time choose the lowest mean billed cost. For the shared FP8 provider use the minimum completion count and mean latency/cost across the two versions. Preserve the prior InferenceNet FP8 winner as a fixed control. SQL correctness does not select routes.",
        "route_stats": list(stats.values()), "shared_provider_candidates": pairs,
        "selected_route_ids": chosen, "shared_fp8_provider": pair,
    })
    for rid in chosen:
        print(rid, stats[rid])
