#!/usr/bin/env python3
"""Summarize measured results and create exportable figures; never invent missing costs."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import textwrap

import numpy as np

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
NAMES = {r["route_id"]: r["label"] for r in json.loads((BASE / "selected-routes.json").read_text())}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def ci_by_question(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row["question_id"]].append(key(row))
    means = np.array([np.mean(groups[qid]) for qid in sorted(groups)])
    rng = np.random.default_rng(20261007)
    indices = rng.integers(0, len(means), size=(5000, len(means)))
    low, high = np.percentile(means[indices].mean(axis=1), [2.5, 97.5])
    return [float(low), float(high)]


def deadline_success(row, seconds):
    return bool(row["correct"] and row["result_ready_ms"] is not None and row["result_ready_ms"] <= seconds * 1000)


def summarize(rows):
    output = []
    for model in sorted({row["route_id"] for row in rows}):
        rs = [r for r in rows if r["route_id"] == model]
        correct = sum(r["correct"] for r in rs)
        known_costs = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
        known_cost = sum(known_costs)
        costs_complete = len(known_costs) == len(rs)
        costs = known_cost if costs_complete else None
        successful = [r["result_ready_ms"] / 1000 for r in rs if r["result_ready_ms"] is not None]
        sql_ready = [r["sql_ready_ms"] / 1000 for r in rs if r["sql_ready_ms"] is not None]
        correct_latencies = [r["result_ready_ms"] / 1000 for r in rs if r["correct"]]
        tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rs)
        cache_tokens = sum((r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for r in rs)
        route = next(r for r in json.loads((BASE / "selected-routes.json").read_text()) if r["route_id"] == model)
        rates = route["pricing_per_token"]
        output_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) or 0 for r in rs)
        uncached_estimate = tokens * float(rates["prompt"]) + output_tokens * float(rates["completion"])
        details = {
            "route_id": model, "model_id": rs[0]["model_id"], "name": NAMES[model], "provider_tag": rs[0]["provider_tag_requested"],
            "precision": rs[0]["quantization_reported"], "attempts": len(rs), "correct": correct,
            "accuracy": correct / len(rs), "accuracy_ci95": ci_by_question(rs, lambda r: float(r["correct"])),
            "sql_ready_completed": len(sql_ready), "database_completed": len(successful),
            "sql_ready_median_s": percentile(sql_ready, 50), "sql_ready_p95_s": percentile(sql_ready, 95),
            "result_ready_median_s": percentile(successful, 50), "result_ready_q25_s": percentile(successful, 25),
            "result_ready_q75_s": percentile(successful, 75), "result_ready_p95_s": percentile(successful, 95),
            "correct_result_median_s": percentile(correct_latencies, 50),
            "known_billed_cost_usd": known_cost, "missing_cost_count": len(rs) - len(known_costs),
            "reserved_or_billed_cost_usd": sum(r["reserved_or_billed_cost_usd"] for r in rs),
            "billed_cost_usd": costs,
            "usd_per_1000_tasks": costs / len(rs) * 1000 if costs is not None else None,
            "usd_per_correct_result": costs / correct if costs is not None and correct else None,
            "api_errors": sum(r["error"] is not None for r in rs),
            "sql_errors": sum(r.get("comparison") == "sql_error" for r in rs),
            "wrong_results": sum(r["result_ready_ms"] is not None and not r["correct"] for r in rs),
            "prompt_tokens": tokens, "cached_prompt_tokens": cache_tokens,
            "cached_prompt_fraction": cache_tokens / tokens if tokens else None,
            "completion_tokens": output_tokens,
            "uncached_token_cost_estimate_usd_known_usage": uncached_estimate,
            "uncached_estimate_usd_per_1000_tasks": uncached_estimate / len(rs) * 1000,
            "endpoint_discount_fraction": rates.get("discount", 0),
            "reasoning_tokens": sum((r.get("usage", {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0 for r in rs),
            "median_visible_generation_tps": percentile([r["visible_generation_tps"] for r in rs if r.get("visible_generation_tps") is not None], 50),
            "first_content_median_s": percentile([r["first_content_ms"] / 1000 for r in rs if r.get("first_content_ms") is not None], 50),
        }
        for seconds in (2, 5, 10):
            count = sum(deadline_success(r, seconds) for r in rs)
            details[f"correct_within_{seconds}s"] = count
            details[f"correct_within_{seconds}s_fraction"] = count / len(rs)
            details[f"correct_within_{seconds}s_ci95"] = ci_by_question(rs, lambda r: float(deadline_success(r, seconds)))
        count5 = details["correct_within_5s"]
        details["usd_per_correct_within_5s"] = costs / count5 if costs is not None and count5 else None
        output.append(details)
    return output


def recommend(summary):
    best_accuracy = max(s["accuracy"] for s in summary)
    accurate = [s for s in summary if s["accuracy"] >= best_accuracy - 0.05 - 1e-12]
    best_deadline = max(s["correct_within_5s_fraction"] for s in accurate)
    eligible = [s for s in accurate if s["correct_within_5s_fraction"] >= best_deadline - 0.05 - 1e-12 and s["usd_per_correct_within_5s"] is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda s: s["usd_per_correct_within_5s"])


def paired_comparisons(rows, winner):
    """Keep task clusters paired when comparing the chosen model with another."""
    output = []
    models = sorted({r["route_id"] for r in rows})
    qids = sorted({r["question_id"] for r in rows})
    rng = np.random.default_rng(20261007)
    samples = rng.integers(0, len(qids), size=(5000, len(qids)))
    for metric, fn in [("accuracy", lambda r: float(r["correct"])), ("correct_within_5s", lambda r: float(deadline_success(r, 5)))]:
        scores = {model: np.array([np.mean([fn(r) for r in rows if r["route_id"] == model and r["question_id"] == qid]) for qid in qids]) for model in models}
        for other in models:
            if other == winner:
                continue
            differences = scores[winner] - scores[other]
            intervals = np.percentile(differences[samples].mean(axis=1), [2.5, 97.5])
            output.append({"recommended_configuration": winner, "other_configuration": other, "metric": metric, "difference_percentage_points": float(differences.mean() * 100), "ci95_low_percentage_points": float(intervals[0] * 100), "ci95_high_percentage_points": float(intervals[1] * 100)})
    return output


def focused_comparisons(rows):
    """Report the study's version, format/provider, and same-format provider contrasts."""
    routes = json.loads((BASE / "selected-routes.json").read_text())
    def pick(version, precision, provider=None):
        return next(r["route_id"] for r in routes if r["model_id"] == version and r["quantization"] == precision and (provider is None or r["provider_tag"] == provider))
    v4, v41 = "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4.1-flash"
    shared = json.loads((RESULTS / "route-selection.json").read_text())["shared_fp8_provider"]
    contrasts = [
        ("V4.1 FP4 route minus original FP8 route; provider confounded", pick(v41, "fp4"), pick(v41, "fp8", "inference-net/fp8")),
        ("V4 FP4 route minus FP8 route; provider confounded", pick(v4, "fp4"), pick(v4, "fp8", shared)),
        ("V4.1 minus V4 at the shared FP8 provider", pick(v41, "fp8", shared), pick(v4, "fp8", shared)),
        ("Shared provider minus InferenceNet, both V4.1 FP8", pick(v41, "fp8", shared), pick(v41, "fp8", "inference-net/fp8")),
    ]
    output = []
    for label, left, right in contrasts:
        for comparison in paired_comparisons(rows, left):
            if comparison["other_configuration"] == right:
                output.append({"contrast": label, "left_configuration": left, "right_configuration": right, **{k: v for k, v in comparison.items() if k not in ("recommended_configuration", "other_configuration")}})
    return output


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    palette = plt.get_cmap("tab10")
    for field, xlabel, filename in [
        ("result_ready_median_s", "Median total delay (seconds; completed database results)", "accuracy-vs-delay"),
        ("usd_per_correct_result", "Billed USD per correct result (log scale)", "accuracy-vs-cost"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6.4))
        fig.subplots_adjust(left=.10, right=.98, top=.90, bottom=.20)
        for i, s in enumerate(summary):
            x, y = s[field], s["accuracy"] * 100
            if x is None:
                continue
            low, high = (100 * v for v in s["accuracy_ci95"])
            ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="o", ms=8, capsize=4, color=palette(i), label=s["name"])
        if field == "usd_per_correct_result":
            ax.set_xscale("log")
        ax.set(xlabel=xlabel, ylabel="Correct first attempts (%)", ylim=(0, 102), title="DeepSeek Flash versions and provider-reported precision")
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=9)
        note = "Five repetitions per task; 95% intervals bootstrap questions. OpenRouter, pinned providers, reasoning disabled."
        if field == "usd_per_correct_result" and any(s[field] is None for s in summary):
            note += " Configurations with unknown total cost omitted."
        fig.text(.10, .025, textwrap.fill(note, width=110), fontsize=8, va="bottom")
        for extension in ("svg", "png"):
            fig.savefig(output / f"{filename}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    positions = np.arange(len(summary))
    for index, seconds in enumerate((2, 5, 10)):
        ax.bar(positions + (index - 1) * .25, [s[f"correct_within_{seconds}s_fraction"] * 100 for s in summary], width=.24, label=f"Within {seconds}s")
    ax.set_xticks(positions, [s["name"] for s in summary], rotation=20, ha="right")
    ax.set(ylabel="All scheduled tasks answered correctly within deadline (%)", ylim=(0, 100), title="Accuracy and total delay together")
    ax.legend(loc="upper center", ncol=3)
    ax.grid(axis="y", alpha=.2)
    for extension in ("svg", "png"):
        fig.savefig(output / f"correct-within-deadline.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "full"], default="full")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    rows = read(RESULTS / f"{args.phase}.jsonl")
    manifest = json.loads((RESULTS / f"manifest-{args.phase}.json").read_text())
    observed = [r["task_id"] for r in rows]
    assert len(observed) == len(set(observed)), "Duplicate attempts"
    if not args.partial:
        assert set(observed) == {t["task_id"] for t in manifest["schedule"]}, "Incomplete run"
    output = RESULTS / (args.phase + ("-partial" if args.partial else "-analysis"))
    output.mkdir(exist_ok=True)
    summary = summarize(rows)
    if args.phase == "full" and not args.partial:
        for model in {r["route_id"] for r in rows}:
            counts = Counter(r["question_id"] for r in rows if r["route_id"] == model)
            assert len(counts) == 60 and set(counts.values()) == {5}, counts
    result = {"phase": args.phase, "partial": args.partial, "first_request_at": rows[0]["started_at"], "last_request_ended_at": rows[-1]["ended_at"], "summary": summary, "recommendation": recommend(summary) if args.phase == "full" and not args.partial else None}
    result["study_plan"] = json.loads((BASE / "study-plan.json").read_text())
    result["previous_winner"] = json.loads((BASE / "previous-winner.json").read_text())
    if result["recommendation"]:
        write_csv(output / "paired-comparisons.csv", paired_comparisons(rows, result["recommendation"]["route_id"]))
    if args.phase == "full" and not args.partial:
        write_csv(output / "focused-comparisons.csv", focused_comparisons(rows))
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    flat = [{k: v for k, v in s.items() if not isinstance(v, list)} for s in summary]
    write_csv(output / "summary.csv", flat)
    tiers = []
    for model in sorted({r["route_id"] for r in rows}):
        for tier in range(1, 7):
            rs = [r for r in rows if r["route_id"] == model and r["tier"] == tier]
            if rs:
                tiers.append({"route_id": model, "tier": tier, "attempts": len(rs), "correct": sum(r["correct"] for r in rs), "accuracy": np.mean([r["correct"] for r in rs]), "correct_within_5s": sum(deadline_success(r, 5) for r in rs)})
    write_csv(output / "by-tier.csv", tiers)
    blocks = []
    for model in sorted({r["route_id"] for r in rows}):
        for repetition in sorted({r["repetition"] for r in rows}):
            rs = [r for r in rows if r["route_id"] == model and r["repetition"] == repetition]
            if not rs:
                continue
            delays = [r["result_ready_ms"] / 1000 for r in rs if r["result_ready_ms"] is not None]
            blocks.append({
                "route_id": model, "repetition": repetition, "attempts": len(rs),
                "accuracy": sum(r["correct"] for r in rs) / len(rs),
                "correct_within_5s_fraction": sum(deadline_success(r, 5) for r in rs) / len(rs),
                "result_ready_median_s": percentile(delays, 50),
                "result_ready_q25_s": percentile(delays, 25), "result_ready_q75_s": percentile(delays, 75),
                "result_ready_p95_s": percentile(delays, 95),
                "api_errors": sum(r["error"] is not None for r in rs),
                "sql_errors": sum(r.get("comparison") == "sql_error" for r in rs),
                "known_billed_cost_usd": sum(r["cost_usd"] for r in rs if r["cost_usd"] is not None),
                "missing_cost_count": sum(r["cost_usd"] is None for r in rs),
            })
    write_csv(output / "by-repetition.csv", blocks)
    question_scores = []
    for model in sorted({r["route_id"] for r in rows}):
        for qid in sorted({r["question_id"] for r in rows}):
            rs = [r for r in rows if r["route_id"] == model and r["question_id"] == qid]
            if rs:
                question_scores.append({"route_id": model, "question_id": qid, "tier": rs[0]["tier"], "attempts": len(rs), "correct": sum(r["correct"] for r in rs), "correct_within_5s": sum(deadline_success(r, 5) for r in rs)})
    write_csv(output / "by-question.csv", question_scores)
    errors = [{"route_id": r["route_id"], "question_id": r["question_id"], "repetition": r["repetition"], "kind": r.get("comparison") or "api_error", "sql": r.get("sql", ""), "error": r.get("error") or r.get("database", {}).get("error", ""), "task_id": r["task_id"]} for r in rows if not r["correct"]]
    if errors:
        write_csv(output / "failures.csv", errors)
    lines = ["# DeepSeek Flash versions and quantizations", "", f"Phase: {args.phase}. Attempts: {len(rows)}. Complete: {not args.partial}.", "", f"Measured {result['first_request_at']} through {result['last_request_ended_at']} from Falkenstein, Germany.", ""]
    lines += ["Each row is a pinned model/provider/precision configuration. FP4 and FP8 use different providers; this experiment cannot attribute their differences solely to quantization. Both versions are also measured at FP8 on a shared provider, and the previous InferenceNet winner is rerun as a control.", ""]
    winner = result["recommendation"]
    if winner:
        lines += [f"Under the predeclared five-second / near-best-accuracy rule, the recommendation is **{winner['name']}** on `{winner['provider_tag']}`.", ""]
    lines += ["| Configuration | Correct | Correct ≤5s | Median total [IQR] | p95 total | USD / 1,000 tasks | USD / correct |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value, digits=3):
        return "unknown" if value is None else f"{value:.{digits}f}"
    def money(value, digits=3):
        return "unknown" if value is None else f"${value:.{digits}f}"
    for s in sorted(summary, key=lambda x: -x["accuracy"]):
        lines.append(f"| {s['name']} | {s['correct']}/{s['attempts']} ({s['accuracy']:.1%}) | {s['correct_within_5s_fraction']:.1%} | {fmt(s['result_ready_median_s'])} [{fmt(s['result_ready_q25_s'])}, {fmt(s['result_ready_q75_s'])}]s | {fmt(s['result_ready_p95_s'])}s | {money(s['usd_per_1000_tasks'])} | {money(s['usd_per_correct_result'],6)} |")
    lines += ["", "Total delay runs from the OpenRouter request start through receipt and parsing of the ClickHouse result. Quantiles cover successful database executions, including incorrect answers. Errors and timeouts remain in accuracy and deadline denominators. Reported costs include failed attempts where billing is known; missing billing remains unknown.", "", "The recommendation first keeps models within five percentage points of the best accuracy, then within five points of that group's best five-second success rate, and chooses the lowest cost per correct result within five seconds. This is a declared workload preference, not a universal ranking.", "", "All models use their pinned provider and the lowest available reasoning mode (disabled here). Cached-token usage and promotional endpoint rates are recorded; prices are not permanent. The fixture and explicit output contracts differ from the earlier published benchmark, so scores are not directly comparable. The observations cover one client region and this measurement period.", "", "Confidence intervals in summary.json and the figures resample questions with their repetitions together. The study has 60 distinct tasks, not 300 independent questions per model."]
    if args.phase == "full" and not args.partial:
        lines += ["", "## Paired accuracy comparisons", "", "| Contrast (left minus right) | Difference | 95% question-bootstrap interval |", "| --- | ---: | ---: |"]
        for comparison in focused_comparisons(rows):
            if comparison["metric"] == "accuracy":
                lines.append(f"| {comparison['contrast']} | {comparison['difference_percentage_points']:+.2f} pp | [{comparison['ci95_low_percentage_points']:+.2f}, {comparison['ci95_high_percentage_points']:+.2f}] pp |")
        lines += ["", "An interval spanning zero does not establish an accuracy advantage. These are descriptive comparisons on this task set, without adjustment for multiple comparisons or provider selection. FP4/FP8 contrasts also change provider. See [paired accuracy and deadline comparisons](focused-comparisons.csv)."]
    lines += ["", "## Providers and generation speed", "", "| Configuration | Pinned provider | Reported precision | Median first content | Median visible tokens/s | API errors | SQL errors |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for s in summary:
        lines.append(f"| {s['name']} | `{s['provider_tag']}` | {s['precision']} | {fmt(s['first_content_median_s'])}s | {fmt(s['median_visible_generation_tps'],1)} | {s['api_errors']} | {s['sql_errors']} |")
    lines += ["", "Precision is reported by OpenRouter, not independently verified. Unknown means the provider does not disclose it. Tokens/s is a streaming estimate and is not directly comparable across tokenizers or chunking schemes; complete-result delay drives the recommendation.", "", "## Billing evidence", "", "| Configuration | Known billed USD | Attempts without cost | Cached input share | Uncached estimate / 1,000 tasks | Endpoint discount field |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for s in summary:
        cache = "unknown" if s["cached_prompt_fraction"] is None else f"{s['cached_prompt_fraction']:.1%}"
        discount = float(s["endpoint_discount_fraction"] or 0)
        lines.append(f"| {s['name']} | {money(s['known_billed_cost_usd'],6)} | {s['missing_cost_count']} | {cache} | {money(s['uncached_estimate_usd_per_1000_tasks'])} | {discount:.0%} |")
    lines += ["", "Uncached estimates apply the saved endpoint rates to returned token counts, without a cache discount. They cover known usage only and cannot resolve missing error billing. Endpoint promotions are already reflected in those saved rates; the discount field is shown separately and is not applied a second time. Infrastructure and screening/pilot costs are excluded from model cost comparisons.", "", "See [methodology](../../README.md), [complete summary](summary.json), [per-tier results](by-tier.csv), [per-question results](by-question.csv), [variation across repetitions](by-repetition.csv), and [individual failures](failures.csv)."]
    if winner:
        winner_tiers = [t for t in tiers if t["route_id"] == winner["route_id"]]
        basic = [t for t in winner_tiers if t["tier"] <= 3]
        advanced = next(t for t in winner_tiers if t["tier"] == 6)
        lines += ["", f"The recommended configuration answered {sum(t['correct'] for t in basic)}/{sum(t['attempts'] for t in basic)} attempts correctly across tiers 1–3, but only {advanced['correct']}/{advanced['attempts']} in tier 6. Its lead does not make advanced ClickHouse SQL reliable without validation.", "", "## Figures", "", "![Accuracy versus total delay](accuracy-vs-delay.png)", "", "![Accuracy versus billed cost per correct result](accuracy-vs-cost.png)", "", "![Correct results within response deadlines](correct-within-deadline.png)"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if args.phase == "full" and not args.partial:
        figures(summary, output)
    print("\n".join(lines[:17]))


if __name__ == "__main__":
    main()
