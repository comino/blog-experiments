# DeepSeek Flash versions and quantizations

An OpenRouter-only extension of the [five-model ClickHouse benchmark](../llm-clickhouse-budget/README.md), requested on 7 October 2026. The user asked to add “Flash 4” and test different quantizations after V4.1 Flash looked promising.

“Flash 4” is interpreted as `deepseek/deepseek-v4-flash` (April 24 release), an explicitly requested older comparison despite the initial recent-model preference. V4.1 Flash is the September 10 release. Both offer provider-reported FP4 and FP8 in the captured endpoint metadata; no FP16/BF16 route is disclosed. Unknown precision is excluded from this quantization study.

The unit of comparison is a **model, provider, reported format, and inference configuration**. No provider in this snapshot serves both FP4 and FP8 for the same version. Differences therefore cannot be attributed solely to quantization: hardware, kernels, serving software, and provider behavior can also differ. Even a shared provider across versions does not establish identical hardware or capacity.

The current [official V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash#model-downloads) describes the instruct checkpoint as mixed precision: FP4 for MoE expert parameters and FP8 for most others. An OpenRouter FP8 label therefore does not, by itself, establish a higher-precision source checkpoint or recovered weight information. Provider checkpoint revisions and the exact conversion/serving pipeline are not exposed by this experiment. The table labels remain provider-reported formats.

## Configurations and provider screening

The main experiment uses five configurations:

1. V4.1 Flash, FP8, InferenceNet—the previous winner, repeated in the same session as the other configurations.
2. V4 Flash, FP4, Relace—selected by screening.
3. V4.1 Flash, FP4, Sail Research—selected by screening.
4. V4 Flash, FP8, Parasail—the selected shared provider.
5. V4.1 Flash, FP8, Parasail—the same provider as configuration 4.

Twelve routes receive five screening prompts each. FP4 screening covers Relace, OpenInference, and AtlasCloud for V4, and Sail Research and Decart for V4.1. Shared FP8 screening covers DeepInfra, StreamLake, and Parasail for both versions. InferenceNet is the fixed control. Account retention restrictions remain enabled and any resulting errors are recorded.

Selection requires at least four completed responses out of five, known billing for all five requests, and no observed reasoning tokens. It favors the highest completion count, then chooses the cheapest route among those within 20% of the fastest median complete-SQL delay. A provider pair uses the smaller completion count and the mean latency/cost across its two versions. SQL correctness does not determine route selection. Five screening requests are a practical selection step, not a conclusive provider benchmark.

Each selected configuration receives twelve separate pilot questions. The full run uses sixty questions five times per configuration: **1,500 scored requests**, plus sixty screening and sixty pilot requests. Model calls run sequentially, with the order of questions and configurations randomized within each repetition. No automatic retries or provider fallbacks are allowed.

## Controls retained from the first run

The question text, output contracts, evaluator, deterministic data generator, schema, and ClickHouse 26.8.19.9 build are unchanged. The replacement CCX13 server is in the same Falkenstein region. Both the API client and database run there, with the database on loopback. Its hardware and software environment are recorded separately.

The fixture contains one million events and ten thousand users. Both table fingerprints must match the first run, and all sixty reference results must match as well as execute twice stably. SQL uses a read-only database user, two threads, a 2 GB memory limit, a ten-second execution limit, fixed result limits, and no query cache. Background merges are stopped; storage caches are warm after validation.

Inference uses reasoning disabled, temperature zero, seed 42 where supported, 4,096 maximum completion tokens, and a sixty-second request timeout. Every request pins its provider and quantization with fallback disabled. Metadata is rechecked before every repetition. Returned provider and tier fields are retained when exposed; provider-reported precision is not independently verified.

Total delay starts immediately before the OpenRouter request and ends after the database result is received and parsed. Scoring and any billing lookup occur afterward. All result rows and columns are scored with explicit ordering and numerical tolerances. The [original methodology](../llm-clickhouse-budget/METHODOLOGY.md) describes these unchanged controls and limitations.

## Analysis and evidence

`analyse.py` groups by complete route ID, so different providers and formats of the same model are never pooled. It reports correctness, complete-result median/IQR/p95, correctness within two/five/ten seconds, actual returned cost, missing billing, cache usage, and an uncached price estimate. Failures remain in accuracy and deadline denominators. Latency quantiles name their completed-response population.

The [decision rule](decision-rule.json) keeps configurations within five percentage points of the best accuracy, then within five points of the best five-second success rate in that group, and chooses the lowest billed cost per correct result within five seconds. Confidence intervals resample sixty question clusters with all their repetitions. This is a declared workload preference, not a universal ranking.

Raw attempts, streamed events, SQL, results, query logs, endpoint snapshots, schedules, and frozen source are retained. `audit.py` recomputes comparisons and verifies the schedule, routing, prompts, and source hashes. The inference cap is **$2** for this extension. The temporary server is deleted after artifact checksums are verified locally.

The full run completed all 1,500 requests and passed its audit. Start with [results and figures](results/full-analysis/RESULTS.md), [interpretation notes](NOTES.md), [selected routes](selected-routes.json), and [audit](results/audit.json). Sail Research FP4 wins the declared five-second cost rule; InferenceNet FP8 offers substantially lower delay. Their two-point observed accuracy difference is inconclusive. First expand `results/full.jsonl.gz` with `gzip -dk results/full.jsonl.gz`. Run `python3 audit.py` and `python3 analyse.py --phase full` inside this directory to reproduce the analysis without model calls. The previous study remains a separate immutable result set.
