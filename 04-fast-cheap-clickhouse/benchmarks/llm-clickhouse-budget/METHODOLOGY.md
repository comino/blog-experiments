# ClickHouse fast and cheap model benchmark

This is a new experiment following the published 22-model benchmark. Model calls use OpenRouter only. The cohort was researched on 7 October 2026 with a release cutoff of 7 August. Exact IDs and release evidence are in [model-shortlist.json](model-shortlist.json).

The author accepted undisclosed provider precision as a labelled exception. Granite uses reported BF16 and DeepSeek uses reported FP8, the highest disclosed formats available for those models in the saved endpoint snapshot. Closed-provider precision remains unknown. These labels come from OpenRouter; the experiment does not independently audit inference arithmetic or weights.

Mistral Large 4 and Qwen3.8 Flash were blocked during screening by the account's zero-data-retention policy. The five accessible models proceed independently. [cohort-notes.json](cohort-notes.json) records the unresolved access decision; no result from the five-model run establishes a seven-model winner.

## Workload and database

The temporary Hetzner CCX13 has two dedicated x86 CPUs and 8 GB RAM in Falkenstein, Germany. Both the Python client and ClickHouse run on it. ClickHouse listens on loopback, runs as a separate system user, and is pinned to **26.8.19.9 LTS**, with an official archive checksum preserved in [clickhouse-binary.json](clickhouse-binary.json).

The deterministic fixture contains one million events and ten thousand users using the original two-table schema. Independent `cityHash64` inputs generate the fields. One hundred users have no events, and a small planted sequence makes the retention and time-window tasks nonempty. Event data covers all 366 days of 2024. The SQL generator and dataset fingerprints are retained.

The query settings are two execution threads, a 2 GB memory limit, a ten-second database execution limit, 100,000 maximum result rows, and 16 MiB maximum result bytes. Exceeding result limits throws an error. Model queries use a read-only user with SELECT access to the benchmark database. Background merges are stopped after fixture preparation; the query cache is disabled. Storage caches are warm from reference validation and pilots. Reference evaluation is outside the measured request path.

## Questions and correctness

There are sixty distinct questions: ten in each of six difficulty tiers. [questions.json](questions.json) preserves the original wording and reference SQL alongside the new version. Every task now declares output columns, ordering, tie-breaks, and limits where needed. Changes resolve hidden limits, nondeterministic ties, duplicate event types, premium-tag substring semantics, and the inactive-user reference. Exact quantiles replace sampling-based references.

All sixty references must execute successfully twice and return stable results before any model calls. Fixture assertions check the million-event count, the hundred inactive users, and a nonempty short event sequence. [reference-checks.json](reference-checks.json) records this validation.

Correctness means matching the reference result on this fixture. Every column and row is compared, including nested arrays and maps. Integer counts and money are exact; floating values allow relative and absolute tolerance of 1e-6. Requested row order and array order matter. Column aliases are ignored. The evaluator does not separately enforce use of a named SQL function when an alternative query produces the required result. Execution on a single fixture is not a proof of equivalence on every possible dataset.

SQL extraction accepts a complete Markdown SQL fence or plain SQL and removes a trailing semicolon. It does not repair model-authored SQL. Regression tests reject a wrong fourth column, wrong requested ordering, duplicate-row changes, approximate integer counts, and extra columns.

## Provider screening and inference

Thirteen candidate routes cover the initial seven models. Each receives the same five screening questions. Provider selection first favors the highest completed-response count, requiring at least four of five completions. Among routes within 20% of the fastest median SQL-ready time, it chooses the lowest observed mean billed cost. Screening correctness is not used to select a route. Five prompts give a practical screening decision, not a definitive provider ranking.

Each selected model then receives twelve pilot questions spanning all six tiers. Screening and pilot responses remain separate from the scored experiment. The full experiment uses all sixty questions five times per model, with a fixed randomized question order per repetition and randomized model order within each question. One request runs at a time. Five repetitions form five successive time blocks in one measurement session.

The run pins the provider slug, reported quantization, service tier, parameters, and price snapshot. Provider fallback and client retries are disabled. Metadata is checked before each repetition; changes in endpoint scope, precision, or rates stop the run. Returned provider and service tier are saved when exposed; some providers omit the tier, and exact underlying endpoint identifiers are not always returned.

All chosen models are run with reasoning disabled. Temperature is zero and seed 42 where the endpoint supports those settings; unsupported settings are omitted. A shared limit of 4,096 completion tokens and a sixty-second model-request wall-clock limit apply. This comparison evaluates these latency-focused settings, not each model's maximum reasoning performance.

## Timing, cost, and recommendation

Both clocks start immediately before sending the OpenRouter request. **SQL ready** ends after the complete response arrives and SQL is extracted. **Result ready** ends after the database result arrives and is parsed. Database time is also recorded separately. Scoring and any billing-metadata lookup occur afterward. Streaming timestamps track first visible content and generation throughput as supplementary diagnostics; tokenization and chunking differ between models.

Latency quantiles cover completed SQL responses or successful database executions, with populations named explicitly. Incorrect but executable answers remain in the latter latency population. Failures and timeouts remain in the denominator for overall correctness and correct results within two, five, and ten seconds. No error is recorded as a zero-second success.

Returned `usage.cost` is the primary billing source, with generation metadata queried if necessary. Missing cost remains unknown in the raw records and receives a conservative reserve for the spending cap. [OpenRouter's zero-completion policy](https://openrouter.ai/docs/guides/features/zero-completion-insurance) generally protects failed requests without output from inference charges; that policy is not substituted for a per-request billing record. Cache token counts and current promotional prices are preserved. Actual billed cost measures this repeated workload; uncached costs are a separate estimate. The inference spending cap is $9.50 across all phases.

The [decision rule](decision-rule.json) was saved before the full run: retain models within five percentage points of the best correctness, then within five points of that group's best five-second success rate, and choose the lowest cost per correct answer delivered within five seconds. The full table remains available to readers with different latency or quality requirements.

Accuracy and deadline confidence intervals use 5,000 bootstrap samples of questions, retaining all repetitions of a sampled question. The experiment has sixty distinct tasks, not three hundred independent questions per model. Repeated outputs, API failures, and timing variation are retained. Results apply to this workload, region, provider configuration, and measurement period; prices and provider performance can change.

## Artifacts and reproduction

The experiment directory contains the fixture, versioned questions, raw SSE events and SQL outputs, database results, endpoint metadata, deterministic schedules, and code hashes. Each attempted request is journalled before sending and durably recorded afterward. Interrupted requests require reconciliation before resuming, preventing accidental duplicate paid calls. Source files used for the full run are copied into `results/source-full/`.

`prepare.py` prepares the fixture and questions; `benchmark.py` configures routes, validates references, screens providers, runs pilots, and performs the main study. `analyse.py` produces summaries, failure tables, bootstrap intervals, and standalone SVG/PNG figures. The temporary server is deleted after artifacts have been copied and verified locally. The original server and published article are unchanged.
