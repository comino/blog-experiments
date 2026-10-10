# Interpretation and examples for the follow-up article

DeepSeek V4.1 Flash on `inference-net/fp8` is the best observed fast-and-cheap choice among the five measured configurations. It scored 264/300 correct, with 263/300 correct results delivered within five seconds. Its median completed-result delay was 0.672 seconds and its p95 was 1.847 seconds. The declared rule selects it without a close cost tie: no other model is within five percentage points of its overall correctness.

This is a result for a model **and its selected provider/settings**. Luna used OpenAI priority service because that route won the latency-focused screening; standard service was cheaper in screening and was not measured in the main study. Solar's saved endpoint rates include a 50% promotion. Mistral Large 4 and Qwen3.8 Flash remain blocked and unranked.

## Cost and uncertainty

DeepSeek's measured inference cost was $0.04135 per 1,000 scheduled tasks and $0.00004699 per correct result. About 82.9% of its input tokens were served from cache. Pricing all observed tokens at the saved uncached rates gives approximately $0.08331 per 1,000 tasks. These repeated-question measurements should not be presented as uncached production costs.

The 95% question-bootstrap interval for DeepSeek's accuracy is 79.3–95.0%. Its observed advantage over Luna is nine percentage points; the paired interval is 1.0–17.3 points. These intervals describe resampling this sixty-question set, not performance on every SQL workload. The paired comparisons are descriptive and are not adjusted for selecting a winner or testing multiple model pairs.

Mercury had 39 upstream rate-limit failures and six timeouts in 300 scheduled requests. Those failures are part of the measured service outcome, not evidence about SQL reasoning on requests it never answered. Their per-request billing was unavailable, so the report does not claim an exact Mercury cost per correct result. Account-wide billing records are excluded from this public snapshot; the per-attempt charges and missing billing are retained in the results.

## Representative failures

The examples below refer to the first full repetition. Raw SQL, returned results, errors, and task IDs are retained in `results/full.jsonl`; all failures are also in [failures.csv](results/full-analysis/failures.csv). No SQL was repaired or retried in the scored study.

| Model / question | Observed failure | Why it matters |
| --- | --- | --- |
| DeepSeek, `t6_01` | Passed the `DateTime64` timestamp directly to `windowFunnel`; this pinned ClickHouse build returned `ILLEGAL_TYPE_OF_ARGUMENT`. | Knowing the function name did not ensure a query that executes. The reference casts to `DateTime`. |
| DeepSeek, `t5_05` | Reused `total_duration` as the alias of `max(total_duration)` while referencing it in `argMax`; ClickHouse returned `ILLEGAL_AGGREGATION`. | Alias substitution can turn plausible SQL into an invalid aggregate expression. |
| DeepSeek, `t6_02` | Counted users independently for each date, returning `[2302, 2387, 2403]` instead of the retention reference's `[2302, 595, 556]`. | Successful execution alone would have accepted a semantically wrong answer. |
| DeepSeek, `t4_10` | A correlated aggregate query returned no rows; the reference returned four rows, including two tied tablet users. | Checking row counts and required ties catches errors that a syntax-only benchmark misses. |
| Luna, `t6_05` | Generated a second `ORDER BY` after `LIMIT 2 BY`; ClickHouse returned a syntax error. | A nearly correct answer still fails the one-attempt user workflow. |

DeepSeek was correct on all 150 attempts in tiers 1–3, but on only 25/50 attempts in tier 6. Seven questions failed in every repetition, and one additional question failed once. The recommendation is strongest for this simpler workload; advanced ClickHouse features still need result validation.

## Scope of an article claim

The defensible headline is that DeepSeek led **these five recent models on this measured ClickHouse workload**, with the selected OpenRouter routes and reasoning disabled. The experiment does not establish a seven-model winner, the best model on OpenRouter as a whole, the highest possible accuracy with reasoning enabled, or an improvement over historical scores measured with a different fixture and evaluator.

Use the [full table and figures](results/full-analysis/RESULTS.md) with the [methodology](METHODOLOGY.md). Treat Mistral and Qwen as outstanding measurements if they remain part of the intended article comparison.
