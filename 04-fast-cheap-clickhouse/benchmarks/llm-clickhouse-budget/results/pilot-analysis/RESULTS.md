# ClickHouse fast and cheap model benchmark

Phase: pilot. Attempts: 60. Complete: True.

Measured 2026-10-07T13:39:51.915894+00:00 through 2026-10-07T13:41:18.306006+00:00 from Falkenstein, Germany.

This run covers the five accessible models. Mistral Large 4, Qwen3.8 Flash remain unmeasured because OpenRouter's account-level zero-data-retention policy blocked their providers. They are not ranked here.

| Model | Correct | Correct ≤5s | Median total [IQR] | p95 total | USD / 1,000 tasks | USD / correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash | 10/12 (83.3%) | 83.3% | 0.945 [0.587, 2.158]s | 3.026s | $0.039 | $0.000046 |
| Granite 4.2 8B | 8/12 (66.7%) | 66.7% | 0.848 [0.541, 0.939]s | 1.120s | $0.039 | $0.000058 |
| GPT-6 Luna | 8/12 (66.7%) | 66.7% | 1.016 [0.934, 1.194]s | 1.822s | $0.141 | $0.000211 |
| Mercury 2.5 | 7/12 (58.3%) | 58.3% | 0.784 [0.650, 0.881]s | 1.253s | unknown | unknown |
| Solar Mini 4 | 6/12 (50.0%) | 41.7% | 2.850 [2.026, 3.278]s | 4.881s | $0.037 | $0.000074 |

Total delay runs from the OpenRouter request start through receipt and parsing of the ClickHouse result. Quantiles cover successful database executions, including incorrect answers. Errors and timeouts remain in accuracy and deadline denominators. Reported costs include failed attempts where billing is known; missing billing remains unknown.

The recommendation first keeps models within five percentage points of the best accuracy, then within five points of that group's best five-second success rate, and chooses the lowest cost per correct result within five seconds. This is a declared workload preference, not a universal ranking.

All models use their pinned provider and the lowest available reasoning mode (disabled here). Cached-token usage and promotional endpoint rates are recorded; prices are not permanent. The fixture and explicit output contracts differ from the earlier published benchmark, so scores are not directly comparable. The observations cover one client region and this measurement period.

Confidence intervals in summary.json and the figures resample questions with their repetitions together. The study has 60 distinct tasks, not 300 independent questions per model.

## Providers and generation speed

| Model | Pinned provider | Reported precision | Median first content | Median visible tokens/s | API errors | SQL errors |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash | `inference-net/fp8` | fp8 | 0.748s | 901.5 | 0 | 2 |
| Granite 4.2 8B | `coreweave/bf16` | bf16 | 0.317s | 139.6 | 0 | 4 |
| Mercury 2.5 | `inception` | unknown | 0.714s | 17432.6 | 2 | 2 |
| GPT-6 Luna | `openai/fast` | unknown | 0.720s | 185.9 | 0 | 4 |
| Solar Mini 4 | `upstage/zdr` | unknown | 1.491s | 46.1 | 0 | 4 |

Precision is reported by OpenRouter, not independently verified. Unknown means the provider does not disclose it. Tokens/s is a streaming estimate and is not directly comparable across tokenizers or chunking schemes; complete-result delay drives the recommendation.

## Billing evidence

| Model | Known billed USD | Attempts without cost | Cached input share | Uncached estimate / 1,000 tasks | Endpoint discount field |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash | $0.000465 | 0 | 82.8% | $0.081 | 0% |
| Granite 4.2 8B | $0.000465 | 0 | 89.5% | $0.062 | 0% |
| Mercury 2.5 | $0.000270 | 2 | 0.0% | $0.023 | 80% |
| GPT-6 Luna | $0.001692 | 0 | 0.0% | $0.141 | 0% |
| Solar Mini 4 | $0.000445 | 0 | 0.0% | $0.037 | 50% |

Uncached estimates apply the saved endpoint rates to returned token counts, without a cache discount. They cover known usage only and cannot resolve missing error billing. Endpoint promotions are already reflected in those saved rates; the discount field is shown separately and is not applied a second time. Infrastructure and screening/pilot costs are excluded from model cost comparisons.

See [methodology](../../METHODOLOGY.md), [complete summary](summary.json), [per-tier results](by-tier.csv), [per-question results](by-question.csv), [variation across repetitions](by-repetition.csv), and [individual failures](failures.csv).
