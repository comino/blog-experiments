# DeepSeek Flash versions and quantizations

Phase: pilot. Attempts: 60. Complete: True.

Measured 2026-10-07T15:00:22.731088+00:00 through 2026-10-07T15:02:54.052986+00:00 from Falkenstein, Germany.

Each row is a pinned model/provider/precision configuration. FP4 and FP8 use different providers; this experiment cannot attribute their differences solely to quantization. Both versions are also measured at FP8 on a shared provider, and the previous InferenceNet winner is rerun as a control.

| Configuration | Correct | Correct ≤5s | Median total [IQR] | p95 total | USD / 1,000 tasks | USD / correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 · FP8 · Parasail | 10/12 (83.3%) | 75.0% | 1.612 [1.454, 1.907]s | 7.930s | $0.063 | $0.000075 |
| V4 · FP4 · Relace | 10/12 (83.3%) | 75.0% | 2.662 [1.735, 3.350]s | 5.051s | $0.074 | $0.000089 |
| V4.1 · FP8 · InferenceNet | 10/12 (83.3%) | 83.3% | 0.973 [0.720, 1.112]s | 2.412s | $0.042 | $0.000050 |
| V4.1 · FP8 · Parasail | 10/12 (83.3%) | 66.7% | 1.268 [0.909, 1.644]s | 14.719s | $0.081 | $0.000097 |
| V4.1 · FP4 · Sail Research | 10/12 (83.3%) | 83.3% | 2.012 [1.743, 3.133]s | 4.910s | $0.035 | $0.000042 |

Total delay runs from the OpenRouter request start through receipt and parsing of the ClickHouse result. Quantiles cover successful database executions, including incorrect answers. Errors and timeouts remain in accuracy and deadline denominators. Reported costs include failed attempts where billing is known; missing billing remains unknown.

The recommendation first keeps models within five percentage points of the best accuracy, then within five points of that group's best five-second success rate, and chooses the lowest cost per correct result within five seconds. This is a declared workload preference, not a universal ranking.

All models use their pinned provider and the lowest available reasoning mode (disabled here). Cached-token usage and promotional endpoint rates are recorded; prices are not permanent. The fixture and explicit output contracts differ from the earlier published benchmark, so scores are not directly comparable. The observations cover one client region and this measurement period.

Confidence intervals in summary.json and the figures resample questions with their repetitions together. The study has 60 distinct tasks, not 300 independent questions per model.

## Providers and generation speed

| Configuration | Pinned provider | Reported precision | Median first content | Median visible tokens/s | API errors | SQL errors |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| V4 · FP8 · Parasail | `parasail/fp8` | fp8 | 0.987s | 93.7 | 0 | 2 |
| V4 · FP4 · Relace | `relace/fp4` | fp4 | 1.250s | 55.9 | 0 | 2 |
| V4.1 · FP8 · InferenceNet | `inference-net/fp8` | fp8 | 0.792s | 255.2 | 0 | 2 |
| V4.1 · FP8 · Parasail | `parasail/fp8` | fp8 | 0.969s | 243.7 | 0 | 2 |
| V4.1 · FP4 · Sail Research | `sail-research/fp4` | fp4 | 1.204s | 72.9 | 0 | 2 |

Precision is reported by OpenRouter, not independently verified. Unknown means the provider does not disclose it. Tokens/s is a streaming estimate and is not directly comparable across tokenizers or chunking schemes; complete-result delay drives the recommendation.

## Billing evidence

| Configuration | Known billed USD | Attempts without cost | Cached input share | Uncached estimate / 1,000 tasks | Endpoint discount field |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4 · FP8 · Parasail | $0.000752 | 0 | 46.1% | $0.078 | 0% |
| V4 · FP4 · Relace | $0.000894 | 0 | 55.3% | $0.074 | 0% |
| V4.1 · FP8 · InferenceNet | $0.000504 | 0 | 75.9% | $0.081 | 0% |
| V4.1 · FP8 · Parasail | $0.000967 | 0 | 82.8% | $0.193 | 0% |
| V4.1 · FP4 · Sail Research | $0.000416 | 0 | 63.5% | $0.055 | 0% |

Uncached estimates apply the saved endpoint rates to returned token counts, without a cache discount. They cover known usage only and cannot resolve missing error billing. Endpoint promotions are already reflected in those saved rates; the discount field is shown separately and is not applied a second time. Infrastructure and screening/pilot costs are excluded from model cost comparisons.

See [methodology](../../README.md), [complete summary](summary.json), [per-tier results](by-tier.csv), [per-question results](by-question.csv), [variation across repetitions](by-repetition.csv), and [individual failures](failures.csv).
