# DeepSeek Flash versions and quantizations

Phase: full. Attempts: 1500. Complete: True.

Measured 2026-10-07T15:04:03.746836+00:00 through 2026-10-07T15:47:24.764365+00:00 from Falkenstein, Germany.

Each row is a pinned model/provider/precision configuration. FP4 and FP8 use different providers; this experiment cannot attribute their differences solely to quantization. Both versions are also measured at FP8 on a shared provider, and the previous InferenceNet winner is rerun as a control.

Under the predeclared five-second / near-best-accuracy rule, the recommendation is **V4.1 · FP4 · Sail Research** on `sail-research/fp4`.

| Configuration | Correct | Correct ≤5s | Median total [IQR] | p95 total | USD / 1,000 tasks | USD / correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 · FP4 · Sail Research | 272/300 (90.7%) | 88.0% | 1.716 [1.418, 2.254]s | 3.743s | $0.028 | $0.000031 |
| V4.1 · FP8 · InferenceNet | 266/300 (88.7%) | 88.7% | 0.800 [0.630, 1.068]s | 1.823s | $0.043 | $0.000049 |
| V4.1 · FP8 · Parasail | 266/300 (88.7%) | 85.3% | 0.786 [0.614, 1.047]s | 4.505s | unknown | unknown |
| V4 · FP4 · Relace | 263/300 (87.7%) | 83.3% | 1.757 [1.188, 2.899]s | 4.982s | $0.084 | $0.000096 |
| V4 · FP8 · Parasail | 262/300 (87.3%) | 85.7% | 1.067 [0.896, 1.385]s | 2.280s | $0.062 | $0.000071 |

Total delay runs from the OpenRouter request start through receipt and parsing of the ClickHouse result. Quantiles cover successful database executions, including incorrect answers. Errors and timeouts remain in accuracy and deadline denominators. Reported costs include failed attempts where billing is known; missing billing remains unknown.

The recommendation first keeps models within five percentage points of the best accuracy, then within five points of that group's best five-second success rate, and chooses the lowest cost per correct result within five seconds. This is a declared workload preference, not a universal ranking.

All models use their pinned provider and the lowest available reasoning mode (disabled here). Cached-token usage and promotional endpoint rates are recorded; prices are not permanent. The fixture and explicit output contracts differ from the earlier published benchmark, so scores are not directly comparable. The observations cover one client region and this measurement period.

Confidence intervals in summary.json and the figures resample questions with their repetitions together. The study has 60 distinct tasks, not 300 independent questions per model.

## Paired accuracy comparisons

| Contrast (left minus right) | Difference | 95% question-bootstrap interval |
| --- | ---: | ---: |
| V4.1 FP4 route minus original FP8 route; provider confounded | +2.00 pp | [-1.00, +6.00] pp |
| V4 FP4 route minus FP8 route; provider confounded | +0.33 pp | [-4.00, +4.67] pp |
| V4.1 minus V4 at the shared FP8 provider | +1.33 pp | [-5.33, +8.33] pp |
| Shared provider minus InferenceNet, both V4.1 FP8 | +0.00 pp | [-2.00, +2.67] pp |

An interval spanning zero does not establish an accuracy advantage. These are descriptive comparisons on this task set, without adjustment for multiple comparisons or provider selection. FP4/FP8 contrasts also change provider. See [paired accuracy and deadline comparisons](focused-comparisons.csv).

## Providers and generation speed

| Configuration | Pinned provider | Reported precision | Median first content | Median visible tokens/s | API errors | SQL errors |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| V4 · FP8 · Parasail | `parasail/fp8` | fp8 | 0.675s | 162.9 | 0 | 22 |
| V4 · FP4 · Relace | `relace/fp4` | fp4 | 1.121s | 88.3 | 0 | 28 |
| V4.1 · FP8 · InferenceNet | `inference-net/fp8` | fp8 | 0.536s | 314.1 | 0 | 25 |
| V4.1 · FP8 · Parasail | `parasail/fp8` | fp8 | 0.521s | 349.7 | 1 | 21 |
| V4.1 · FP4 · Sail Research | `sail-research/fp4` | fp4 | 1.042s | 82.3 | 0 | 24 |

Precision is reported by OpenRouter, not independently verified. Unknown means the provider does not disclose it. Tokens/s is a streaming estimate and is not directly comparable across tokenizers or chunking schemes; complete-result delay drives the recommendation.

## Billing evidence

| Configuration | Known billed USD | Attempts without cost | Cached input share | Uncached estimate / 1,000 tasks | Endpoint discount field |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4 · FP8 · Parasail | $0.018610 | 0 | 53.6% | $0.079 | 0% |
| V4 · FP4 · Relace | $0.025181 | 0 | 54.9% | $0.084 | 0% |
| V4.1 · FP8 · InferenceNet | $0.012952 | 0 | 79.0% | $0.083 | 0% |
| V4.1 · FP8 · Parasail | $0.027723 | 1 | 83.5% | $0.205 | 0% |
| V4.1 · FP4 · Sail Research | $0.008464 | 0 | 93.2% | $0.058 | 0% |

Uncached estimates apply the saved endpoint rates to returned token counts, without a cache discount. They cover known usage only and cannot resolve missing error billing. Endpoint promotions are already reflected in those saved rates; the discount field is shown separately and is not applied a second time. Infrastructure and screening/pilot costs are excluded from model cost comparisons.

See [methodology](../../README.md), [complete summary](summary.json), [per-tier results](by-tier.csv), [per-question results](by-question.csv), [variation across repetitions](by-repetition.csv), and [individual failures](failures.csv).

The recommended configuration answered 150/150 attempts correctly across tiers 1–3, but only 31/50 in tier 6. Its lead does not make advanced ClickHouse SQL reliable without validation.

## Figures

![Accuracy versus total delay](accuracy-vs-delay.png)

![Accuracy versus billed cost per correct result](accuracy-vs-cost.png)

![Correct results within response deadlines](correct-within-deadline.png)
