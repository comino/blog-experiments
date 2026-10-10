# Interpretation of the DeepSeek extension

The main run completed 1,500 requests between 15:04 and 15:47 UTC on 7 October 2026. Each of five pinned configurations answered the same sixty questions five times. [Results](results/full-analysis/RESULTS.md), [raw attempts](results/full.jsonl), and the [audit](results/audit.json) retain the evidence. Screening and pilot calls are separate.

## Two useful choices

**DeepSeek V4.1 Flash on Sail Research FP4 wins the predeclared five-second cost rule.** It answered 272/300 correctly (90.7%), with 264/300 correct within five seconds (88.0%). Median complete-result delay was 1.716 seconds and p95 was 3.743 seconds. Observed inference cost was $0.0282 per 1,000 scheduled queries, or $0.0321 per 1,000 correct results delivered within five seconds, counting the cost of unsuccessful attempts.

**V4.1 Flash on InferenceNet FP8 is the stronger choice when interactive delay matters most.** It answered 266/300 correctly (88.7%), all within five seconds, at a 0.800-second median and 1.823-second p95. It delivered correct answers within two seconds on 85.3% of all attempts, compared with Sail's 60.0%. Its observed cost was $0.0432 per 1,000 scheduled queries. The extra cost over Sail is approximately $0.015 per 1,000 queries on this workload.

Sail's observed accuracy lead is only two percentage points. Its paired 95% question-bootstrap interval is −1 to +6 points. Its five-second success difference from InferenceNet is −0.67 points, with an interval of −4.33 to +3.33. These sixty tasks do not establish that Sail is more accurate. The choice under the declared rule rests on cost while remaining close on accuracy and five-second success.

The InferenceNet control scored 88.0% in the preceding five-model study and 88.7% here. Its median moved from 0.672 to 0.800 seconds and its cost from $0.0413 to $0.0432 per 1,000. Use the concurrent control for this comparison; the earlier run had another measurement window and server, even though the fixture and references match.

## What V4 and the other provider add

V4 Flash on Parasail FP8 reached 87.3% accuracy at a 1.067-second median and $0.0620 per 1,000 queries. V4 on Relace FP4 reached 87.7% at 1.757 seconds and $0.0839. Neither improves the observed speed, cost, or accuracy of the concurrent V4.1 InferenceNet route. The V4 FP4 route is slower and more expensive than V4 FP8 in this comparison; lower bit width alone does not imply a better API service.

V4.1 on Parasail FP8 tied InferenceNet's 88.7% accuracy and had a slightly lower median of 0.786 seconds. Its p95 was 4.505 seconds, compared with InferenceNet's 1.823, and one request timed out after sixty seconds. Its 85.3% five-second success rate was lower than InferenceNet's 88.7%. Known charges already amount to $0.0924 per 1,000 attempts; exact cost remains unknown because the timeout returned neither usage nor a generation ID. Missing billing is not zero.

All four planned paired accuracy contrasts have intervals spanning zero. Even the shared-provider comparison, V4.1 versus V4 at Parasail FP8, cannot establish an accuracy improvement here: +1.33 points, interval −5.33 to +8.33. [Focused comparisons](results/full-analysis/focused-comparisons.csv) also include deadline success.

## Cache and precision limits

Sail cached 93.2% of input tokens and InferenceNet 79.0%. Applying uncached endpoint rates to the returned token counts gives $0.0582 and $0.0832 per 1,000 attempts respectively. Those are price estimates, not separate uncached performance measurements. Repeated questions and a common schema make this a cache-friendly workload. All selected endpoint discount fields were zero at capture.

FP4 and FP8 are OpenRouter's provider-reported labels. No selected provider offers both formats for the same model version, so format is confounded with provider hardware and serving behavior. The official V4 checkpoint itself uses mixed FP4/FP8 parameters, as discussed in the [methodology](README.md). Do not describe these results as evidence that FP4 intrinsically improves accuracy or that an FP8 label establishes uncompressed source weights.

The run disabled reasoning, observed zero reported reasoning tokens, and pinned providers without fallbacks. Returned models and providers matched when exposed. Endpoint metadata stayed stable across all five blocks. All 1,499 completed model responses omitted the returned service-tier field: the requested tier is recorded, but actual tier cannot be independently confirmed. This is one client region and one measurement period.

## Failure examples for the follow-up

Sail answered all 150 attempts in tiers 1–3 correctly, but only 31/50 in tier 6. Every configuration failed all five `t6_01` attempts; the saved ClickHouse errors show `DateTime64` passed directly to `windowFunnel`. All V4.1 routes failed `t6_03` five times, including the timestamp type error for `sequenceMatch`.

Much of Sail's small advantage comes from individual advanced tasks. On `t6_06`, it returned all 31 requested January days in all five attempts. InferenceNet succeeded once; one saved failure used the exclusive fill endpoint January 31 and omitted that final day. Conversely, on `t5_09`, Sail succeeded once and InferenceNet three times. Aggregate accuracy conceals these task-specific differences. The scorer compares results on this fixture; it does not prove equivalence on arbitrary data or enforce every requested function name.

Known charges across screening, pilot, and the main run total **$0.0997574974**. The key-wide usage increase agrees to within $0.000000003, but does not identify the cost of individual failed requests. Nineteen screening failures and one main-run timeout still have unknown per-request billing. The [artifact checksums](artifacts-sha256.json) describe the original backup. Account-wide billing and machine-cleanup records are excluded from this public snapshot.
