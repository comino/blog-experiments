# Exp03 LLM — Fix Results (v2)

## Fix 7: medianExactIf Existence

**`medianExact` exists** as a base aggregate function. The `-If` combinator works with it:
```sql
SELECT medianExactIf(number, number > 5) FROM numbers(10)  -- returns 8 ✓
```

**Available median functions:**
- median, medianExact, medianExactHigh, medianExactLow, medianExactWeighted, medianExactWeightedInterpolated
- medianTDigest, medianTDigestWeighted, medianBFloat16, medianBFloat16Weighted
- medianTiming, medianTimingWeighted, medianDD, medianDeterministic, medianGK, medianInterpolatedWeighted

**Verdict:** `medianExactIf` is valid ClickHouse SQL (aggregate combinator pattern). No issue here.

## Fix 8: arrayExists Validity

```sql
SELECT arrayExists(x -> startsWith(x, 'vip'), ['vip_user', 'normal', 'vip_admin'])
-- Result: 1 ✓
```

**Verdict:** `arrayExists` with lambda works correctly. The LLM-generated query is valid.
