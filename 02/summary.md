# Experiment 02: Projections vs Materialized Views — Summary

**ClickHouse Version:** 25.11.3.54  
**Dataset:** 200M web analytics rows (6 columns)  
**Server:** 16 vCPU, 30GB RAM (Hetzner CX53)  
**Date:** 2026-02-14

## Setup

| Table | Engine | Rows | Disk Size |
|-------|--------|------|-----------|
| web_analytics_base | MergeTree ORDER BY (page, timestamp) | 200M | 1.25 GiB |
| web_analytics_proj | MergeTree + 2 projections | 200M | 2.86 GiB |
| web_analytics_mv_source | MergeTree (feeds MV) | 200M | 1.25 GiB |
| hourly_stats_mv_target | AggregatingMergeTree | 8.76M | 103 MiB |

**Projections on `web_analytics_proj`:**
- `proj_country_time`: Re-sort by `(country, timestamp)` — all columns
- `proj_hourly_stats`: Aggregating — page × hour with count/avg/sum

## Correctness

✅ All queries return identical results across base, projection, and MV variants.

## Query Benchmark Results

### Q1: Dashboard Rollup (single page, single day, hourly aggregation)
Small result set, well-served by primary key (page prefix).

| Variant | Cold (median) | Warm (median) | Rows Read |
|---------|--------------|--------------|-----------|
| base | 32ms | 6ms | 8,192 |
| proj | 34ms | 7ms | 8,192 |
| mv | 12ms | 6ms | 16,384 |

**Finding:** No meaningful difference. Base table's ORDER BY (page, timestamp) already optimal for page-filtered queries. Projection not used (EXPLAIN confirms: reads from base sort). MV reads slightly more rows (pre-aggregated granules) but latency similar.

### Q2: Country Filter (WHERE country = 'DE', 1 month)
Tests re-sort projection benefit.

| Variant | Cold (median) | Warm (median) | Rows Read |
|---------|--------------|--------------|-----------|
| base | 41ms | 14ms | ~1M (varies) |
| proj | 38ms | 12ms | 352K |

**Finding:** Projection `proj_country_time` **is used** (EXPLAIN confirms: `ReadFromMergeTree (proj_country_time)`). Reads 3-4× fewer rows. Latency improvement modest (~15-20%) because even full scan is fast at this scale.

### Q3: Top-K Pages by avg duration (full table aggregation)
Tests aggregating projection benefit.

| Variant | Cold (median) | Warm (median) | Rows Read |
|---------|--------------|--------------|-----------|
| base | 441ms | 229ms | 200M |
| proj | 78ms | 82ms | 8.76M |
| mv | 75ms | 50ms | 8.76M |

**Finding:** **Massive win for both projection and MV.** Both read only 8.76M pre-aggregated rows instead of 200M. **5-6× faster cold, 3-5× faster warm.** EXPLAIN confirms projection `proj_hourly_stats` is used. MV slightly faster warm (AggregatingMergeTree optimized for merge operations).

## EXPLAIN Verification

| Query | Projection Used? | Details |
|-------|-----------------|---------|
| Q1 on proj | ❌ No | Base sort (page, timestamp) already optimal |
| Q2 on proj | ✅ Yes | `proj_country_time` — re-sort projection |
| Q3 on proj | ✅ Yes | `proj_hourly_stats` — aggregating projection |

## Storage Overhead

| Configuration | Disk Size | Overhead vs Base |
|---------------|-----------|-----------------|
| Base only | 1.25 GiB | — |
| Base + 2 projections | 2.86 GiB | **+129%** |
| Base + MV (source + target) | 1.25 GiB + 103 MiB | **+8%** |

**Key insight:** Projections store a complete copy of data (re-sorted or aggregated) within the same part. The re-sort projection alone roughly doubles storage. MV target table is much smaller because it only stores the aggregated result.

## Ingest Impact

10M rows per INSERT, 3 repetitions. ⚠️ Concurrent workload from Experiment 01 was active during this test, so absolute numbers are reduced but relative comparisons remain valid.

| Scenario | Median rows/s | vs Base Only |
|----------|--------------|-------------|
| Base only (no projections/MV) | 2,965K | — |
| Base + 2 projections | 730K | **-75%** |
| Base + MV | 878K | **-70%** |

**Key insight:** Both projections and MVs dramatically reduce ingest throughput. Projections are slightly worse than MV in this test because they maintain 2 additional sort orders. The re-sort projection (full data copy) is particularly expensive.

## Decision Tree

```
Need to accelerate queries on your ClickHouse table?
│
├─ Is the query pattern a different sort order (e.g., filter by non-key column)?
│   │
│   ├─ YES → Is storage a concern?
│   │   ├─ YES → Consider a separate table or MV (only stores needed columns)
│   │   └─ NO → **Use a re-sort PROJECTION**
│   │         ✓ Zero operational overhead (no separate table to manage)
│   │         ✓ Optimizer picks it automatically
│   │         ⚠ ~100% storage overhead per re-sort projection
│   │         ⚠ Significant ingest slowdown (~75%)
│   │
│   └─ NO → Continue below
│
├─ Is the query pattern a pre-aggregation (GROUP BY)?
│   │
│   ├─ Can the query be rewritten to use -State/-Merge functions?
│   │   ├─ YES → **Use a Materialized View** to AggregatingMergeTree
│   │   │     ✓ Minimal storage overhead (~8%)
│   │   │     ✓ Independent table = flexible schema
│   │   │     ✓ Can be queried directly with -Merge functions
│   │   │     ⚠ Requires -State/-Merge boilerplate
│   │   │     ⚠ Ingest slowdown (~70%)
│   │   │
│   │   └─ NO → **Use an aggregating PROJECTION**
│   │         ✓ Transparent to queries (no rewrite needed)
│   │         ✓ Optimizer uses it automatically
│   │         ⚠ More storage than MV (stored per-part)
│   │         ⚠ Ingest slowdown (~75%)
│   │
│   └─ Is the base table's ORDER BY already good enough?
│       └─ YES → Do nothing. ClickHouse is fast.
│
└─ Summary rules of thumb:
    • Projection = convenience (transparent, auto-selected) but costly storage
    • MV = efficiency (minimal storage, flexible) but requires query adaptation
    • Both hurt ingest equally (~70-75% reduction)
    • For read-heavy, write-light workloads → either works
    • For write-heavy workloads → minimize projections/MVs
    • Always verify with EXPLAIN that projections are actually used!
```

## Key Takeaways

1. **Projections work transparently** — the optimizer automatically selects them when beneficial. No query changes needed.
2. **MVs require query adaptation** (-State/-Merge functions) but offer better storage efficiency.
3. **Storage: Projections are expensive** (+129% for 2 projections) vs MV (+8% for equivalent aggregation).
4. **Ingest: Both hurt equally** (~70-75% throughput reduction).
5. **For full-table aggregation, both deliver 5-6× speedup** over base table scan.
6. **Always verify with EXPLAIN** — projections are not always used (Q1 example: base ORDER BY already optimal).
7. **The optimizer is smart** — it won't use a projection when the base sort order already serves the query well.

---

## Extended Results (2026-02-14)

### Extension 1: Scaling Analysis (1M → 200M rows)

**Q3 (Top-K full aggregation) — Median cold latency (seconds):**

| Size | Base | Projection | MV | Proj Speedup | MV Speedup |
|------|------|-----------|-----|-------------|------------|
| 1M | 0.019 | 0.014 | 0.028 | 1.4× | 0.7× |
| 10M | 0.019 | 0.073 | 0.077 | 0.3× | 0.2× |
| 50M | 0.135 | 0.083 | 0.075 | 1.6× | 1.8× |
| 200M | 0.447 | 0.103 | 0.042 | 4.3× | 10.6× |

**Q2 (Country filter) — Median cold latency (seconds):**

| Size | Base | Projection |
|------|------|-----------|
| 1M | 0.019 | 0.013 |
| 10M | 0.022 | 0.014 |
| 50M | 0.009 | 0.017 |
| 200M | 0.010 | 0.033 |

**Finding:** For full-table aggregation (Q3), projections and MVs show increasing benefit as data grows. At 200M rows, projection is 4× faster and MV is 10× faster than base. Below 10M, the overhead of maintaining separate structures isn't worth it — base table scans are already fast enough. For country filters (Q2), the re-sort projection doesn't consistently help because ClickHouse's parallel scan is already fast for filtered queries.

**Storage overhead scales linearly:**

| Size | Base (MiB) | Proj (MiB) | Overhead | MV Target (MiB) | Overhead |
|------|-----------|-----------|---------|-----------------|---------|
| 1M | 10 | 31 | +210% | 9.3 | -7% |
| 10M | 92 | 263 | +186% | 62 | -33% |
| 50M | 392 | 947 | +141% | 94 | -76% |
| 200M | 1,280 | 2,930 | +129% | 103 | -92% |

### Extension 2: Extended Query Patterns (8 queries, 200M rows)

All times are median of 5 cold runs (seconds):

| Query | Description | Base | Proj | MV | Proj Used? |
|-------|-----------|------|------|-----|-----------|
| Q1 | Dashboard rollup (1 day) | 0.007 | 0.009 | 0.006 | ❌ base sort optimal |
| Q2 | Dashboard rollup (30 days) | 0.007 | 0.010 | 0.007 | ❌ base sort optimal |
| Q3 | Country filter (exact) | 0.011 | 0.010 | — | ✅ proj_country_time |
| Q4 | Country + time range | 0.013 | 0.011 | — | ✅ proj_country_time |
| Q5 | Top-K by avg duration | 0.208 | 0.040 | 0.040 | ✅ proj_hourly_stats |
| Q6 | Top-K with HAVING | 0.183 | 0.045 | 0.046 | ✅ proj_hourly_stats |
| Q7 | Cardinality (uniqExact) | 0.485 | 0.519 | — | ❌ no matching proj |
| Q8 | Multi-dim GROUP BY | 1.540 | 1.438 | — | ❌ no matching proj |

**Key findings:**
- Q5/Q6: Aggregating projection provides **5× speedup** (reads pre-aggregated data). MV equally fast.
- Q3/Q4: Re-sort projection used but marginal benefit — both are already fast.
- Q7 (uniqExact): No projection can help — requires raw user_id values. Both ~0.5s.
- Q8 (multi-dimension): No matching projection. Full scan at 1.3-1.5s. Would need a dedicated projection.
- Q1/Q2: Base table's ORDER BY (page, timestamp) already optimal for page-filtered queries.

### Extension 3: Projection Count Scaling (10M rows)

| Projections | Ingest (rows/s) | vs Baseline | Storage (MiB) | vs Baseline |
|-------------|----------------|------------|--------------|------------|
| 0 | 3,087K | — | 92 | — |
| 1 (re-sort) | 1,965K | **-36%** | 201 | **+118%** |
| 3 (2 re-sort + 1 agg) | 538K | **-83%** | 382 | **+315%** |
| 5 (3 re-sort + 2 agg) | 386K | **-87%** | 470 | **+411%** |

**Finding:** Each additional re-sort projection roughly doubles the write cost (full copy of data in new sort order). Going from 0→1 projections costs -36% ingest. Going from 1→3 is catastrophic: -83%. At 5 projections, ingest drops to **1/8th** of baseline. Rule of thumb: **1-2 projections maximum** for write-heavy workloads.

### Extension 4: MV Freshness

10 batches of 100K rows inserted sequentially, queried immediately after each INSERT.

| Batch | Expected | Projection Count | MV Count | Both Correct? |
|-------|----------|-----------------|----------|--------------|
| 1 | 100K | 100,000 | 100,000 | ✅ |
| 5 | 500K | 500,000 | 500,000 | ✅ |
| 10 | 1M | 1,000,000 | 1,000,000 | ✅ |

**Both projections and MVs are immediately consistent after INSERT completes.** There is no lag. ClickHouse processes both synchronously during the INSERT operation. Average query time: ~5ms for both.

### Extension 5: EXPLAIN Analysis Summary

| Query | On Proj Table | Projection Used? | Details |
|-------|--------------|-----------------|---------|
| Q1 | web_analytics_proj | ❌ | Base sort (page, timestamp) already optimal |
| Q2 | web_analytics_proj | ❌ | Base sort (page, timestamp) already optimal |
| Q3 | proj_country_time | ✅ | Re-sort by (country, timestamp) — prewhere filter |
| Q4 | proj_country_time | ✅ | Re-sort by (country, timestamp) — prewhere filter |
| Q5 | proj_hourly_stats | ✅ | Aggregating projection — reads AggregateFunction states |
| Q6 | proj_hourly_stats | ✅ | Aggregating projection — reads AggregateFunction states |
| Q7 | web_analytics_proj | ❌ | uniqExact needs raw data, no projection matches |
| Q8 | web_analytics_proj | ❌ | Multi-dim GROUP BY, no projection matches |

**Optimizer behavior:** The query planner only uses a projection when it provides a strict advantage. For Q1/Q2, the base ORDER BY already enables efficient key-range reads. For Q7/Q8, no projection covers the required columns/aggregation. The optimizer correctly selects `proj_country_time` for country filters and `proj_hourly_stats` for page-level aggregations.

Full EXPLAIN outputs are in `data/explain_outputs/`.

### Updated Decision Guidelines

Based on the extended analysis:

1. **Projections pay off at ≥50M rows** for full-table aggregation queries. Below that, base scans are fast enough.
2. **MVs are more storage-efficient at every scale** — only store the aggregated result, not a full data copy.
3. **Limit projections to 1-2 per table** — each additional projection has severe ingest impact.
4. **Both are synchronously consistent** — no freshness concern for either approach.
5. **Projections only help when the optimizer can match them** — verify with EXPLAIN before assuming benefit.
6. **For cardinality queries (uniqExact) or multi-dimension GROUP BYs**, neither projections nor MVs help unless you create a dedicated one.
