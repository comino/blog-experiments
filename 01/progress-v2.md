# V2 Progress

## Status: Complete ✅

### Done
- [x] Read both reviews (GPT5 + Gemini)
- [x] Read both drafts + STYLE.md
- [x] Export DDL for all 5 exp01 variants
- [x] Export DDL for all exp02 tables
- [x] system.columns per-column compression data (confirms 872× and 843×)
- [x] Gorilla distribution data collected (explains 0.87× vs 1.26×)
- [x] EXPLAIN for exp01 Q1 (v1 + v4)
- [x] EXPLAIN for exp02 Q2 (country filter → proj_country_time used)
- [x] EXPLAIN for exp02 Q3 agg (→ proj_hourly_stats used)
- [x] MV target row count: 8,760,000
- [x] 10-run benchmarks for exp01 (Q1-Q3, all 5 variants, cold+warm)
- [x] 10-run benchmarks for exp02 (Q1-Q3, base/proj/mv, cold+warm)
- [x] Compute median/IQR/stddev from benchmark results
- [x] Write 01-compression-shootout-v2.md
- [x] Write 02-projections-vs-mvs-v2.md

### Not Done (would need more time)
- [ ] Ingest with various batch sizes (10K, 100K, 1M, 10M)
- [ ] _part_offset projection test
- [ ] 10M scaling anomaly: verify OPTIMIZE FINAL
- [ ] Fair 1-to-1 comparison: 1 aggregating projection vs 1 MV (separate tables)
