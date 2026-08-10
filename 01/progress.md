# Experiment 01: Progress

## Status: COMPLETE
- **Started:** 2026-02-14 15:38 UTC
- **Original complete:** ~15:50 UTC
- **Extension complete:** ~16:30 UTC

## Completed Phases
- [x] Original experiment (5 variants, 100M rows, 3 queries)
- [x] Scaling analysis (1M, 10M, 100M, 500M × 5 variants)
  - [x] Source tables created
  - [x] Variant tables created + loaded
  - [x] Query benchmark (6 queries × 4 sizes × 5 variants × 5 runs × cold/warm)
  - [x] Ingest benchmark (1M, 10M, 100M × 5 variants × 3 runs)
- [x] Distribution analysis (monotone, sinus, spiky, random × 5 variants)
  - [x] Source tables created
  - [x] Variant tables loaded + optimized
  - [x] Storage collected
  - [x] Query benchmark (DQ1, DQ2 × 4 dists × 5 variants × 3 runs × cold/warm)
- [x] Extended queries (Q1-Q6)
- [x] CPU profiling (from query_log ProfileEvents)
- [x] Visualizations (7 plots)
- [x] Combined CSVs (scaling.csv, distributions.csv)

## Output Files
- `data/scaling.csv` — Combined scaling results (20 rows)
- `data/distributions.csv` — Combined distribution results (20 rows)
- `data/scaling_storage.csv` — Raw per-column storage data
- `data/scaling_queries.csv` — Raw query benchmark data (1198 measurements)
- `data/scaling_ingest.csv` — Raw ingest measurements (45 runs)
- `plots/scaling_storage.png` — Storage by dataset size (log scale)
- `plots/scaling_queries.png` — Query latency by dataset size (6 subplots)
- `plots/scaling_ingest.png` — Ingest throughput by dataset size
- `plots/scaling_cpu.png` — CPU time by dataset size (6 subplots)
- `plots/distribution_storage_heatmap.png` — Storage heatmap (4 dists × 5 variants)
- `plots/distribution_dq1_cold_heatmap.png` — DQ1 query latency heatmap
- `plots/distribution_dq2_cold_heatmap.png` — DQ2 query latency heatmap

## Notes
- 500M ingest benchmark skipped (too slow for meaningful comparison)
- 500M tables not OPTIMIZE FINAL'd (CH killed long merges); auto-merged
- ZSTD(9) vs ZSTD(3) difference negligible across all sizes
