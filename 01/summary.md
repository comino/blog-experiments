# Experiment 01: Compression Shootout — Results

**Date:** 2026-02-14  
**ClickHouse:** 25.11.3.54  
**Server:** CX53 (16 vCPU, 30GB RAM), Hetzner  
**Data:** 100M rows × 6 columns, simulated Prometheus-style time series  
**Settings:** `max_threads=auto(16)`, `index_granularity=8192`  
**ORDER BY:** `(metric_name, host, timestamp)` for all variants

## Variants

| Variant | Codec Strategy |
|---------|---------------|
| V1 | Default (LZ4) |
| V2 | ZSTD(3) on all columns |
| V3 | Per-column: DoubleDelta+LZ4 (ts), Delta+ZSTD (counter), Gorilla+LZ4 (value), LZ4 (tags) |
| V4 | Per-column: DoubleDelta+ZSTD(3) (ts), Delta+ZSTD(3) (counter), Gorilla+ZSTD(3) (value), ZSTD(3) (tags) |
| V5 | Aggressive: DoubleDelta+ZSTD(9) (ts), Delta+ZSTD(9) (counter), Gorilla+ZSTD(3) (value), ZSTD(9) (tags) |

## Storage Results

### Total Table Size (compressed)

| Variant | Total Compressed | Compression Ratio | vs V1 |
|---------|----------------:|------------------:|------:|
| V1 (LZ4) | 1,774 MB | 1.31x | 1.00x |
| V2 (ZSTD) | 864 MB | 2.69x | 2.05x |
| V3 (per-col LZ4) | 673 MB | 3.46x | 2.64x |
| V4 (per-col ZSTD) | 543 MB | 4.28x | 3.27x |
| V5 (aggressive) | 543 MB | 4.28x | 3.27x |

### Per-Column Compression Ratios

| Column | Type | V1 | V2 | V3 | V4 | V5 |
|--------|------|---:|---:|---:|---:|---:|
| **timestamp** | DateTime | 1.00x | 1.40x | **838.80x** | **872.35x** | **872.35x** |
| **counter** | UInt64 | 1.90x | 5.77x | 20.66x | **843.18x** | **848.65x** |
| **value** | Float64 | 1.45x | 1.82x | 1.26x | 1.48x | 1.48x |
| **metric_name** | LC(String) | 207.89x | 908.46x | 204.61x | 907.66x | 952.61x |
| **host** | LC(String) | 207.44x | 897.90x | 204.06x | 897.64x | 945.58x |
| **region** | LC(String) | 206.49x | 868.47x | 202.93x | 866.81x | 918.44x |

**Key findings:**
- DoubleDelta achieves **838–872x** compression on sorted timestamps (from 400MB → 459KB!)
- Delta+ZSTD on monotone counters: **843–849x** (from 800MB → 943KB)
- Gorilla codec on Float64 gauge values actually **hurts** (1.26x vs 1.45x for plain LZ4) — the simulated data has too much noise for Gorilla
- LowCardinality strings are already well-compressed (~207x with LZ4), ZSTD pushes to ~900x

## Query Performance (median, ms)

| Query | V1 | V2 | V3 | V4 | V5 |
|-------|---:|---:|---:|---:|---:|
| **Q1** range+agg (cold) | **57** | 87 | 94 | 119 | 115 |
| **Q1** range+agg (warm) | **53** | 84 | 99 | 119 | 119 |
| **Q2** top-k (cold) | 141 | 192 | 247 | **100** | 104 |
| **Q2** top-k (warm) | 138 | 190 | 253 | **106** | 113 |
| **Q3** wide scan (cold) | **37** | 62 | 93 | 86 | 77 |
| **Q3** wide scan (warm) | **38** | 66 | 95 | 77 | 80 |

**Key findings:**
- **V1 (LZ4) is fastest for Q1 and Q3** — decompression overhead of specialized codecs exceeds I/O savings at this data size
- **V4/V5 win Q2 dramatically** (100ms vs 247ms for V3!) — Delta+ZSTD on counter column makes `sum(counter)` much faster due to tiny compressed size
- V3 is **worst for Q2** — Delta+ZSTD(1) without proper ZSTD level leaves counter at only 20x compression vs 843x for V4
- Cold vs warm difference is minimal — data fits in page cache

### CPU Time (median, µs)

| Query | V1 | V2 | V3 | V4 | V5 |
|-------|---:|---:|---:|---:|---:|
| Q1 cold | 526K | 896K | 1,049K | 1,234K | 1,284K |
| Q2 cold | 1,692K | 2,307K | 3,058K | 1,191K | 1,196K |
| Q3 cold | 358K | 692K | 965K | 883K | 832K |

- Decompression CPU cost: ZSTD uses ~1.7x more CPU than LZ4
- Gorilla+LZ4 (V3) is expensive: 2.7x more CPU than plain LZ4 for Q3

## Ingest Throughput (10M rows, median)

| Variant | Rows/s | vs V1 |
|---------|-------:|------:|
| V1 (LZ4) | **8,116K** | 1.00x |
| V2 (ZSTD) | 5,989K | 0.74x |
| V3 (per-col LZ4) | 6,502K | 0.80x |
| V4 (per-col ZSTD) | 6,494K | 0.80x |
| V5 (aggressive) | 6,297K | 0.78x |

- LZ4 is **26% faster for ingest** than ZSTD(3)
- Per-column codecs cost ~20% ingest performance vs plain LZ4
- V5 (ZSTD(9)) is only marginally slower than V4 (ZSTD(3)) — the specialized pre-codecs dominate

## Hypothesis Validation

| # | Hypothesis | Result |
|---|-----------|--------|
| H1 | DoubleDelta+LZ4 for timestamps >50x compression | ✅ **839x** — far exceeded expectations |
| H2 | Gorilla+LZ4 for gauge floats 3-5x better than LZ4 | ❌ **0.87x** — Gorilla is *worse* on noisy float data |
| H3 | Delta+ZSTD for counters >100x compression | ✅ **843x** (V4) — but only with ZSTD(3)+, not ZSTD(1) |
| H4 | ZSTD 10-20% slower ingest, 20-40% less storage | ✅ **26% slower ingest**, **51% less storage** (V2 vs V1) |
| H5 | Per-column tuning beats global ZSTD | ✅ V4 is **37% smaller** than V2 and **faster on Q2** |

## Recommendation Matrix

| Column Type | Best Codec | Why |
|------------|-----------|-----|
| **Timestamp (sorted)** | `DoubleDelta, ZSTD(3)` | 872x compression, negligible size |
| **Counter (monotone UInt64)** | `Delta, ZSTD(3)` | 843x compression; ZSTD(9) gives only marginal gains |
| **Gauge (noisy Float64)** | `LZ4` (default) | Gorilla hurts; ZSTD(3) gives modest 1.48x but costs CPU |
| **LowCardinality String** | `ZSTD(3)` | 4x better than LZ4 (900x vs 207x) |
| **Overall best** | **V4** | Best storage (tied with V5), fastest Q2, acceptable Q1/Q3 |

### When to Pick What

| Priority | Recommendation |
|----------|---------------|
| **Max ingest speed** | V1 (default LZ4) — 8.1M rows/s |
| **Best compression** | V4 or V5 — 3.27x better than V1 |
| **Fastest queries** | V1 for scan-heavy, V4/V5 for aggregation on compressed columns |
| **Best all-around** | **V4 (per-column + ZSTD(3))** — best storage, best Q2, 20% slower ingest |

### Surprise Finding: Gorilla Codec

Gorilla encoding is designed for IEEE 754 floats with small deltas between consecutive values (XOR-based). Our gauge data (`sin(x) * 50 + 50 + noise`) has too much variation between consecutive values after sorting by `(metric_name, host, timestamp)`, making Gorilla produce *larger* output than raw LZ4. **Only use Gorilla when consecutive float values have very small deltas** (e.g., temperature readings with 0.01° changes).

### Surprise Finding: V4 ≈ V5 for Storage

ZSTD(9) vs ZSTD(3) made virtually no difference (543MB vs 543MB). The pre-codecs (DoubleDelta, Delta) do the heavy lifting — ZSTD just handles the residual. **Don't bother with ZSTD levels above 3 for time series.**

---

## Reproducibility

```
results/01/
├── scripts/
│   ├── benchmark.sh          ← Main benchmark runner (query + ingest)
│   ├── load_variants.sh      ← Load data into all variant tables
│   └── heatmap.py            ← Generate visualization
├── sql/
│   ├── 01_create_tables.sql  ← CREATE DATABASE + all 6 tables
│   ├── 02_generate_data.sql  ← INSERT 100M rows into source
│   ├── 03_load_variants.sql  ← Load + OPTIMIZE all variants
│   ├── 04_measure_storage.sql ← Query system.columns for storage stats
│   ├── 05_benchmark_queries.sql ← The 3 benchmark queries (template)
│   ├── 06_collect_query_results.sql ← Extract from system.query_log
│   └── 07_query_summary.sql  ← Median/IQR aggregation
├── data/
│   ├── storage.csv           ← Per-column compression data
│   ├── queries.csv           ← 150 query benchmark measurements
│   └── ingest.csv            ← 15 ingest measurements
├── plots/
│   └── heatmap.png           ← Compression + query latency heatmaps
├── progress.md
└── summary.md                ← This file
```

---

## Extended Results: Scaling Analysis

**Sizes:** 1M, 10M, 100M, 500M rows × 5 codec variants  
**Queries:** Q1 (range 1h), Q2 (range 7d), Q3 (top-K), Q4 (point lookup), Q5 (full scan w/ filter), Q6 (multi-column agg)

### Compression Ratio Scales Linearly

| Size | V1 | V2 | V3 | V4 | V5 |
|------|---:|---:|---:|---:|---:|
| 1M | 1.67x | 2.62x | 3.41x | 4.21x | 4.22x |
| 10M | 1.67x | 2.66x | 3.42x | 4.23x | 4.23x |
| 100M | 1.67x | 2.66x | 3.42x | 4.23x | 4.23x |
| 500M | 1.67x | 2.66x | 3.42x | 4.23x | 4.23x |

**Key finding:** Compression ratios are **virtually constant** across all dataset sizes. The codec overhead is per-block, not per-dataset.

### Storage (total compressed, MB)

| Size | V1 (LZ4) | V2 (ZSTD) | V3 (per-col) | V4 (per-col ZSTD) | V5 (aggressive) |
|------|--------:|--------:|--------:|--------:|--------:|
| 1M | 14 | 9 | 7 | 5 | 5 |
| 10M | 137 | 87 | 67 | 54 | 54 |
| 100M | 1,374 | 865 | 674 | 543 | 543 |
| 500M | 6,872 | 4,323 | 3,367 | 2,717 | 2,717 |

### Ingest Throughput (median, M rows/s)

| Size | V1 | V2 | V3 | V4 | V5 |
|------|---:|---:|---:|---:|---:|
| 1M | 5.5 | 4.3 | 4.5 | 4.7 | 4.6 |
| 10M | 8.1 | 5.9 | 6.5 | 5.9 | 6.1 |
| 100M | 8.5 | 6.1 | 6.5 | 6.3 | 6.2 |

**Key finding:** LZ4 ingest advantage is consistent (~30% faster) across sizes.

---

## Extended Results: Distribution Analysis

**Distributions:** monotone, sinus, spiky, random × 5 variants, 10M rows each  
**Queries:** DQ1 (7-day range agg), DQ2 (full scan agg)

### Storage by Distribution (total compressed, MB)

| Distribution | V1 | V2 | V3 | V4 | V5 |
|-------------|---:|---:|---:|---:|---:|
| **Monotone** | 120 | 46 | 5 | **1** | **1** |
| **Sinus** | 161 | 113 | 85 | **76** | **76** |
| **Spiky** | 81 | 37 | 4 | **0.4** | **0.4** |
| **Random** | 201 | 187 | 163 | **163** | **163** |

**Key findings:**
- **Monotone data** (perfect for Delta): V4/V5 achieve **217x** compression (1 MB for 200 MB!) — Delta+ZSTD is ideal
- **Spiky data** (99% zeros): V4 achieves **308x** — the pre-codecs exploit the constant runs brilliantly
- **Sinus/noise**: Only 2.75x for V4 — Gorilla can't handle the random noise component
- **Random data**: Incompressible. Even V4/V5 only 1.29x. Gorilla makes it **worse** (0.97x)

### Query Performance by Distribution (median, ms)

| Distribution | DQ1: V1/V2/V3/V4/V5 | DQ2: V1/V2/V3/V4/V5 |
|-------------|---------------------|---------------------|
| **Monotone** | 9/9/9/7/9 | 29/37/38/**23**/**22** |
| **Sinus** | 8/9/9/10/10 | **19**/34/45/37/40 |
| **Spiky** | 7/8/8/7/7 | **18**/23/31/**12**/15 |
| **Random** | 9/**7**/13/11/11 | **23**/**12**/86/35/39 |

**Key findings:**
- On **monotone/spiky** data, V4 wins DQ2 decisively — tiny compressed size → less I/O
- On **sinus** (noisy float) data, V1 (plain LZ4) is fastest — decompression overhead outweighs I/O savings
- On **random** data, V2 (ZSTD) wins DQ2 — Gorilla (V3-V5) adds CPU overhead with no compression benefit
- V3 (Gorilla+LZ4) is **consistently worst** for DQ2 — worst of both worlds on non-ideal float data

### Distribution Lessons

| Data Pattern | Best Codec | Compression | Query Speed |
|-------------|-----------|------------|------------|
| Monotone counter | Delta+ZSTD(3) (V4) | **217x** | Fastest |
| Mostly-zero spiky | Delta+ZSTD(3) (V4) | **308x** | Fastest |
| Sinusoidal + noise | LZ4 (V1) or ZSTD(3) (V2) | 1.3-1.9x | V1 fastest |
| Random (incompressible) | ZSTD(3) (V2) | 1.1x | V2 fastest |

---

## Extended Queries (Q1-Q6) — Definition

| Query | Description |
|-------|-----------|
| Q1 | Range 1h: `WHERE timestamp BETWEEN ... AND ... GROUP BY toStartOfHour(timestamp)` |
| Q2 | Range 7d: Same as Q1 but 7-day window |
| Q3 | Top-K: `GROUP BY host ORDER BY sum(counter) DESC LIMIT 10` |
| Q4 | Point Lookup: `WHERE host = 'host-7' AND timestamp BETWEEN 12:00 AND 12:05` |
| Q5 | Full Scan w/ filter: `WHERE metric_name IN (...) AND value > 50 AND region = ...` |
| Q6 | Multi-Column: `GROUP BY host, region, metric_name` with 6 aggregations |

---

## Visualizations

See `plots/` directory:
- `scaling_storage.png` — Log-scale line chart of compressed size vs dataset size
- `scaling_queries.png` — 6 subplots of query latency vs dataset size
- `scaling_ingest.png` — Ingest throughput vs dataset size
- `scaling_cpu.png` — CPU time vs dataset size (from ProfileEvents)
- `distribution_storage_heatmap.png` — 4×5 heatmap of compressed size
- `distribution_dq1_cold_heatmap.png` — Query DQ1 latency heatmap
- `distribution_dq2_cold_heatmap.png` — Query DQ2 latency heatmap

---

### How to reproduce
```bash
# 1. Create tables
cat sql/01_create_tables.sql | ssh thesis-clickhouse "clickhouse-client --multiquery"

# 2. Generate 100M rows
ssh thesis-clickhouse "clickhouse-client --query \"$(cat sql/02_generate_data.sql)\""

# 3. Load + optimize all variants
scp scripts/load_variants.sh thesis-clickhouse:/tmp/
ssh thesis-clickhouse "bash /tmp/load_variants.sh"

# 4. Run benchmark
scp scripts/benchmark.sh thesis-clickhouse:/tmp/
ssh thesis-clickhouse "bash /tmp/benchmark.sh"

# 5. Collect results
ssh thesis-clickhouse "clickhouse-client --query \"$(cat sql/04_measure_storage.sql)\"" > data/storage.csv
ssh thesis-clickhouse "clickhouse-client --query \"$(cat sql/06_collect_query_results.sql)\"" > data/queries.csv

# 6. Generate plots
python3 scripts/heatmap.py
```
