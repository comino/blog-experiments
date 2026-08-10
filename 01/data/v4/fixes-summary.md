# Exp01 Compression — Fix Results (v4)

## Fix 1: V5 DDL vs Prose Contradiction

**Actual codecs on `v5_aggressive`:**

| Column | Codec |
|--------|-------|
| timestamp | `CODEC(DoubleDelta, ZSTD(9))` |
| metric_name | `CODEC(ZSTD(9))` |
| **value** | **`CODEC(Gorilla(8), ZSTD(3))`** |
| host | `CODEC(ZSTD(9))` |
| region | `CODEC(ZSTD(9))` |
| counter | `CODEC(Delta(8), ZSTD(9))` |

**Verdict:** The `value` column uses `ZSTD(3)`, not `ZSTD(9)`. If the DDL in the blog says ZSTD(9) for value, the DDL is wrong. If the prose says ZSTD(9) for value, the prose is wrong. The actual DB has **ZSTD(3)** on value.

## Fix 2: Gorilla LZ4-Only Control

**No new tables needed!** The existing `dist_*_v1` tables already serve as LZ4-only controls (default codec = LZ4). The `dist_*_v3` tables use `Gorilla(8)+LZ4`.

### Compression comparison (value column only):

| Distribution | LZ4 (v1) | Gorilla+LZ4 (v3) | ZSTD(3) (v2) | Gorilla+ZSTD(3) (v4) |
|---|---|---|---|---|
| **monotone** | 114.79 MiB (1.74×) | 4.48 MiB (44.66×) | 43.57 MiB (4.60×) | 944.53 KiB (217.15×) |
| **spiky** | 76.97 MiB (1.62×) | 4.05 MiB (30.80×) | 35.23 MiB (3.54×) | 415.09 KiB (308.08×) |
| **sinus+noise** | 153.20 MiB (1.31×) | 80.73 MiB (2.48×) | 107.62 MiB (1.86×) | 72.73 MiB (2.75×) |
| **random** | 191.63 MiB (1.05×) | 155.54 MiB (1.29×) | 177.90 MiB (1.13×) | 155.20 MiB (1.29×) |

**Key insight:** Gorilla dominates on monotone/spiky (25-190× better than LZ4 alone). On random data, Gorilla adds only marginal improvement (~23% better). On sinus+noise, Gorilla helps moderately (1.9× better than LZ4 alone).

## Fix 3: V3 Per-Column Breakdown

| Column | Compressed | Uncompressed | Ratio |
|--------|-----------|-------------|-------|
| counter | 642.35 MiB | 2.14 GiB | 3.42 |
| host | 642.35 MiB | 2.14 GiB | 3.42 |
| metric_name | 642.35 MiB | 2.14 GiB | 3.42 |
| region | 642.35 MiB | 2.14 GiB | 3.42 |
| timestamp | 642.35 MiB | 2.14 GiB | 3.42 |
| value | 642.35 MiB | 2.14 GiB | 3.42 |

**⚠️ Suspicious:** All columns show identical sizes (642.35 MiB compressed, 3.42× ratio). This likely means the table was created without per-column codecs — all columns use the same default LZ4 compression. The "percolumn" in the name may be misleading, or the table wasn't set up correctly for per-column codec testing.

## Fix 4: V4 vs V5 Exact Bytes

| Table | Total Compressed Bytes |
|-------|----------------------|
| v4_percolumn_zstd | 3,260,425,146 |
| v5_aggressive | 3,260,284,080 |

**Difference:** 141,066 bytes (0.004%) — v5_aggressive is marginally smaller.
**Verdict:** Nearly identical. The aggressive per-column codec strategy (v5) provides negligible additional compression over uniform ZSTD (v4) for this dataset.
