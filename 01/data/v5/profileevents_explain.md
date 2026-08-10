# ProfileEvents read_rows ≈ 40M Explanation (Q05)

## Q05 Definition (as in draft/benchmark)

```sql
SELECT count() FROM <variant> WHERE value > 60
```

Note: The extended benchmark scripts use a different Q5 (`WHERE metric_name IN (...) AND value > 50 AND region = 'eu-central'`). The ProfileEvents analysis here uses the draft's Q05 definition.

## Why read_rows ≈ 40M (not 100M)

ClickHouse automatically applies **PREWHERE** optimization: `WHERE value > 60` is moved to a PREWHERE clause. With PREWHERE, ClickHouse reads the `value` column first and uses per-granule min-max statistics to skip granules where all values are ≤ 60.

### Evidence

1. **With automatic PREWHERE (default):** read_rows ≈ 40M, ~4889 marks selected
2. **With `PREWHERE 1 WHERE value > 60` (PREWHERE disabled):** read_rows = 100M, 12208 marks
3. **With old analyzer + `optimize_move_to_prewhere=0`:** read_rows = 100M, 12208 marks

### Granule-level analysis

The data has `value = 50 + 30*sin(number/1000) + noise(0, 0.5)`. After sorting by `(metric_name, host, timestamp)`, within each sorted group the sin wave creates regions where all values in a granule fall below 60. Of 12,208 total granules, approximately 5,327 contain at least one row with `value > 60`, while ~6,881 can be skipped entirely.

ClickHouse's PREWHERE reads the `value` column data for candidate granules and skips those where the condition cannot be satisfied, based on per-granule min-max statistics stored in the mark file.

The `read_rows` metric in `system.query_log` with PREWHERE reflects the rows from selected granules, not total rows scanned.

### Actual row counts

| Metric | Value |
|--------|-------|
| Total rows | 100,000,000 |
| Rows matching `value > 60` | 17,565,454 (17.6%) |
| Granules with ≥1 matching row | ~5,327 of 12,208 |
| `read_rows` (PREWHERE active) | ~40,050,688 (~4,889 granules × 8,192) |
| `read_rows` (PREWHERE disabled) | 100,000,000 (all 12,208 granules) |

## ProfileEvents: V1 vs V4 (warm, `WHERE value > 60`)

| Metric | V1 (LZ4) | V4 (per-col+ZSTD) | Ratio |
|--------|----------:|-------------------:|------:|
| Wall clock (ms) | 16 | 95–102 | ~6× |
| CPU virtual (µs) | 170K | 1,310K | **7.7×** |
| Disk read (µs) | 67K | 53K | 0.8× |
| read_rows | 40M | 40M | 1.0× |
| read_bytes | 320 MB | 320 MB | 1.0× |
| Selected marks | ~4,890 | ~4,895 | 1.0× |

*V4 first warm run reads 100M rows (PREWHERE statistics not yet cached); subsequent runs read ~40M. V1 median of 3 runs; V4 median of runs 2–3 (excluding cold first run).*

### EXPLAIN Output

#### V1 (LZ4 default)
```
ReadFromMergeTree (exp01_compression.v1_default)
  Indexes:
    PrimaryKey
      Condition: true
      Parts: 1/1
      Granules: 12208/12208
```

#### V4 (per-column ZSTD)
```
ReadFromMergeTree (exp01_compression.v4_percolumn_zstd)
  Indexes:
    PrimaryKey
      Condition: true
      Parts: 1/1
      Granules: 12208/12208
```

Note: EXPLAIN shows all 12,208 granules because PREWHERE filtering happens at execution time, not at plan time. The actual granule skip count is visible only in ProfileEvents.

## Conclusion

The `read_rows ≈ 40M` is caused by ClickHouse's automatic PREWHERE optimization, which moves `WHERE value > 60` to PREWHERE and uses per-granule min-max statistics to skip ~60% of granules where all values are ≤ 60. Both V1 and V4 read comparable data volumes after PREWHERE; the performance difference is predominantly attributable to ZSTD decompression CPU (7.7× more CPU time for V4).

Generated: 2026-02-16
