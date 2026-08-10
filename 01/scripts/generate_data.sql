-- =============================================================
-- Exp01 Data Generator: 100M Prometheus-style Metrics
-- =============================================================
-- Environment: ClickHouse 25.11.3.54, single node, 32 GB RAM
-- Reproduces the exact dataset used in the compression benchmark
--
-- Schema: 6 columns simulating Prometheus scrape data
--   - timestamp: 1-second intervals starting 2024-01-01
--   - metric_name: 5 distinct values (cpu_usage, mem_used, disk_io, net_rx, net_tx)
--   - value: sin(t) + uniform_noise(0, 0.5), range ~[0, 100]
--   - host: 10 distinct values (host_0 .. host_9)
--   - region: 4 distinct values (us-east, us-west, eu-central, ap-south)
--   - counter: monotonically increasing per (metric_name, host) group
--
-- Total: 100,000,000 rows = 5 metrics × 10 hosts × 2,000,000 timestamps
-- Time span: 2024-01-01 00:00:00 to 2024-01-24 03:33:19 (2M seconds)

-- Step 1: Create source table
CREATE DATABASE IF NOT EXISTS exp01_compression;

CREATE TABLE IF NOT EXISTS exp01_compression.source
(
    timestamp DateTime,
    metric_name LowCardinality(String),
    value Float64,
    host LowCardinality(String),
    region LowCardinality(String),
    counter UInt64
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp);

-- Step 2: Generate 100M rows
-- Uses ClickHouse's numbers() table function for zero-copy generation
INSERT INTO exp01_compression.source
SELECT
    toDateTime('2024-01-01 00:00:00') + intDiv(number, 50) AS timestamp,
    ['cpu_usage', 'mem_used', 'disk_io', 'net_rx', 'net_tx'][1 + (number % 5)] AS metric_name,
    50 + 30 * sin(number / 1000.0) + (rand(number) % 1000) / 2000.0 AS value,
    concat('host_', toString(intDiv(number % 50, 5))) AS host,
    ['us-east', 'us-west', 'eu-central', 'ap-south'][1 + (rand(number + 1) % 4)] AS region,
    intDiv(number, 50) AS counter
FROM numbers(100000000);

-- Step 3: Optimize to single part
OPTIMIZE TABLE exp01_compression.source FINAL;

-- Step 4: Create variant tables and populate
-- V1: Default (LZ4)
CREATE TABLE IF NOT EXISTS exp01_compression.v1_default AS exp01_compression.source;
INSERT INTO exp01_compression.v1_default SELECT * FROM exp01_compression.source;
OPTIMIZE TABLE exp01_compression.v1_default FINAL;

-- V2: ZSTD(3) table-level
CREATE TABLE IF NOT EXISTS exp01_compression.v2_zstd
(
    timestamp DateTime CODEC(ZSTD(3)),
    metric_name LowCardinality(String) CODEC(ZSTD(3)),
    value Float64 CODEC(ZSTD(3)),
    host LowCardinality(String) CODEC(ZSTD(3)),
    region LowCardinality(String) CODEC(ZSTD(3)),
    counter UInt64 CODEC(ZSTD(3))
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp);
INSERT INTO exp01_compression.v2_zstd SELECT * FROM exp01_compression.source;
OPTIMIZE TABLE exp01_compression.v2_zstd FINAL;

-- V3: Per-column specialized (DoubleDelta + Gorilla + Delta)
CREATE TABLE IF NOT EXISTS exp01_compression.v3_percolumn
(
    timestamp DateTime CODEC(DoubleDelta, LZ4),
    metric_name LowCardinality(String) CODEC(LZ4),
    value Float64 CODEC(Gorilla(8), LZ4),
    host LowCardinality(String) CODEC(LZ4),
    region LowCardinality(String) CODEC(LZ4),
    counter UInt64 CODEC(Delta(8), ZSTD(1))
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp);
INSERT INTO exp01_compression.v3_percolumn SELECT * FROM exp01_compression.source;
OPTIMIZE TABLE exp01_compression.v3_percolumn FINAL;

-- V4: Per-column + ZSTD(3) backend
CREATE TABLE IF NOT EXISTS exp01_compression.v4_percolumn_zstd
(
    timestamp DateTime CODEC(DoubleDelta, ZSTD(3)),
    metric_name LowCardinality(String) CODEC(ZSTD(3)),
    value Float64 CODEC(Gorilla(8), ZSTD(3)),
    host LowCardinality(String) CODEC(ZSTD(3)),
    region LowCardinality(String) CODEC(ZSTD(3)),
    counter UInt64 CODEC(Delta(8), ZSTD(3))
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp);
INSERT INTO exp01_compression.v4_percolumn_zstd SELECT * FROM exp01_compression.source;
OPTIMIZE TABLE exp01_compression.v4_percolumn_zstd FINAL;

-- V5: Aggressive (same codecs as V4 but ZSTD(9))
CREATE TABLE IF NOT EXISTS exp01_compression.v5_aggressive
(
    timestamp DateTime CODEC(DoubleDelta, ZSTD(9)),
    metric_name LowCardinality(String) CODEC(ZSTD(9)),
    value Float64 CODEC(Gorilla(8), ZSTD(9)),
    host LowCardinality(String) CODEC(ZSTD(9)),
    region LowCardinality(String) CODEC(ZSTD(9)),
    counter UInt64 CODEC(Delta(8), ZSTD(9))
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp);
INSERT INTO exp01_compression.v5_aggressive SELECT * FROM exp01_compression.source;
OPTIMIZE TABLE exp01_compression.v5_aggressive FINAL;
