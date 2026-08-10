-- Experiment 01: Compression Shootout
-- Step 1: Create database and all tables
-- Run: cat sql/01_create_tables.sql | ssh thesis-clickhouse "clickhouse-client --multiquery"

CREATE DATABASE IF NOT EXISTS exp01_compression;

-- Source table (data generation target)
DROP TABLE IF EXISTS exp01_compression.source;
CREATE TABLE exp01_compression.source
(
    timestamp DateTime,
    metric_name LowCardinality(String),
    value Float64,
    host LowCardinality(String),
    region LowCardinality(String),
    counter UInt64
)
ENGINE = MergeTree()
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V1: Default (LZ4)
DROP TABLE IF EXISTS exp01_compression.v1_default;
CREATE TABLE exp01_compression.v1_default
(
    timestamp DateTime,
    metric_name LowCardinality(String),
    value Float64,
    host LowCardinality(String),
    region LowCardinality(String),
    counter UInt64
)
ENGINE = MergeTree()
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V2: ZSTD(3) global
DROP TABLE IF EXISTS exp01_compression.v2_zstd;
CREATE TABLE exp01_compression.v2_zstd
(
    timestamp DateTime CODEC(ZSTD(3)),
    metric_name LowCardinality(String) CODEC(ZSTD(3)),
    value Float64 CODEC(ZSTD(3)),
    host LowCardinality(String) CODEC(ZSTD(3)),
    region LowCardinality(String) CODEC(ZSTD(3)),
    counter UInt64 CODEC(ZSTD(3))
)
ENGINE = MergeTree()
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V3: Per-column optimized (DoubleDelta+LZ4 for ts, Delta+ZSTD for counter, Gorilla+LZ4 for gauge, LZ4 for tags)
DROP TABLE IF EXISTS exp01_compression.v3_percolumn;
CREATE TABLE exp01_compression.v3_percolumn
(
    timestamp DateTime CODEC(DoubleDelta, LZ4),
    metric_name LowCardinality(String) CODEC(LZ4),
    value Float64 CODEC(Gorilla, LZ4),
    host LowCardinality(String) CODEC(LZ4),
    region LowCardinality(String) CODEC(LZ4),
    counter UInt64 CODEC(Delta, ZSTD(1))
)
ENGINE = MergeTree()
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V4: Per-column with ZSTD instead of LZ4
DROP TABLE IF EXISTS exp01_compression.v4_percolumn_zstd;
CREATE TABLE exp01_compression.v4_percolumn_zstd
(
    timestamp DateTime CODEC(DoubleDelta, ZSTD(3)),
    metric_name LowCardinality(String) CODEC(ZSTD(3)),
    value Float64 CODEC(Gorilla, ZSTD(3)),
    host LowCardinality(String) CODEC(ZSTD(3)),
    region LowCardinality(String) CODEC(ZSTD(3)),
    counter UInt64 CODEC(Delta, ZSTD(3))
)
ENGINE = MergeTree()
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V5: Aggressive compression
DROP TABLE IF EXISTS exp01_compression.v5_aggressive;
CREATE TABLE exp01_compression.v5_aggressive
(
    timestamp DateTime CODEC(DoubleDelta, ZSTD(9)),
    metric_name LowCardinality(String) CODEC(ZSTD(9)),
    value Float64 CODEC(Gorilla, ZSTD(3)),
    host LowCardinality(String) CODEC(ZSTD(9)),
    region LowCardinality(String) CODEC(ZSTD(9)),
    counter UInt64 CODEC(Delta, ZSTD(9))
)
ENGINE = MergeTree()
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;
