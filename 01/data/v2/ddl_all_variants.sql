-- V1: Default (LZ4)
CREATE TABLE exp01_compression.v1_default
(
    `timestamp` DateTime,
    `metric_name` LowCardinality(String),
    `value` Float64,
    `host` LowCardinality(String),
    `region` LowCardinality(String),
    `counter` UInt64
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V2: ZSTD(3) on all columns
CREATE TABLE exp01_compression.v2_zstd
(
    `timestamp` DateTime CODEC(ZSTD(3)),
    `metric_name` LowCardinality(String) CODEC(ZSTD(3)),
    `value` Float64 CODEC(ZSTD(3)),
    `host` LowCardinality(String) CODEC(ZSTD(3)),
    `region` LowCardinality(String) CODEC(ZSTD(3)),
    `counter` UInt64 CODEC(ZSTD(3))
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V3: Per-column specialized + LZ4
CREATE TABLE exp01_compression.v3_percolumn
(
    `timestamp` DateTime CODEC(DoubleDelta, LZ4),
    `metric_name` LowCardinality(String) CODEC(LZ4),
    `value` Float64 CODEC(Gorilla(8), LZ4),
    `host` LowCardinality(String) CODEC(LZ4),
    `region` LowCardinality(String) CODEC(LZ4),
    `counter` UInt64 CODEC(Delta(8), ZSTD(1))
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V4: Per-column specialized + ZSTD(3)
CREATE TABLE exp01_compression.v4_percolumn_zstd
(
    `timestamp` DateTime CODEC(DoubleDelta, ZSTD(3)),
    `metric_name` LowCardinality(String) CODEC(ZSTD(3)),
    `value` Float64 CODEC(Gorilla(8), ZSTD(3)),
    `host` LowCardinality(String) CODEC(ZSTD(3)),
    `region` LowCardinality(String) CODEC(ZSTD(3)),
    `counter` UInt64 CODEC(Delta(8), ZSTD(3))
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;

-- V5: Aggressive — same as V4 but ZSTD(9) (except value stays ZSTD(3))
CREATE TABLE exp01_compression.v5_aggressive
(
    `timestamp` DateTime CODEC(DoubleDelta, ZSTD(9)),
    `metric_name` LowCardinality(String) CODEC(ZSTD(9)),
    `value` Float64 CODEC(Gorilla(8), ZSTD(3)),
    `host` LowCardinality(String) CODEC(ZSTD(9)),
    `region` LowCardinality(String) CODEC(ZSTD(9)),
    `counter` UInt64 CODEC(Delta(8), ZSTD(9))
)
ENGINE = MergeTree
ORDER BY (metric_name, host, timestamp)
SETTINGS index_granularity = 8192;
