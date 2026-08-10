-- Experiment 01 Extended: Distribution Analysis
-- 4 data distributions × 5 codec variants = 20 tables
-- Each distribution: 10M rows, single Float64 value column + timestamp + counter
-- Focus: How do codecs perform on different data patterns?

CREATE DATABASE IF NOT EXISTS exp01_compression;

-- ══════════════════════════════════════════════
-- DISTRIBUTION SOURCE TABLES (10M rows each)
-- ══════════════════════════════════════════════

-- D1: Monotone counter (0, 1, 2, 3, ...)
DROP TABLE IF EXISTS exp01_compression.dist_monotone;
CREATE TABLE exp01_compression.dist_monotone
(
    timestamp DateTime,
    value Float64,
    counter UInt64,
    tag LowCardinality(String)
)
ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192;

INSERT INTO exp01_compression.dist_monotone
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(number) AS timestamp,
    toFloat64(number) AS value,
    number AS counter,
    concat('tag-', toString(number % 100)) AS tag
FROM numbers(10000000);

-- D2: Sinusoidal + noise (gauge-like)
DROP TABLE IF EXISTS exp01_compression.dist_sinus;
CREATE TABLE exp01_compression.dist_sinus
(
    timestamp DateTime,
    value Float64,
    counter UInt64,
    tag LowCardinality(String)
)
ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192;

INSERT INTO exp01_compression.dist_sinus
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(number) AS timestamp,
    sin(number / 1000.0) * 1000 + (rand() % 1000) / 10.0 AS value,
    number AS counter,
    concat('tag-', toString(number % 100)) AS tag
FROM numbers(10000000);

-- D3: Spiky (mostly 0, periodic bursts)
DROP TABLE IF EXISTS exp01_compression.dist_spiky;
CREATE TABLE exp01_compression.dist_spiky
(
    timestamp DateTime,
    value Float64,
    counter UInt64,
    tag LowCardinality(String)
)
ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192;

INSERT INTO exp01_compression.dist_spiky
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(number) AS timestamp,
    if(number % 1000 < 10, toFloat64(rand() % 10000), 0.0) AS value,
    number AS counter,
    concat('tag-', toString(number % 100)) AS tag
FROM numbers(10000000);

-- D4: High cardinality random (worst case for compression)
DROP TABLE IF EXISTS exp01_compression.dist_random;
CREATE TABLE exp01_compression.dist_random
(
    timestamp DateTime,
    value Float64,
    counter UInt64,
    tag LowCardinality(String)
)
ENGINE = MergeTree() ORDER BY timestamp SETTINGS index_granularity = 8192;

INSERT INTO exp01_compression.dist_random
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(number) AS timestamp,
    reinterpretAsFloat64(rand64()) AS value,
    rand64() AS counter,
    concat('tag-', toString(number % 100)) AS tag
FROM numbers(10000000);

-- ══════════════════════════════════════════════
-- 5 CODEC VARIANTS × 4 DISTRIBUTIONS = 20 tables
-- ══════════════════════════════════════════════
-- Naming: dist_{distribution}_{variant}
-- Distributions: monotone, sinus, spiky, random
-- Variants: v1 (LZ4), v2 (ZSTD3), v3 (per-col LZ4), v4 (per-col ZSTD), v5 (aggressive)
