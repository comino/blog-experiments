-- Experiment 01 Extended: Scaling Analysis
-- Creates source tables at 1M, 10M, 100M, 500M rows
-- Then 5 codec variants × 4 sizes = 20 tables

CREATE DATABASE IF NOT EXISTS exp01_compression;

-- ══════════════════════════════════════════════
-- SOURCE TABLES (different sizes)
-- ══════════════════════════════════════════════

DROP TABLE IF EXISTS exp01_compression.source_1m;
CREATE TABLE exp01_compression.source_1m AS exp01_compression.source;
INSERT INTO exp01_compression.source_1m
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(intDiv(number, 10)) AS timestamp,
    arrayElement(['cpu_usage','mem_free','disk_io','net_bytes_sent','http_requests_total'], (number % 5) + 1) AS metric_name,
    CASE WHEN number % 5 < 2 THEN sin(number / 1000.0) * 50 + 50 + (rand() % 100) / 100.0
         ELSE toFloat64(number % 5) * 10 + (rand() % 1000) / 100.0 END AS value,
    concat('host-', toString(number % 50)) AS host,
    arrayElement(['us-east','us-west','eu-central','ap-south'], (number % 4) + 1) AS region,
    number AS counter
FROM numbers(1000000);

DROP TABLE IF EXISTS exp01_compression.source_10m;
CREATE TABLE exp01_compression.source_10m AS exp01_compression.source;
INSERT INTO exp01_compression.source_10m
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(intDiv(number, 10)) AS timestamp,
    arrayElement(['cpu_usage','mem_free','disk_io','net_bytes_sent','http_requests_total'], (number % 5) + 1) AS metric_name,
    CASE WHEN number % 5 < 2 THEN sin(number / 1000.0) * 50 + 50 + (rand() % 100) / 100.0
         ELSE toFloat64(number % 5) * 10 + (rand() % 1000) / 100.0 END AS value,
    concat('host-', toString(number % 50)) AS host,
    arrayElement(['us-east','us-west','eu-central','ap-south'], (number % 4) + 1) AS region,
    number AS counter
FROM numbers(10000000);

-- source (100M) already exists from original experiment

DROP TABLE IF EXISTS exp01_compression.source_500m;
CREATE TABLE exp01_compression.source_500m AS exp01_compression.source;
INSERT INTO exp01_compression.source_500m
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(intDiv(number, 10)) AS timestamp,
    arrayElement(['cpu_usage','mem_free','disk_io','net_bytes_sent','http_requests_total'], (number % 5) + 1) AS metric_name,
    CASE WHEN number % 5 < 2 THEN sin(number / 1000.0) * 50 + 50 + (rand() % 100) / 100.0
         ELSE toFloat64(number % 5) * 10 + (rand() % 1000) / 100.0 END AS value,
    concat('host-', toString(number % 50)) AS host,
    arrayElement(['us-east','us-west','eu-central','ap-south'], (number % 4) + 1) AS region,
    number AS counter
FROM numbers(500000000);
