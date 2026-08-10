-- Experiment 01: Measure storage per column per variant
-- Run: ssh thesis-clickhouse "clickhouse-client --query \"$(cat sql/04_measure_storage.sql)\""
-- Output: storage.csv

SELECT
    name AS column,
    table AS variant,
    data_compressed_bytes,
    data_uncompressed_bytes,
    round(data_uncompressed_bytes / data_compressed_bytes, 2) AS ratio
FROM system.columns
WHERE database = 'exp01_compression' AND table LIKE 'v%'
ORDER BY table, name
FORMAT CSVWithNames;
