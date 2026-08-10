-- Experiment 01: Generate 100M rows of simulated time series data
-- Run: ssh thesis-clickhouse "clickhouse-client --query \"$(cat sql/02_generate_data.sql)\""
-- Duration: ~5 minutes on CX53

INSERT INTO exp01_compression.source
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(intDiv(number, 10)) AS timestamp,
    arrayElement(
        ['cpu_usage', 'mem_free', 'disk_io', 'net_bytes_sent', 'http_requests_total'],
        (number % 5) + 1
    ) AS metric_name,
    CASE
        -- Gauge metrics (cpu_usage, mem_free): sinusoidal + noise
        WHEN number % 5 < 2 THEN sin(number / 1000.0) * 50 + 50 + (rand() % 100) / 100.0
        -- Counter/other metrics: stepped + noise
        ELSE toFloat64(number % 5) * 10 + (rand() % 1000) / 100.0
    END AS value,
    concat('host-', toString(number % 50)) AS host,
    arrayElement(
        ['us-east', 'us-west', 'eu-central', 'ap-south'],
        (number % 4) + 1
    ) AS region,
    number AS counter
FROM numbers(100000000);
