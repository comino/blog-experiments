-- Experiment 01: Extract benchmark results from system.query_log
-- Run after benchmark.sh completes
-- Output: queries.csv

SELECT
    extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][2] AS query,
    extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][3] AS variant,
    extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][4] AS run,
    extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][1] AS temp,
    query_duration_ms,
    read_rows,
    read_bytes,
    ProfileEvents['OSCPUVirtualTimeMicroseconds'] AS cpu_us
FROM system.query_log
WHERE query_id LIKE 'exp01_%' AND type = 'QueryFinish'
  AND extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][1] != ''
ORDER BY query, variant, run, temp
FORMAT CSVWithNames;
