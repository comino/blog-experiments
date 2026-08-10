-- Experiment 01: Summary statistics (median, p25, p75) per query × variant × temp
-- For the summary.md report

SELECT
    extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][2] AS query,
    extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][3] AS variant,
    extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][1] AS temp,
    median(query_duration_ms) AS median_ms,
    quantile(0.25)(query_duration_ms) AS p25_ms,
    quantile(0.75)(query_duration_ms) AS p75_ms,
    median(ProfileEvents['OSCPUVirtualTimeMicroseconds']) AS median_cpu_us
FROM system.query_log
WHERE query_id LIKE 'exp01_%' AND type = 'QueryFinish'
  AND extractAllGroupsVertical(query_id, 'exp01_(cold|warm)_(Q[123])_(v[0-9a-z_]+)_([0-9]+)_')[1][1] != ''
GROUP BY query, variant, temp
ORDER BY query, variant, temp
FORMAT PrettyCompact;
