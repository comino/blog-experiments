-- Correctness Test 1: Q1 Dashboard Rollup - Base vs Projection vs MV
-- Q1 on base table
SELECT 'Q1_base',
    toStartOfHour(timestamp) AS hour,
    page,
    count() AS hits,
    avg(duration_ms) AS avg_duration
FROM exp02_projections.web_analytics_base
WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02'
GROUP BY hour, page
ORDER BY hour
FORMAT TabSeparatedWithNames;

-- Q1 on proj table (should use proj_hourly_stats)
SELECT 'Q1_proj',
    toStartOfHour(timestamp) AS hour,
    page,
    count() AS hits,
    avg(duration_ms) AS avg_duration
FROM exp02_projections.web_analytics_proj
WHERE page = '/page/42' AND timestamp >= '2024-06-01' AND timestamp < '2024-06-02'
GROUP BY hour, page
ORDER BY hour
FORMAT TabSeparatedWithNames;

-- Q1 on MV
SELECT 'Q1_mv',
    page,
    hour,
    countMerge(hits) AS hits,
    avgMerge(avg_duration) AS avg_duration
FROM exp02_projections.hourly_stats_mv_target
WHERE page = '/page/42' AND hour >= '2024-06-01' AND hour < '2024-06-02'
GROUP BY page, hour
ORDER BY hour
FORMAT TabSeparatedWithNames;

-- Q2: Country filter - Base vs Proj
SELECT 'Q2_base', count(), avg(duration_ms)
FROM exp02_projections.web_analytics_base
WHERE country = 'DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'
FORMAT TabSeparatedWithNames;

SELECT 'Q2_proj', count(), avg(duration_ms)
FROM exp02_projections.web_analytics_proj
WHERE country = 'DE' AND timestamp >= '2024-03-01' AND timestamp < '2024-04-01'
FORMAT TabSeparatedWithNames;

-- Q3: Top-K pages by avg duration - Base vs Proj vs MV
SELECT 'Q3_base', page, avg(duration_ms) AS avg_dur, count() AS hits
FROM exp02_projections.web_analytics_base
GROUP BY page ORDER BY avg_dur DESC LIMIT 10
FORMAT TabSeparatedWithNames;

SELECT 'Q3_proj', page, avg(duration_ms) AS avg_dur, count() AS hits
FROM exp02_projections.web_analytics_proj
GROUP BY page ORDER BY avg_dur DESC LIMIT 10
FORMAT TabSeparatedWithNames;

SELECT 'Q3_mv', page, avgMerge(avg_duration) AS avg_dur, countMerge(hits) AS hits
FROM exp02_projections.hourly_stats_mv_target
GROUP BY page ORDER BY avg_dur DESC LIMIT 10
FORMAT TabSeparatedWithNames;
