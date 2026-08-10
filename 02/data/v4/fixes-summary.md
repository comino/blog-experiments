# Exp02 Projections — Fix Results (v4)

## Fix 5: MV Target Row Count

- **Row count:** 8,760,000
- **Unique pages:** 1,000
- **Unique hours:** 8,760

This makes sense: 1,000 pages × 8,760 hours (365 days × 24h) = 8,760,000 rows.

## Fix 6: Unified Storage (from system.parts_columns)

### All tables:

| Table | Data Compressed | Data Uncompressed | Marks | Columns | Rows |
|-------|----------------|-------------------|-------|---------|------|
| fair_mv_source | 8,056,011,024 | 18,908,750,898 | 1,443,294 | 6 | 200,000,000 |
| fair_mv_target | 539,889,405 | 1,040,234,975 | 57,870 | 5 | 8,760,000 |
| fair_proj | 8,056,129,518 | 18,908,749,002 | 1,444,188 | 6 | 200,000,000 |
| hourly_stats_mv_target | 539,889,405 | 1,040,234,975 | 57,870 | 5 | 8,760,000 |
| scale_10m_base | 402,630,144 | 900,692,574 | 74,010 | 6 | 10,000,000 |
| scale_10m_mv_source | 402,653,244 | 900,667,266 | 74,202 | 6 | 10,000,000 |
| scale_10m_mv_target | 26,968,510 | 50,369,445 | 5,535 | 5 | 437,930 |
| scale_10m_proj | 402,663,000 | 900,692,688 | 73,884 | 6 | 10,000,000 |
| web_analytics_base | 8,054,814,174 | 18,902,251,602 | 1,440,252 | 6 | 200,000,000 |
| web_analytics_mv_source | 8,054,844,456 | 18,902,251,602 | 1,440,558 | 6 | 200,000,000 |
| web_analytics_proj | 8,054,842,800 | 18,902,251,602 | 1,440,432 | 6 | 200,000,000 |

### Projection storage (inside web_analytics_proj):

| Projection | Compressed | Uncompressed | Marks | Rows |
|-----------|-----------|-------------|-------|------|
| proj_country_time | 9,726,181,242 | 18,919,159,890 | 1,463,136 | 200,000,000 |
| proj_hourly_stats | 539,889,405 | 1,040,234,975 | 57,870 | 8,760,000 |

**Note:** The projection storage is NOT included in the base parts_columns numbers. The `web_analytics_proj` base data is 8.05 GB compressed, but the projections add another ~10.3 GB (proj_country_time is actually larger than the base data due to different sort order).

### fair_proj projections:

| Projection | Compressed | Uncompressed | Marks | Rows |
|-----------|-----------|-------------|-------|------|
| proj_hourly_stats | 539,889,405 | 1,040,234,975 | 57,870 | 8,760,000 |

### scale_10m_proj projections:

| Projection | Compressed | Uncompressed | Marks | Rows |
|-----------|-----------|-------------|-------|------|
| proj_country_time | 428,809,860 | 900,678,576 | 74,676 | 10,000,000 |
| proj_hourly_stats | 30,599,725 | 50,374,995 | 5,295 | 437,916 |
