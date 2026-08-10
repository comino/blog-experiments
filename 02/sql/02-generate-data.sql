-- Generate 200M rows into web_analytics_base
INSERT INTO exp02_projections.web_analytics_base
SELECT
    toDateTime('2024-01-01') + toIntervalSecond(rand() % (365 * 86400)) AS timestamp,
    rand() % 1000000 + 1 AS user_id,
    concat('/page/', toString(rand() % 1000)) AS page,
    50 + rand() % 9950 AS duration_ms,
    arrayElement(['US','DE','UK','FR','JP','BR','IN','CA','AU','MX','IT','ES','KR','NL','SE','CH','AT','BE','PL','CZ','DK','NO','FI','PT','IE','RO','HU','GR','BG','HR','SK','SI','LT','LV','EE','IL','TR','ZA','NG','EG','KE','AR','CL','CO','PE','TH','VN','MY','PH','ID'], (rand() % 50) + 1) AS country,
    arrayElement(['desktop','mobile','tablet','smart_tv','wearable'], (rand() % 5) + 1) AS device_type
FROM numbers(200000000);
