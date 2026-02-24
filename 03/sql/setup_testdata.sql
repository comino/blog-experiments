-- Exp03: Setup database and generate 1M test rows
CREATE DATABASE IF NOT EXISTS exp03_llm;

-- Drop existing tables
DROP TABLE IF EXISTS exp03_llm.events;
DROP TABLE IF EXISTS exp03_llm.users;

-- Create users table
CREATE TABLE exp03_llm.users
(
    user_id    UInt64,
    created_at DateTime64(3, 'UTC'),
    plan       LowCardinality(String),
    email      String,
    age        Nullable(UInt8),
    country    LowCardinality(String),
    metadata   Map(String, String)
)
ENGINE = MergeTree()
ORDER BY user_id;

-- Create events table
CREATE TABLE exp03_llm.events
(
    event_id       UInt64,
    timestamp      DateTime64(3, 'UTC'),
    user_id        UInt64,
    event_type     LowCardinality(String),
    page           String,
    duration_ms    Nullable(UInt32),
    properties     Map(String, String),
    tags           Array(String),
    country        LowCardinality(String),
    device         LowCardinality(String),
    revenue_cents  Nullable(Int64)
)
ENGINE = MergeTree()
ORDER BY (timestamp, user_id)
PARTITION BY toYYYYMM(timestamp);

-- Insert 10K users
INSERT INTO exp03_llm.users
SELECT
    number + 1 AS user_id,
    toDateTime64('2023-01-01 00:00:00', 3, 'UTC') + toIntervalSecond(rand(number) % (2 * 365 * 86400)) AS created_at,
    arrayElement(['free', 'pro', 'enterprise'], (rand(number + 1) % 3) + 1) AS plan,
    concat('user', toString(number + 1), '@example.com') AS email,
    if(rand(number + 2) % 5 = 0, NULL, toUInt8(18 + rand(number + 3) % 50)) AS age,
    arrayElement(['US', 'DE', 'GB', 'FR', 'JP', 'BR', 'IN', 'CA', 'AU', 'NL'], (rand(number + 4) % 10) + 1) AS country,
    map(
        'signup_source', arrayElement(['organic', 'paid', 'referral', 'social'], (rand(number + 5) % 4) + 1)
    ) AS metadata
FROM numbers(10000);

-- Insert 1M events
INSERT INTO exp03_llm.events
SELECT
    number + 1 AS event_id,
    toDateTime64('2024-01-01 00:00:00', 3, 'UTC') + toIntervalSecond(rand(number) % (365 * 86400)) AS timestamp,
    (rand(number + 1) % 10000) + 1 AS user_id,
    arrayElement(['pageview', 'click', 'purchase', 'signup', 'logout'], (rand(number + 2) % 5) + 1) AS event_type,
    concat('/', arrayElement(['home', 'about', 'pricing', 'docs', 'blog', 'dashboard', 'settings', 'checkout'], (rand(number + 3) % 8) + 1)) AS page,
    if(rand(number + 4) % 4 = 0, NULL, toUInt32(100 + rand(number + 5) % 9900)) AS duration_ms,
    if(rand(number + 6) % 3 = 0,
        map('campaign', arrayElement(['summer_sale', 'black_friday', 'spring_promo', 'launch'], (rand(number + 7) % 4) + 1),
            'source', arrayElement(['google', 'facebook', 'twitter', 'email', 'direct'], (rand(number + 8) % 5) + 1)),
        map('source', arrayElement(['google', 'facebook', 'twitter', 'email', 'direct'], (rand(number + 9) % 5) + 1))
    ) AS properties,
    if(rand(number + 10) % 3 = 0,
        arrayFilter(x -> x != '', [
            if(rand(number + 11) % 4 = 0, 'promo', ''),
            if(rand(number + 12) % 5 = 0, 'vip_member', ''),
            if(rand(number + 13) % 6 = 0, 'premium_user', ''),
            if(rand(number + 14) % 3 = 0, 'mobile', ''),
            if(rand(number + 15) % 7 = 0, 'new', '')
        ]),
        []
    ) AS tags,
    arrayElement(['US', 'DE', 'GB', 'FR', 'JP', 'BR', 'IN', 'CA', 'AU', 'NL'], (rand(number + 16) % 10) + 1) AS country,
    arrayElement(['desktop', 'mobile', 'tablet'], (rand(number + 17) % 3) + 1) AS device,
    if(rand(number + 18) % 5 = 0, toInt64(100 + rand(number + 19) % 9900), NULL) AS revenue_cents
FROM numbers(1000000);
