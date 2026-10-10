CREATE TABLE IF NOT EXISTS exploration.ais_positions
(
    source_date Date,
    vessel_id String,
    observed_at DateTime('UTC'),
    latitude Float64,
    longitude Float64,
    speed_knots Nullable(Float32),
    course_deg Nullable(Float32),
    heading_deg Nullable(Float32),
    vessel_name String,
    vessel_type LowCardinality(String),
    vessel_type_code Nullable(UInt16),
    navigation_status Nullable(UInt8),
    length_m Nullable(Float32),
    draft_m Nullable(Float32)
)
ENGINE = MergeTree
PARTITION BY source_date
ORDER BY (vessel_type, vessel_id, observed_at)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS exploration.storm_tracks
(
    storm_id String,
    storm_name LowCardinality(String),
    observed_at DateTime('UTC'),
    latitude Float64,
    longitude Float64,
    wind_knots Nullable(Float32),
    pressure_hpa Nullable(Float32),
    category LowCardinality(String),
    category_code Nullable(Int8),
    nature LowCardinality(String),
    basin LowCardinality(String),
    season UInt16,
    wind_source LowCardinality(String),
    usa_interpolation LowCardinality(String),
    usa_record LowCardinality(String),
    radius_34_ne_km Nullable(Float32),
    radius_34_se_km Nullable(Float32),
    radius_34_sw_km Nullable(Float32),
    radius_34_nw_km Nullable(Float32)
)
ENGINE = MergeTree
ORDER BY (storm_name, storm_id, observed_at);
