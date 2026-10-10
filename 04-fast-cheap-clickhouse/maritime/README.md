# Ships & Storms

This is a reproducible historical dataset for the Data Canvas explorer: NOAA
vessel broadcasts in the Gulf of Mexico, Florida, and nearby Atlantic waters,
alongside NOAA IBTrACS North Atlantic storm tracks. It supports actual vessel
trajectories, storm intensity, geographic traffic density, time comparisons,
and proximity queries across two sources.

## Coverage and sources

- **AIS:** September 15 through October 15, 2024, inclusive; longitude −98 to −75,
  latitude 18 to 33. The 31 source files total 8.39 GB. All qualifying reported
  positions are retained; there is no vessel sample or additional downsampling.
  The source is already downsampled to approximately one-minute broadcasts.
- **Storms:** the full 2024 North Atlantic season, primary (`MAIN`) tracks from
  IBTrACS v04r01. Full tracks are retained, including points outside the AIS
  footprint; this allows a storm to enter and leave the map honestly.
- **Time:** UTC throughout. This is a historical replay, not live vessel tracking.
- **Readiness:** `dataset.json` records measured row counts, daily coverage,
  geographic/time bounds, and validation checks. `complete: true` is emitted
  only when every manifested AIS date is loaded and all checks pass.

[NOAA AIS source and documentation](https://github.com/ocm-marinecadastre/ais-vessel-traffic)
(CC0 1.0) and
[NOAA IBTrACS source](https://www.ncei.noaa.gov/products/international-best-track-archive)
are recorded in `source-manifest.json` with individual URLs, ETags, byte lengths,
last-modified dates, citations, and transformation notes. The data can change at
source; an ETag mismatch stops ingestion rather than silently mixing revisions.

The optional satellite layer uses **NASA GIBS / Aqua MODIS** true-colour tiles for
the selected UTC date. It is a daily satellite image, not an hourly image matched
to each ship or storm point. Its exact service template, attribution, usage
policy, and a verified October 9 tile are also recorded in the source manifest.

## Tables and units

The full contract is in `schema.sql`:

- `exploration.ais_positions`: `vessel_id` is the nine-digit MMSI stored as text;
  `observed_at`, latitude/longitude, speed in knots, course and heading in degrees,
  vessel name, readable vessel class, standard type/status codes, length/draft
  in metres, and source date. Invalid coordinates, invalid MMSIs, and timestamps
  outside the corresponding source day are excluded. Unknown numeric values
  remain null. AIS speed 102.2 means **102.2 knots or more**, and draft 25.5 means
  **25.5 metres or more**; those capped source values remain unchanged.
  Known type codes 1–99 and status codes 0–14 are retained; unknown/reserved
  values outside these ranges become null.
- `exploration.storm_tracks`: storm identifier/name, timestamp, position,
  one-minute sustained wind from `USA_WIND` in knots, `USA_PRES` in hPa,
  Saffir–Simpson category/code, nature, season, source agency, source interpolation
  and special-event flags, and the four quadrant 34-knot wind radii in kilometres.
  A wind radius is a quadrant maximum extent, not a predicted impact boundary
  or forecast cone. Millibars and hPa are numerically equivalent.

Vessel classification and sentinel handling follow the
[US Coast Guard AIS message definitions](https://www.navcen.uscg.gov/ais-class-a-reports).
Storm variables follow the
[IBTrACS column documentation](https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf).
The loader does not create interpolated positions. IBTrACS itself interpolates
some records: `usa_interpolation` retains the first `IFLAG` character (`O`, `P`,
`I`, `V`, or `_`). Charts should not represent every source point as an original
instrument observation.

## Reproduce

On the existing owned host, using its private `runtime/credentials.json` loader
account and private ClickHouse HTTP port:

```sh
python3 maritime/load.py --workers 2
python3 maritime/load.py --validate-only
python3 maritime/verify_queries.py
python3 maritime/verify_ingest.py --check-resume
```

The only Python dependency is `requests`. The loader creates only the two new
maritime tables, uses at most two simultaneous imports, two ClickHouse threads
and 4 GB per import, and streams the official Parquet URLs directly into
ClickHouse. It does not download AIS caches, replace tables, alter existing
wildlife/taxi data, or change services. Completed per-day imports are verified
against the local ignored `load.jsonl`; an existing date without a successful
record is an ambiguous partial import and is deliberately refused. Inspect that
condition before resuming. Keep `load.jsonl` alongside the host data between runs.

To smoke-test only one date before a full import:

```sh
python3 maritime/load.py --only-date 2024-10-09 --workers 1
```

That creates a report with `complete: false`. Run the full import afterward to
fill the remaining dates. Source-normalization checks run locally with:

```sh
python3 -m unittest discover -s maritime -p 'test_*.py' -v
```

Run the ingestion audit after completion while the ClickHouse query-log entries
are retained. It compares all 32 successful inserts with table counts and the
loader ledger, records compressed storage, and verifies no import is running.
The ledger stays local; compact reviewed evidence is saved as
`ingest-validation.json`. Concurrent imports are refused using a process lock.

## Interpretation

AIS measures received reports. Receiver outages, transmission gaps, carriage
requirements, and vessels leaving the footprint affect counts. A reduction in
reports does not by itself prove that a storm reduced traffic. September's
baseline also includes Hurricane Helene; it is not an unaffected control period.
Use distinct vessel counts for activity, preserve gaps in playback, and bound
nearest-storm comparisons in both space and time. Slow or stationary reports do
not alone establish anchoring, waiting for a berth, or a port closure. Vessel
names and type/status fields are transmitted metadata and may be incorrect.
The source can have multiple positions for the same vessel and timestamp; the
loader preserves these records. Playback should select a coherent whole point
tuple per vessel/frame, including null fields, and should not bridge long gaps.

`example-queries.json` contains executable examples with bounded result sets;
`query-validation.json` records checks against the loaded data. These are query
correctness checks, not LLM or production latency benchmarks. Raw query result
counts and row samples help reviewers assess exactly what was checked.
