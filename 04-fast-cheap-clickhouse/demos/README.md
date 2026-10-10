# Four live canvas demo stories

Use a deployment configured with both datasets and a separate test account. The public article includes recordings of the completed sequences.

Start a **New chat** for each story, choose its dataset, and keep Animation set to **Flowing**. Wait for each view to settle. Questions add views; select the intended source before an explicit change or a marked follow-up. Use **Fit** to return from a focused card to the complete canvas.

Lead with the monarch story for the fastest complete visual progression. Keep ELSA as the longer advanced example: its final rehearsal had substantial model latency. Exact UTC boundaries and the explicit ranking instructions are part of the rehearsed prompts.

## 1 A year of monarch sightings moves north

**Dataset:** Wildlife worldwide.

Turn a map into a seasonal landscape, then open the actual photographs behind one high point.

**Reveal:** The selected 2025 cohort contains 66,712 reports. The August cell from 40° to 45° north contains 10,225 of them; the final photographs come from that exact marked cell.

**Prompt 1**

```text
Create a 5-degree density map of monarch observations in 2025: Danaus plexippus, taxon_id 48662, longitude -130 to -60 and latitude 0 inclusive to 60 exclusive. Use bin centers and the full wildlife table, count reported observations, and include cell_size_deg=5.
```

**Prompt 2**

```text
Add a seasonal wheel of monthly reported observations for exactly the same 2025 monarch cohort and region. Keep the map.
```

**Prompt 3**

```text
Add a heatmap for the same 2025 monarch cohort and region: month on the horizontal axis, 5-degree latitude-bin centers on the vertical axis, and reported observation count as color. Name the columns month, latitude_band and observations.
```

Select the month/latitude heatmap and focus it. Enable Mark, then click month 8 and latitude-band center 42.5. Check that one point is marked before continuing.

**Prompt 4 · local view change**

```text
make it 3d
```

Play Orbit for about three seconds, then pause. The same marked cell survives this local 3D conversion.

Keep the marked surface selected.

**Prompt 5**

```text
Add 12 licensed photos from this marked cell, ordered by observation date then observation ID. Keep the existing views.
```

Enlarge a photograph, close it, then Fit. The gallery includes caterpillars as well as adult butterflies.

These are reports of different butterflies and life stages. They show a seasonal observation pattern, not individually tracked migration routes or population size.

## 2 Two hemispheres have different fungal calendars

**Dataset:** Wildlife worldwide.

Raw report totals would swamp the comparison. Normalize the two regions, rotate their calendars, then zoom into the Australian peak and reveal unfamiliar fungi.

**Reveal:** The Northern Europe box peaks in October, with 25.42% of its annual reports. The southeastern Australia box peaks in May, with 21.53%. The Sydney-area cell contains 6,202 May reports, and the gallery shows twelve licensed examples with at most two per scientific name.

**Prompt 1**

```text
Map 2025 fungi observations in two temperate boxes: Northern Europe box, latitude 48 to 60 and longitude -10 to 30; Southeastern Australia box, latitude -40 to -30 and longitude 140 to 154. Use a 2-degree density grid with bin centers, count reported observations from the full wildlife table, and include cell_size_deg=2.
```

**Prompt 2**

```text
Add a two-series monthly line chart comparing those same two fungi boxes in 2025. For each box separately, divide each month's observation count by that box's total for the year and multiply by 100. Return month, region, observations and annual_share_pct; x=month, y=annual_share_pct, z=region. Label it as each region's share of annual reported observations. Use region labels Northern Europe box and Southeastern Australia box.
```

Let the two overlaid curves establish the comparison before changing their presentation.

Select the monthly comparison.

**Prompt 3 · local view change**

```text
make it a seasonal wheel
```

Switch the wheel group from Northern Europe box to Southeastern Australia box, pausing on the October and May peaks. Each region totals 100%; the Australian May sector accounts for 21.53%.

**Prompt 4**

```text
Add a close-up density map of fungi observations in the Southeastern Australia box during May 2025, latitude -40 to -30 and longitude 140 to 154. Use 1-degree bin centers, count reported observations and include cell_size_deg=1.
```

Select and focus the May Australia map. Hover the Sydney-area cell at longitude 151.5, latitude −33.5; its tooltip should show 6,202 observations. Click to mark it.

**Prompt 5**

```text
Add exactly 12 licensed photos in total from this marked cell. Within each scientific name, rank observations by date then observation ID and keep its first two; from those, take the first 12 overall in the same order. Keep the existing views.
```

Enlarge a photo and close it, then Fit. As an optional finale, choose a red Aseroe rubra example; the recorded rehearsal opened the first, orange Agaricales image.

Monthly shares are calculated within each rectangular region. Normalization removes the difference in total report volume, but it does not remove observer bias or measure fungal abundance. Some scientific names identify broader taxa.

## 3 A busy river map with nearly stationary ships

**Dataset:** Ships & Storms.

Six vessels share a map, but their speed histories split into two very different stories. Then ask a real sequence question: which slow spells persisted without a reporting gap?

**Reveal:** Three selected vessels travel along the river; AFRICAN TEAL, CSSC TAI YUAN and GENCO MARY report almost zero speed. Each appears among low-speed runs lasting about 23.95 hours. The run query must split at faster reports, unknown speed, or gaps longer than ten minutes.

**Prompt 1**

```text
Replay these six vessels around the lower Mississippi: AQUAVITA TRADER (538090649), CHEM RANGER (538004628), FAIRCHEM FORTITUDE (563105500), AFRICAN TEAL (311001216), CSSC TAI YUAN (477615900), and GENCO MARY (538009752). Use the full UTC day from 2024-10-09T00:00:00Z inclusive to 2024-10-10T00:00:00Z exclusive. Keep each vessel's entire available journey for that day.
```

Play for a few seconds and pause. For an optional river close-up, turn Storms off, drag the Louisiana cluster to the map centre, then click the map’s + control a few times. The mouse wheel zooms the whole canvas; hiding Storms alone does not refit the map. Focus the speed chart next to see the moving and nearly stationary groups clearly.

**Prompt 2**

```text
Add a line chart of the same six vessels' mean reported speed in knots, from 2024-10-09T00:00:00Z inclusive to 2024-10-10T00:00:00Z exclusive. Return one row per hour and vessel_id: group only by hour and vessel_id, and use any(vessel_name) only as a display label. Show a separate named series for each vessel. Name the time column hour, keep missing hours missing, and keep the replay.
```

Click the replay and scrub once. The speed chart cursor should follow the same UTC moment. After pausing animated playback, nudge the time slider one tick for an exact comparison frame; the chart cursor can otherwise retain its last throttled frame.

Clear any mark before asking about all ships in the box.

**Prompt 3**

```text
Now look across all cargo ships and tankers in the lower Mississippi box, latitude 28.5 to 30.2 and longitude -90.5 to -88.6, from 2024-10-09T00:00:00Z inclusive to 2024-10-10T00:00:00Z exclusive. Add a table of the 10 longest runs of below-1-knot reports lasting at least six hours. Within each vessel's chronological reports in this box, break a run when speed reaches 1 knot, speed is unknown, or the gap exceeds 10 minutes. Use elapsed seconds for the six-hour test and ranking; round hours only for display. Include vessel name, start, end, duration, and report count. Label these low-speed runs.
```

Pause on the names and durations. This step uses chronological segmentation, rather than subtracting the first and last low-speed timestamps.

**Prompt 4**

```text
Add a heatmap of the lower Mississippi area's moving cargo ships and tankers, from 2024-10-01T00:00:00Z inclusive to 2024-10-16T00:00:00Z exclusive. Use latitude 28.5 to 30.2 and longitude -90.5 to -88.6, independently of the six replay vessels or the low-speed table. Count distinct vessel IDs with a reported speed of at least 3 knots in each cell. Use calendar day on x and UTC hour on y.
```

Select the day/hour heatmap.

**Prompt 5 · local view change**

```text
make it 3d
```

Play Orbit for about three seconds, pause, then Fit.

A low-speed reporting run does not prove anchoring, waiting, or congestion. The replay covers each vessel’s whole available day; the wider analyses use the explicitly defined lower Mississippi box.

## 4 ELSA crosses the Gulf during Milton

**Dataset:** Ships & Storms.

A moving ship, a moving storm, and a coastline whose reported vessel activity changes. Start with the voyage, then ask whether the pattern is visible in the wider data.

**Reveal:** ELSA reaches her westernmost reported longitude in the October 10 04:00 UTC hourly bin, then returns east. On October 9, the Tampa box reports 54.4% of its own October 1–5 mean daily vessel count; the Port Everglades box reports 91.4%.

**Prompt 1**

```text
Create one replay view for cargo ship ELSA, vessel ID 538006273, for the full six-day UTC interval: start 2024-10-07T00:00:00Z inclusive, end 2024-10-13T00:00:00Z exclusive. Keep her entire available Gulf journey.
```

Play the clean voyage first. Pause and show satellite imagery briefly, then switch it off for clearer tracks. The replay already overlays nearby storm tracks. The satellite layer is a dated daily NASA Aqua image.

**Prompt 2**

```text
Add ELSA's reported longitude over UTC time for exactly the same six days: start 2024-10-07T00:00:00Z inclusive, end 2024-10-13T00:00:00Z exclusive. Use the last observed longitude in each hourly bin and name the time column hour. Keep missing hours missing and keep the replay.
```

The new line should contain 137 observed hours. Missing hours remain gaps. Focus the line and drag straight down inside the plot from about −81° to −91°. This zooms only the longitude axis, keeping all six days visible; the default zero baseline compresses this coordinate range.

Click the ELSA replay. Scrub near October 10 and watch the linked cursor on the longitude chart.

**Prompt 3**

```text
Add a two-series line chart of daily distinct vessel IDs, from 2024-10-01T00:00:00Z inclusive to 2024-10-16T00:00:00Z exclusive. Compare Tampa Bay (latitude 27.5 to 28.1, longitude -82.9 to -82.35) with the Port Everglades area (latitude 25.95 to 26.25, longitude -80.2 to -79.8). These are IDs reported in geographic boxes; label them that way. Include all vessels in the boxes, independently of ELSA.
```

Select the two-area comparison chart first.

**Prompt 4**

```text
Change this comparison chart to an activity index. For each area separately, set its mean daily distinct-vessel count from 2024-10-01T00:00:00Z inclusive to 2024-10-06T00:00:00Z exclusive equal to 100. Plot each daily count divided by its own baseline, times 100. Keep the same boxes and all 15 days, from 2024-10-01T00:00:00Z inclusive to 2024-10-16T00:00:00Z exclusive. Keep the other views.
```

Pause on October 9 so the difference between the two indexed series is visible.

**Prompt 5**

```text
Add a heatmap of distinct vessel IDs reported in Tampa Bay, latitude 27.5 to 28.1 and longitude -82.9 to -82.35. Use the full six-day UTC interval from 2024-10-07T00:00:00Z inclusive to 2024-10-13T00:00:00Z exclusive. Put calendar day on x, UTC hour on y, and distinct vessel count in the color. Include all vessels in the box and keep the other views.
```

Select the Tampa day/hour heatmap.

**Prompt 6 · local view change**

```text
make it 3d
```

Play Orbit briefly, pause, then Fit to show all four views.

AIS counts identify vessels reporting inside the stated boxes. The boxes are not official port boundaries, and the observations do not establish why ELSA took that route or what caused the counts to change.

## Rehearsal and verification

The rehearsal runner creates a fresh chat for each story on the separate QA account, submits real prompts to the pinned OpenRouter route, waits for rendered results, and records a video at normal speed. It checks card counts, selected-card updates, exact row counts, marked row identity, loaded photos, Orbit motion, dated satellite tiles and linked time cursors. It restores the QA account’s originally active chat afterward.

Run a new numbered attempt from the repository root. Playwright Chromium and the ignored local deployment credentials are required.

```sh
python3 rehearse_stories.py --name story-rehearsal-07
```

Use `--stories` followed by story IDs to rehearse a subset. The runner keeps failed attempts and refuses to overwrite an evidence directory. Data queries, local presentation changes and animation completion are timed separately. These are prepared demos, not a blind model benchmark.

## Results from the live rehearsals

All four sequences passed **76 automated checks**, including comparisons with independent query references. The selected complete runs used **17 model calls costing $0.008428159** in OpenRouter-reported inference charges. The four local chart conversions made no model calls and took **138–257 ms** to instrumented readiness.

| Story | Model calls | Observed time to data | Inference cost |
| --- | ---: | ---: | ---: |
| Monarch year | 4 | 1.75–3.23 s | $0.001718391 |
| Opposite fungal seasons | 4 | 1.87–2.35 s | $0.001936424 |
| Moving and nearly stationary ships | 4 | 2.01–4.11 s | $0.002048166 |
| ELSA and Milton | 5 | 5.75–16.62 s | $0.002725178 |

Across these seventeen prepared requests, the observed median was **2.35 seconds**. The 16.62-second request spent 15.43 seconds in the model call and 0.72 seconds in the data query. These are individual application measurements; they do not establish a latency percentile or reliably instantaneous inference. Flowing arrival/framing animations and external image readiness are separate from the time-to-data measurement. ClickHouse query timing includes MCP overhead; gallery timing includes photo metadata lookup.

The first attempts matter. One fungi query returned 773 photos instead of twelve; the first ELSA queries excluded October 12; and the first six-vessel speed query grouped by an aggregate alias and failed. Explicit bounds, grouping and ranking instructions resolved those prompt failures. An overlay request also created a second card, and one recording process stopped before completion. Finally, the app itself treated “Change this comparison chart” as Add because of the noun “comparison”; that interaction bug is fixed in deployed build `20261007-stories1`. All earlier attempts remain in the [rehearsal summary](../performance/demo-stories-01/summary.json). The 33 captured calls across attempts cost $0.016058554; an interrupted step without a captured request ID could be absent from that subtotal.

[Independent validation](../performance/demo-stories-01/validation.json) covers the returned aggregates, cohorts, dates, gallery records/licenses, exact local data reuse and captured replay samples. The [compressed retained results](rehearsal-evidence.json.gz) and [independent reference data](reference-data.json.gz) support an offline recheck. Photo pixels and presentation gestures are checked separately in the browser. The verifier checks these prepared cases, not arbitrary SQL equivalence.

```sh
python3 verify_demo_stories.py demos/rehearsal-evidence.json.gz
python3 -m unittest discover -s tests -p 'test_demo_stories.py' -v
```

Use the repository's locked canvas Python environment for those commands. Fifteen tests deliberately corrupt dates, aggregates, photo IDs, grouping, gap handling, duration thresholds and replay positions to verify that the audit rejects incorrect results. Sixty backend tests also pass. The deployment preserved both workspaces and all 35 chats present before the change.

## Recordings

The two published recordings retain the real requests and waits at their original speed. The article hosts the videos; this repository contains the scripts and compact validation data.

- [Monarch year](https://sveneliasson.de/media/clickhouse-fast-cheap-exploration/monarch-demo.mp4) · 1 min 27 s
- Opposite fungal seasons · 1 min 36 s; recording retained privately.
- Moving and nearly stationary ships · 1 min 42 s; recording retained privately.
- [ELSA and Milton](https://sveneliasson.de/media/clickhouse-fast-cheap-exploration/elsa-milton-demo.mp4) · 2 min 11 s

The [media check](../performance/demo-stories-01/media-validation.json) verifies that each MP4 keeps its source recording's duration and 1600 × 1000 resolution.

The separately retained ELSA presentation clip shows the longitude zoom and dated satellite toggle using retained query results, with **zero inference**. The [presentation checks](../performance/story-presentation-01/summary.json) also cover river pan/zoom, playback and the red Aseroe photo. These local gestures preserve the saved queries and data. A real 86-minute ELSA reporting gap correctly hides the vessel head; use the observed October 9 afternoon fixes for the clear ship-and-storm frame.
