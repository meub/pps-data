# Notes for Claude

## Deploying

**Do not run `./deploy.sh` unless the user explicitly asks.** Make changes
locally — including regenerating `web/data.json` and committing — but
stop short of deploying. The user batches deploys themselves.

`./deploy.sh` syncs `web/index.html`, `web/feeders.html`, `web/data.json`,
and the favicons to S3 bucket `ppsdata.info` and invalidates CloudFront
distribution `E1Y8ZR63ATQJ9S`. The script is gitignored since it contains
infra IDs.

Typical flow after a data or UI change:

```bash
.venv/bin/python scripts/export_web.py   # regen web/data.json if pipeline ran
# stop here — wait for the user to ask before running ./deploy.sh
```

## Pipeline shape

The data flow is a three-stage pipeline. Most fetchers are independent;
merges and the web export run last.

1. **Fetch / parse** (`scripts/fetch_*.py`, `scripts/parse_*.py`) — pull
   raw sources into `data/raw/`. Each is idempotent.
2. **Build master** (`scripts/build_master.py`) — assemble
   `data/pps_schools.csv` (83 rows × 89 cols, 74 in-scope + 9 high
   schools kept for reference).
3. **Spatial merges** — add catchment-aggregated columns to the master:
   `merge_housing.py`, `merge_permits.py`, `merge_bli_forecast.py`. All
   use `boundary_join.BoundaryIndex` for point-in-polygon with a 1-mile
   haversine fallback for schools without a published catchment.
4. **Export** (`scripts/export_web.py`) — filter to the 74 in-scope
   schools and write `web/data.json`. Run this after any master-CSV
   change to refresh the dashboard.

## BLI housing forecast (recent addition)

The "Housing growth forecast" section is backed by Metro's 2045 BLI
housing-allocation grid, area-weighted to each school's catchment:

- `scripts/fetch_metro_bli.py` pulls the grid from the City of Portland
  ArcGIS service (`MapServer/88`) into a ~30 MB GeoJSON at
  `data/raw/metro_bli_housing_allocation.geojson`. The file is
  gitignored — regenerate locally, don't commit it.
- `scripts/merge_bli_forecast.py` joins each grid cell's
  `Forecast_Units_Prop` to overlapping catchments weighted by
  `intersection.area / cell.area`. Adds
  `bli_forecast_units_within_catchment` to the master CSV.
- The dashboard ranks schools by projected new residential units within
  catchment by ~2035. Schools without a catchment (ACCESS, focus-option
  programs, some alternatives) fall back to a 1-mile buffer and are
  flagged in a separate teal color in the chart legend.

## Feeders page

`web/feeders.html` is a standalone page at `/feeders.html` — not linked
from the main nav. One Mermaid flowchart per high school showing its
elementary → middle → HS feeder chain. Uses the same dark theme as the
main dashboard but only loads the Mermaid CDN (no Plotly, no Leaflet, no
`data.json`). If you add a new high school or change a feeder, edit the
Mermaid blocks in `web/feeders.html` directly.

## Responsive rank charts

The four ranking charts that were vertical-bar on desktop
(`plot-ranking`, `plot-forecast`, `plot-ratio`, plus anything else using
that pattern) swap to horizontal bars below 700 px window width, and
the chart-container height scales with row count (`rows.length * 18 +
padding`). Any new rank chart of this style should follow the same
mobile branch — see `renderRanking` in `web/index.html` as the
template. Legends on mobile are centered horizontally
(`x: 0.5, xanchor: "center"`); guideline annotations
(median, 100% capacity, in-scope total) are anchored to the bottom of
the plot area on mobile (`yref: "paper", y: 0, yanchor: "top"`) so they
don't overlap the bars at the top.

## Hidden sections

A few sections are in `web/index.html` but currently wrapped in
`<section style="display:none;">` (Outperformers, Clusters) or gated
behind a `getElementById` existence check (Ventilation — the ranking
and scatter were removed from the DOM, but the `airflow_*` columns
still live in the master CSV and the table). Don't wire them back up
without asking.
