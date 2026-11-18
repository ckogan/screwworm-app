# Data files read by the review app

Copies, unchanged, so that the browser build of the app (which ships this folder only) can read them.
Do not edit them here. Rebuild them at the source and copy again.

| File | Source | Used for |
|---|---|---|
| `detections_official.csv` | `changes/2026-11-18-county-tailoring/data/` | nearby-pressure index; last detection per county |
| `county_centroids_all.csv` | `changes/2026-11-19-linked-counties/data/` | nearby-pressure index |
| `county_adjacency.csv` | `changes/2026-11-19-linked-counties/data/` | reference level of the index |
| `index_reference.json` | `changes/2026-11-19-linked-counties/results/` | factor for the introduction-rate option; counts on the 2026 tab |
| `linked_counties.csv`, `linked_group.csv`, `linked_summary.json` | `changes/2026-11-19-linked-counties/results/` | "Neighbouring counties" tab |
| `backcheck_group.csv`, `backcheck_texas.json` | `changes/2026-11-19-linked-counties/results/` | "Check against 2026" tab |
| `calibration_2026_summary.json` | a few values taken from `fit.json` and `outcomes_summary.json` in `changes/2026-11-19-calibration-2026/results/` | "Check against 2026" tab; help text of the what-if input |
| `county_temperature_this_year.csv` | `changes/2026-11-19-observed-weather/data/` | "2026 observed" weather option |

Copied on 18 November 2026.
