"""Nearby-pressure index: turns the official confirmed detections into the `nearby_pressure`
input of the introduction module (intro_design.md, "Nearby detections" and "Linked counties").

For a county and a date the raw value is a sum over confirmed animal cases in OTHER counties in the
30 days up to that date. Each case counts kernel(distance) * south_weight(direction), with the
distance and direction measured between county internal points. The index is the raw value divided
by its reference level: the mean raw value over the county-days that produced the reference
introduction rate (counties with no case of their own that adjoin a county with a case, 13 June to
5 September 2026; priors_design.md, `mean_intro_interval`). So the index is 1, on average, under the
conditions in which the reference rate was counted.

Pure Python (csv, math, datetime only) and no network, so it can run in the browser build of the app.
The data come from CSV files in the repository; the default paths are below.
Numbers and checks: changes/2026-11-19-linked-counties/README.md.
"""
from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path

_EX = Path(__file__).resolve().parents[2]
CASES_CSV = _EX / "changes/2026-11-18-county-tailoring/data/detections_official.csv"
CENTROIDS_CSV = _EX / "changes/2026-11-19-linked-counties/data/county_centroids_all.csv"
ADJACENCY_CSV = _EX / "changes/2026-11-19-linked-counties/data/county_adjacency.csv"

# Defaults and the ranges to explore. Both dials have thin data behind them (see intro_design.md).
DEFAULTS = dict(
    kernel_km=150.0,         # distance scale of the kernel, km: the historical seasonal front moved
                             # 80 to 160 km per fly generation, about 100 to 200 km per 30 days
    south_weight=2.0,        # weight of a case in a county due south, relative to due north (1 = none)
    kernel_shape="power",    # "power": 1 / (1 + (d / L)^2); "exponential": exp(-d / L)
    window_days=30,          # cases confirmed in the 30 days up to the date
)
RANGES = dict(kernel_km=(30.0, 300.0), south_weight=(1.0, 5.0))
# The period and the county set behind the reference introduction rate (priors_design.md): 85 days
# after 12 June 2026, counties with no case so far that adjoin a county with a case.
REF_START = date(2026, 6, 13)
REF_END = date(2026, 9, 5)
MONTH_DAYS = 30.0


# ---------------------------------------------------------------------------- data loading
def load_cases(path=CASES_CSV) -> list:
    """Confirmed animal cases as (date, (state, county)); fly-trap detections are left out."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["case_type"] == "Fly Trap":
                continue
            out.append((date.fromisoformat(r["confirmed_date"]), (r["state"], r["county"])))
    return sorted(out)


def load_centroids(path=CENTROIDS_CSV) -> dict:
    """County internal points: {(state, county): (lat, lon)}."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["state"], r["county"])] = (float(r["lat"]), float(r["lon"]))
    return out


def load_adjacency(path=ADJACENCY_CSV, centroids_path=CENTROIDS_CSV) -> dict:
    """County adjacency as {(state, county): set of (state, county)}. Neighbours outside the three
    states of the centroid file are dropped."""
    key = {}
    with open(centroids_path, newline="") as f:
        for r in csv.DictReader(f):
            key[r["geoid"]] = (r["state"], r["county"])
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["geoid"] in key and r["neighbor_geoid"] in key:
                out.setdefault(key[r["geoid"]], set()).add(key[r["neighbor_geoid"]])
    return out


# ---------------------------------------------------------------------------- geometry and weights
def distance_km(a: tuple, b: tuple) -> float:
    """Great-circle distance between two (lat, lon) points."""
    p1, l1, p2, l2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(x))


def southness(target: tuple, source: tuple) -> float:
    """How far south of the target the source lies, 0 to 1: the cosine of the angle between the
    direction to the source and due south, and 0 when the source is not south of the target."""
    p1, p2 = math.radians(target[0]), math.radians(source[0])
    dl = math.radians(source[1] - target[1])
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = math.atan2(y, x)                 # 0 = north, pi = south
    return max(0.0, -math.cos(bearing))


def kernel(d_km: float, kernel_km: float, shape: str = "power") -> float:
    if shape == "power":
        return 1.0 / (1.0 + (d_km / kernel_km) ** 2)
    if shape == "exponential":
        return math.exp(-d_km / kernel_km)
    raise ValueError("kernel_shape must be 'power' or 'exponential'")


def pair_weight(target: tuple, source: tuple, kernel_km=DEFAULTS["kernel_km"],
                south_weight=DEFAULTS["south_weight"], kernel_shape=DEFAULTS["kernel_shape"]) -> float:
    """Weight of one case at `source` for a county at `target` (both (lat, lon))."""
    return (kernel(distance_km(target, source), kernel_km, kernel_shape)
            * (1.0 + (south_weight - 1.0) * southness(target, source)))


# ---------------------------------------------------------------------------- the index
def raw_index(county: tuple, on: date, cases: list, centroids: dict, exclude=(), **dials) -> float:
    """Weighted count of cases in other counties confirmed in the window ending on `on` (inclusive).
    `exclude` lists further counties to leave out (the members of a linked group, whose pressure
    comes from the model instead)."""
    d = dict(DEFAULTS, **dials)
    start = on - timedelta(days=int(d["window_days"]))
    skip = {county, *exclude}
    here = centroids[county]
    total, cache = 0.0, {}
    for when, src in cases:
        if when <= start or when > on or src in skip:
            continue
        if src not in cache:
            cache[src] = pair_weight(here, centroids[src], d["kernel_km"], d["south_weight"],
                                     d["kernel_shape"])
        total += cache[src]
    return total


def index_breakdown(county: tuple, on: date, cases: list, centroids: dict, reference: float = 1.0,
                    exclude=(), **dials) -> list:
    """The index split by source county, largest first: one dict per county with cases in the window
    (cases, km, southness, weight per case, contribution). The contributions sum to the index."""
    d = dict(DEFAULTS, **dials)
    start = on - timedelta(days=int(d["window_days"]))
    skip = {county, *exclude}
    here = centroids[county]
    n = {}
    for when, src in cases:
        if start < when <= on and src not in skip:
            n[src] = n.get(src, 0) + 1
    rows = []
    for src, k in n.items():
        w = pair_weight(here, centroids[src], d["kernel_km"], d["south_weight"], d["kernel_shape"])
        rows.append(dict(state=src[0], county=src[1], cases=k,
                         km=distance_km(here, centroids[src]), southness=southness(here, centroids[src]),
                         weight_per_case=w / reference, contribution=k * w / reference))
    return sorted(rows, key=lambda r: -r["contribution"])


def at_risk_counties(on: date, cases: list, adjacency: dict) -> list:
    """Counties with no case up to `on` that adjoin a county with a case up to `on`."""
    hit = {src for when, src in cases if when <= on}
    return sorted(c for c, nb in adjacency.items() if c not in hit and nb & hit)


def reference_level(cases: list, centroids: dict, adjacency: dict, start: date = REF_START,
                    end: date = REF_END, **dials) -> dict:
    """Mean raw index over the at-risk county-days of the reference period, with the counts behind
    it: county-days, the daily number of at-risk counties, and the first detections that fell on an
    at-risk county (the events the reference rate counted)."""
    first = {}
    for when, src in cases:
        first.setdefault(src, when)
    total, n, per_day, events = 0.0, 0, [], []
    day = start
    while day <= end:
        risk = at_risk_counties(day - timedelta(days=1), cases, adjacency)
        per_day.append(len(risk))
        for c in risk:
            total += raw_index(c, day, cases, centroids, **dials)
            n += 1
            if first.get(c) == day:
                events.append((day.isoformat(), c[1]))
        day += timedelta(days=1)
    return dict(level=total / n, county_days=n, at_risk_min=min(per_day), at_risk_max=max(per_day),
                at_risk_mean=n / len(per_day), days=len(per_day), events=events)


def nearby_index(county: tuple, on: date, cases: list, centroids: dict, reference: float,
                 exclude=(), **dials) -> float:
    """The index for `county` on a date: raw value over the reference level (same dials for both)."""
    return raw_index(county, on, cases, centroids, exclude=exclude, **dials) / reference


def index_by_segment(county: tuple, start: date, n_segments: int, cases: list, centroids: dict,
                     reference: float, data_end: date | None = None, exclude=(), **dials) -> list:
    """One index value per 30-day segment of a run that starts on `start`. Each segment uses the
    cases known when it begins (the 30 days up to its first day), so nothing looks ahead. Segments
    that begin after `data_end` (default: the last case date) keep the value at `data_end`."""
    data_end = data_end or max(when for when, _ in cases)
    out = []
    for j in range(n_segments):
        on = min(start + timedelta(days=int(MONTH_DAYS) * j), data_end)
        out.append(nearby_index(county, on, cases, centroids, reference, exclude=exclude, **dials))
    return out


def schedule_with_index(intro_module, m: dict, horizon_days: float, nearby: list,
                        wildlife_follows_index: bool = False) -> dict:
    """The rate schedule of intro_module.schedule with `nearby_pressure` set segment by segment.
    `nearby` has one value per 30-day segment; a list of equal values reproduces
    intro_module.schedule. With `wildlife_follows_index` the wildlife share is scaled by the index
    too (an option examined in the 2026 check; off by default, as in the module)."""
    n_seg = max(1, int(math.ceil(float(horizon_days) / MONTH_DAYS - 1e-9)))
    if len(nearby) < n_seg:
        raise ValueError("need one nearby value per 30-day segment")
    start = int(m["start_month"]) - 1
    shift = float(m.get("temp_shift", 0.0))
    season_on = bool(m.get("season_on", True))
    times, intro, spread, rows = [], [], [], []
    for j in range(n_seg):
        k = (start + j) % 12
        act = (intro_module.season_activity(m["monthly_tmax"][k] + shift, m["monthly_tmin"][k] + shift)
               if season_on else 1.0)
        mj = dict(m, nearby_pressure=float(nearby[j]))
        pw = intro_module.pathway_rates(mj, act)
        wild = pw["wild"] * (max(0.0, float(nearby[j])) if wildlife_follows_index else 1.0)
        a = pw["fly"] + pw["move"] + wild
        s = intro_module.spread_multiplier(mj, act)
        times.append(j * MONTH_DAYS)
        intro.append(a)
        spread.append(s)
        rows.append(dict(month=intro_module.MONTH_NAMES[k], activity=act, nearby=float(nearby[j]),
                         fly=pw["fly"], move=pw["move"], wild=wild, intro=a, spread=s))
    return dict(schedule_times=times, intro_multiplier=intro, spread_multiplier=spread, rows=rows)
