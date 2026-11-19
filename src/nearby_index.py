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

Option, off by default (2026-11-20): cases in Mexico within 100 miles of the US border can be added
as a second case source (`extra=`, built by `load_mexico_cases`). The public list gives only the
Mexican state and the approximate miles from the border, so each case is spread over hand-built
anchor points along the border and moved into Mexico by the stated miles. With `extra` left out,
every function here returns exactly what it returned before the option existed.
Numbers, limits and checks: changes/2026-11-20-mexico-border-series/README.md.
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
MEXICO_CSV = _EX / "changes/2026-11-20-mexico-border-series/data/mexico_rows_2026-11-18.csv"
ANCHORS_CSV = _EX / "changes/2026-11-20-mexico-border-series/data/border_anchors.csv"

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

# The Mexican border-area option (off unless a caller passes `extra`). All four are assumptions.
MEXICO_DEFAULTS = dict(
    mexico_weight=1.0,       # one listed Mexican case counts as this many US confirmed cases
    placement="spread",      # "spread": evenly over the anchors nearest the state, moved inland by the
                             # stated miles; "border": the same with no move inland; "midpoint": all at
                             # the state's middle anchor, moved inland; "own_frontage": only anchors on
                             # the state's own border, moved inland
    repeats="copies",        # "copies": a row repeated n times in the source counts n; "distinct": once
    dropped="keep",          # "keep" or "drop" the rows no longer in the latest source pull
)
MEXICO_RANGES = dict(mexico_weight=(0.25, 2.0))
MILE_KM = 1.609344


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


# ---------------------------------------------------------------------------- Mexican cases (option)
def bearing_deg(a: tuple, b: tuple) -> float:
    """Initial great-circle bearing from a to b, degrees clockwise from north."""
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def destination(a: tuple, bearing: float, km: float) -> tuple:
    """The point `km` from a along a great circle that leaves a at `bearing` degrees."""
    p1, l1, b, ang = math.radians(a[0]), math.radians(a[1]), math.radians(bearing), km / 6371.0
    p2 = math.asin(math.sin(p1) * math.cos(ang) + math.cos(p1) * math.sin(ang) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(ang) * math.cos(p1),
                         math.cos(ang) - math.sin(p1) * math.sin(p2))
    return (math.degrees(p2), math.degrees(l2))


def load_anchors(path=ANCHORS_CSV) -> dict:
    """Hand-built border anchor points by Mexican state, ordered west to east:
    {state: [dict(name, point, inland, own, mid, max_miles)]}. `inland` is the bearing into Mexico:
    the direction of the border at the anchor (the chord between the anchors before and after it,
    one-sided at the ends) turned 90 degrees clockwise, since Mexico is on the right going east."""
    by = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["state"], []).append(dict(
                order=int(r["order"]), name=r["name"], point=(float(r["lat"]), float(r["lon"])),
                own=r["own_frontage"] == "1", mid=r["mid"] == "1",
                max_miles=float(r["max_miles"]) if r["max_miles"].strip() else None))
    for rows in by.values():
        rows.sort(key=lambda a: a["order"])
        for i, a in enumerate(rows):
            before = rows[max(i - 1, 0)]["point"]
            after = rows[min(i + 1, len(rows) - 1)]["point"]
            a["inland"] = (bearing_deg(before, after) + 90.0) % 360.0 if before != after else 180.0
    return by


def place_mexico_case(state: str, miles: float, anchors: dict, placement: str = "spread") -> tuple:
    """Where one listed case is put: a tuple of (lat, lon, share), shares summing to 1."""
    rows = anchors[state]
    if placement in ("spread", "border"):
        use = [a for a in rows if a["max_miles"] is None or miles <= a["max_miles"]] or rows
    elif placement == "midpoint":
        use = [a for a in rows if a["mid"]]
    elif placement == "own_frontage":
        use = [a for a in rows if a["own"]]
    else:
        raise ValueError("placement must be 'spread', 'border', 'midpoint' or 'own_frontage'")
    km = 0.0 if placement == "border" else miles * MILE_KM
    out = []
    for a in use:
        lat, lon = destination(a["point"], a["inland"], km) if km else a["point"]
        out.append((round(lat, 6), round(lon, 6), 1.0 / len(use)))
    return tuple(out)


def load_mexico_cases(path=MEXICO_CSV, anchors_path=ANCHORS_CSV, placement=MEXICO_DEFAULTS["placement"],
                      repeats=MEXICO_DEFAULTS["repeats"], dropped=MEXICO_DEFAULTS["dropped"]) -> list:
    """The public list of cases in Mexico within 100 miles of the US border, as the `extra` case
    source of the index: (date, weight, placements, state), sorted by date. The list has no place
    below the state, so `placements` come from `place_mexico_case`. `copies` is the number of times
    an identical row stood in the latest source pull; it is blank for a row no longer in that pull."""
    if repeats not in ("copies", "distinct") or dropped not in ("keep", "drop"):
        raise ValueError("repeats must be 'copies' or 'distinct'; dropped must be 'keep' or 'drop'")
    anchors = load_anchors(anchors_path)
    out, placed = [], {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            copies = r["copies"].strip()
            if not copies and dropped == "drop":
                continue
            weight = float(copies) if copies and repeats == "copies" else 1.0
            mm, dd, yy = (int(x) for x in r["Date (mm/dd/yy)"].strip().split("/"))
            state, miles = r["State"].strip(), float(r["Approximate Miles From US"])
            if (state, miles) not in placed:
                placed[(state, miles)] = place_mexico_case(state, miles, anchors, placement)
            out.append((date(2000 + yy, mm, dd), weight, placed[(state, miles)], state))
    return sorted(out, key=lambda c: (c[0], c[3], c[2]))


def _place_weight(here: tuple, places: tuple, d: dict) -> float:
    """Weight of one case spread over `places` for a county at `here`: the share-weighted mean of
    the same pair weight a US case gets."""
    return sum(share * pair_weight(here, (lat, lon), d["kernel_km"], d["south_weight"], d["kernel_shape"])
               for lat, lon, share in places)


def _extra_raw(county: tuple, here: tuple, on: date, start: date, extra: list, d: dict, memo) -> float:
    memo = {} if memo is None else memo
    total = 0.0
    for when, weight, places, _ in extra:
        if when <= start or when > on:
            continue
        key = (county, places)
        if key not in memo:
            memo[key] = _place_weight(here, places, d)
        total += weight * memo[key]
    return total


# ---------------------------------------------------------------------------- the index
def raw_index(county: tuple, on: date, cases: list, centroids: dict, exclude=(), extra=None,
              mexico_weight: float = 1.0, _memo=None, **dials) -> float:
    """Weighted count of cases in other counties confirmed in the window ending on `on` (inclusive).
    `exclude` lists further counties to leave out (the members of a linked group, whose pressure
    comes from the model instead). `extra` (optional, from `load_mexico_cases`) adds cases in Mexico,
    each through the same kernel and southern weight, times `mexico_weight`. `_memo` is a cache a
    caller may share between calls that use the same dials."""
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
    if extra:
        if mexico_weight < 0:
            raise ValueError("mexico_weight must not be negative")
        if mexico_weight > 0:
            total += mexico_weight * _extra_raw(county, here, on, start, extra, d, _memo)
    return total


def index_breakdown(county: tuple, on: date, cases: list, centroids: dict, reference: float = 1.0,
                    exclude=(), extra=None, mexico_weight: float = 1.0, **dials) -> list:
    """The index split by source county, largest first: one dict per county with cases in the window
    (cases, km, southness, weight per case, contribution). The contributions sum to the index.
    With `extra`, one more row per Mexican state (state "Mexico"; km and southness are means over
    the placed points, weighted by cases)."""
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
    if extra and mexico_weight > 0:
        by = {}
        for when, weight, places, state in extra:
            if start < when <= on:
                s = by.setdefault(state, dict(n=0.0, w=0.0, km=0.0, south=0.0))
                s["n"] += weight
                s["w"] += weight * _place_weight(here, places, d)
                s["km"] += weight * sum(sh * distance_km(here, (la, lo)) for la, lo, sh in places)
                s["south"] += weight * sum(sh * southness(here, (la, lo)) for la, lo, sh in places)
        for state, s in by.items():
            rows.append(dict(state="Mexico", county=state, cases=s["n"], km=s["km"] / s["n"],
                             southness=s["south"] / s["n"],
                             weight_per_case=mexico_weight * s["w"] / s["n"] / reference,
                             contribution=mexico_weight * s["w"] / reference))
    return sorted(rows, key=lambda r: -r["contribution"])


def at_risk_counties(on: date, cases: list, adjacency: dict) -> list:
    """Counties with no case up to `on` that adjoin a county with a case up to `on`."""
    hit = {src for when, src in cases if when <= on}
    return sorted(c for c, nb in adjacency.items() if c not in hit and nb & hit)


def reference_level(cases: list, centroids: dict, adjacency: dict, start: date = REF_START,
                    end: date = REF_END, extra=None, mexico_weight: float = 1.0, **dials) -> dict:
    """Mean raw index over the at-risk county-days of the reference period, with the counts behind
    it: county-days, the daily number of at-risk counties, and the first detections that fell on an
    at-risk county (the events the reference rate counted).
    With `extra`, the Mexican cases enter the raw value on those same county-days, so the index
    still averages 1 where the reference rate was counted; the county-days themselves are set by US
    cases only. The result then also carries `level_us_only` and `mexico_share_of_level`."""
    first = {}
    for when, src in cases:
        first.setdefault(src, when)
    total, n, per_day, events = 0.0, 0, [], []
    total_us, memo = 0.0, {}
    day = start
    while day <= end:
        risk = at_risk_counties(day - timedelta(days=1), cases, adjacency)
        per_day.append(len(risk))
        for c in risk:
            us = raw_index(c, day, cases, centroids, **dials)
            total += us
            if extra:
                total_us += us
                total += raw_index(c, day, [], centroids, extra=extra, mexico_weight=mexico_weight,
                                   _memo=memo, **dials)
            n += 1
            if first.get(c) == day:
                events.append((day.isoformat(), c[1]))
        day += timedelta(days=1)
    out = dict(level=total / n, county_days=n, at_risk_min=min(per_day), at_risk_max=max(per_day),
               at_risk_mean=n / len(per_day), days=len(per_day), events=events)
    if extra:
        out.update(level_us_only=total_us / n, mexico_share_of_level=(total - total_us) / total)
    return out


def nearby_index(county: tuple, on: date, cases: list, centroids: dict, reference: float,
                 exclude=(), extra=None, mexico_weight: float = 1.0, _memo=None, **dials) -> float:
    """The index for `county` on a date: raw value over the reference level (same dials for both,
    and the same `extra` and `mexico_weight` for both when the Mexican option is on)."""
    if not extra:
        return raw_index(county, on, cases, centroids, exclude=exclude, **dials) / reference
    return raw_index(county, on, cases, centroids, exclude=exclude, extra=extra,
                     mexico_weight=mexico_weight, _memo=_memo, **dials) / reference


def index_by_segment(county: tuple, start: date, n_segments: int, cases: list, centroids: dict,
                     reference: float, data_end: date | None = None, exclude=(), extra=None,
                     mexico_weight: float = 1.0, **dials) -> list:
    """One index value per 30-day segment of a run that starts on `start`. Each segment uses the
    cases known when it begins (the 30 days up to its first day), so nothing looks ahead. Segments
    that begin after `data_end` (default: the last case date) keep the value at `data_end`."""
    data_end = data_end or max(when for when, _ in cases)
    out, memo = [], {}
    for j in range(n_segments):
        on = min(start + timedelta(days=int(MONTH_DAYS) * j), data_end)
        out.append(nearby_index(county, on, cases, centroids, reference, exclude=exclude, extra=extra,
                                mexico_weight=mexico_weight, _memo=memo, **dials))
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
