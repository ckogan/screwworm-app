"""Introduction module: turns season, sterile-fly coverage, nearby detections and mitigations into
the rate schedule the two model builds read (model_design.md, "Rate schedule"; intro_design.md).

The module sits outside the builds. Its only outputs are, month by month, a multiplier on the zone
introduction rate and a multiplier on between-premises spread. Under the reference conditions (a warm
month, sterile-fly coverage at its reference share, nearby detections as when the reference rate was
estimated, no added mitigation) both multipliers are exactly 1, so the reference introduction rate
is reproduced (the calibration rule of issue #55).

Season term: `season_model` = "curve" (the default: the development-rate curve at the month's own
temperature) or "degree_days" (issue #68: generation time and survival from degree days, with a lag of
one generation; intro_design.md, "Degree-day season option").

Pure Python (math only), so the same file runs in the browser build of the app.
Sources for every constant: intro_design.md and changes/2026-11-18-county-tailoring/lit_notes.md.
"""
from __future__ import annotations

import math

MONTH_DAYS = 30.0          # schedule segments are 30 days, so breakpoints sit on the inspection grid
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Development-rate curve of the fly (Gutierrez and Ponti 2014, Eqn 1): rate per hour at temperature T.
_A, _T_LOW, _B, _T_UP = 0.00065, 14.5, 1.75, 40.0
T_REF = 27.2               # reference temperature (the source's survival optimum), where activity = 1


def development_rate(temp_c: float) -> float:
    if temp_c <= _T_LOW:
        return 0.0
    return _A * (temp_c - _T_LOW) / (1.0 + _B ** (temp_c - _T_UP))


def season_activity(t_max_c: float, t_min_c: float) -> float:
    """Fly activity in [0, 1] for a month, from its mean daily maximum and minimum temperature:
    the development rate averaged over the two, relative to the rate at T_REF, capped at 1.
    Averaging the daily extremes (an assumption) keeps some activity in months whose mean sits near
    the 14.5 C threshold, which a rate evaluated at the monthly mean would set to zero."""
    r = 0.5 * (development_rate(t_max_c) + development_rate(t_min_c))
    return max(0.0, min(1.0, r / development_rate(T_REF)))


# ---------------------------------------------------------------- degree-day season option (issue #68)
# Constants from Gutierrez and Ponti (2014): degree days above 14.5 C for the pupa (95) and the adult
# before first egg laying (61); the daily increment of degree days x(T) = 24 R(T) X_PREOVIP (text below
# Eqn 1); daily mortality of adults, also used there for pupae (Eqn 2). HOST_DAYS is the fixed time on
# the host (egg hatch 8 to 24 h, larvae 5 to 5.7 d, at body temperature). The paper's upper threshold of
# 43.5 C is not a parameter here: the upper limit comes from the 40 C term of development_rate.
X_PUPA, X_PREOVIP = 95.0, 61.0
X_OFF_HOST = X_PUPA + X_PREOVIP          # 156 degree days off the host per generation
HOST_DAYS = 6.0
_MU_A, _MU_B = 0.00036, 0.0035           # mortality per day = _MU_A (T - T_REF)^2 + _MU_B, at most 1
SINE_POINTS = 24                         # points of the daily temperature wave
MAX_OFF_HOST_DAYS = 360.0                # a generation that needs longer does not complete
YEAR_DAYS = 12 * MONTH_DAYS              # the module's year: twelve 30-day segments
SEASON_MODELS = ("curve", "degree_days")


def daily_mortality(temp_c: float) -> float:
    """Share of pupae and adults dying per day at temperature T (Gutierrez and Ponti 2014, Eqn 2)."""
    return min(1.0, _MU_A * (temp_c - T_REF) ** 2 + _MU_B)


def degree_days_per_day(t_max_c: float, t_min_c: float) -> float:
    """Degree days above 14.5 C added in one day with the given maximum and minimum: the source's
    increment 24 R(T) X_PREOVIP averaged over a single sine wave between the two (24 points). With
    maximum equal to minimum this is the increment at that temperature."""
    mean = 0.5 * (t_max_c + t_min_c)
    amp = 0.5 * (t_max_c - t_min_c)
    if amp == 0.0:
        return 24.0 * X_PREOVIP * development_rate(mean)
    tot = 0.0
    for i in range(SINE_POINTS):
        tot += development_rate(mean + amp * math.cos(2.0 * math.pi * (i + 0.5) / SINE_POINTS))
    return 24.0 * X_PREOVIP * tot / SINE_POINTS


def cohort(dd: list, mu: list, seg_days: float, t: float, backward: bool, periodic: bool = True):
    """Generation time (days) and off-host survival of one cohort, or None if it does not complete.

    `dd` and `mu` give the degree days per day and the mortality per day on consecutive segments of
    `seg_days` days (months of 30 days, or single days), starting at time 0. With `backward` the
    cohort is the one whose females start laying at time `t`: its off-host period ends at `t` and
    the time on the host lies before that. Otherwise the eggs are laid at `t`, the host time
    follows, and the off-host period starts at `t + HOST_DAYS`. `periodic` repeats the lists."""
    n = len(dd)
    need, days, log_s = X_OFF_HOST, 0.0, 0.0
    pos = float(t) if backward else float(t) + HOST_DAYS
    while True:
        if backward:
            i = int(math.ceil(pos / seg_days)) - 1
            avail = pos - i * seg_days
        else:
            i = int(math.floor(pos / seg_days))
            avail = (i + 1) * seg_days - pos
        if not periodic and (i < 0 or i >= n):
            return None
        d, m = dd[i % n], mu[i % n]
        step = need / d if d > 0.0 and need <= d * avail else avail
        if m >= 1.0:
            if step > 0.0:
                return None                       # nothing survives a day of certain death
        else:
            log_s += step * math.log(1.0 - m)
        days += step
        if d > 0.0 and need <= d * avail:
            return HOST_DAYS + days, math.exp(log_s)
        need -= d * avail
        pos = i * seg_days if backward else (i + 1) * seg_days
        if days > MAX_OFF_HOST_DAYS:
            return None


_DD_REF = degree_days_per_day(T_REF, T_REF)
_MU_REF = daily_mortality(T_REF)


def degree_day_factor(dd: list, mu: list, seg_days: float, t: float, lag: bool = True,
                      periodic: bool = True) -> float:
    """Season factor in [0, 1] at time `t`: surviving flies produced per parent per day (off-host
    survival divided by generation time) relative to the same at a constant T_REF, capped at 1.
    With `lag` (the module's setting) the cohort is the one that arrives at `t`, so the factor
    reflects the generation before `t`; without it, the cohort that starts at `t`.
    The reference is walked over the same segments, so the factor is exactly 1 at T_REF."""
    c = cohort(dd, mu, seg_days, t, lag, periodic)
    if c is None:
        return 0.0
    n = len(dd)
    r = cohort([_DD_REF] * n, [_MU_REF] * n, seg_days, t, lag, True)
    return max(0.0, min(1.0, (c[1] / c[0]) / (r[1] / r[0])))


def monthly_tables(monthly_tmax: list, monthly_tmin: list) -> tuple:
    """Degree days per day and mortality per day for each of the twelve months. Mortality is read
    at the mean of the daily maximum and minimum."""
    dd = [degree_days_per_day(a, b) for a, b in zip(monthly_tmax, monthly_tmin)]
    mu = [daily_mortality(0.5 * (a + b)) for a, b in zip(monthly_tmax, monthly_tmin)]
    return dd, mu


_FACTOR_CACHE: dict = {}


def season_factor_degree_days(monthly_tmax: list, monthly_tmin: list, lag: bool = True) -> list:
    """The twelve monthly season factors, January first: the mean of the daily factor over the
    30 days of each month (read at the middle of each day). Temperatures repeat every year."""
    key = (tuple(monthly_tmax), tuple(monthly_tmin), bool(lag))
    if key not in _FACTOR_CACHE:
        dd, mu = monthly_tables(monthly_tmax, monthly_tmin)
        days = int(MONTH_DAYS)
        out = []
        for k in range(12):
            tot = 0.0
            for j in range(days):
                tot += degree_day_factor(dd, mu, MONTH_DAYS, k * MONTH_DAYS + j + 0.5, lag)
            out.append(tot / days)
        if len(_FACTOR_CACHE) > 64:
            _FACTOR_CACHE.clear()
        _FACTOR_CACHE[key] = out
    return list(_FACTOR_CACHE[key])


def generation_days(monthly_tmax: list, monthly_tmin: list, t: float, backward: bool = False):
    """Generation time in days of the cohort that starts (or, with `backward`, arrives) on day `t`
    of the module's 360-day year; None if it does not complete."""
    dd, mu = monthly_tables(monthly_tmax, monthly_tmin)
    c = cohort(dd, mu, MONTH_DAYS, t, backward)
    return None if c is None else c[0]


def generations_per_year(monthly_tmax: list, monthly_tmin: list) -> float:
    """Generations in 365 days: the sum over the days of the year of 1 / (forward generation time)."""
    dd, mu = monthly_tables(monthly_tmax, monthly_tmin)
    tot = 0.0
    for j in range(int(YEAR_DAYS)):
        c = cohort(dd, mu, MONTH_DAYS, j + 0.5, False)
        if c is not None:
            tot += 1.0 / c[0]
    return tot * 365.0 / YEAR_DAYS


def cycles_to_days(monthly_tmax: list, monthly_tmin: list, start_month: int, n_cycles: int):
    """Days taken by `n_cycles` fly generations run one after another, from the first day of
    `start_month` (1 to 12); None if one of them does not complete."""
    dd, mu = monthly_tables(monthly_tmax, monthly_tmin)
    t0 = (int(start_month) - 1) * MONTH_DAYS
    t = t0
    for _ in range(int(n_cycles)):
        c = cohort(dd, mu, MONTH_DAYS, t, False)
        if c is None:
            return None
        t += c[0]
    return t - t0


def season_summary(m: dict) -> dict:
    """For the app's caption: generations per year and the days taken by 3 and 4 life cycles from
    the start month, at the temperatures the schedule uses (`temp_shift` added)."""
    shift = float(m.get("temp_shift", 0.0))
    tmax = [x + shift for x in m["monthly_tmax"]]
    tmin = [x + shift for x in m["monthly_tmin"]]
    return dict(generations_per_year=generations_per_year(tmax, tmin),
                cycles_3=cycles_to_days(tmax, tmin, m["start_month"], 3),
                cycles_4=cycles_to_days(tmax, tmin, m["start_month"], 4))


def sterile_fertile_fraction(coverage: float, kappa: float) -> float:
    """Share of matings that stay fertile when `coverage` of the sterile-fly release target is met:
    1 / (1 + kappa * coverage) (Knipling's ratio rule with competitiveness folded into kappa)."""
    return 1.0 / (1.0 + max(0.0, kappa) * max(0.0, coverage))


def pathway_rates(m: dict, activity: float) -> dict:
    """Relative introduction rate by pathway for one month (reference conditions sum to 1)."""
    share_fly = float(m["share_fly"])
    share_move = float(m["share_move"])
    share_wild = max(0.0, 1.0 - share_fly - share_move)
    sterile = (sterile_fertile_fraction(m["release_coverage"], m["release_kappa"])
               / sterile_fertile_fraction(m["release_coverage_ref"], m["release_kappa"]))
    nearby = max(0.0, float(m["nearby_pressure"]))
    treat = 1.0 - float(m["treat_share"]) * float(m["treat_effect"])
    wound = float(m["wound_factor"])
    fly = share_fly * activity * sterile * nearby
    move = share_move * nearby * (1.0 - float(m["move_check_effect"]))
    wild = share_wild * activity
    return dict(fly=fly * treat * wound, move=move * treat * wound, wild=wild * treat * wound,
                sterile=sterile, treat=treat)


def spread_multiplier(m: dict, activity: float) -> float:
    """Multiplier on between-premises spread for one month: season, sterile-fly coverage relative to
    the reference, preventive treatment, and movement checks on the moved-animal share of spread."""
    sterile = (sterile_fertile_fraction(m["release_coverage"], m["release_kappa"])
               / sterile_fertile_fraction(m["release_coverage_ref"], m["release_kappa"]))
    treat = 1.0 - float(m["treat_share"]) * float(m["treat_effect"])
    move = 1.0 - float(m["spread_move_share"]) * float(m["move_check_effect"])
    return activity * sterile * treat * move


def schedule(m: dict, horizon_days: float) -> dict:
    """The rate schedule over [0, horizon): one segment per 30 days from `start_month` (1 to 12).

    `m` carries the module inputs (the app input ids listed in presets.MODULE_DEFAULTS) and
    `monthly_tmax`, `monthly_tmin` (12 values each, deg C, January first), with `temp_shift` added to
    both. `season_model` chooses the season term; in a row, `activity` holds whichever term was used.
    Returns schedule_times, intro_multiplier, spread_multiplier and a per-segment breakdown."""
    n_seg = max(1, int(math.ceil(float(horizon_days) / MONTH_DAYS - 1e-9)))
    start = int(m["start_month"]) - 1
    shift = float(m.get("temp_shift", 0.0))
    season_on = bool(m.get("season_on", True))
    # season_model: "curve" (default, also when the key is absent) or "degree_days" (issue #68)
    factors = None
    if season_on and m.get("season_model", "curve") == "degree_days":
        factors = season_factor_degree_days([x + shift for x in m["monthly_tmax"]],
                                            [x + shift for x in m["monthly_tmin"]])
    times, intro, spread, rows = [], [], [], []
    for j in range(n_seg):
        k = (start + j) % 12
        if factors is not None:
            act = factors[k]
        else:
            act = (season_activity(m["monthly_tmax"][k] + shift, m["monthly_tmin"][k] + shift)
                   if season_on else 1.0)
        pw = pathway_rates(m, act)
        a = pw["fly"] + pw["move"] + pw["wild"]
        s = spread_multiplier(m, act)
        times.append(j * MONTH_DAYS)
        intro.append(a)
        spread.append(s)
        rows.append(dict(month=MONTH_NAMES[k], activity=act, fly=pw["fly"], move=pw["move"],
                         wild=pw["wild"], intro=a, spread=s))
    return dict(schedule_times=times, intro_multiplier=intro, spread_multiplier=spread, rows=rows)
