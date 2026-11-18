"""Introduction module: turns season, sterile-fly coverage, nearby detections and mitigations into
the rate schedule the two model builds read (model_design.md, "Rate schedule"; intro_design.md).

The module sits outside the builds. Its only outputs are, month by month, a multiplier on the zone
introduction rate and a multiplier on between-premises spread. Under the reference conditions (a warm
month, sterile-fly coverage at its reference share, nearby detections as when the reference rate was
estimated, no added mitigation) both multipliers are exactly 1, so the reference introduction rate
is reproduced (the calibration rule of issue #55).

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
    both. Returns schedule_times, intro_multiplier, spread_multiplier and a per-segment breakdown."""
    n_seg = max(1, int(math.ceil(float(horizon_days) / MONTH_DAYS - 1e-9)))
    start = int(m["start_month"]) - 1
    shift = float(m.get("temp_shift", 0.0))
    season_on = bool(m.get("season_on", True))
    times, intro, spread, rows = [], [], [], []
    for j in range(n_seg):
        k = (start + j) % 12
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
