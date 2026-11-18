"""Scenario presets for the nws-freedom review app: the ONE table the app, the stage-5 scenario
runner (and any future video data pass) all read, so they cannot drift.

Every entry carries a COMPLETE input set (keys are app input ids, not model parameter names), so
loading one leaves nothing behind from the previous scenario. `preset_to_params()` is the single
mapping from app inputs to the lane's parameter dict; app.py uses it for its sliders and the
scenario runner uses it for the presets.

Priors: priors_design.md, first-pass literature review folded 2026-09-08 (within doubling 4.5 d,
within recovery 7 d, f_clinical_max 0.9, rho_r 0.33 /d; zone rows per cluster C). Decisions F1 (the
four screwworm presets), A8 (default horizon 180 d, marks 30/60/90/120/180), C11 (clearance on).

v0.6 (pre-seeded start, 2026-09-09): every pack carries the initial state `k0_mean` / `age0_scale`,
the four zone presets start with leftovers (1 premises, mean age 14 d, both guessed), the clean start
stays the default so the scenario-check presets are the reduction anchor (scenarios.md #16), and
`preset_to_params()` passes the initial state and `release_threshold` through to the lane.

2026-10-23 (parameter-pack correction, accepted 2026-10-23, changes/2026-10-24-clearance-and-doubling/):
(1) zone clearance OFF in the default pack (`clearance_on=False`, so `mean_recovery_time = Inf`): on
the freedom path every premises still counted is undetected, and the 21-d clearance came from the
release rule for detected, treated premises. The 21 d stays as the slider value for when the box is
ticked. Scenario-check presets whose scenarios.md entry says "clearance on" now say so explicitly, and
`six_months_clear` is the clearance-on twin for checks 4 and 9. (2) `doubling_time_within` 4.5 to
3.6 d, re-derived with the model's own within-premises recovery time (7 d) instead of about 14 d.

2026-10-24 (lane change, accepted 2026-10-28, changes/2026-10-24-lane-clearance/): (1) `course_end_on`
(lane `course_end_exit`): an undetected premises leaves the count when its own within-premises course
has run out (fewer than half an animal still infested); (2) `release_on` / `release_mean` (lane
`mean_release_time`): after a detection (D = 1) counted premises clear on the 21-d release clock, as
treated premises do. Both ON in this pack; both OFF at the lane defaults, so regional-freedom and any
pack without them is unchanged. The analytic-oracle presets (checks 2, 3, 4 no-detection pair, 14) set
`course_end_on=False` explicitly: their textbook formulas are for the chain without the exit, and the
exit has its own oracle check (19).
"""
from __future__ import annotations

import math

from intro_module import schedule as _intro_schedule

# APHIS freedom marks (E6: shown, never tuned to). 24-d life cycle at about 22 C. Since 2026-11-17 the
# app no longer draws them (policy dates, not model results); the scenario-check figures still use them.
LIFE_CYCLE_D = 24.0
MARKS = [
    (3 * LIFE_CYCLE_D, "3 life cycles (72 d)"),
    (90.0, "APHIS 3-month rule"),
    (180.0, "technically free (6 mo)"),
    (360.0, "officially free (12 mo)"),
]

# The app's default state = the `six_months` preset (A8: 180 d, monthly rounds).
_BASE = dict(
    # within-premises course (first-pass review, 2026-09-08)
    mean_latent_time=2.0, doubling_time_within=3.6,  # 2026-10-23: was 4.5 (T_rec 7 d, not ~14)
    recovery_within_on=True, mean_recovery_time_within=7.0,
    N_herd=100, f_clinical_max=0.9, mean_clinical_onset_time=3.0,
    p_latent_seed=0.01, p_infectious_seed=0.0, age_window=60,
    # detection: scheduled inspection + owner reporting (alert zone: 3-day mean reporting delay)
    Se_animal=0.85, n_animals_per_round=30, rho_a=0.70, rho_r=0.33,
    # days between the owner's close looks at the animals (issue #48). 0 here, so every
    # scenario-check preset keeps its numbers; the screwworm presets carry the ranch value below.
    owner_contact_interval=0.0,
    # zone (first-pass review, 2026-09-08: a county adjacent to active zones)
    # intro_rate: the rate of newly infected premises (introductions), per day
    M=100, design_prevalence=0.0, intro_rate=round(1.0 / 130.0, 4),  # 0.0077 /d ~ 130 d
    spread_on=True, doubling_time=24.0,  # 2026-10-24 (accepted 2026-10-28): the net figure (was 10, gross with 21-d clearance subtracted; 30 before that)
    clearance_on=False, clearance_mean=21, beta_lockdown_multiplier=0.25,  # 2026-10-23: clearance off (was on, C11)
    # 2026-10-24 (accepted 2026-10-28): course-end exit, and the release clock after a detection
    course_end_on=True, release_on=True, release_mean=21,
    # schedule
    # 2026-11-17: the schedule is a period covered and a spacing; rounds = floor(period / spacing)
    period_days=180, round_interval=30, n_farms_per_round=20,
    # initial state (v0.6): the CLEAN start is the default, so every scenario-check preset below
    # inherits k0_mean = 0 and reproduces the 2026-09-08 numbers exactly (scenarios.md #16, the
    # reduction anchor). Only the four zone presets carry leftovers.
    k0_mean=0.0, age0_scale=0.0,
    # release question
    release_threshold=0.95,
    # costs (placeholders; phase 2)
    c_lost_freedom=1_000_000, c_per_case=20_000, c_per_test=5, c_per_round=2_000,
    # introduction module (issues #55 to #58; intro_design.md). OFF here, so every scenario-check
    # preset keeps constant rates and its earlier numbers. The county presets switch it on.
    module_on=False, season_on=True, start_month=6, temp_shift=0.0,
    share_fly=0.60, share_move=0.25,                 # wildlife share = 1 - both (assumed shares)
    release_coverage=0.5, release_coverage_ref=0.5, release_kappa=0.75,
    nearby_pressure=1.0, wound_factor=1.0,
    treat_share=0.0, treat_effect=0.3, move_check_effect=0.0, spread_move_share=0.3,
)

# Worked-example county (issue #50): Zavala County, Texas. Sources and pull dates:
# changes/2026-11-18-county-tailoring/data/README.md.
COUNTY = dict(
    name="Zavala County, Texas",
    M=114,                       # operations with cattle, 2022 Census of Agriculture
    cattle=38198, mean_herd=335, median_class="20 to 49 head",
    area_km2=3360.3,
    last_detection="11 June 2026", start_month=6,
    station="Crystal City (USC00412160), 1991 to 2020 normals",
    # mean daily maximum and minimum temperature by month, deg C, January first
    monthly_tmax=[19.3, 22.3, 26.1, 30.1, 32.8, 35.8, 36.3, 37.3, 33.8, 29.4, 23.4, 19.8],
    monthly_tmin=[6.6, 9.2, 12.8, 16.6, 20.9, 23.7, 24.2, 24.3, 21.9, 17.6, 11.8, 7.3],
)


def _s(**over):
    return dict(_BASE, **over)


# A premises that carries larvae from the moment it is infested: needed wherever a scenario
# depends on inspection sensitivity being effectively 1.
_DETECTABLE = dict(mean_latent_time=0.0, recovery_within_on=False,
                   p_latent_seed=0.0, p_infectious_seed=0.5)

# The outbreak's leftovers at the last detection (v0.6). BOTH NUMBERS ARE GUESSED PLACEHOLDERS,
# not lit-derived: priors_design.md, "Initial state (v0.6)" (the review found no estimate of the
# undetected remainder; one premises is a starting point and 14 d is about one within-premises
# course). Carried by the four zone presets only; `early_warning` keeps the clean start.
_LEFTOVERS = dict(k0_mean=1.0, age0_scale=14.0)

# Large ranches: the owner looks the animals over closely about once a week (issue #48). An assumed
# value with little data behind it; the range to explore is 1 to 30 days.
_RANCH = dict(owner_contact_interval=7.0)

# The county presets: real premises count, the introduction module on, the clock starting in the
# month of the county's last detection. N_herd is set in _COUNTY_HERD (see intro_design.md).
_COUNTY_HERD = 100
_ZAVALA = dict(M=COUNTY["M"], N_herd=_COUNTY_HERD, n_farms_per_round=23, module_on=True,
               start_month=COUNTY["start_month"])

SCENARIOS = {
    # ---- county presets (issue #50) ----
    "county_12mo":         _s(period_days=360, round_interval=30, **_LEFTOVERS, **_RANCH, **_ZAVALA),
    "county_6mo":          _s(**_LEFTOVERS, **_RANCH, **_ZAVALA),
    "county_3yr":          _s(period_days=1080, round_interval=30, **_LEFTOVERS, **_RANCH, **_ZAVALA),
    # ---- screwworm presets (F1) ----
    "six_months":          _s(**_LEFTOVERS, **_RANCH),                                 # 180 d, 6 monthly rounds
    "zone_last_detection": _s(period_days=90, round_interval=30, **_LEFTOVERS, **_RANCH),  # 90 d, the APHIS release window
    "three_cycles":        _s(period_days=72, round_interval=24, **_LEFTOVERS, **_RANCH),  # 72 d, one round per life cycle
    "twelve_months":       _s(period_days=360, round_interval=30, **_LEFTOVERS, **_RANCH),  # 360 d
    "early_warning":       _s(intro_rate=round(1.0 / 1000.0, 4), **_RANCH),             # still-free county 200 km north (A2)
    # ---- scenario-check presets (scenarios.md numbering) ----
    "s1":                  _s(intro_rate=0.0),                                                    # 1
    "s2":                  _s(Se_animal=0.0, rho_r=0.0, clearance_on=False, course_end_on=False),  # 2 (oracle: no exit)
    "s3":                  _s(Se_animal=0.0, rho_r=0.0, spread_on=False, clearance_on=True,
                              course_end_on=False),                                                # 3 (clearance on; oracle: no exit)
    "s3_noclear":          _s(Se_animal=0.0, rho_r=0.0, spread_on=False, clearance_on=False,
                              course_end_on=False),                                                # 4 (twin of 3)
    "six_months_noclear":  _s(clearance_on=False),                                                 # 4 (= the clean default since 2026-10-23)
    "six_months_clear":    _s(clearance_on=True),                                                  # 4, 9 (clearance-on twin, 2026-10-23)
    "s2c_se":              _s(Se_animal=0.0, rho_r=0.0, clearance_on=True),                                         # 6
    "s2c_anim":            _s(rho_r=0.0, n_animals_per_round=0, clearance_on=True),                                 # 6
    "s2c_farm":            _s(rho_r=0.0, n_farms_per_round=0, clearance_on=True),                                   # 6
    "s5":                  _s(intro_rate=0.05, rho_r=0.0, Se_animal=1.0, clearance_on=True,
                              n_animals_per_round=100, n_farms_per_round=100, **_DETECTABLE),      # 7
    "s6":                  _s(rho_r=0.0, spread_on=False, clearance_on=False, period_days=30,
                              round_interval=30, n_farms_per_round=20, Se_animal=1.0,
                              n_animals_per_round=100, **_DETECTABLE),                             # 8
    "s_wild":              _s(intro_rate=round(1.0 / 65.0, 4), n_farms_per_round=50, clearance_on=True),            # 9 (doubled introductions)
    "s_wild_nocap":        _s(intro_rate=round(1.0 / 65.0, 4), n_farms_per_round=100, clearance_on=True),            # 9
    "s_wild_capzero":      _s(intro_rate=round(1.0 / 65.0, 4), n_farms_per_round=0, clearance_on=True),            # 9
    "s8_decay":            _s(age_window=120),                                                     # 10
    "s8_logistic":         _s(age_window=200, mean_latent_time=0.0, recovery_within_on=False,
                              p_latent_seed=0.0, p_infectious_seed=0.01),                          # 10
    "s_harder":            _s(rho_r=0.0, n_animals_per_round=5),                                   # 11
    "s11":                 _s(design_prevalence=0.0),                                              # 12
    # ---- pre-seeded start (v0.6); the schedule and clearance come from _BASE ----
    "s14":                 _s(k0_mean=3.0, age0_scale=0.0, intro_rate=0.0, spread_on=False,
                              Se_animal=0.0, rho_r=0.0, clearance_on=True, course_end_on=False),   # 14 (oracle: no exit)
    # ---- course-end exit (2026-10-24): its own oracle, no clearance, nothing detects or spreads ----
    "s19_fresh":           _s(k0_mean=3.0, age0_scale=0.0, intro_rate=0.0, spread_on=False,
                              Se_animal=0.0, rho_r=0.0, clearance_on=False),                       # 19
    "s19_aged":            _s(k0_mean=3.0, age0_scale=14.0, intro_rate=0.0, spread_on=False,
                              Se_animal=0.0, rho_r=0.0, clearance_on=False),                       # 19
    "six_months_noexit":   _s(course_end_on=False, release_on=False, **_LEFTOVERS),               # 19: the 2026-10-23 pack
    "s15_k3":              _s(k0_mean=3.0, age0_scale=14.0),                                       # 15
    "s18_age0":            _s(k0_mean=1.0, age0_scale=0.0),                                        # 18
    "s18_age14":           _s(k0_mean=1.0, age0_scale=14.0),                                       # 18 (= six_months)
    "s18_age60":           _s(k0_mean=1.0, age0_scale=60.0),                                       # 18
}

SCENARIO_LABELS = {
    "county_12mo":         "Zavala County: 12 months from the last detection (monthly rounds)",
    "county_6mo":          "Zavala County: 6 months from the last detection",
    "county_3yr":          "Zavala County: 3 years from the last detection",
    "six_months":          "6 months clean: technically-free mark (180 d, monthly) [default]",
    "zone_last_detection": "Zone at last detection: the APHIS 3-month release window (90 d)",
    "three_cycles":        "3 life cycles clean: one round per 24-d cycle (72 d)",
    "twelve_months":       "12 months clean: officially-free mark (360 d, monthly)",
    "early_warning":       "Early warning: a still-free county 200 km north (introductions every 1,000 d)",
    "s1":                  "1: No introductions",
    "s2":                  "2: No detection, no clearance (pure exponential)",
    "s3":                  "3: Clearance on, no detection, no spread (independent coins)",
    "s3_noclear":          "4: ... same with clearance off (twin of 3)",
    "six_months_noclear":  "4: 6 months, clearance off (the clean default since 2026-10-23)",
    "six_months_clear":    "4: 6 months, clearance on at 21 d (the pre-2026-10-23 default)",
    "s2c_se":              "6: Learn nothing: blind inspection",
    "s2c_anim":            "6: Learn nothing: no animals inspected",
    "s2c_farm":            "6: Learn nothing: no premises visited",
    "s5":                  "7: Census with a perfect inspection",
    "s6":                  "8: One clean inspection round",
    "s_wild":              "9: Wildlife: doubled introductions (65 d), half the premises inspectable",
    "s_wild_nocap":        "9: Wildlife: doubled introductions (65 d), census (no cap)",
    "s_wild_capzero":      "9: Wildlife: doubled introductions (65 d), nothing inspectable",
    "s8_decay":            "10: Inside one premises: the reviewed course (near critical)",
    "s8_logistic":         "10: Inside one premises: logistic (recovery off, latent 0)",
    "s_harder":            "11: Harder inspection cuts two ways (owner reporting off, 5 animals)",
    "s11":                 "12: Design prevalence anchor",
    "s14":                 "14: Poisson-thinning oracle (leftovers clear like independent coins)",
    "s19_fresh":           "19: Course-end exit oracle, fresh leftovers (all leave at the course-end age)",
    "s19_aged":            "19: Course-end exit oracle, leftovers aged 14 d",
    "six_months_noexit":   "19: 6 months without the exit and the release clock (the 2026-10-23 pack)",
    "s15_k3":              "15: Curve starts at e^-3 (k0_mean = 3)",
    "s18_age0":            "18: Leftovers aged 0 d (k0_mean = 1)",
    "s18_age14":           "18: Leftovers aged 14 d (k0_mean = 1)",
    "s18_age60":           "18: Leftovers aged 60 d (k0_mean = 1)",
}

# Input ids by widget type (the loader needs to call the right update function).
CHECKBOXES = ("recovery_within_on", "spread_on", "clearance_on", "course_end_on", "release_on",
              "module_on", "season_on")
NUMERICS = ("c_lost_freedom", "c_per_case", "c_per_test", "c_per_round")


def rounds_in(pack: dict) -> int:
    """Number of inspection rounds in the period covered: floor(period_days / round_interval).
    A last partial round is dropped. A pack that still carries an explicit `n_rounds` keeps it."""
    if "n_rounds" in pack:
        return int(pack["n_rounds"])
    interval = float(pack["round_interval"])
    if interval <= 0.0:
        return 0
    return int(math.floor(float(pack["period_days"]) / interval + 1e-9))


def preset_to_params(pack: dict, dt: float) -> dict:
    """Map an app input pack to the lane's parameter dict. The ONLY such mapping."""
    n_rounds = rounds_in(pack)
    interval = float(pack["round_interval"])
    round_times = [round(interval * (i + 1), 6) for i in range(n_rounds)]
    horizon = max(round_times) if round_times else 0.0
    mean_recovery = float(pack["clearance_mean"]) if pack["clearance_on"] else math.inf
    recovery_within = (float(pack["mean_recovery_time_within"]) if pack["recovery_within_on"]
                       else math.inf)
    doubling_between = float(pack["doubling_time"]) if pack["spread_on"] else math.inf
    # 2026-10-24: a pack without the new keys (any pack written before them) keeps both switches off
    course_end = bool(pack.get("course_end_on", False))
    release = (float(pack["release_mean"]) if pack.get("release_on", False) else None)
    rate = float(pack["intro_rate"])
    intro_interval = math.inf if rate <= 0.0 else 1.0 / rate
    sched = {}
    if pack.get("module_on", False) and horizon > 0.0:
        m = dict(pack, monthly_tmax=COUNTY["monthly_tmax"], monthly_tmin=COUNTY["monthly_tmin"])
        sc = _intro_schedule(m, horizon)
        sched = dict(schedule_times=sc["schedule_times"], intro_multiplier=sc["intro_multiplier"],
                     spread_multiplier=sc["spread_multiplier"])
    return dict(
        **sched,
        mean_latent_time=float(pack["mean_latent_time"]),
        doubling_time_within=float(pack["doubling_time_within"]),
        mean_recovery_time_within=recovery_within,
        N_herd=int(pack["N_herd"]),
        p_latent_seed=float(pack["p_latent_seed"]),
        p_infectious_seed=float(pack["p_infectious_seed"]),
        f_clinical_max=float(pack["f_clinical_max"]),
        mean_clinical_onset_time=float(pack["mean_clinical_onset_time"]),
        rho_a=float(pack["rho_a"]),
        rho_r=float(pack["rho_r"]),
        owner_contact_interval=float(pack.get("owner_contact_interval", 0.0)),   # issue #48
        Se_animal=float(pack["Se_animal"]),
        n_animals_per_round=int(pack["n_animals_per_round"]),
        mean_intro_interval=intro_interval,
        doubling_time=doubling_between,
        mean_recovery_time=mean_recovery,
        beta_lockdown_multiplier=float(pack["beta_lockdown_multiplier"]),
        course_end_exit=course_end,       # 2026-10-24 (accepted 2026-10-28)
        mean_release_time=release,        # 2026-10-24: None = the lane's single clearance rate
        design_prevalence=float(pack["design_prevalence"]),
        M=int(pack["M"]),
        round_times=round_times,
        n_farms_per_round=int(pack["n_farms_per_round"]),
        horizon=horizon,
        # initial state and the release floor (v0.6): the lane reads all three. Before today the
        # threshold stayed in the app, so the lane's t_release ran on its own default.
        k0_mean=float(pack["k0_mean"]),
        age0_scale=float(pack["age0_scale"]),
        release_threshold=float(pack["release_threshold"]),
        dt=float(dt),                     # inherited from the lane, never restated
        c_lost_freedom=float(pack["c_lost_freedom"] or 0.0),
        c_per_case=float(pack["c_per_case"] or 0.0),
        c_per_test=float(pack["c_per_test"] or 0.0),
        c_per_round=float(pack["c_per_round"] or 0.0),
    )
