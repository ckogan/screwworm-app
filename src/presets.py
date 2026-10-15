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
"""
from __future__ import annotations

import math

# APHIS freedom marks (E6: shown, never tuned to). 24-d life cycle at about 22 C.
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
    mean_latent_time=2.0, doubling_time_within=4.5,
    recovery_within_on=True, mean_recovery_time_within=7.0,
    N_herd=100, f_clinical_max=0.9, mean_clinical_onset_time=3.0,
    p_latent_seed=0.01, p_infectious_seed=0.0, age_window=60,
    # detection: scheduled inspection + owner reporting (alert zone: 3-day mean reporting delay)
    Se_animal=0.85, n_animals_per_round=30, rho_a=0.70, rho_r=0.33,
    # zone (first-pass review, 2026-09-08: a county adjacent to active zones)
    M=100, design_prevalence=0.0, intro_rate=round(1.0 / 130.0, 4),  # 0.0077 /d ~ 130 d
    spread_on=True, doubling_time=10.0,                              # gross, no control (was 30, a net figure)
    clearance_on=True, clearance_mean=21, beta_lockdown_multiplier=0.25,
    # schedule
    n_rounds=6, round_interval=30, n_farms_per_round=20,
    # initial state (v0.6): the CLEAN start is the default, so every scenario-check preset below
    # inherits k0_mean = 0 and reproduces the 2026-09-08 numbers exactly (scenarios.md #16, the
    # reduction anchor). Only the four zone presets carry leftovers.
    k0_mean=0.0, age0_scale=0.0,
    # release question
    release_threshold=0.95,
    # costs (placeholders; phase 2)
    c_lost_freedom=1_000_000, c_per_case=20_000, c_per_test=5, c_per_round=2_000,
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

SCENARIOS = {
    # ---- screwworm presets (F1) ----
    "six_months":          _s(**_LEFTOVERS),                                 # 180 d, 6 monthly rounds
    "zone_last_detection": _s(n_rounds=3, round_interval=30, **_LEFTOVERS),  # 90 d, the APHIS release window
    "three_cycles":        _s(n_rounds=3, round_interval=24, **_LEFTOVERS),  # 72 d, one round per life cycle
    "twelve_months":       _s(n_rounds=12, round_interval=30, **_LEFTOVERS),  # 360 d
    "early_warning":       _s(intro_rate=round(1.0 / 1000.0, 4)),             # still-free county 200 km north (A2)
    # ---- scenario-check presets (scenarios.md numbering) ----
    "s1":                  _s(intro_rate=0.0),                                                    # 1
    "s2":                  _s(Se_animal=0.0, rho_r=0.0, clearance_on=False),                       # 2
    "s3":                  _s(Se_animal=0.0, rho_r=0.0, spread_on=False),                          # 3 (clearance on)
    "s3_noclear":          _s(Se_animal=0.0, rho_r=0.0, spread_on=False, clearance_on=False),      # 4
    "six_months_noclear":  _s(clearance_on=False),                                                 # 4
    "s2c_se":              _s(Se_animal=0.0, rho_r=0.0),                                           # 6
    "s2c_anim":            _s(rho_r=0.0, n_animals_per_round=0),                                   # 6
    "s2c_farm":            _s(rho_r=0.0, n_farms_per_round=0),                                     # 6
    "s5":                  _s(intro_rate=0.05, rho_r=0.0, Se_animal=1.0,
                              n_animals_per_round=100, n_farms_per_round=100, **_DETECTABLE),      # 7
    "s6":                  _s(rho_r=0.0, spread_on=False, clearance_on=False, n_rounds=1,
                              round_interval=30, n_farms_per_round=20, Se_animal=1.0,
                              n_animals_per_round=100, **_DETECTABLE),                             # 8
    "s_wild":              _s(intro_rate=round(1.0 / 65.0, 4), n_farms_per_round=50),              # 9 (doubled introductions)
    "s_wild_nocap":        _s(intro_rate=round(1.0 / 65.0, 4), n_farms_per_round=100),             # 9
    "s_wild_capzero":      _s(intro_rate=round(1.0 / 65.0, 4), n_farms_per_round=0),               # 9
    "s8_decay":            _s(age_window=120),                                                     # 10
    "s8_logistic":         _s(age_window=200, mean_latent_time=0.0, recovery_within_on=False,
                              p_latent_seed=0.0, p_infectious_seed=0.01),                          # 10
    "s_harder":            _s(rho_r=0.0, n_animals_per_round=5),                                   # 11
    "s11":                 _s(design_prevalence=0.0),                                              # 12
    # ---- pre-seeded start (v0.6); the schedule and clearance come from _BASE ----
    "s14":                 _s(k0_mean=3.0, age0_scale=0.0, intro_rate=0.0, spread_on=False,
                              Se_animal=0.0, rho_r=0.0),                                           # 14
    "s15_k3":              _s(k0_mean=3.0, age0_scale=14.0),                                       # 15
    "s18_age0":            _s(k0_mean=1.0, age0_scale=0.0),                                        # 18
    "s18_age14":           _s(k0_mean=1.0, age0_scale=14.0),                                       # 18 (= six_months)
    "s18_age60":           _s(k0_mean=1.0, age0_scale=60.0),                                       # 18
}

SCENARIO_LABELS = {
    "six_months":          "6 months clean: technically-free mark (180 d, monthly) [default]",
    "zone_last_detection": "Zone at last detection: the APHIS 3-month release window (90 d)",
    "three_cycles":        "3 life cycles clean: one round per 24-d cycle (72 d)",
    "twelve_months":       "12 months clean: officially-free mark (360 d, monthly)",
    "early_warning":       "Early warning: a still-free county 200 km north (introductions every 1,000 d)",
    "s1":                  "1: No introductions",
    "s2":                  "2: No detection, no clearance (pure exponential)",
    "s3":                  "3: Clearance on, no detection, no spread (independent coins)",
    "s3_noclear":          "4: ... same with clearance off (twin of 3)",
    "six_months_noclear":  "4: 6 months, clearance off (twin of default)",
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
    "s15_k3":              "15: Curve starts at e^-3 (k0_mean = 3)",
    "s18_age0":            "18: Leftovers aged 0 d (k0_mean = 1)",
    "s18_age14":           "18: Leftovers aged 14 d (k0_mean = 1)",
    "s18_age60":           "18: Leftovers aged 60 d (k0_mean = 1)",
}

# Input ids by widget type (the loader needs to call the right update function).
CHECKBOXES = ("recovery_within_on", "spread_on", "clearance_on")
NUMERICS = ("c_lost_freedom", "c_per_case", "c_per_test", "c_per_round")


def preset_to_params(pack: dict, dt: float) -> dict:
    """Map an app input pack to the lane's parameter dict. The ONLY such mapping."""
    n_rounds = int(pack["n_rounds"])
    interval = float(pack["round_interval"])
    round_times = [round(interval * (i + 1), 6) for i in range(n_rounds)]
    horizon = max(round_times) if round_times else 0.0
    mean_recovery = float(pack["clearance_mean"]) if pack["clearance_on"] else math.inf
    recovery_within = (float(pack["mean_recovery_time_within"]) if pack["recovery_within_on"]
                       else math.inf)
    doubling_between = float(pack["doubling_time"]) if pack["spread_on"] else math.inf
    rate = float(pack["intro_rate"])
    intro_interval = math.inf if rate <= 0.0 else 1.0 / rate
    return dict(
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
        Se_animal=float(pack["Se_animal"]),
        n_animals_per_round=int(pack["n_animals_per_round"]),
        mean_intro_interval=intro_interval,
        doubling_time=doubling_between,
        mean_recovery_time=mean_recovery,
        beta_lockdown_multiplier=float(pack["beta_lockdown_multiplier"]),
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
