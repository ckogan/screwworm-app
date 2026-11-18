"""
Zone Freedom from New World Screwworm: Review App (model-cycle stage 4, Shiny for Python).

Spec: ../../app_design.md  ·  Model lane: model.py,
THIS PACK'S OWN closed-form lane since 2026-09-09 (forked from regional-freedom `8f9d7b5` for the
pre-seeded start and edited in place under stage 2), no longer the sibling example's by reference
(decision E1, 2026-09-07, superseded). The status line in the UI names the lane's commit and the
stage-3 verdict that stands behind it.

Two plain panels, live, plus the release question:
  1. The premises (mechanism): within-premises + detection sliders -> detectability-vs-age curves
     p_shed(tau) (larvae-bearing), p_clinical(tau) (visible wound), HSe(tau) (inspection),
     h_pass(tau) (owner reporting).
  2. The zone (outcomes): zone + surveillance sliders -> the P_free running estimate with the
     inspection rounds and the APHIS freedom marks (72 / 90 / 180 / 360 d, never tuned to), the
     release-threshold readout (first round at which P_free reaches the chosen floor; P_free at
     90 d as the APHIS 3-month point), and the scorecard of design-time projections.

Initial state (v0.6, 2026-09-09): the zone may start with the outbreak's leftovers, so the sliders
carry an "Initial state" group (k0_mean undetected infested premises at the last detection, mean age
age0_scale) and the curve starts at P_free(0) = e^{-k0_mean} instead of 1. The release readout is
the lane's own t_release output at the slider's floor.

Dense curve (2026-09-15, changes/2026-09-09-eval-times, option A): ONE lane call per curve. The grid of
plotted times goes to the lane as `eval_times` and every point is the running estimate read off the
single full-horizon march, so the between-round shape is exact, under aged leftovers too. Before this
the curve re-evaluated at a truncated horizon per point (about 30 calls, the slow panel) and the early
points sat low when the leftovers were aged.

What-if mode only (conditional P(free|theta) at the pinned slider values). The integrated headline
is deferred until the priors are lit-reviewed (placeholder table, priors_design.md).

Presets live in presets.py (one table; the stage-5 scenario runner reads the same one).

Presentation (2026-10-29, app_design.md "Presentation modes"): plain wording is the default, for
readers new to the model; NWS_REVIEW=1 in the environment shows the full reviewer text. NWS_DEMO is
kept as an alias for plain. Both modes run the same inputs, model call and numbers.

Run:  shiny run examples/nws-freedom/apps/review_app/app.py                  (plain, repo root)
      NWS_REVIEW=1 shiny run examples/nws-freedom/apps/review_app/app.py     (reviewer text)
"""

from __future__ import annotations

# Cap BLAS threads BEFORE numpy import: the lane runs many small matrix exponentials, which
# thrash multi-core BLAS. Single-thread is ~40x faster in wall time here.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools
import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shiny import App, reactive, render, ui

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from intro_module import MONTH_NAMES, schedule as intro_schedule  # noqa: E402
from presets import (COUNTY, CHECKBOXES, NUMERICS, SCENARIO_LABELS, SCENARIOS,  # noqa: E402
                     preset_to_params, rounds_in)

# --------------------------------------------------------------------------------------
# Load this example's own closed-form lane (forked from regional-freedom 2026-09-09, v0.6).
# --------------------------------------------------------------------------------------
_EXAMPLE = _HERE  # shinylive bundle
_LANE_DIR = _HERE  # shinylive bundle: model.py sits next to app.py
_MODEL_PATH = _LANE_DIR / "model.py"
_spec = importlib.util.spec_from_file_location("rf_model", _MODEL_PATH)
model = importlib.util.module_from_spec(_spec)
sys.modules["rf_model"] = model
_spec.loader.exec_module(model)


def _lane_commit() -> str:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(_LANE_DIR)],
                             cwd=str(_EXAMPLE), capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        return sha or "9b6ccbe (recorded)"
    except Exception:
        return "9b6ccbe (recorded)"


LANE_COMMIT = _lane_commit()
DT = float(model.DEFAULTS["dt"])


STATUS_LINE = (
    f"**Model lane:** `model.py`, **this pack's own lane at "
    f"v0.6 (pre-seeded start, edited 2026-09-09)**, forked from regional-freedom `8f9d7b5`, at "
    f"`{LANE_COMMIT}`, `dt = {DT}` (inherited from the lane). **Two-lane agreement check (stage 3), "
    f"v0.5: run on this pack's screwworm grid 2026-09-07, PASS** (368 evaluable cells, max |z| 3.12, "
    f"one band-edge cell at the parent's known corner `doubling_time_within = 0.1 d`; "
    f"`validation/stage3_verdict_2026-09-07.md`). **v0.6 (both edited lanes, 471-cell grid, 2026-09-09 to "
    f"10): PASS** (418 evaluable cells, max |z| 3.87, one cell beyond 3 sigma against an allowance of 2, "
    f"sign test p = 0.59; reduction anchor exact on all 421 clean-start cells; Poisson-thinning anchor "
    f"within 3 sigma; `validation/stage3_verdict_2026-09-09.md`). "
    f"**Priors: first-pass literature review 2026-09-07** (`priors_design.md`; not yet "
    f"reviewed), and the initial state in the zone presets is a **guessed placeholder**."
)


def _frozen(p: dict):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in p.items()))


@functools.lru_cache(maxsize=512)
def _evaluate_cached(key):
    p = {k: (list(v) if isinstance(v, tuple) else v) for k, v in key}
    return model.evaluate(p)


def evaluate(p: dict) -> dict:
    return _evaluate_cached(_frozen(p))


# --------------------------------------------------------------------------------------
# Presentation mode (2026-10-29, app_design.md "Presentation modes")
# --------------------------------------------------------------------------------------
# Plain presentation is the default: result text only, in plain words, for readers new to the
# model. NWS_REVIEW=1 in the environment restores the full reviewer text (status line, tab notes,
# symbol names, every hint). NWS_DEMO (2026-09-21) is kept as an alias for plain and wins over
# NWS_REVIEW if both are set. The mode changes wording and layout only: the inputs, their ranges
# and defaults, the model call and every computed number are the same in both modes.
def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")


REVIEW = _env_on("NWS_REVIEW") and not _env_on("NWS_DEMO")
PLAIN = not REVIEW
# Default scenario: the county worked example in the plain layout; the generic six-month zone in the
# reviewer layout (its checks are written against it).
DEFAULT_SCENARIO = "county_12mo" if PLAIN else "six_months"
_BASE = SCENARIOS[DEFAULT_SCENARIO]

# Slider ranges, shared by both layouts so the two modes cannot drift: id -> (min, max, step).
# Defaults always come from presets._BASE.
RANGES = dict(
    mean_latent_time=(0.0, 8.0, 0.1), doubling_time_within=(0.3, 30.0, 0.1),
    mean_recovery_time_within=(2.0, 1000.0, 0.5), N_herd=(10, 1000, 10),
    f_clinical_max=(0.0, 1.0, 0.01), mean_clinical_onset_time=(0.1, 10.0, 0.1),
    p_latent_seed=(0.0, 1.0, 0.01), p_infectious_seed=(0.0, 1.0, 0.01), age_window=(5, 200, 5),
    Se_animal=(0.0, 1.0, 0.01), n_animals_per_round=(0, 100, 1),
    rho_a=(0.0, 1.0, 0.01), rho_r=(0.0, 1.0, 0.001), owner_contact_interval=(0.0, 30.0, 0.5),
    M=(20, 500, 10), design_prevalence=(0.0, 0.05, 0.005), intro_rate=(0.0, 0.2, 0.0001),
    doubling_time=(0.01, 150.0, 0.01), clearance_mean=(1, 365, 1), release_mean=(1, 365, 1),
    beta_lockdown_multiplier=(0.0, 1.0, 0.05),
    period_days=(1, 1080, 1), round_interval=(1, 60, 1), n_farms_per_round=(0, 100, 1),
    k0_mean=(0.0, 20.0, 0.1), age0_scale=(0, 180, 1),
    release_threshold=(0.5, 0.999, 0.001),
    # introduction module (issues #55 to #58)
    start_month=(1, 12, 1), temp_shift=(-5.0, 5.0, 0.5),
    share_fly=(0.0, 1.0, 0.05), share_move=(0.0, 1.0, 0.05),
    release_coverage=(0.0, 1.5, 0.05), release_coverage_ref=(0.0, 1.5, 0.05), release_kappa=(0.0, 9.0, 0.05),
    nearby_pressure=(0.0, 5.0, 0.1), wound_factor=(0.25, 3.0, 0.05),
    treat_share=(0.0, 1.0, 0.05), treat_effect=(0.0, 1.0, 0.05),
    move_check_effect=(0.0, 1.0, 0.05), spread_move_share=(0.0, 1.0, 0.05),
)


def _sl(input_id: str, label: str):
    lo, hi, step = RANGES[input_id]
    return ui.input_slider(input_id, label, lo, hi, _BASE[input_id], step=step)


def _cb(input_id: str, label: str):
    return ui.input_checkbox(input_id, label, value=bool(_BASE[input_id]))


def _num(input_id: str, label: str):
    return ui.input_numeric(input_id, label, _BASE[input_id], min=0)


def _review_sidebar():
    """The reviewer layout (NWS_REVIEW=1): every input with its symbol and hint, as before 2026-10-29."""
    return ui.sidebar(
        ui.input_select("scenario", "Scenario preset", SCENARIO_LABELS, selected="six_months"),
        ui.input_action_button("apply_preset", "Load scenario into ALL inputs",
                               class_="btn-sm btn-outline-primary"),
        ui.help_text("Writes every input, including the ones this scenario does not use, so nothing "
                     "carries over. The four screwworm presets share the placeholder priors and differ "
                     "in horizon and round schedule; the numbered ones reproduce scenarios.md."),
        ui.accordion(
            ui.accordion_panel(
                "Within-premises course",
                _sl("mean_latent_time", "Egg laid to larvae visible (d)"),
                ui.help_text("0 = larvae visible at once."),
                _sl("doubling_time_within", "Within-premises doubling time (d)"),
                ui.help_text("Local fly population building on one premises (mass action, an assumption). "
                             "Value 3.6 d (2026-10-23), derived from observed net doubling 9 to 20 d with the "
                             "7 d recovery below: R about 1.35, net doubling about 14 d. The 4.5 d before it "
                             "matched a 14 d recovery; the 14 d placeholder made the course decay."),
                _cb("recovery_within_on", "Within-premises recovery (else never)"),
                _sl("mean_recovery_time_within", "Infestation ends by treatment, healing or death (d)"),
                ui.help_text("Unchecked = recovery OFF exactly (mean = Inf). Needed for the logistic scenario."),
                _sl("N_herd", "Animals on the premises (N_herd)"),
                ui.help_text("Floor 10: the mean-field course is only valid for a herd (C14)."),
                _sl("f_clinical_max", "Fraction of infested animals whose wound becomes visible"),
                _sl("mean_clinical_onset_time", "Days until the wound is obvious"),
                _sl("p_latent_seed", "Seeded latent fraction (p_latent_seed)"),
                _sl("p_infectious_seed", "Seeded larvae-bearing fraction (p_infectious_seed)"),
                _sl("age_window", "Premises-age display window (d)"),
            ),
            ui.accordion_panel(
                "Detection",
                ui.help_text("Scheduled inspection"),
                _sl("Se_animal", "Inspection finds an active myiasis (Se_animal)"),
                _sl("n_animals_per_round", "Animals inspected per premises per round"),
                ui.help_text("0 = no animals inspected, so a round carries no information."),
                ui.help_text("Owner reporting"),
                _sl("rho_a", "Owner notices a visible wound (rho_a)"),
                _sl("rho_r", "Reporting rate once noticed (rho_r, /d)"),
                _sl("owner_contact_interval", "Days between the owner's close looks (0 = every day)"),
                ui.help_text("1/7 = a one-week mean reporting delay."),
            ),
            ui.accordion_panel(
                "Zone design",
                _sl("M", "Premises in the zone (M)"),
                _sl("design_prevalence", "Design prevalence"),
                ui.help_text("0 = absolute freedom, P(K=0) (C12)."),
                _sl("intro_rate", "Zone introduction rate (/day)"),
                ui.help_text("An introduction is a newly infected premises; intro_rate is their rate per day."),
                ui.output_ui("intro_caption"),
                ui.help_text("Fly immigration + infested-animal movement + wildlife, one zone rate."),
                _cb("spread_on", "Between-premises spread (else none)"),
                _sl("doubling_time", "Between-premises doubling time (d)"),
                ui.help_text("Unchecked = spread OFF exactly. Below 0.25 d the lane floors the doubling time."),
                _cb("clearance_on", "Zone clearance: a treated premises leaves the count (else never)"),
                _sl("clearance_mean", "Mean time to clear a premises (d)"),
                ui.help_text("OFF by default since 2026-10-23: before a detection every counted premises is "
                             "undetected, and the 21 d is the release rule for detected, treated premises. "
                             "Ticking it applies 21 d to every premises, found or not (the pre-2026-10-23 pack)."),
                _cb("course_end_on", "Course end: a premises leaves the count when its own infestation "
                                     "has run its course (else never)"),
                ui.help_text("2026-10-24 (accepted 2026-10-28): the premises leaves at the age when fewer than half "
                             "an animal is still infested under the within-premises course above."),
                _cb("release_on", "Release clock after a detection (else the zone clearance above)"),
                _sl("release_mean", "Mean time to release a premises after a detection (d)"),
                ui.help_text("Applies only once a detection has happened (it moves P(R>0), cost and incidence, "
                             "not P_free)."),
                _sl("beta_lockdown_multiplier", "Response multiplier after a new detection (SIT + movement controls)"),
                ui.help_text("Inert for P_free (which conditions on no detection) but it moves the burden outputs."),
            ),
            ui.accordion_panel(
                "Inspection schedule",
                _sl("period_days", "Period covered (days)"),
                _sl("round_interval", "Days between rounds"),
                ui.output_ui("rounds_caption"),
                ui.help_text("Rounds = floor(period / interval); a last partial round is dropped. 180 / 30 = 6 "
                             "(default); 72 / 24 = 3 (3 life cycles); 360 / 30 = 12."),
                _sl("n_farms_per_round", "Premises visited per round"),
                ui.help_text("0 = no premises visited. Set to M for a census."),
            ),
            ui.accordion_panel(
                "Initial state: leftovers at the last detection (v0.6)",
                _sl("k0_mean", "Expected undetected infested premises at the start (k0_mean)"),
                _sl("age0_scale", "Average age of those undetected infestations at the start (age0_scale, d)"),
                ui.help_text("The outbreak's leftovers: premises still infested and not yet found when the "
                             "clock starts. The values in the zone presets (1 premises, 14 d) are guessed "
                             "placeholders, not literature (priors_design.md). 0 = the clean start, which "
                             "reproduces v0.5 exactly. The curve starts at P_free(0) = e^-k0_mean "
                             "(0.3679 at 1, 0.0498 at 3), so the release question is about the climb."),
            ),
            ui.accordion_panel(
                "Introduction module (issues #55 to #58)",
                _cb("module_on", "Module on"), _cb("season_on", "Season on"),
                _sl("start_month", "Start month"), _sl("temp_shift", "Temperature shift (deg C)"),
                _sl("share_fly", "Share: flies"), _sl("share_move", "Share: moved animals"),
                _sl("release_coverage", "Sterile-fly coverage"), _sl("release_coverage_ref", "Coverage (reference)"),
                _sl("release_kappa", "kappa"), _sl("nearby_pressure", "Nearby pressure"),
                _sl("wound_factor", "Wound factor"), _sl("treat_share", "Treated share"),
                _sl("treat_effect", "Treatment effect"), _sl("move_check_effect", "Movement-check effect"),
                _sl("spread_move_share", "Moved-animal share of spread"),
            ),
            ui.accordion_panel(
                "Release question",
                _sl("release_threshold", "P_free floor for releasing the zone"),
                ui.help_text("The APHIS 3-month rule is shown as one point on the curve; the 6- and 12-month marks "
                             "are shown but nothing is tuned to them (E6)."),
            ),
            ui.accordion_panel(
                "Cost coefficients (phase 2 placeholders)",
                _num("c_lost_freedom", "Cost of losing freedom"),
                _num("c_per_case", "Cost per infested premises"),
                _num("c_per_test", "Cost per animal inspected"),
                _num("c_per_round", "Cost per round (fixed)"),
            ),
            # the initial-state group opens too: it is the v0.6 change and it moves the curve's start
            open=["Zone design", "Inspection schedule",
                  "Initial state: leftovers at the last detection (v0.6)"],
            multiple=True,
        ),
        width=380,
        title="Inputs (what-if, pinned theta)",
    )


# Plain scenario menu: the five screwworm scenarios only (the numbered model-check presets are
# reviewer material and stay in the reviewer menu).
PLAIN_SCENARIOS = {
    "county_12mo": "Zavala County: 12 months from the last detection (default)",
    "county_3yr": "Zavala County: 3 years from the last detection",
    "county_6mo": "Zavala County: 6 months from the last detection",
    "six_months": "Generic zone, constant rates: 6 months",
    "zone_last_detection": "Generic zone, constant rates: 3 months",
    "three_cycles": "Generic zone, constant rates: 3 fly life cycles (72 days)",
    "twelve_months": "Generic zone, constant rates: 12 months",
    "early_warning": "Early warning: a still-free county farther north",
}


def _sub(text: str):
    return ui.tags.div(text, style="font-weight:600; margin:10px 0 4px;")


def _plain_sidebar():
    """The plain layout (default): the inputs that move the answer most are open, the rest folded.

    Open: introduction interval, owner reporting, inspection schedule, zone clearance, the four
    leading parameters of the joint sensitivity ranking (validation page, 2026-09-22) plus the
    schedule a program sets. Everything else sits under one collapsed "More settings" group.
    """
    return ui.sidebar(
        ui.input_select("scenario", "Scenario", PLAIN_SCENARIOS, selected=DEFAULT_SCENARIO),
        ui.input_action_button("apply_preset", "Load scenario", class_="btn-sm btn-outline-primary"),
        ui.accordion(
            ui.accordion_panel(
                "Introductions",
                _sl("intro_rate", "Introductions into the zone in a warm month (per day)"),
                ui.help_text("An introduction is a newly infected premises. This is the rate under the "
                             "reference conditions: a warm month, half the sterile-fly target met, "
                             "nearby detections as in summer 2026."),
                ui.output_ui("intro_caption"),
                _sl("release_coverage", "Share of the sterile-fly release target being met"),
                _sl("nearby_pressure", "Detections in nearby counties, relative to summer 2026 (1 = the same)"),
                ui.help_text("Counties to the south count most. 0 = no active zones nearby."),
                _sub("Where introductions come from (shares in a warm month)"),
                _sl("share_fly", "Flies arriving"),
                _sl("share_move", "Moved animals"),
                ui.output_ui("share_caption"),
                _sl("wound_factor", "Open wounds on animals, relative to usual (1 = usual)"),
            ),
            ui.accordion_panel(
                "Season",
                _cb("season_on", "Rates change with the season"),
                _sl("start_month", "Month of the last detection (1 = January)"),
                _sl("temp_shift", "Warmer or cooler than normal (deg C)"),
                ui.help_text(f"Monthly temperatures: {COUNTY['station']}."),
            ),
            ui.accordion_panel(
                "Mitigations",
                _sl("treat_share", "Share of animals given preventive treatment"),
                _sl("treat_effect", "How well treatment protects an animal (0 to 1)"),
                _sl("move_check_effect", "Share of infested moved animals stopped by movement checks"),
                ui.help_text("Treatment lowers introductions and spread between premises. Movement "
                             "checks act on moved animals."),
            ),
            ui.accordion_panel(
                "Owner reporting",
                _sl("owner_contact_interval", "Days between the owner's close looks at the animals"),
                ui.help_text("On large ranches an owner may see the animals closely about once a week. "
                             "Little data: an assumed value."),
                _sl("rho_r", "Reporting rate once a wound is noticed (per day)"),
                ui.output_ui("delay_caption"),
                _sl("rho_a", "Chance an owner notices a visible wound"),
            ),
            ui.accordion_panel(
                "Inspection schedule",
                _sl("period_days", "Period covered (days)"),
                _sl("round_interval", "Days between rounds"),
                ui.output_ui("rounds_caption"),
                _sl("n_farms_per_round", "Premises inspected per round"),
            ),
            ui.accordion_panel(
                "Infestation on a premises",
                _sl("k0_mean", "Undetected infested premises at the start (expected number)"),
                _sl("age0_scale", "Average age of those undetected infestations at the start (days)"),
                _cb("recovery_within_on", "Infested animals recover"),
                _sl("mean_recovery_time_within", "Average time until an infestation ends (days)"),
            ),
            ui.accordion_panel(
                "More settings",
                _sub("Inspection"),
                _sl("Se_animal", "Chance an inspection finds an infested animal"),
                _sl("n_animals_per_round", "Animals inspected per premises per round"),
                _sub("Zone"),
                _cb("module_on", "Use the introduction module (season, pathways, mitigations)"),
                _sl("release_coverage_ref", "Sterile-fly share met when the reference rate was estimated"),
                _sl("release_kappa", "Strength of sterile-fly effect (fertile share = 1 / (1 + k x share))"),
                _sl("spread_move_share", "Share of spread between premises carried by moved animals"),
                _sl("M", "Premises in the zone"),
                _cb("clearance_on", "Infested premises clear after a set time"),
                _sl("clearance_mean", "Average time to clear (days)"),
                _cb("spread_on", "Spread between premises"),
                _sl("doubling_time", "Doubling time of spread between premises (days)"),
                _cb("course_end_on", "A premises recovers when its infestation runs its course"),
                _cb("release_on", "Premises are released on a set clock after a detection"),
                _sl("release_mean", "Average time to release after a detection (days)"),
                _sl("beta_lockdown_multiplier", "Spread remaining under the response after a detection (fraction)"),
                _sl("design_prevalence", "Design prevalence (0 = no infested premises at all)"),
                _sub("Inside one premises"),
                _sl("mean_latent_time", "Egg laid to larvae visible (days)"),
                _sl("doubling_time_within", "Doubling time of the infestation on a premises (days)"),
                _sl("N_herd", "Animals on a premises"),
                _sl("f_clinical_max", "Fraction of infested animals whose wound becomes visible"),
                _sl("mean_clinical_onset_time", "Days until a wound is obvious"),
                _sl("p_latent_seed", "Fraction of animals infested at introduction, larvae not yet visible"),
                _sl("p_infectious_seed", "Fraction of animals carrying larvae at introduction"),
                _sl("age_window", "Days shown on the premises chart"),
                _sub("Reference line"),
                _sl("release_threshold", "Reference line (for illustration)"),
                _sub("Costs (illustrative)"),
                _num("c_lost_freedom", "Cost of a confirmed detection ($)"),
                _num("c_per_case", "Cost per infested premises ($)"),
                _num("c_per_test", "Cost per animal inspected ($)"),
                _num("c_per_round", "Fixed cost per inspection round ($)"),
            ),
            open=["Introductions", "Season", "Owner reporting", "Inspection schedule"],
            multiple=True,
        ),
        width=360,
        title="Settings",
    )


# Plot text sizes for phone-width plots in plain mode (2026-10-29).
_NARROW_RC = {"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5,
              "ytick.labelsize": 6.5, "legend.fontsize": 6.5}

# What the detection box assumes (2026-11-17). P_lost_freedom is P(at least one alarm from an
# inspection round or an owner report over [0, horizon]), horizon = the last round, over all
# outcomes: it is not conditioned on the inspection results. It counts leftovers at the start,
# introductions and spread. A free zone never alarms (L_r(0) = 1 and the owner-report hazard is
# zero at K = 0), so every detection is real: no false positives.
DETECTION_NOTE = ("Computed before any inspection results are counted: the chance that inspections or "
                  "owner reports confirm at least one infested premises by the last round. A confirmed "
                  "detection is treated as real (no false positives).")
# Issue #53: the plain layout shows the complement, under a planning heading.
NO_DETECTION_NOTE = ("Worked out before any inspection results are in: the chance that neither the "
                     "inspections nor owner reports confirm an infested premises by the last round. "
                     "It includes cases where the zone is infested and nothing was found.")
FREE_MEANS = ("**What \"free\" means here:** no infested premises in the zone. It does not mean there "
              "are no flies in the environment. A positive fly trap counts as a detection; empty traps "
              "are given no credit.")
BY_HAND = ("**Conditions this model does not cover, to confirm by hand:** enough roadside checkpoints; "
           "adequate outreach to producers and veterinarians; no pending or unresolved laboratory "
           "samples; wildlife surveillance clean during the window.")

_CAPTION_STYLE = ("font-size:0.95rem; color:#0b5394; padding:4px 10px; margin:2px 0 8px; "
                  "border-left:4px solid #0b5394; background:#eef5fb; border-radius:4px;")

PLAIN_NOTE = ("Draft model estimate based on literature-derived parameters; "
              "for discussion, not a determination.")
FREEDOM_PLOT_TITLE = "Zone freedom over time, with the inspection rounds and the reference line"

if PLAIN:
    app_ui = ui.page_sidebar(
        _plain_sidebar(),
        # plots scale down with the card on narrow screens instead of overflowing it
        ui.tags.style("#pfree_plot img, #premises_plot img { max-width: 100%; height: auto !important; } "
                      ".bslib-page-title { white-space: normal; }"),
        ui.div(PLAIN_NOTE, style=("font-size:0.92rem; color:#5a4a00; background:#fff8e1; "
                                  "border-left:4px solid #e0a800; padding:6px 10px; margin:0 0 8px; "
                                  "border-radius:4px;")),
        ui.navset_card_tab(
            ui.nav_panel(
                "Zone freedom",
                ui.output_ui("release_readout"),
                ui.div(ui.markdown(FREE_MEANS), style=_CAPTION_STYLE),
                ui.output_plot("pfree_plot", height="440px"),
                ui.output_ui("scorecard"),
                ui.div(ui.markdown(BY_HAND), style=_CAPTION_STYLE),
            ),
            ui.nav_panel(
                "Introductions by month",
                ui.p("Where new infested premises come from in each month of the run, as the "
                     "introduction module works it out from the season, sterile-fly coverage, nearby "
                     "detections and mitigations.", style="margin:4px 0 6px;"),
                ui.output_plot("module_plot", height="420px"),
                ui.output_ui("module_note"),
            ),
            ui.nav_panel(
                "What would have to be true",
                ui.output_ui("must_be_true"),
            ),
            ui.nav_panel(
                "Inside one premises",
                ui.p("How an infested premises looks, and how likely it is to be found, as the "
                     "infestation ages.", style="margin:4px 0 6px;"),
                ui.output_plot("premises_plot", height="640px"),
            ),
        ),
        title=f"Zone freedom from New World screwworm: {COUNTY['name']} (worked example)",
        fillable=False,
    )
else:
    app_ui = ui.page_sidebar(
        _review_sidebar(),
        ui.markdown(STATUS_LINE),
        ui.navset_card_tab(
            ui.nav_panel(
                "1 · The premises (mechanism)",
                ui.markdown(
                    "*A freshly infested premises has nothing visible; its catchability moves with time since "
                    "infestation as eggs hatch and wounds become obvious. Under the placeholder priors the "
                    "larvae-bearing fraction **decays** from the seed (subcritical course) rather than building.*"
                ),
                ui.output_plot("premises_plot", height="640px"),
            ),
            ui.nav_panel(
                "2 · The zone (outcomes)",
                ui.output_ui("release_readout"),
                ui.output_ui("scorecard"),
                ui.output_plot("pfree_plot", height="400px"),
                ui.markdown(
                    "*`P_free(t)` is the **running estimate**: the probability the zone has no infested premises "
                    "given every inspection round through `t` came back clean and no owner report was confirmed. "
                    "**The dense curve is exact**: every plotted point, between rounds too, is the running "
                    "estimate read off one full-horizon evaluation of the model (`eval_times`), not a "
                    "re-evaluation at a shorter horizon. "
                    "The faint dashed line is the **unconditional** `P(K=0)(t) = P_free * (1 - P_lost)`; with "
                    "clearance on it settles toward a plateau instead of decaying. Short arrows under the curve "
                    "mark the inspection rounds. Since v0.6 the curve starts at "
                    "`P_free(0) = e^-k0_mean`, the chance the last detection left nothing behind, so with "
                    "leftovers the first stretch is the climb the release rule waits out.*"
                ),
                ui.markdown(
                    "*Scorecard burden and cost figures are **design-time projections at t = 0** over the full "
                    "(K, D) mixture including the post-detection response branch.*"
                ),
            ),
        ),
        title="Zone freedom from New World screwworm: review app (what-if, placeholder priors)",
        fillable=False,
    )


# --------------------------------------------------------------------------------------
# Plain wording helpers (2026-10-29): percentages and the headline sentence
# --------------------------------------------------------------------------------------
def _pct(x: float, near: float | None = None) -> str:
    """A probability as a percentage. Whole percent, one decimal place when the whole-percent
    rounding would blur the reading (below 1 percent, above 99.5 percent, or within one point of
    `near`, the reference line), never "0%" for a positive value nor "100%" for one short of 1."""
    v = 100.0 * float(x)
    if x >= 1.0:
        return "100%"
    if v >= 99.95:
        return ">99.9%"
    if 0.0 < v < 0.05:
        return "<0.1%"
    if 0.0 < v < 1.0:
        return f"{v:.1f}%"
    if v >= 99.5 or (near is not None and abs(v - 100.0 * near) < 1.0):
        return f"{v:.1f}%"
    return f"{v:.0f}%"


def _held_from(out: dict, rounds: list):
    """The first round from which the running chance of freedom stays at or above the reference
    line through the last round, or None. With seasonal rates a winter crossing does not count
    unless it holds to the end of the run."""
    thr = float(out["release_threshold"])
    held = None
    for t in sorted(rounds, reverse=True):
        if float(out["p_free_by_time"][float(t)]) >= thr:
            held = float(t)
        else:
            break
    return held


def _season_headline(out: dict, rounds: list, rows: list) -> str:
    """Plain headline when rates change with the season (issues #56, #59): the value at the last
    round, the range over the run from day 60 on with the months, and what the winter rise means."""
    thr = float(out["release_threshold"])
    pf = out["p_free_by_time"]
    last = float(max(rounds))
    month = lambda t: rows[min(len(rows) - 1, max(0, int(math.ceil(t / 30.0 - 1e-9)) - 1))]["month"]
    later = [float(t) for t in rounds if float(t) >= 60.0] or [last]
    lo = min(later, key=lambda t: pf[t]); hi = max(later, key=lambda t: pf[t])
    first = (f"Chance the zone is free at the last inspection round (day {last:g}, {month(last)}): "
             f"**{_pct(pf[last], thr)}**.")
    if abs(pf[hi] - pf[lo]) < 0.005:
        second = "It is steady over the run."
    else:
        second = (f"Over the run it moves between **{_pct(pf[lo], thr)}** ({month(lo)}) and "
                  f"**{_pct(pf[hi], thr)}** ({month(hi)}). The rise in the cold months comes from fewer "
                  f"flies arriving, not from stronger evidence, and it falls back when the weather warms. "
                  f"Read the warm-season value.")
    held = _held_from(out, rounds)
    line = f"the {_pct(thr)} reference line (for illustration)"
    third = (f"It stays at or above {line} from day {held:g} to the end of the run." if held is not None
             else f"It does not stay at or above {line} to the end of the run.")
    return f"{first} {second} {third}"


def _plain_headline(out: dict, rounds: list) -> str:
    """The plain-mode headline: chance free at the APHIS 3- and 6-month marks (and 12 months
    when the schedule reaches it), then whether and when the reference line is reached. Every
    number is read off the same model output the reviewer readout uses."""
    thr = float(out["release_threshold"])
    pf_by_t = out["p_free_by_time"]
    last = max(rounds) if rounds else 0.0
    parts = []
    for tm, name in ((90.0, "3 months"), (180.0, "6 months"), (360.0, "12 months")):
        if tm in pf_by_t:
            parts.append(f"after {name}: **{_pct(pf_by_t[tm], thr)}**")
    if parts:
        first = "Chance the zone is free " + "; ".join(parts) + "."
    else:
        first = (f"Chance the zone is free at the last inspection round (day {last:g}): "
                 f"**{_pct(pf_by_t.get(float(last), pf_by_t[0.0]), thr)}**.")
    t_rel = out["t_release"]
    line = f"the {_pct(thr)} reference line (for illustration)"
    if t_rel is not None:
        second = f"It reaches {line} at the day-{float(t_rel):g} inspection round."
    else:
        second = f"It does not reach {line} within {last:g} days"
        prev = [t for t in rounds if float(t) < last]
        if prev and last in pf_by_t:
            step = float(pf_by_t[last]) - float(pf_by_t[float(max(prev))])
            second += ("; it is still rising" if step > 1e-4 else
                       "; it is falling" if step < -1e-4 else "; it has levelled off")
        second += "."
    return f"{first} {second}"


# --------------------------------------------------------------------------------------
# Debounce helper (Shiny for Python has no reactive.debounce)
# --------------------------------------------------------------------------------------
def _debounce(delay_secs: float):
    def wrapper(f):
        when = reactive.value(None)
        trigger = reactive.value(0)

        @reactive.effect
        def _schedule():
            f()
            when.set(time.time() + delay_secs)

        @reactive.effect
        def _fire():
            w = when()
            if w is None:
                return
            remaining = w - time.time()
            if remaining <= 0:
                with reactive.isolate():
                    when.set(None)
                    trigger.set(trigger() + 1)
            else:
                reactive.invalidate_later(remaining)

        @reactive.calc
        @reactive.event(trigger, ignore_none=False)
        def settled():
            with reactive.isolate():
                return f()

        return settled
    return wrapper


# --------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------
def _dense_times(p: dict):
    """The dense evaluation grid of the freedom curve and the round times (one lane call serves the
    curve, the boxes and the headline: zone_outputs and pfree_trajectory pass the same eval_times,
    so the second is a cache hit)."""
    dt = float(p["dt"])
    horizon = float(p["horizon"])
    rounds = sorted(float(t) for t in p["round_times"] if float(t) <= horizon + 1e-9)
    interior = 3 if len(rounds) <= 6 else 1           # sparser for long horizons (app_design.md)
    pts = {0.0}
    prev = 0.0
    for tr in rounds:
        for j in range(1, interior + 1):
            t = round(round((prev + (tr - prev) * j / (interior + 1)) / dt) * dt, 6)
            if prev < t < tr:
                pts.add(t)
        pre = round(tr - dt, 6)
        if pre > prev:
            pts.add(pre)
        pts.add(tr)
        prev = tr
    ts = sorted(x for x in pts if 0.0 <= x <= horizon + 1e-9)
    return ts, rounds


def server(input, output, session):

    @reactive.calc
    def pack() -> dict:
        return dict(
            mean_latent_time=input.mean_latent_time(), doubling_time_within=input.doubling_time_within(),
            recovery_within_on=input.recovery_within_on(),
            mean_recovery_time_within=input.mean_recovery_time_within(),
            N_herd=input.N_herd(), f_clinical_max=input.f_clinical_max(),
            mean_clinical_onset_time=input.mean_clinical_onset_time(),
            p_latent_seed=input.p_latent_seed(), p_infectious_seed=input.p_infectious_seed(),
            age_window=input.age_window(),
            Se_animal=input.Se_animal(), n_animals_per_round=input.n_animals_per_round(),
            rho_a=input.rho_a(), rho_r=input.rho_r(),
            owner_contact_interval=input.owner_contact_interval(),
            module_on=input.module_on(), season_on=input.season_on(), start_month=input.start_month(),
            temp_shift=input.temp_shift(), share_fly=input.share_fly(), share_move=input.share_move(),
            release_coverage=input.release_coverage(), release_coverage_ref=input.release_coverage_ref(),
            release_kappa=input.release_kappa(), nearby_pressure=input.nearby_pressure(),
            wound_factor=input.wound_factor(), treat_share=input.treat_share(),
            treat_effect=input.treat_effect(), move_check_effect=input.move_check_effect(),
            spread_move_share=input.spread_move_share(),
            M=input.M(), design_prevalence=input.design_prevalence(), intro_rate=input.intro_rate(),
            spread_on=input.spread_on(), doubling_time=input.doubling_time(),
            clearance_on=input.clearance_on(), clearance_mean=input.clearance_mean(),
            course_end_on=input.course_end_on(), release_on=input.release_on(),
            release_mean=input.release_mean(),
            beta_lockdown_multiplier=input.beta_lockdown_multiplier(),
            period_days=input.period_days(), round_interval=input.round_interval(),
            n_farms_per_round=input.n_farms_per_round(),
            k0_mean=input.k0_mean(), age0_scale=input.age0_scale(),   # initial state (v0.6)
            release_threshold=input.release_threshold(),
            c_lost_freedom=input.c_lost_freedom(), c_per_case=input.c_per_case(),
            c_per_test=input.c_per_test(), c_per_round=input.c_per_round(),
        )

    @reactive.calc
    def params() -> dict:
        return preset_to_params(pack(), DT)

    params_d = _debounce(0.25)(params)

    @reactive.effect
    def _cap_farms_per_round():
        ui.update_slider("n_farms_per_round", min=0, max=int(input.M()))

    # ---- panel 1 ----
    @reactive.calc
    def premises_curves():
        window = float(input.age_window())
        age_grid = np.linspace(0.0, window, 240)
        return age_grid, model.within_farm_curves(params_d(), age_grid)

    # ---- panel 2: one evaluate() -> P_free at rounds + design-time scalars ----
    @reactive.calc
    def zone_outputs():
        p = params_d()
        out = evaluate(dict(p, eval_times=_dense_times(p)[0]))   # same call as the curve
        n_rounds = len([t for t in p["round_times"] if t <= p["horizon"] + 1e-9])
        n_prem = min(int(p["n_farms_per_round"]), int(p["M"]))
        n_insp = n_rounds * n_prem * int(p["n_animals_per_round"])
        breakdown = dict(
            lost_freedom_cost=p["c_lost_freedom"] * out["P_lost_freedom"],
            case_cost=p["c_per_case"] * out["cumulative_incidence"],
            test_cost=p["c_per_test"] * n_insp,
            round_cost=p["c_per_round"] * n_rounds,
            n_insp=n_insp, n_rounds=n_rounds,
        )
        return dict(out=out, breakdown=breakdown, round_times=sorted(p["round_times"]))

    # ---- dense P_free(t) trajectory: ONE evaluate() call per curve with eval_times (2026-09-15) ----
    # The lane reads the running P(K <= K* | no alarm so far) and P(no alarm so far) off its single
    # full-horizon march at every listed time, so every dense point is the exact running estimate.
    # Before this the curve re-evaluated at a truncated horizon per point (about 30 lane calls, the
    # slow panel), and under aged leftovers the early points sat low because the lane truncates the
    # initial-age law at the evaluation horizon. The point rule is unchanged: `interior` points per
    # interval snapped to dt, the t - dt point before each round, the rounds, and 0.
    @reactive.calc
    def pfree_trajectory():
        p = params_d()
        ts, rounds = _dense_times(p)
        # One lane call: the grid goes in as eval_times (every entry a multiple of dt by
        # construction, which the lane checks). The round values and every scalar output of this
        # run are identical to zone_outputs' plain run (lane self-test, check_eval_times (a)).
        o = evaluate(dict(p, eval_times=ts))
        pf_t, pn_t = o["p_free_by_time"], o["p_noalarm_by_time"]
        # t = 0 is the lane's own P_free(0) = P(K_0 <= K*) (v0.6): with leftovers the curve starts
        # at e^{-k0_mean}, not at 1. D(0) = 0, so P_lost(0) = 0 and the unconditional line starts
        # at the same point (model_design.md, "Initial state"). At t > 0 the unconditional
        # P(K <= K*, no alarm so far)(t) = P_free(t) * P(no alarm so far)(t), the second factor the
        # lane's p_noalarm_by_time (P0.sum() at that step), where the truncated route used
        # 1 - P_lost of each shortened run.
        p_free_0 = float(pf_t[0.0])
        pfree, puncond = [], []
        for t in ts:
            if t <= 0.0:
                pfree.append(p_free_0); puncond.append(p_free_0); continue
            pf = float(pf_t[t])
            pfree.append(pf); puncond.append(pf * float(pn_t[t]))
        return dict(t=ts, pfree=pfree, puncond=puncond, rounds=rounds)

    # ---------------------------------------------------------------- panel 1 plot
    def _narrow(output_id: str) -> bool:
        """Plain mode on a phone-width plot: smaller text so titles and legends fit."""
        if not PLAIN:
            return False
        try:
            return float(session.clientdata.output_width(output_id)) < 600
        except Exception:
            return False

    @render.plot
    def premises_plot():
        narrow = _narrow("premises_plot")
        with plt.rc_context(_NARROW_RC if narrow else {}):
            return _premises_fig(narrow)

    def _premises_fig(narrow: bool):
        tau, wf = premises_curves()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
        if PLAIN:
            lab = dict(shed="Animals carrying larvae", clin="Animals with a visible wound",
                       y1="Fraction of animals", t1="Inside one infested premises",
                       hse="Chance one inspection round finds it",
                       hpass="Owner report rate (per day)", x="Days since the premises was infested",
                       y2="Chance or rate", t2="How likely the premises is to be found")
        else:
            lab = dict(shed=r"$p_{shed}(\tau)$ larvae-bearing fraction",
                       clin=r"$p_{clinical}(\tau)$ visible-wound fraction",
                       y1="within-premises fraction", t1="Within-premises course vs premises age",
                       hse=r"$HSe(\tau)$ inspection (per round)",
                       hpass=r"$h_{pass}(\tau)$ owner-reporting hazard (/d)",
                       x=r"premises age $\tau$ (days since infestation)",
                       y2="detectability", t2="Detectability vs premises age")
        ax1.plot(tau, wf["p_shed"], color="#1f77b4", lw=2, label=lab["shed"])
        ax1.plot(tau, wf["p_clinical"], color="#d62728", lw=2, label=lab["clin"])
        ax1.set_ylabel(lab["y1"])
        ax1.set_title(lab["t1"])
        ymax = max(0.02, float(np.nanmax(wf["p_shed"])) * 1.15) if len(wf["p_shed"]) else 1.0
        ax1.set_ylim(0, min(1.02, ymax) if ymax < 0.5 else 1.02)
        ax1.legend(loc="upper right", fontsize=6.5 if narrow else 9)
        ax1.grid(alpha=0.25)
        ax2.plot(tau, wf["HSe"], color="#2ca02c", lw=2, label=lab["hse"])
        ax2.plot(tau, wf["h_pass"], color="#9467bd", lw=2, label=lab["hpass"])
        ax2.set_xlabel(lab["x"])
        ax2.set_ylabel(lab["y2"])
        ax2.set_title(lab["t2"])
        ax2.set_ylim(bottom=0)
        _w = float(tau[-1]) if len(tau) else 1.0
        ax2.set_xticks([_w * k / 4.0 for k in range(5)])
        ax2.legend(loc="upper right", fontsize=6.5 if narrow else 9)
        ax2.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------- panel 2 plot
    @render.plot
    def pfree_plot():
        narrow = _narrow("pfree_plot")
        with plt.rc_context(_NARROW_RC if narrow else {}):
            return _pfree_fig(narrow)

    def _pfree_fig(narrow: bool):
        tr = pfree_trajectory()
        ts, pfree, puncond, rounds = tr["t"], tr["pfree"], tr["puncond"], tr["rounds"]
        thr = float(input.release_threshold())
        fig, ax = plt.subplots(figsize=(9, 4.4 if PLAIN else 4.0))
        if PLAIN:
            lab_free = "Chance the zone is free, given no detection so far"
            lab_unc = "Baseline, before counting the inspection results: chance the zone is free and nothing detected so far"
            lab_thr = f"{_pct(thr)} reference line (for illustration)"
            xlab = "Days since the last detection"
        else:
            lab_free = r"$P_{free}(t)$ (given no confirmed detection yet)"
            lab_unc = r"Baseline, before counting the inspection results: $P_{free}\cdot(1-P_{lost})$"
            lab_thr = f"{thr:.3g} reference line (for illustration)"
            xlab = "time since the last detection, t (days)"
        ax.plot(ts, pfree, color="#1f77b4", lw=2, label=lab_free)
        ax.plot(ts, puncond, color="#1f77b4", lw=1.3, ls="--", alpha=0.5, label=lab_unc)
        # Inspection rounds: a dot on the curve and a short upward arrow just under it (no
        # full-height lines, so the curve crosses nothing). The policy milestones are not drawn
        # (2026-11-17): they are policy dates, not model results.
        for t in rounds:
            idx = min(range(len(ts)), key=lambda i: abs(ts[i] - t))
            ax.plot(ts[idx], pfree[idx], "o", color="#1f77b4", ms=5)
            ax.annotate("", xy=(ts[idx], pfree[idx] - 0.025), xytext=(ts[idx], pfree[idx] - 0.10),
                        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0, mutation_scale=8))
        if rounds:
            ax.plot([], [], ls="none", marker=r"$\uparrow$", color="0.35", ms=9,
                    label="Inspection round" if PLAIN else "inspection round")
        # Season shading (issue #56): the colder the month, the darker the band; month names on top.
        rows = module_rows() if PLAIN else None
        if rows is not None and input.season_on():
            for j, row in enumerate(rows):
                cold = 1.0 - float(row["activity"])
                if cold > 0.02:
                    ax.axvspan(30.0 * j, min(30.0 * (j + 1), ts[-1]), color="#6baed6", alpha=0.35 * cold, lw=0)
            every = max(1, len(rows) // 12)
            for j, row in enumerate(rows):
                if j % every == 0 and 30.0 * j + 15.0 < ts[-1]:
                    ax.text(30.0 * j + 15.0, 1.03, row["month"], ha="center", va="bottom",
                            fontsize=6 if narrow else 7.5, color="0.35")
            ax.fill_between([], [], [], color="#6baed6", alpha=0.3, label="Cold months: low fly activity")
        ax.axhline(thr, color="#2ca02c", lw=1.0, ls="--", alpha=0.7)
        ax.text(0.5, thr + 0.01, lab_thr, color="#2ca02c", fontsize=6 if narrow else 8)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Chance the zone is free")
        # narrow screens (phones): wrap the title and stack the legend so nothing is cut off
        ax.set_title(FREEDOM_PLOT_TITLE.replace(", with", ",\nwith") if narrow else FREEDOM_PLOT_TITLE,
                     pad=16 if rows is not None else 6)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        ticks = [0.0] + list(rounds)
        ax.set_xticks(ticks[::max(1, len(ticks) // 13)])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        if PLAIN:
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        if PLAIN:   # below the axes, so the legend never covers the curve
            ax.legend(loc="upper left" if narrow else "upper center",
                      bbox_to_anchor=(-0.2, -0.16) if narrow else (0.5, -0.16), ncol=1 if narrow else 2,
                      fontsize=7 if narrow else 9, frameon=False)
        else:
            ax.legend(loc="lower left", fontsize=9)
        ax.grid(axis="y", alpha=0.25)   # no vertical grid lines: the curve crosses nothing
        fig.tight_layout()
        return fig

    # (The v0.6 "dense points sit low under aged leftovers" caption that stood here was removed on
    # 2026-09-15: the single-call route above makes every dense point exact, so it no longer applies.)

    # ---------------------------------------------------------------- release readout
    @render.ui
    def release_readout():
        r = zone_outputs()
        out = r["out"]
        if PLAIN:
            rows = module_rows()
            text = (_season_headline(out, r["round_times"], rows)
                    if rows is not None and input.season_on() and r["round_times"]
                    else _plain_headline(out, r["round_times"]))
            return ui.div(ui.markdown(text),
                          style=("font-size:1.1rem; padding:8px 12px; margin:4px 0 8px; "
                                 "border-left:4px solid #2ca02c; background:#f0f8f0; border-radius:4px;"))
        # v0.6: the floor and the crossing are the LANE's own (release_threshold now reaches it
        # through preset_to_params), so the readout cannot disagree with the model about either.
        thr = float(out["release_threshold"])
        pf_by_t = out["p_free_by_time"]
        rounds = r["round_times"]
        p_free_0 = float(pf_by_t[0.0])                      # P(K_0 <= K*), the curve's starting point
        t_rel = out["t_release"]                            # None when the floor is never reached
        last = max(rounds) if rounds else 0.0
        if t_rel is not None:
            first_txt = (f"**first reached at round day {float(t_rel):g}** "
                         f"(`P_free = {pf_by_t[float(t_rel)]:.4f}`)")
        else:
            # not reached: say where the curve ended up and whether it is still climbing, so the
            # reader can tell a slow transient from a curve that has settled below the floor.
            pf_last = float(pf_by_t.get(float(last), p_free_0))
            prev = [t for t in rounds if float(t) < last]
            if prev:
                step = pf_last - float(pf_by_t[float(max(prev))])
                if step > 1e-4:
                    trend = f"still climbing, {step:+.4f} over the last round"
                elif step < -1e-4:
                    trend = f"falling, {step:+.4f} over the last round"
                else:
                    trend = f"settled (the last round moved it {step:+.4f})"
            else:
                trend = "one round only"
            first_txt = (f"**not reached within the {last:g}-d horizon**: at the last round "
                         f"(day {last:g}) `P_free = {pf_last:.4f}`, {trend}")
        if 90.0 in pf_by_t:
            aphis = f"`P_free(90 d) = {pf_by_t[90.0]:.4f}` (the APHIS 3-month point)"
        else:
            last = max(rounds)
            aphis = (f"90 d is not a round in this schedule; last round day {last:g}: "
                     f"`P_free = {pf_by_t[float(last)]:.4f}`")
        text = (f"**Release question.** The curve starts at `P_free(0) = {p_free_0:.4f}` "
                f"(the leftovers at the last detection). Floor `P_free >= {thr:.3f}`: {first_txt}. "
                f"{aphis}.")
        return ui.div(ui.markdown(text),
                      style=("font-size:1.0rem; padding:6px 10px; margin:4px 0 8px; "
                             "border-left:4px solid #2ca02c; background:#f0f8f0; border-radius:4px;"))

    # ---------------------------------------------------------------- scorecard
    @render.ui
    def scorecard():
        r = zone_outputs()
        s, cb = r["out"], r["breakdown"]
        money = lambda x: f"${x:,.0f}"
        t_rel = s["t_release"]                              # v0.6: lane output, None = never reached
        if PLAIN:
            last = max(r["round_times"]) if r["round_times"] else 0.0
            thr = float(s["release_threshold"])
            # owners never notice: inspections only. A second lane call, so skipped on long runs.
            weak = evaluate(dict(params_d(), rho_a=0.0)) if last <= 400.0 else None
            results = ui.layout_columns(
                ui.value_box("Chance free at the start", _pct(s["p_free_by_time"][0.0], thr),
                             theme="bg-gradient-indigo-blue"),
                ui.value_box(f"Chance free at the last round (day {last:g})", _pct(s["P_free"], thr),
                             theme="bg-gradient-blue-purple"),
                ui.value_box(f"The same, if owners never report (day {last:g})",
                             _pct(weak["P_free"], thr) if weak is not None else "not computed",
                             ui.p("Inspection rounds alone. The gap to the box on the left is how much "
                                  "of the result rests on owner reporting."
                                  if weak is not None else
                                  "Computed for runs of up to 400 days, to keep long runs quick.",
                                  style="font-size:0.8rem; margin:0;"),
                             theme="bg-gradient-blue-purple"),
                (ui.value_box(f"{_pct(thr)} reference line held to the end of the run",
                              "no" if _held_from(s, r["round_times"]) is None
                              else f"from day {_held_from(s, r['round_times']):g}",
                              theme="bg-gradient-blue-purple")
                 if module_rows() is not None else
                 ui.value_box(f"{_pct(thr)} reference line first reached",
                              f"not by day {last:g}" if t_rel is None else f"day {float(t_rel):g}",
                              theme="bg-gradient-blue-purple")),
                col_widths=(3, 3, 3, 3),
            )
            planning = ui.layout_columns(
                ui.value_box(f"Chance of no detection by day {last:g} (before any results)",
                             _pct(1.0 - float(s["P_lost_freedom"])),
                             ui.p(NO_DETECTION_NOTE, style="font-size:0.8rem; margin:0;"),
                             theme="bg-gradient-orange-yellow"),
                ui.value_box(f"Expected infested premises by day {last:g}",
                             f"{s['cumulative_incidence']:.1f}", theme="bg-gradient-orange-yellow"),
                ui.value_box("Chance of any spread between premises", _pct(s["P_R_gt0"]),
                             theme="bg-gradient-purple-red"),
                ui.value_box("Expected cost per premises (illustrative)", money(s["cost_share_per_farm"]),
                             theme="bg-gradient-teal-cyan"),
                ui.value_box("Expected total cost (illustrative)", money(s["expected_cost"]),
                             theme="bg-gradient-cyan-blue"),
                col_widths=(4, 4, 4, 6, 6),
            )
            return ui.TagList(
                results,
                ui.tags.h5("Planning", style="margin:18px 0 2px;"),
                ui.p("Projections made at the start, before any inspection results. They are not "
                     "part of the freedom estimate above.", style="font-size:0.9rem; margin:0 0 8px;"),
                planning,
            )
        boxes = ui.layout_columns(
            ui.value_box("P_free at start (t = 0)", f"{s['p_free_by_time'][0.0]:.3f}",
                         theme="bg-gradient-indigo-blue"),
            ui.value_box(f"t_release (floor {s['release_threshold']:.3f})",
                         "not reached" if t_rel is None else f"{float(t_rel):g} d",
                         theme="bg-gradient-blue-purple"),
            ui.value_box("P_free (final round)", f"{s['P_free']:.3f}", theme="bg-gradient-blue-purple"),
            ui.value_box("P_lost_freedom (confirmed detection over horizon)", f"{s['P_lost_freedom']:.3f}",
                         ui.p(DETECTION_NOTE, style="font-size:0.8rem; margin:0;"),
                         theme="bg-gradient-red-orange"),
            ui.value_box("Cumulative incidence", f"{s['cumulative_incidence']:.2f} premises",
                         theme="bg-gradient-orange-yellow"),
            ui.value_box("P(R>0) any between-premises spread", f"{s['P_R_gt0']:.3f}", theme="bg-gradient-purple-red"),
            ui.value_box("Cost share / premises", money(s["cost_share_per_farm"]), theme="bg-gradient-teal-cyan"),
            ui.value_box("Expected cost (total)", money(s["expected_cost"]), theme="bg-gradient-cyan-blue"),
            col_widths=(4, 4, 4, 4, 4, 4, 4, 4),   # 8 boxes since v0.6: three, three, two
        )
        breakdown = ui.markdown(
            f"**Cost breakdown** (placeholder weights): lost freedom {money(cb['lost_freedom_cost'])} · "
            f"cases {money(cb['case_cost'])} · inspections {money(cb['test_cost'])} "
            f"({int(cb['n_insp'])} animal inspections) · rounds {money(cb['round_cost'])} ({cb['n_rounds']} rounds)."
        )
        return ui.TagList(boxes, breakdown)

    @render.ui
    def intro_caption():
        r = float(input.intro_rate())
        k0 = float(input.k0_mean())
        if PLAIN:
            if r > 0.0:
                text = f"About **{1.0 / r:.0f} days** between introductions."
            elif k0 <= 0.0:
                text = "No introductions: the zone stays free."
            else:
                text = ("No introductions. The chance of freedom rises as any undetected infested "
                        "premises at the start are found or recover.")
            return ui.div(ui.markdown(text), style=_CAPTION_STYLE)
        if r > 0.0:
            text = f"about **{1.0 / r:.0f} days** between introductions into the zone (lambda = {r:.4f}/day)."
        elif k0 <= 0.0:
            text = "**no introductions**: `P_free(t) = 1` at all t."
        else:
            # v0.6: with leftovers the curve starts below 1 even with nothing arriving, and only
            # climbs as they clear or are caught. The old wording was true only at k0_mean = 0.
            text = ("**no introductions**, but the zone starts with leftovers: `P_free` climbs from "
                    "`e^-k0_mean` as they clear or are caught, and nothing pushes it back down.")
        return ui.div(ui.markdown(text),
                      style=("font-size:1.0rem; color:#0b5394; font-weight:500; padding:6px 10px; "
                             "margin:4px 0 2px; border-left:4px solid #0b5394; background:#eef5fb; border-radius:4px;"))

    @reactive.calc
    def module_rows():
        pk = pack()
        if not pk["module_on"]:
            return None
        m = dict(pk, monthly_tmax=COUNTY["monthly_tmax"], monthly_tmin=COUNTY["monthly_tmin"])
        return intro_schedule(m, float(params()["horizon"]))["rows"]

    @render.plot
    def module_plot():
        rows = module_rows()
        fig, ax = plt.subplots(figsize=(9, 4.2))
        if rows is None:
            ax.text(0.5, 0.5, "The introduction module is off in this scenario:\nthe introduction rate is constant.",
                    ha="center", va="center"); ax.axis("off")
            return fig
        base = float(input.intro_rate()) * 30.0
        x = list(range(len(rows)))
        fly = [r["fly"] * base for r in rows]; mv = [r["move"] * base for r in rows]
        wl = [r["wild"] * base for r in rows]
        ax.bar(x, fly, color="#1f77b4", label="Flies arriving")
        ax.bar(x, mv, bottom=fly, color="#ff7f0e", label="Moved animals")
        ax.bar(x, wl, bottom=[a + b for a, b in zip(fly, mv)], color="#2ca02c", label="Wildlife")
        step = max(1, len(rows) // 18)
        ax.set_xticks(x[::step]); ax.set_xticklabels([r["month"] for r in rows][::step])
        ax.set_ylabel("Expected new infested premises per month")
        ax.set_xlabel("Month of the run")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    @render.ui
    def module_note():
        rows = module_rows()
        if rows is None:
            return ui.div()
        lo = min(rows, key=lambda r: r["intro"]); hi = max(rows, key=lambda r: r["intro"])
        text = (f"Lowest month: **{lo['month']}**, {100 * lo['intro']:.0f}% of the reference rate "
                f"(fly activity {100 * lo['activity']:.0f}%). Highest: **{hi['month']}**, "
                f"{100 * hi['intro']:.0f}%. Fly activity comes from monthly temperatures for "
                f"{COUNTY['name']}. Rain, wind and terrain are not modelled.")
        return ui.div(ui.markdown(text), style=_CAPTION_STYLE)

    @render.ui
    def must_be_true():
        """Issue #59: the conditions and data a freedom statement from this run rests on, written
        from the current inputs."""
        pk = pack()
        r = zone_outputs()
        last = max(r["round_times"]) if r["round_times"] else 0.0
        n_r = len(r["round_times"])
        share = 100.0 * min(int(pk["n_farms_per_round"]), int(pk["M"])) / max(1, int(pk["M"]))
        delay = (1.0 / float(pk["rho_r"]) if float(pk["rho_r"]) > 0 else float("inf"))
        rows = module_rows()
        warm = (sum(1 for x in rows if x["activity"] >= 0.9) if rows is not None else None)
        season = (f"The run covers **{warm}** warm month{'' if warm == 1 else 's'} (fly activity at "
                  f"90% or more). A clean cold month counts for little."
                  if warm is not None else
                  "Rates are constant in this scenario, so the season is not considered.")
        text = f"""
The chance of freedom shown for this run (**{_pct(r['out']['P_free'])}** at day {last:g}) holds only if all of
the following are true.

**The surveillance happened as entered**
- {n_r} inspection round{'' if n_r == 1 else 's'} over {last:g} days, each visiting {share:.0f}% of premises
  and {int(pk['n_animals_per_round'])} animals on each.
- Owners look their animals over closely at least every {float(pk['owner_contact_interval']):g} days and
  report a wound within about {delay:.0f} days. Most of the result rests on this.
- No confirmed detection of any kind in the zone over the whole period: no inspection finding, no owner
  report, and no positive fly trap.

**The pressure from outside is as entered**
- About one new infested premises every {1.0 / float(pk['intro_rate']) if float(pk['intro_rate']) > 0 else float('inf'):.0f}
  days in a warm month, under the reference conditions.
- Sterile-fly release meets {100 * float(pk['release_coverage']):.0f}% of its target.
- Detections in nearby counties are at {float(pk['nearby_pressure']):g} times their summer 2026 level.
- {season}

**Conditions the model does not cover are confirmed by hand**
- Enough roadside checkpoints.
- Adequate outreach to producers and veterinarians.
- No pending or unresolved laboratory samples.
- Wildlife surveillance clean during the window.
- The neighbouring zones, and zones to the south, are clear for the same window.

**Data that would tighten the answer most**
- How often owners of large ranches see their animals closely, and how quickly they report.
- The share of past detections found by inspection and by owner report.
- Sterile-fly release numbers against the target.
- The share of infestations in wildlife.

One county is a worked example. A statement about the whole state needs many counties and the spread
between them.
"""
        return ui.markdown(text)

    @render.ui
    def share_caption():
        w = max(0.0, 1.0 - float(input.share_fly()) - float(input.share_move()))
        return ui.div(ui.markdown(f"Wildlife: the remaining **{100 * w:.0f}%**. Assumed shares; little data."),
                      style=_CAPTION_STYLE)

    @render.ui
    def rounds_caption():
        n = rounds_in(dict(period_days=input.period_days(), round_interval=input.round_interval()))
        return ui.div(ui.markdown(f"= **{n}** inspection round{'' if n == 1 else 's'}"), style=_CAPTION_STYLE)

    @render.ui
    def delay_caption():
        r = float(input.rho_r())
        tc = float(input.owner_contact_interval())
        if r <= 0.0:
            text = "Owners never report."
        elif tc > 0.0:
            text = (f"Average delay about **{1.0 / r + tc / 2.0:.0f} days** from a visible wound to a "
                    f"report: {tc / 2.0:.1f} days until the owner's next close look, then "
                    f"{1.0 / r:.0f} days to report.")
        else:
            text = f"Average delay about **{1.0 / r:.0f} days** from noticing to reporting."
        return ui.div(ui.markdown(text), style=_CAPTION_STYLE)

    # ---- scenario loader: writes EVERY input ----
    # It iterates the preset's own keys, so the v0.6 ids `k0_mean` and `age0_scale` are written
    # like any other slider: they are keys of every pack in presets.py and neither is a checkbox
    # or a numeric. Anything added to `_BASE` there must have an input id here with the same name.
    @reactive.effect
    @reactive.event(input.apply_preset)
    def _load_scenario():
        pk = SCENARIOS[input.scenario()]
        ui.update_slider("M", value=pk["M"])              # first: farms/round is capped at M
        for k, v in pk.items():
            if k == "M":
                continue
            if k in CHECKBOXES:
                ui.update_checkbox(k, value=bool(v))
            elif k in NUMERICS:
                ui.update_numeric(k, value=v)
            else:
                ui.update_slider(k, value=v)


app = App(app_ui, server)
