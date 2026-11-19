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
import re

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
import streams_illustration as streams  # noqa: E402  (illustration tab only; not the model)

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


# The scenario with no surveillance: nothing is detected. The first two entries switch detection
# off; the rest pin inputs that then have no effect, so the cached run is reused.
NO_SURVEILLANCE = dict(Se_animal=0.0, rho_r=0.0, rho_a=0.0, n_farms_per_round=0,
                       n_animals_per_round=0, owner_contact_interval=0.0)


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
DOUBLING_WARN_BELOW = 3.0     # days; see doubling_caption
# Default scenario: the county worked example in the plain layout; the generic six-month zone in the
# reviewer layout (its checks are written against it).
DEFAULT_SCENARIO = "county_12mo" if PLAIN else "six_months"
_BASE = SCENARIOS[DEFAULT_SCENARIO]

# --------------------------------------------------------------------------------------
# Options and precomputed tables added 2026-11-19 (changes/2026-11-19-linked-counties,
# changes/2026-11-19-observed-weather). Every data file is a copy under data/ (data/README.md), so
# the browser build, which ships this folder only, can read them. All options default to off.
# --------------------------------------------------------------------------------------
import csv  # noqa: E402
import json  # noqa: E402
from datetime import date, datetime  # noqa: E402

import nearby_index  # noqa: E402
from presets import (COUNTY_OBSERVED, INTRO_REF_DAYS, INTRO_REF_FACTOR, OPTION_DEFAULTS,  # noqa: E402
                     intro_ref_factor, module_schedule, renewed_rate)

_DATA = _HERE / "data"
INDEX_COUNTY = ("Texas", "Zavala")          # the worked-example county (presets.COUNTY)
INDEX_PULLED = date(2026, 11, 17)           # export pull date: changes/2026-11-18-county-tailoring/data/README.md
PRECOMPUTED_ON = date(2026, 11, 18)         # date of the linked-counties and 2026-check result files
# Official declaration rules count three or four 21-day life cycles after the last detection;
# older practice used six and twelve months. Policy dates, shown lightly, never tuned to.
RULE_DATES = [(63.0, "3 cycles"), (84.0, "4 cycles"), (180.0, "6 months"), (360.0, "12 months")]
OBSERVED_CAVEAT = ("The in-county station stopped reporting in 2008. The 2026 values come from a "
                   "station 21.5 km away, adjusted.")


def _day(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def _read_csv(name: str) -> list:
    with open(_DATA / name, newline="") as f:
        return list(csv.DictReader(f))


try:
    _CASES = nearby_index.load_cases(_DATA / "detections_official.csv")
    _CENTROIDS = nearby_index.load_centroids(_DATA / "county_centroids_all.csv")
    _ADJACENCY = nearby_index.load_adjacency(_DATA / "county_adjacency.csv",
                                             _DATA / "county_centroids_all.csv")
    INDEX_DATE = max(when for when, _ in _CASES)       # latest confirmation in the export
    INDEX_OK = True
except (OSError, KeyError, ValueError):
    _CASES, _CENTROIDS, _ADJACENCY, INDEX_DATE, INDEX_OK = [], {}, {}, None, False


try:
    # First day of the worked example's run: the county's last detection (presets.COUNTY).
    INDEX_START = datetime.strptime(COUNTY["last_detection"], "%d %B %Y").date()
except (KeyError, ValueError):
    INDEX_START = None


def _index_periods() -> int:
    """Number of 30-day periods of the run that begin on or before the latest confirmation date."""
    if INDEX_START is None or INDEX_DATE is None or INDEX_DATE < INDEX_START:
        return 0
    return (INDEX_DATE - INDEX_START).days // int(nearby_index.MONTH_DAYS) + 1


def _ref_days_text() -> str:
    """The interval of the adjacency-count option as a reader sees it, rounded and exact, for
    example "400 days (397)". Built from days_per_event_counted in data/index_reference.json."""
    return f"{round(INTRO_REF_DAYS, -1):.0f} days ({INTRO_REF_DAYS:.0f})"


@functools.lru_cache(maxsize=256)
def county_index(kernel_km: float, south_weight: float) -> dict:
    """The nearby-pressure index for the county on the data date, with its three largest source
    counties. The reference level is recomputed with the same two dials (nearby_index.py)."""
    dials = dict(kernel_km=float(kernel_km), south_weight=float(south_weight))
    ref = nearby_index.reference_level(_CASES, _CENTROIDS, _ADJACENCY, **dials)["level"]
    rows = nearby_index.index_breakdown(INDEX_COUNTY, INDEX_DATE, _CASES, _CENTROIDS, ref, **dials)
    value = nearby_index.nearby_index(INDEX_COUNTY, INDEX_DATE, _CASES, _CENTROIDS, ref, **dials)
    return dict(value=value, top=rows[:3], n_sources=len(rows),
                series=nearby_index.index_by_segment(INDEX_COUNTY, INDEX_START, _index_periods(),
                                                     _CASES, _CENTROIDS, ref, **dials))


def _load_tables() -> dict | None:
    """The precomputed linked-counties run and the reference counts, or None if a file is missing."""
    try:
        with open(_DATA / "linked_summary.json") as f:
            summary = json.load(f)
        with open(_DATA / "index_reference.json") as f:
            reference = json.load(f)
        return dict(
            summary=summary, reference=reference,
            counties=[r for r in _read_csv("linked_counties.csv") if r["run"] == "main"],
            group=[r for r in _read_csv("linked_group.csv") if r["run"] == "main"],
        )
    except (OSError, KeyError, ValueError):
        return None


TABLES = _load_tables()
_MISSING = "The data files for this view were not found in the app's data folder."


def _load_calibration() -> dict | None:
    """The summary of the held-out calibration against 2026 (data/calibration_2026_summary.json, a
    few values copied from the results of changes/2026-11-19-calibration-2026), or None if the file
    is missing or incomplete."""
    try:
        with open(_DATA / "calibration_2026_summary.json") as f:
            cal = json.load(f)
        for key in ("test_rows", "differences", "fitted_interval_days", "fitted_interval_profile_95",
                    "timing", "earlier_case", "no_earlier_case"):
            cal[key]
        return cal
    except (OSError, KeyError, ValueError):
        return None


CALIBRATION = _load_calibration()


def _load_fair() -> dict | None:
    """The check against 2026 on equal windows (data/backcheck_fair.json, a short copy of the
    corrected numbers of changes/2026-11-19-linked-counties), or None if the file is missing or
    incomplete. The first version of the check (backcheck_texas.json) is superseded and not read."""
    try:
        with open(_DATA / "backcheck_fair.json") as f:
            fair = json.load(f)
        for window, groups in (("whole_period", ("june_to_august", "september_to_mid_november")),
                               ("autumn_window", ("no_case_to_31_aug", "first_wave_case"))):
            w = fair[window]
            for key in ("label", "ratio_expected_to_seen", "brier_model", "brier_base_rate",
                        "base_rate_scores_better", "rank_score"):
                w[key]
            for g in (w["total"],) + tuple(w["by_group"][k] for k in groups):
                float(g["expected"]), int(g["seen"]), int(g["counties"])
        return fair
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _load_fair_seven() -> list | None:
    """The autumn-window rows of the county and its neighbours (data/backcheck_fair_seven.csv, seven
    rows copied from results/backcheck_fair.csv of the same change), or None if the file is missing."""
    try:
        rows = _read_csv("backcheck_fair_seven.csv")
        for r in rows:
            float(r["a_p_new_detection_autumn"]), int(r["a_seen"]), r["county"], r["group"]
        return rows or None
    except (OSError, KeyError, TypeError, ValueError):
        return None


FAIR = _load_fair()
FAIR_SEVEN = _load_fair_seven()
RENEWED_RANGE = (0.0, 0.30, 0.01)       # the what-if input: chance per 30 days, off at 0
_CAL_LABELS = {
    "unfitted": "Model as built, nothing fitted",
    "fitted_A_off": "Model fitted on June to August",
    "base_rate_A": "One base rate for every county",
    "adjacency_rate_A": "Base rate by whether the county adjoins a county with cases",
}


def _renewed_help() -> str:
    """Help text of the what-if input. The two counts are read from the calibration summary."""
    seen = ""
    if CALIBRATION is not None:
        ec = CALIBRATION["earlier_case"]
        seen = (f"In 2026, Texas counties with an earlier case had a renewed detection in "
                f"{int(ec['observed'])} of {int(ec['county_periods'])} county-periods of 30 days. "
                f"The model expected about {float(ec['expected']):.0f} of those, even after its "
                f"introduction rate was fitted to the summer data. ")
    return ("Off at 0. " + seen +
            "The cause is not known: persistence the model does not capture, undetected "
            "infestations, wildlife, or re-introduction. The value is added as a constant rate of new "
            "infested premises. Season, sterile flies, nearby detections and mitigations do not change "
            "it. Owners and rounds must still find a renewed infestation, so the chance of a renewed "
            "detection the model gives is lower than the value set here in the first month or two, then about equal. The value cannot "
            "be tested until new months of data arrive.")


def _intro_caution() -> str:
    """Standing caution under the headline of a scenario that uses the introduction module. The
    numbers are read from the check on equal windows and are given only when that file is present
    and shows the stated direction. Without it the sentence stands without numbers."""
    times, season, earlier = "more", "", ""
    if FAIR is not None:
        tot = FAIR["whole_period"]["total"]
        if float(tot["expected"]) > int(tot["seen"]):
            times = f"about {float(FAIR['whole_period']['ratio_expected_to_seen']):.1f} times as many"
            season = (f" ({float(tot['expected']):.0f} Texas counties expected with a first "
                      f"detection, {int(tot['seen'])} seen)")
        fw = FAIR["autumn_window"]["by_group"]["first_wave_case"]
        if float(fw["expected"]) < int(fw["seen"]):
            earlier = f" (about {float(fw['expected']):.0f} expected in autumn, {int(fw['seen'])} seen)"
    return ("**Caution:** the introduction side of this model has not been validated against "
            f"outcomes. In a first check against 2026 it expected {times} new detections "
            f"{'as' if FAIR is not None and season else 'than'} occurred over the season{season}, "
            f"and too few in counties that already had a case{earlier}. "
            "See the \"Check against 2026\" tab.")


def _scores_line(w: dict) -> str:
    """One sentence on the model against one base rate for a window of the check on equal windows.
    The direction comes from the flag in the file."""
    m, b = float(w["brier_model"]), float(w["brier_base_rate"])
    if w["base_rate_scores_better"]:
        return (f"One base rate for every county scores better than the model here "
                f"(Brier score {b:.3f} against {m:.3f} for the model; lower is better).")
    how = "slightly better" if (b - m) < 0.2 * b else "better"
    return (f"On this window the model scores {how} than one base rate for every county "
            f"(Brier score {m:.3f} against {b:.3f}; lower is better).")


def _too(g: dict) -> str:
    e, o = float(g["expected"]), int(g["seen"])
    return "too many" if e > o else "too few" if e < o else "the same"


def check_2026_block():
    """The check on equal windows on the "Check against 2026" tab: the whole period, the autumn
    window, the county and its neighbours, the correction note. Every number is read from
    data/backcheck_fair.json and data/backcheck_fair_seven.csv."""
    if FAIR is None:
        return ui.div(_MISSING, style=_CAPTION_STYLE)
    wp, au = FAIR["whole_period"], FAIR["autumn_window"]
    wt, at = wp["total"], au["total"]
    ja, sn = wp["by_group"]["june_to_august"], wp["by_group"]["september_to_mid_november"]
    nc, fw = au["by_group"]["no_case_to_31_aug"], au["by_group"]["first_wave_case"]
    head = (f"**Over the season the model expected about {float(wp['ratio_expected_to_seen']):.1f} "
            f"times as many new detections as occurred.** In autumn it expected "
            f"{float(au['ratio_expected_to_seen']):.1f} times as many overall, and {_too(fw)} in "
            f"counties that already had a case.")
    whole = (f"**Whole period, on equal terms.** {wp['label']}. The first confirmed detection in "
             f"the period is counted. Over {int(wt['counties'])} Texas counties the model expected "
             f"**{float(wt['expected']):.1f}** counties with a detection. **{int(wt['seen'])}** had "
             f"one. That is a factor of {float(wp['ratio_expected_to_seen']):.1f}. June to August: "
             f"{float(ja['expected']):.1f} expected, {int(ja['seen'])} seen. September to mid "
             f"November: {float(sn['expected']):.1f} expected, {int(sn['seen'])} seen. "
             + _scores_line(wp))
    rank = float(au["rank_score"])
    autumn = (f"**Autumn window.** {au['label']}. The model expected **{float(at['expected']):.1f}** "
              f"counties with a new confirmed detection. **{int(at['seen'])}** had one. That is a "
              f"factor of {float(au['ratio_expected_to_seen']):.1f}. In the {int(nc['counties'])} "
              f"counties with no case to 31 August it expected {float(nc['expected']):.1f} and "
              f"{int(nc['seen'])} came: {_too(nc)}. In the {int(fw['counties'])} counties with a "
              f"first-wave case it expected {float(fw['expected']):.1f} and {int(fw['seen'])} came: "
              f"{_too(fw)}. " + _scores_line(au)
              + (f" Its ordering of counties is {'fair' if rank >= 0.7 else 'weak'}: in about "
                 f"{100.0 * rank:.0f} percent of pairs of one county with a detection and one "
                 f"without, the county with the detection had the higher chance."))
    adj = au.get("brier_adjacency_base_rate")
    if adj is not None and float(adj) < float(au["brier_model"]):
        autumn += (f" A base rate that knows only the county's state on 31 August scores "
                   f"{float(adj):.3f}, which is better than the model; its rates are taken from "
                   f"the same outcomes, which favours it.")
    parts = [
        ui.div(ui.markdown(head), style=("font-size:1.1rem; padding:8px 12px; margin:4px 0 8px; "
                                         "border-left:4px solid #b45f06; background:#fdf3e7; "
                                         "border-radius:4px;")),
        ui.markdown(whole),
        ui.markdown(autumn),
    ]
    if FAIR_SEVEN is not None:
        ps = [float(r["a_p_new_detection_autumn"]) for r in FAIR_SEVEN]
        seen = sum(int(r["a_seen"]) for r in FAIR_SEVEN)
        rows = [[r["county"], r["group"], _pct(float(r["a_p_new_detection_autumn"])),
                 (r.get("a_first_autumn_case") or "yes") if int(r["a_seen"]) else "none"]
                for r in FAIR_SEVEN]
        none_chance = math.prod(1.0 - p for p in ps)
        tail = (f" By the model's own numbers that happens with chance {none_chance:.2f}, so these "
                f"counties say nothing either way." if seen == 0 else "")
        parts += [
            ui.p(f"{INDEX_COUNTY[1]} County and its neighbours, on the autumn window.",
                 style="margin:8px 0 2px;"),
            _table(["County", "State on 31 August", "Model: chance of a new detection in autumn",
                    "New detection in autumn"], rows),
            ui.markdown(f"The model's chances for these {len(rows)} counties add up to {sum(ps):.2f} "
                        f"expected. {seen if seen else 'None'} of them had a new detection.{tail}"),
        ]
    sup = FAIR.get("superseded") or {}
    if "expected" in sup and "seen" in sup:
        when = re.search(r"Corrected on (\d{1,2} [A-Z][a-z]+ \d{4})", str(FAIR.get("correction_note", "")))
        parts.append(ui.markdown(
            f"**Correction{', ' + when.group(1) if when else ''}:** the first version of this check "
            f"compared unequal windows (its expected count included June to August risk for counties "
            f"selected for having no case in those months) and overstated the over-prediction; it gave "
            f"{float(sup['expected']):.1f} expected against {int(sup['seen'])} seen. It was corrected "
            f"after an independent re-run."))
    if TABLES is not None:
        ref = TABLES["reference"]
        parts.append(ui.markdown(
            f"**What the check points to.** The reference introduction rate looks about three times "
            f"too high: the adjacency file gives {int(ref['at_risk_counties_min'])} to "
            f"{int(ref['at_risk_counties_max'])} counties at risk in summer 2026, not the 12 to 20 "
            f"assumed when the rate was set. The model also has no term for a county's own history, "
            f"so it expects too few new detections where there was an earlier case. The check does "
            f"not show the leftovers assumed at the last detection to be too high."))
    parts.append(ui.markdown(
        f"Precomputed on {_day(PRECOMPUTED_ON)} for the default settings, with the analytic build of "
        f"the model. One season, {int(at['seen'])} events in autumn and {int(wt['seen'])} in all, in "
        f"counties that are not independent: a check, not a calibration. The introduction-rate "
        f"option on the left applies the lower rate"
        + (f", about one introduction per {_ref_days_text()}." if INTRO_REF_DAYS else ".")))
    return ui.TagList(*parts)


def calibration_2026_block():
    """The held-out calibration on the "Check against 2026" tab: verdict first, then the scores, the
    fitted introduction interval and the two findings. Every number is read from the summary file.
    Returns None, so the tab shows nothing more, when the file is missing."""
    cal = CALIBRATION
    if cal is None:
        return None
    lo_b, hi_b = cal["differences"]["base_rate_A"]["boot_95"]          # fitted minus base rate, Brier
    lo_a, hi_a = cal["differences"]["adjacency_rate_A"]["boot_95"]
    if lo_b <= 0.0 <= hi_b and lo_a > 0.0:
        verdict = ("**A calibration fitted on June to August and tested on September to mid November "
                   "does not beat the base rates.** It ties one base rate for every county and is "
                   "worse than a base rate that knows only whether the county adjoins a county with "
                   "cases.")
    else:
        verdict = ("**A calibration was fitted on June to August and tested on September to mid "
                   "November.** The scores are in the table below.")
    rows = [[_CAL_LABELS.get(r["key"], r["key"]), f"{float(r['expected']):.1f}", f"{int(r['observed'])}",
             f"{float(r['brier']):.5f}"] for r in cal["test_rows"]]
    n = int(cal["test_rows"][0]["county_periods"])
    tw = cal.get("test_window")
    window = (f" from {_day(date.fromisoformat(tw['start']))} to {_day(date.fromisoformat(tw['end']))}"
              if tw else "")
    lo, hi = cal["fitted_interval_profile_95"]
    interval = (f"**Fitted introduction interval:** about {float(cal['fitted_interval_days']):.0f} days "
                f"between introductions ({float(lo):.0f} to {float(hi):.0f}). This value and its range "
                f"were fitted after seeing the 2026 data. The app does not use them.")
    t = cal["timing"]
    timing = ("**Timing runs the wrong way.** Over the test periods the fitted model expected "
              + ", then ".join(f"{float(r['expected']):.1f}" for r in t)
              + " county-periods with a detection. The counts seen were "
              + ", then ".join(f"{int(r['observed'])}" for r in t) + ".")
    ec, nc = cal["earlier_case"], cal["no_earlier_case"]
    earlier = (f"**Counties with an earlier case are under-predicted.** {int(ec['observed'])} of "
               f"{int(ec['county_periods'])} such county-periods had a renewed detection "
               f"({_pct(ec['observed'] / ec['county_periods'])}). The fitted model expected "
               f"{float(ec['expected']):.1f}. Where there had been no case the share was "
               f"{100.0 * nc['observed'] / nc['county_periods']:.1f}%. The model has no term for this. "
               f"The what-if input under Introductions adds one.")
    return ui.TagList(
        ui.tags.h5("Calibration, tested on later months", style="margin:18px 0 4px;"),
        ui.div(ui.markdown(verdict), style=("font-size:1.1rem; padding:8px 12px; margin:4px 0 8px; "
                                            "border-left:4px solid #b45f06; background:#fdf3e7; "
                                            "border-radius:4px;")),
        ui.p(f"Test window{window}: {n} county-periods of up to 30 days in Texas. Expected and seen "
             f"count county-periods with a new confirmed detection. A lower Brier score is better.",
             style="margin:8px 0 2px;"),
        _table(["Predictor", "Expected", "Seen", "Brier score"], rows),
        ui.markdown(interval),
        ui.markdown(timing),
        ui.markdown(earlier),
    )


def _table(header: list, rows: list, bold_last: bool = False):
    """A small plain table."""
    body = []
    for i, row in enumerate(rows):
        style = "font-weight:600; border-top:2px solid #999;" if bold_last and i == len(rows) - 1 else ""
        body.append(ui.tags.tr(*[ui.tags.td(c) for c in row], style=style))
    return ui.tags.div(
        ui.tags.table(ui.tags.thead(ui.tags.tr(*[ui.tags.th(h) for h in header])),
                      ui.tags.tbody(*body), class_="table table-sm",
                      style="font-size:0.9rem; width:auto;"),
        style="overflow-x:auto;")


def _intro_ref_inputs():
    """The reference-rate option (both layouts): which count of counties at risk the rate rests on."""
    alt = (f"Counties at risk counted from the adjacency file: about one introduction per "
           f"{_ref_days_text()}" if INTRO_REF_DAYS else "Adjacency count (data file missing)")
    mult = (f" by {INTRO_REF_FACTOR:.3g}. At the default rate that is about one introduction per "
            f"{_ref_days_text()}." if INTRO_REF_DAYS else ".")
    return [
        ui.input_radio_buttons(
            "intro_ref", "Counties at risk behind this rate",
            {"record": "Current value (about one introduction per 130 days)",
             "adjacency": alt},
            selected=OPTION_DEFAULTS["intro_ref"]),
        ui.help_text("The check against 2026 suggests the current value is about three times too "
                     "high. The choice is open. The second option multiplies the rate entered above" + mult),
    ]


def _nearby_source_inputs(*hand):
    """The source of nearby pressure (both layouts): the choice, the hand slider passed in as `hand`,
    the readout, and the index's two dials in a fold."""
    lo_k, hi_k = nearby_index.RANGES["kernel_km"]
    lo_s, hi_s = nearby_index.RANGES["south_weight"]
    return [
        ui.input_radio_buttons(
            "nearby_source", "Detections in nearby counties: source",
            {"hand": "Set by hand", "index": "From official detections (index)"},
            selected=OPTION_DEFAULTS["nearby_source"]),
        *hand,
        ui.output_ui("index_readout"),
        ui.accordion(
            ui.accordion_panel(
                "Assumptions with little data",
                ui.tags.div("Nearby-pressure index", style="font-weight:600; margin:2px 0 4px;"),
                ui.input_slider("kernel_km", "Distance scale (km)", lo_k, hi_k,
                                OPTION_DEFAULTS["kernel_km"], step=10),
                ui.input_slider("south_weight", "Weight on counties to the south (1 = none)", lo_s, hi_s,
                                OPTION_DEFAULTS["south_weight"], step=0.5),
                ui.help_text("Both rest on thin data. The distance scale follows the speed of the "
                             "historical seasonal front, which is not a measured dispersal distance. "
                             "The size of the southern weight is assumed."),
                ui.tags.div("Renewed infestations (what-if)", style="font-weight:600; margin:14px 0 4px;"),
                ui.input_slider("renewed_chance",
                                "Chance per 30 days of a renewed infestation from causes the model "
                                "does not represent (what-if)", RENEWED_RANGE[0], RENEWED_RANGE[1],
                                OPTION_DEFAULTS["renewed_chance"], step=RENEWED_RANGE[2]),
                ui.help_text(_renewed_help()),
                value="index_dials_panel",
            ),
            id="index_dials", open=False,
        ),
    ]


def _weather_inputs():
    """The weather option (both layouts)."""
    return [
        ui.input_radio_buttons(
            "weather_source", "Monthly temperatures",
            {"normals": "30-year normals", "observed": "2026 observed where available"},
            selected=OPTION_DEFAULTS["weather_source"]),
        ui.help_text(OBSERVED_CAVEAT + " June to November 2026 are observed. November is a part "
                     "month. The other months keep the normals."),
    ]


def _neighbours_panel():
    return ui.nav_panel(
        "Neighbouring counties",
        ui.output_ui("neighbours_intro"),
        ui.output_ui("neighbours_table"),
        ui.output_plot("neighbours_plot", height="420px"),
        ui.output_ui("neighbours_note"),
    )


def _check_panel():
    return ui.nav_panel(
        "Check against 2026",
        ui.output_ui("check_2026"),
        ui.output_ui("calibration_2026"),
    )

# --------------------------------------------------------------------------------------
# "Traps and wildlife (illustration)" tab (2026-11-19). A PROPOSED design, not built: see
# model_design.md, "Additional evidence streams". The tab has its own inputs (ids start with
# "st_"), reads no model output and feeds nothing to the model. streams_illustration.py holds the
# arithmetic and the proposed values and ranges.
# --------------------------------------------------------------------------------------
STREAMS_TAB = "Traps and wildlife (illustration)"
STREAMS_BANNER = ("Illustration of a proposed design. Not part of the model. Nothing on this tab "
                  "changes the result on the other tabs.")
_STREAM_COLOURS = {"trap": "#1f77b4", "wild": "#2ca02c", "both": "#222222"}


def _stream_inputs(key: str, switch_label: str, coverage_label: str, rate_label: str, delay_label: str):
    s = streams.STREAMS[key]
    c, q, d = s["coverage"], s["rate"], s["delay"]
    return ui.div(
        ui.input_checkbox(f"st_{key}_on", switch_label, value=False),
        ui.panel_conditional(
            f"input.st_{key}_on",
            ui.input_slider(f"st_{key}_coverage", coverage_label, min=c["low"], max=c["high"],
                            value=c["value"], step=c["step"]),
            ui.input_slider(f"st_{key}_rate", rate_label, min=q["low"], max=q["high"],
                            value=q["value"], step=q["step"]),
            ui.output_ui(f"st_{key}_week"),
            ui.input_slider(f"st_{key}_delay", delay_label, min=d["low"], max=d["high"],
                            value=d["value"], step=d["step"]),
        ),
        style="flex:1 1 300px; min-width:260px; padding:8px 12px; border:1px solid #ddd; border-radius:6px;")


def _streams_panel():
    return ui.nav_panel(
        STREAMS_TAB,
        ui.div(ui.tags.strong(STREAMS_BANNER),
               style=("font-size:1rem; color:#7a1f1f; background:#fdecea; border-left:4px solid #c0392b; "
                      "padding:8px 12px; margin:0 0 10px; border-radius:4px;")),
        ui.p("The model does not yet use fly traps or wildlife findings as evidence. A positive trap "
             "means a problem. This tab shows how much empty traps and clean wildlife checks could "
             "count under a proposed design that is not built. Both are switched off to start with, "
             "so empty traps and clean wildlife checks earn nothing until you switch them on.",
             style="margin:4px 0 8px;"),
        ui.div(
            _stream_inputs("trap", "Use fly traps",
                           "Coverage: share of infested premises the traps could ever find",
                           "Rate per day of a confirmed catch, for a premises within reach",
                           "Delay in days before an infested premises sends out adult flies"),
            _stream_inputs("wild", "Use wildlife and companion-animal findings",
                           "Coverage: share of infested premises with a wildlife or dog case someone could find",
                           "Rate per day that such a case is found and confirmed",
                           "Delay in days before such a case can exist and be seen"),
            ui.div(
                ui.input_slider("st_hidden", "Expected hidden infested premises at day 0",
                                min=streams.HIDDEN["low"], max=streams.HIDDEN["high"],
                                value=streams.HIDDEN["value"], step=streams.HIDDEN["step"]),
                ui.help_text("Used by the second plot and by the credit rows of the table only. "
                             "The sliders start at the proposed values and span the proposed ranges."),
                style="flex:1 1 300px; min-width:260px; padding:8px 12px; border:1px solid #ddd; border-radius:6px;"),
            style="display:flex; flex-wrap:wrap; gap:10px; margin:0 0 10px;"),
        ui.tags.h5("Chance that one infested premises would have been found"),
        ui.output_plot("st_found_plot", height="380px"),
        ui.div("Empty results earn nothing before the delay, a little after it, and never more than "
               "the coverage. The dashed lines are the coverage ceilings.", style=_ST_CAPTION),
        ui.tags.h5("Simplest case: chance the zone is free, given every check so far was negative"),
        ui.output_plot("st_free_plot", height="380px"),
        ui.div("The simplest case only: the hidden premises are present from day 0, with no owner "
               "reporting, no inspection rounds, no new introductions and no premises recovering. "
               "This is how much empty traps and clean wildlife checks alone could move the answer "
               "under the proposed design. In the full model owners and inspection rounds already "
               "find most infestations, so the added credit would be smaller. That combination is "
               "not built.", style=_ST_CAPTION),
        ui.tags.h5("The numbers, and the spread across the proposed ranges"),
        ui.output_ui("st_table"),
        ui.tags.h5("Thin data, and what the proposed design leaves out"),
        ui.p("No measured trap sensitivity for this species against a wild population at low density "
             "is on file, and no rate at which wildlife or dog cases reveal an infested premises. "
             "Five of the six starting values are guesses."),
        ui.p("The proposed design leaves out a trap catching a fly that flew in when no premises is "
             "infested, and wildlife as hosts with their own spread."),
    )


_ST_CAPTION = ("font-size:0.95rem; color:#0b5394; padding:4px 10px; margin:2px 0 12px; "
               "border-left:4px solid #0b5394; background:#eef5fb; border-radius:4px;")

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
    doubling_time=(0.25, 150.0, 0.05), clearance_mean=(1, 365, 1), release_mean=(1, 365, 1),
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
                ui.output_ui("doubling_caption"),
                ui.help_text("Unchecked = spread OFF exactly. The slider stops at 0.25 d; both builds would evaluate a shorter doubling time as written down to their shared floor of 0.1 d."),
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
                _sl("release_kappa", "kappa"),
                *_nearby_source_inputs(_sl("nearby_pressure", "Nearby pressure (set by hand)")),
                *_intro_ref_inputs(), *_weather_inputs(),
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
                             "reference conditions: a warm month, 0.3 of the sterile-fly release target met, "
                             "nearby detections as in summer 2026."),
                ui.output_ui("intro_caption"),
                # The picture of the module is on the report page. The browser build runs the app in
                # a frame one folder below the site, so the report is at ../report/.
                ui.tags.p(ui.tags.a(
                    "See the picture of how introductions are built", href="../report/#intro",
                    target="_blank", rel="noopener"),
                    style="margin:4px 0 8px"),
                *_intro_ref_inputs(),
                _sl("release_coverage", "Share of the sterile-fly release target being met"),
                *_nearby_source_inputs(
                    _sl("nearby_pressure", "Detections in nearby counties, relative to summer 2026 (1 = the same)"),
                    ui.help_text("Counties to the south count most. 0 = no active zones nearby.")),
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
                *_weather_inputs(),
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
                ui.output_ui("doubling_caption"),
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

BASELINE_NOTE = (
    "**The dashed line is the baseline:** the chance the zone is free if nobody were looking, with no "
    "inspection rounds and no owner reports. It is a comparison line, not a forecast of what doing "
    "nothing would achieve. The gap between the two lines is what the surveillance evidence adds: "
    "inspection rounds that found nothing and no confirmed owner report.\n\n"
    "**Why it falls and then rises:** in this model an infestation that nobody finds ends by itself "
    "after about 166 days (about five and a half months) at the default settings. The infestations "
    "assumed left behind at day 0 end mostly between days 135 and 166. On the county scenario the cold "
    "months do most of the later rise, and the fall at the end comes with the return of warm months.\n\n"
    "**How firm it is:** the baseline leans on that self-ending, which has little data behind it. Over "
    "the range for the length of an unfound infestation (97 to 271 days) the baseline at the last day "
    "runs from 9% to 47% on the six-month scenario (25% at the defaults) and from 12% to 44% on the "
    "county scenario (32% at the defaults).")

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
                ui.output_ui("intro_caution"),
                ui.div(ui.markdown(FREE_MEANS), style=_CAPTION_STYLE),
                ui.output_plot("pfree_plot", height="440px"),
                ui.div(ui.markdown(BASELINE_NOTE), style=_CAPTION_STYLE),
                ui.output_ui("rule_table"),
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
            _neighbours_panel(),
            _check_panel(),
            ui.nav_panel(
                "Inside one premises",
                ui.p("How an infested premises looks, and how likely it is to be found, as the "
                     "infestation ages.", style="margin:4px 0 6px;"),
                ui.output_plot("premises_plot", height="640px"),
            ),
            _streams_panel(),
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
                ui.output_ui("intro_caution"),
                ui.output_ui("scorecard"),
                ui.output_plot("pfree_plot", height="400px"),
                ui.output_ui("rule_table"),
                ui.markdown(
                    "*`P_free(t)` is the **running estimate**: the probability the zone has no infested premises "
                    "given every inspection round through `t` came back clean and no owner report was confirmed. "
                    "**The dense curve is exact**: every plotted point, between rounds too, is the running "
                    "estimate read off one full-horizon evaluation of the model (`eval_times`), not a "
                    "re-evaluation at a shorter horizon. "
                    "The faint dashed line is the chance, seen from day 0, that the zone is free at `t` **and** "
                    "nothing has been detected by then: `P_free * (1 - P_lost)`. It is not the unconditional "
                    "`P(K=0)(t)`, and it is not the no-surveillance baseline of the plain view; with "
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
            _neighbours_panel(),
            _check_panel(),
            _streams_panel(),
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
                  f"**{_pct(pf[hi], thr)}** ({month(hi)}). The rise in the cold months comes from the "
                  f"assumption that fewer flies arrive then, not from stronger evidence, and it falls "
                  f"back when the weather warms. "
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
    pts.update(_rule_times(p).values())               # the rule dates, so their table reads exact days
    ts = sorted(x for x in pts if 0.0 <= x <= horizon + 1e-9)
    return ts, rounds


def _rule_times(p: dict) -> dict:
    """Rule date -> the evaluated time that stands for it (snapped to dt), for the dates inside the run."""
    dt = float(p["dt"])
    horizon = float(p["horizon"])
    return {t: round(round(t / dt) * dt, 6) for t, _ in RULE_DATES if t <= horizon + 1e-9}


def server(input, output, session):

    @reactive.calc
    def index_now():
        """The index at the two dials, or None when the data files are missing."""
        if not INDEX_OK:
            return None
        return county_index(float(input.kernel_km()), float(input.south_weight()))

    @reactive.calc
    def nearby_value() -> float:
        """Nearby pressure as the model receives it: the hand slider, or the index when selected."""
        if input.nearby_source() == "index" and INDEX_OK:
            return float(index_now()["value"])
        return input.nearby_pressure()

    @reactive.calc
    def pack() -> dict:
        return dict(
            # options added 2026-11-19 (presets.OPTION_DEFAULTS); the defaults change nothing
            intro_ref=input.intro_ref(), weather_source=input.weather_source(),
            renewed_chance=input.renewed_chance(),      # what-if term, 0 = off (presets.renewed_rate)
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
            release_kappa=input.release_kappa(), nearby_pressure=nearby_value(),
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
        with reactive.isolate():        # the what-if value these parameters were built with
            renewed = float(pack().get("renewed_chance", 0.0) or 0.0)
        return dict(out=out, breakdown=breakdown, round_times=sorted(p["round_times"]), renewed=renewed)

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
        # at e^{-k0_mean}, not at 1. D(0) = 0, so P_lost(0) = 0 and the joint line (free and nothing
        # detected so far) starts at the same point (model_design.md, "Initial state"). At t > 0
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
        # Baseline of the plain graph (2026-11-20): the chance the zone is free with no surveillance.
        # The same scenario with nothing able to detect: no inspection finds anything
        # (Se_animal = 0) and no owner report is ever made (rho_r = 0). Infestations still arrive,
        # spread and end as in the scenario; with no detection no control starts. One more lane
        # call on the same grid. The inputs that only act through detection are pinned first, so
        # moving an inspection or owner slider reuses the cached baseline (not the round interval,
        # which sets the time grid)
        # (changes/2026-11-20-baseline-no-surveillance: the line does not move with them).
        pbase = None
        if PLAIN:
            ob = evaluate(dict(p, **NO_SURVEILLANCE, eval_times=ts))
            pb_t = ob["p_free_by_time"]
            pbase = [float(pb_t[0.0 if t <= 0.0 else t]) for t in ts]
        return dict(t=ts, pfree=pfree, puncond=puncond, pbase=pbase, rounds=rounds)

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
            lab_unc = "Baseline: chance the zone is free with no surveillance"
            lab_thr = f"{_pct(thr)} reference line (for illustration)"
            xlab = "Days since the last detection"
        else:
            lab_free = r"$P_{free}(t)$ (given no confirmed detection yet)"
            lab_unc = r"Free and nothing detected so far, seen from day 0: $P_{free}\cdot(1-P_{lost})$"
            lab_thr = f"{thr:.3g} reference line (for illustration)"
            xlab = "time since the last detection, t (days)"
        ax.plot(ts, pfree, color="#1f77b4", lw=2, label=lab_free)
        # plain view: the no-surveillance baseline; technical view: the joint chance of being free
        # and undetected, under its exact label
        ax.plot(ts, tr["pbase"] if PLAIN else puncond, color="#1f77b4", lw=1.3, ls="--", alpha=0.5,
                label=lab_unc)
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
        # Official rule dates (2026-11-19): light dotted marks with one legend entry. Policy dates,
        # not model results; the table under the plot gives the model's value on each.
        marks = [(t, name) for t, name in RULE_DATES if t <= ts[-1] + 1e-9]
        for t, name in marks:
            ax.axvline(t, color="0.5", lw=0.8, ls=":", alpha=0.8, zorder=1)
            ax.annotate(name, xy=(t, 0.03), xytext=(-2, 0), textcoords="offset points", rotation=90,
                        ha="right", va="bottom", fontsize=6 if narrow else 7.5, color="0.4")
        if marks:
            ax.plot([], [], color="0.5", lw=0.8, ls=":",
                    label="Official rule dates (policy dates, not model results)")
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
        # one line under the headline whenever the what-if term is on, so that a screenshot
        # cannot be mistaken for the default
        whatif = ("" if r["renewed"] <= 0.0 else
                  f"\n\n**What-if:** this result includes a what-if term for renewed infestations, "
                  f"set at a chance of {r['renewed']:.2f} per 30 days. It is not part of the default and cannot be tested until new data arrive.")
        if PLAIN:
            rows = module_rows()
            text = (_season_headline(out, r["round_times"], rows)
                    if rows is not None and input.season_on() and r["round_times"]
                    else _plain_headline(out, r["round_times"]))
            return ui.div(ui.markdown(text + whatif),
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
        return ui.div(ui.markdown(text + whatif),
                      style=("font-size:1.0rem; padding:6px 10px; margin:4px 0 8px; "
                             "border-left:4px solid #2ca02c; background:#f0f8f0; border-radius:4px;"))

    @render.ui
    def intro_caution():
        # standing caution for scenarios that use the introduction module (the county scenarios);
        # nothing for the scenario checks and any scenario with the module off
        if not input.module_on():
            return None
        return ui.div(ui.markdown(_intro_caution()), style=_CAPTION_STYLE)

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
    def doubling_caption():
        # The closed form overstates freedom when spread between premises is very fast (the
        # shared-age closure; changes/2026-11-18-county-tailoring/divergence/README.md).
        if not input.spread_on() or float(input.doubling_time()) >= DOUBLING_WARN_BELOW:
            return None
        return ui.div(ui.markdown(
            f"**Approximate here.** Below {DOUBLING_WARN_BELOW:g} days this build of the model "
            "reads the chance of freedom too high after the first rounds, by 0.002 to 0.02. The "
            "simulator build of the model is the reference in this range."),
            style="color:#8a4b00; font-size:0.9em;")

    @render.ui
    def intro_caption():
        factor = intro_ref_factor(dict(intro_ref=input.intro_ref()))
        r = float(input.intro_rate()) * factor
        k0 = float(input.k0_mean())
        if PLAIN:
            if r > 0.0:
                text = f"About **{1.0 / r:.0f} days** between introductions."
                if factor != 1.0:
                    text += f" The rate above is multiplied by {factor:.3g} (the option below)."
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
        return module_schedule(pk, float(params()["horizon"]))["rows"]   # as preset_to_params builds it

    @render.plot
    def module_plot():
        rows = module_rows()
        fig, ax = plt.subplots(figsize=(9, 4.2))
        if rows is None:
            ax.text(0.5, 0.5, "The introduction module is off in this scenario:\nthe introduction rate is constant.",
                    ha="center", va="center"); ax.axis("off")
            return fig
        base = float(input.intro_rate()) * intro_ref_factor(pack()) * 30.0
        x = list(range(len(rows)))
        fly = [r["fly"] * base for r in rows]; mv = [r["move"] * base for r in rows]
        wl = [r["wild"] * base for r in rows]
        ax.bar(x, fly, color="#1f77b4", label="Flies arriving")
        ax.bar(x, mv, bottom=fly, color="#ff7f0e", label="Moved animals")
        ax.bar(x, wl, bottom=[a + b for a, b in zip(fly, mv)], color="#2ca02c", label="Wildlife")
        renewed = renewed_rate(pack()) * 30.0       # what-if term: constant, drawn only when on
        if renewed > 0.0:
            ax.bar(x, [renewed] * len(rows), bottom=[a + b + c for a, b, c in zip(fly, mv, wl)],
                   color="#d9d9d9", edgecolor="#555555", hatch="//", linewidth=0.6,
                   label=(f"Renewed infestations, what-if (chance {float(pack()['renewed_chance']):.2f} "
                          f"per 30 days, {renewed:.2f} expected)"))
        step = max(1, len(rows) // 18)
        ax.set_xticks(x[::step]); ax.set_xticklabels([r["month"] for r in rows][::step])
        ax.set_ylabel("Expected new infested premises per month")
        ax.set_xlabel("Month of the run")
        ax.legend(frameon=False, ncol=3 if renewed <= 0.0 else 2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    def _renewed_sentence(pk: dict) -> str:
        """One sentence on the what-if term in use, or "" when it is off."""
        h30 = renewed_rate(pk) * 30.0
        if h30 <= 0.0:
            return ""
        return (f"**What-if term in use:** a chance of {float(pk['renewed_chance']):.2f} per 30 days of "
                f"a renewed infestation from causes the model does not represent, which is "
                f"{h30:.2f} expected renewed infestations per 30 days (some periods have more than "
                f"one). It is the same in every month, "
                f"whatever the season, sterile-fly coverage, nearby detections or mitigations.")

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
        if any(r.get("observed") for r in rows):
            seen = [r["month"] for r in rows if r.get("observed")]
            text += (f" {seen[0]} to {seen[-1]} use 2026 observed temperatures. " + OBSERVED_CAVEAT
                     if len(seen) > 1 else f" {seen[0]} uses 2026 observed temperatures. " + OBSERVED_CAVEAT)
        whatif = _renewed_sentence(pack())
        if whatif:
            text += ("\n\n" + whatif + " It is the hatched bar on the chart. The percentages above "
                     "are for the module's own schedule and leave it out.")
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
        rate = float(pk["intro_rate"]) * intro_ref_factor(pk)
        nearby_how = ("the index from official detections" if input.nearby_source() == "index" and INDEX_OK
                      else "set by hand")
        season = (f"The run covers **{warm}** warm month{'' if warm == 1 else 's'} (fly activity at "
                  f"90% or more). A clean cold month counts for little."
                  if warm is not None else
                  "Rates are constant in this scenario, so the season is not considered.")
        whatif = _renewed_sentence(pk)
        whatif = f"\n- {whatif}" if whatif else ""
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
- About one new infested premises every {1.0 / rate if rate > 0 else float('inf'):.0f}
  days in a warm month, under the reference conditions.
- Sterile-fly release meets {100 * float(pk['release_coverage']):.0f}% of its target.
- Detections in nearby counties are at {round(float(pk['nearby_pressure']), 2):g} times their summer 2026 level
  ({nearby_how}).
- {season}{whatif}

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

    # ---------------------------------------------------------------- options added 2026-11-19
    @render.ui
    def index_readout():
        """What the model is given for nearby pressure, and where the index comes from."""
        if not INDEX_OK:
            return ui.div("The detection files for the index were not found. The hand-set value is used.",
                          style=_CAPTION_STYLE)
        ix = index_now()
        if input.nearby_source() != "index":
            text = (f"Using the hand-set value, **{float(input.nearby_pressure()):g}**. The index from "
                    f"official detections would give {ix['value']:.2f}.")
            return ui.div(ui.markdown(text), style=_CAPTION_STYLE)
        top = "; ".join(f"{r['county']} {r['contribution']:.2f} ({r['cases']} case"
                        f"{'' if r['cases'] == 1 else 's'}, {r['km']:.0f} km)" for r in ix["top"])
        text = (f"Using the index: **{ix['value']:.2f}** for {INDEX_COUNTY[1]} County. The slider above is "
                f"not used. Data: confirmed animal cases in other counties in the 30 days to "
                f"{_day(INDEX_DATE)}, the latest confirmation in the official export pulled "
                f"{_day(INDEX_PULLED)}. Largest sources: {top}. The app uses one fixed value for the "
                f"whole run: the index on that latest confirmation date. In the data the value "
                f"changes month by month.")
        if ix["series"]:
            text += (f" {INDEX_COUNTY[1]} County, 30-day periods from {_day(INDEX_START)}: "
                     + ", ".join(f"{v:.2f}" for v in ix["series"])
                     + f". On {_day(INDEX_DATE)}: {ix['value']:.2f}.")
        text += " That is a scenario, not a forecast."
        return ui.div(ui.markdown(text), style=_CAPTION_STYLE)

    @render.ui
    def rule_table():
        """The model's chance of freedom on the official rule dates, read off the dense curve."""
        r = zone_outputs()
        pf = r["out"]["p_free_by_time"]
        thr = float(r["out"]["release_threshold"])
        at = _rule_times(params_d())
        rows = []
        for t, name in RULE_DATES:
            if t in at:
                key = min(pf, key=lambda x: abs(float(x) - at[t]))
                rows.append([name, f"day {t:g}", _pct(pf[key], thr) if PLAIN else f"{pf[key]:.4f}"])
            else:
                rows.append([name, f"day {t:g}", "after the end of this run"])
        return ui.div(
            ui.p("Official declaration rules count three or four 21-day fly life cycles after the last "
                 "detection. Older practice used six and twelve months. The model's value on each date:",
                 style="font-size:0.9rem; margin:6px 0 2px;"),
            _table(["Rule date", "Days after the last detection",
                    "Chance the zone is free, given no detection so far"], rows))

    def _linked_by_county():
        by = {}
        for row in TABLES["counties"]:
            by.setdefault(row["county"], []).append(row)
        return {c: sorted(v, key=lambda x: int(x["round"])) for c, v in by.items()}

    @render.ui
    def neighbours_intro():
        if TABLES is None:
            return ui.div(_MISSING, style=_CAPTION_STYLE)
        sm = TABLES["summary"]
        start = date.fromisoformat(sm["start"])
        text = (f"**Precomputed on {_day(PRECOMPUTED_ON)}** for the default settings, with the analytic build "
                f"of the model. The simulator build agrees on the seven county runs, given their "
                f"schedules. The link between counties was computed from the analytic build only. "
                f"Nothing here changes with the settings on the left. "
                f"{sm['group'][0]} County and the {len(sm['group']) - 1} counties that adjoin it, each run "
                f"for {int(sm['horizon_days'])} days from {_day(start)} with monthly rounds. A county's "
                f"nearby pressure is the index from detections outside the group plus a link to the "
                f"model's own result for the other counties. This run starts later than the scenario on "
                f"the first tab, so its {sm['group'][0]} row is a different run.")
        return ui.div(ui.markdown(text), style=_CAPTION_STYLE)

    @render.ui
    def neighbours_table():
        if TABLES is None:
            return None
        sm = TABLES["summary"]
        by = _linked_by_county()
        last_case = {}
        for when, src in _CASES:
            last_case[src] = when
        month = lambda iso: f"{date.fromisoformat(iso):%b %Y}"
        rows = []
        for c in sm["group"]:
            rr = by[c]
            low = min(rr, key=lambda x: float(x["p_free"]))
            links = [float(x["link_from_group"]) for x in rr]
            seen = last_case.get((INDEX_COUNTY[0], c))
            rows.append([c, f"{int(sm['premises'][c]):,}", _day(seen) if seen else "none",
                         sm["start_state"][c].split(":")[0],
                         f"{float(sm['index_from_detections_outside_group'][c]):.2f}",
                         f"{min(links):.2f} to {max(links):.2f}",
                         _pct(float(rr[-1]["p_free"])),
                         f"{_pct(float(low['p_free']))} ({month(low['date'])})"])
        g = TABLES["group"]
        g_last = max(g, key=lambda x: int(x["round"]))
        g_low = min(g, key=lambda x: float(x["group_free_if_independent"]))
        span = lambda x: (f"{_pct(float(x['group_free_if_independent']))} to "
                          f"{_pct(float(x['group_free_upper_bound']))}")
        rows.append([f"All {len(sm['group'])} counties free", "", "", "", "", "", span(g_last),
                     f"{span(g_low)} ({month(g_low['date'])})"])
        return _table(["County", "Premises", "Last detection", "Start", "Index from detections",
                       "Link from the group (lowest to highest month)",
                       f"Chance free at the last round ({month(g_last['date'])})",
                       "Lowest chance free (month)"], rows, bold_last=True)

    @render.plot
    def neighbours_plot():
        fig, ax = plt.subplots(figsize=(9, 4.2))
        if TABLES is None:
            ax.text(0.5, 0.5, _MISSING, ha="center", va="center"); ax.axis("off")
            return fig
        by = _linked_by_county()
        g = sorted(TABLES["group"], key=lambda x: int(x["round"]))
        x = [int(r["round"]) for r in g]
        ax.fill_between(x, [float(r["group_free_if_independent"]) for r in g],
                        [float(r["group_free_upper_bound"]) for r in g], color="0.6", alpha=0.25, lw=0,
                        label="All counties free: between independent counties and the lowest county")
        for c in TABLES["summary"]["group"]:
            lead = c == TABLES["summary"]["group"][0]
            ax.plot([int(r["round"]) for r in by[c]], [float(r["p_free"]) for r in by[c]],
                    lw=2.4 if lead else 1.2, marker="o", ms=3, label=c, zorder=3 if lead else 2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{date.fromisoformat(r['date']):%b}\n{date.fromisoformat(r['date']):%Y}"
                            if i in (0, len(g) - 1) or r["date"][5:7] == "01"
                            else f"{date.fromisoformat(r['date']):%b}" for i, r in enumerate(g)])
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_ylabel("Chance free, given no detection there so far")
        ax.set_xlabel("Inspection round (month)")
        ax.set_title("Chance each county is free, and all of them together")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        return fig

    @render.ui
    def neighbours_note():
        if TABLES is None:
            return None
        text = ("**Reading the group row.** The lower end is the product of the county values, which "
                "would hold if the counties were independent. They share weather and outside pressure, "
                "so the true value is higher. It cannot be above the lowest county, which is the upper "
                "end. Each county value assumes no detection in that county so far. The results of "
                "inspections in the neighbours are not used. After the data date the outside pressure "
                "is held at its last value, which is a scenario and not a forecast. The link between "
                "counties rests on assumptions with little data.")
        return ui.div(ui.markdown(text), style=_CAPTION_STYLE)

    @render.ui
    def check_2026():
        return check_2026_block()             # the check on equal windows; a note when the file is missing

    @render.ui
    def calibration_2026():
        return calibration_2026_block()       # None when the summary file is missing

    # ---- "Traps and wildlife (illustration)" tab: a proposed design, not built ----
    # Reads only its own "st_" inputs. No model output is read and none is changed.
    def _st_stream(key: str) -> tuple:
        return (float(input[f"st_{key}_coverage"]()), float(input[f"st_{key}_rate"]()),
                float(input[f"st_{key}_delay"]()))

    @reactive.calc
    def st_in_use() -> dict:
        """The streams switched on: key -> (coverage, rate per day, delay in days)."""
        return {k: _st_stream(k) for k in streams.STREAM_KEYS if input[f"st_{k}_on"]()}

    def _st_week(key: str):
        q = float(input[f"st_{key}_rate"]())
        return ui.help_text(f"A rate of {q:g} per day is a chance of {_st_pct(streams.per_week(q))} "
                            f"per week, and a mean wait of {1.0 / q:.0f} days.")

    def _st_pct(x: float) -> str:
        return f"{100.0 * x:.1f} percent"

    @render.ui
    def st_trap_week():
        return _st_week("trap")

    @render.ui
    def st_wild_week():
        return _st_week("wild")

    def _st_lines(use: dict) -> list:
        """(label, colour, list of streams) for each line of the plots."""
        lines = [(streams.STREAMS[k]["label"], _STREAM_COLOURS[k], [use[k]]) for k in use]
        if len(use) == 2:
            lines.append(("Both together", _STREAM_COLOURS["both"], list(use.values())))
        return lines

    @render.plot
    def st_found_plot():
        use = st_in_use()
        days = np.arange(0, 361)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        if not use:
            ax.plot(days, np.zeros(len(days)), color="#888888", lw=2,
                    label="Both switched off: nothing is found")
        for label, colour, ss in _st_lines(use):
            ax.plot(days, [streams.found(t, ss) for t in days], color=colour, lw=2, label=label)
            ceil = streams.ceiling_found(ss)
            ax.axhline(ceil, color=colour, ls="--", lw=1, alpha=0.7)
            ax.annotate(f"ceiling {ceil:.2f}", xy=(360, ceil), xytext=(-4, 3),
                        textcoords="offset points", ha="right", fontsize=8, color=colour)
        ax.set_xlim(0, 360); ax.set_ylim(0, 1.02)
        ax.set_xticks(range(0, 361, 30))
        ax.set_xlabel("Days since the premises became infested")
        ax.set_ylabel("Chance it has been found")
        ax.set_title("Illustration of a proposed design, not built", fontsize=10, loc="left")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    @render.plot
    def st_free_plot():
        use = st_in_use()
        m = float(input.st_hidden())
        days = np.arange(0, 361)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        none = streams.p_free_none(m)
        ax.axhline(none, color="#888888", lw=2, label=f"No traps, no wildlife checks: {none:.3f}")
        if use:
            ss = list(use.values())
            name = "Both together" if len(use) == 2 else streams.STREAMS[next(iter(use))]["label"]
            ax.plot(days, [streams.p_free(t, m, ss) for t in days], color="#b5179e", lw=2,
                    label=f"{name}, every check negative")
            ceil = streams.p_free_ceiling(m, ss)
            ax.axhline(ceil, color="#b5179e", ls="--", lw=1, alpha=0.8, label=f"Ceiling: {ceil:.3f}")
        ax.set_xlim(0, 360); ax.set_ylim(0, 1.02)
        ax.set_xticks(range(0, 361, 30))
        ax.set_xlabel("Days with every trap check and wildlife check negative")
        ax.set_ylabel("Chance the zone is free")
        ax.set_title(f"Illustration of a proposed design, not built. Simplest case, {m:g} expected "
                     f"hidden infested premises at day 0", fontsize=10, loc="left")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    @render.ui
    def st_table():
        use = st_in_use()
        m = float(input.st_hidden())
        keys = list(use) or list(streams.STREAM_KEYS)       # for the range columns
        cols = [streams.summary(m, list(use.values()))]
        cols += [streams.summary(m, [streams.setting(k, w) for k in keys]) for w in ("value", "low", "high")]
        rows = [[f"Chance one infested premises is found by day {d}"]
                + [f"{c['found'][d]:.2f}" for c in cols] for d in streams.TABLE_DAYS_FOUND]
        rows += [[f"Credit at day {d}: rise in the chance the zone is free, simplest case"]
                 + [f"{c['credit'][d]:+.3f}" for c in cols] for d in streams.TABLE_DAYS_CREDIT]
        which = " and ".join(streams.STREAMS[k]["short"] for k in keys)
        if not use:
            lead = ("Both are switched off, so your settings earn nothing. The other three columns "
                    "show traps and wildlife both in use.")
        else:
            lead = f"All four columns use what is switched on: {which}."
        return ui.TagList(
            _table(["", "Your settings", "Proposed starting values", "Low end of the proposed ranges",
                    "High end of the proposed ranges"], rows),
            ui.help_text(f"{lead} The low end takes the lowest coverage, the lowest rate and the "
                         f"longest delay. The high end takes the highest coverage, the highest rate "
                         f"and the shortest delay. With {m:g} expected hidden infested premises at "
                         f"day 0 and no traps or wildlife checks, the chance the zone is free in the "
                         f"simplest case is {streams.p_free_none(m):.3f}. The spread is wide because "
                         f"the starting values are mostly guesses. Proposed design, not built."),
        )

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
        # the options added 2026-11-19 are not pack keys: a scenario always loads with them off
        for k in ("nearby_source", "intro_ref", "weather_source"):
            ui.update_radio_buttons(k, selected=OPTION_DEFAULTS[k])
        for k in ("kernel_km", "south_weight", "renewed_chance"):
            ui.update_slider(k, value=OPTION_DEFAULTS[k])


app = App(app_ui, server)
