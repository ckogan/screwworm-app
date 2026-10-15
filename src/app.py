"""
Zone Freedom from New World Screwworm: Review App (model-cycle stage 4, Shiny for Python).

Spec: ../../app_design.md  ·  Model lane: examples/nws-freedom/implementations/claude/model.py,
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
is deferred until the priors are reviewed (placeholder table).

Presets live in presets.py (one table; the stage-5 scenario runner reads the same one).

Run:  shiny run app.py
"""

from __future__ import annotations

# Cap BLAS threads BEFORE numpy import: the lane runs many small matrix exponentials, which
# thrash multi-core BLAS. Single-thread is ~40x faster in wall time here.
import os
os.environ.setdefault("NWS_DEMO", "1")  # shinylive bundle: demo mode

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
from presets import (CHECKBOXES, MARKS, NUMERICS, SCENARIO_LABELS, SCENARIOS,  # noqa: E402
                     preset_to_params)

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
_BASE = SCENARIOS["six_months"]

STATUS_LINE = (
    f"**Model lane:** `examples/nws-freedom/implementations/claude/model.py`, **this pack's own lane at "
    f"v0.6 (pre-seeded start, edited 2026-09-09)**, forked from regional-freedom `8f9d7b5`, at "
    f"`{LANE_COMMIT}`, `dt = {DT}` (inherited from the lane). **Two-lane agreement check (stage 3), "
    f"v0.5: run on this pack's screwworm grid 2026-09-07, PASS** (368 evaluable cells, max |z| 3.12, "
    f"one band-edge cell at the parent's known corner `doubling_time_within = 0.1 d`; "
    f"`validation/stage3_verdict_2026-09-07.md`). **v0.6 (both edited lanes, 471-cell grid, 2026-09-09 to "
    f"10): PASS** (418 evaluable cells, max |z| 3.87, one cell beyond 3 sigma against an allowance of 2, "
    f"sign test p = 0.59; reduction anchor exact on all 421 clean-start cells; Poisson-thinning anchor "
    f"within 3 sigma; `validation/stage3_verdict_2026-09-09.md`). "
    f"**Priors: first-pass literature review 2026-09-07** (not yet expert-reviewed), and the initial state in the zone presets is a **guessed placeholder**."
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
# UI
# --------------------------------------------------------------------------------------
sidebar = ui.sidebar(
    ui.input_select("scenario", "Scenario preset", SCENARIO_LABELS, selected="six_months"),
    ui.input_action_button("apply_preset", "Load scenario into ALL inputs",
                           class_="btn-sm btn-outline-primary"),
    ui.help_text("Writes every input, including the ones this scenario does not use, so nothing "
                 "carries over. The four screwworm presets share the placeholder priors and differ "
                 "in horizon and round schedule; the numbered ones reproduce scenarios.md."),
    ui.accordion(
        ui.accordion_panel(
            "Within-premises course",
            ui.input_slider("mean_latent_time", "Egg laid to larvae visible (d)", 0.0, 8.0,
                            _BASE["mean_latent_time"], step=0.1),
            ui.help_text("0 = larvae visible at once."),
            ui.input_slider("doubling_time_within", "Within-premises doubling time (d)",
                            0.3, 30.0, _BASE["doubling_time_within"], step=0.1),
            ui.help_text("Local fly population building on one premises (mass action, an assumption). "
                         "Reviewed value 4.5 d against 7 d recovery: near critical (R about 1.1). "
                         "The old placeholder, 14 d, made the course decay."),
            ui.input_checkbox("recovery_within_on", "Within-premises recovery (else never)", value=True),
            ui.input_slider("mean_recovery_time_within", "Infestation ends by treatment, healing or death (d)",
                            2.0, 1000.0, _BASE["mean_recovery_time_within"], step=0.5),
            ui.help_text("Unchecked = recovery OFF exactly (mean = Inf). Needed for the logistic scenario."),
            ui.input_slider("N_herd", "Animals on the premises (N_herd)", 10, 1000, _BASE["N_herd"], step=10),
            ui.help_text("Floor 10: the mean-field course is only valid for a herd (C14)."),
            ui.input_slider("f_clinical_max", "Fraction of infested animals whose wound becomes visible",
                            0.0, 1.0, _BASE["f_clinical_max"], step=0.01),
            ui.input_slider("mean_clinical_onset_time", "Days until the wound is obvious",
                            0.1, 10.0, _BASE["mean_clinical_onset_time"], step=0.1),
            ui.input_slider("p_latent_seed", "Seeded latent fraction (p_latent_seed)",
                            0.0, 1.0, _BASE["p_latent_seed"], step=0.01),
            ui.input_slider("p_infectious_seed", "Seeded larvae-bearing fraction (p_infectious_seed)",
                            0.0, 1.0, _BASE["p_infectious_seed"], step=0.01),
            ui.input_slider("age_window", "Premises-age display window (d)", 5, 200, _BASE["age_window"], step=5),
        ),
        ui.accordion_panel(
            "Detection",
            ui.help_text("Scheduled inspection"),
            ui.input_slider("Se_animal", "Inspection finds an active myiasis (Se_animal)",
                            0.0, 1.0, _BASE["Se_animal"], step=0.01),
            ui.input_slider("n_animals_per_round", "Animals inspected per premises per round",
                            0, 100, _BASE["n_animals_per_round"], step=1),
            ui.help_text("0 = no animals inspected, so a round carries no information."),
            ui.help_text("Owner reporting"),
            ui.input_slider("rho_a", "Owner notices a visible wound (rho_a)", 0.0, 1.0, _BASE["rho_a"], step=0.01),
            ui.input_slider("rho_r", "Reporting rate once noticed (rho_r, /d)", 0.0, 1.0, _BASE["rho_r"], step=0.001),
            ui.help_text("1/7 = a one-week mean reporting delay."),
        ),
        ui.accordion_panel(
            "Zone design",
            ui.input_slider("M", "Premises in the zone (M)", 20, 500, _BASE["M"], step=10),
            ui.input_slider("design_prevalence", "Design prevalence", 0.0, 0.05, _BASE["design_prevalence"], step=0.005),
            ui.help_text("0 = absolute freedom, P(K=0) (C12)."),
            ui.input_slider("intro_rate", "Zone introduction rate (/day)", 0.0, 0.2, _BASE["intro_rate"], step=0.0001),
            ui.output_ui("intro_caption"),
            ui.help_text("Fly immigration + infested-animal movement + wildlife, one zone rate."),
            ui.input_checkbox("spread_on", "Between-premises spread (else none)", value=True),
            ui.input_slider("doubling_time", "Between-premises doubling time (d)",
                            0.01, 150.0, _BASE["doubling_time"], step=0.01),
            ui.help_text("Unchecked = spread OFF exactly. Below 0.25 d the lane floors the doubling time."),
            ui.input_checkbox("clearance_on", "Zone clearance: a treated premises leaves the count (else never)",
                              value=True),
            ui.input_slider("clearance_mean", "Mean time to clear a premises (d)", 1, 365, _BASE["clearance_mean"], step=1),
            ui.help_text("ON by default for screwworm (C11, 21 d). Off recovers the FMD default."),
            ui.input_slider("beta_lockdown_multiplier", "Response multiplier after a new detection (SIT + movement controls)",
                            0.0, 1.0, _BASE["beta_lockdown_multiplier"], step=0.05),
            ui.help_text("Inert for P_free (which conditions on no detection) but it moves the burden outputs."),
        ),
        ui.accordion_panel(
            "Inspection schedule",
            ui.input_slider("n_rounds", "Number of inspection rounds", 1, 12, _BASE["n_rounds"], step=1),
            ui.input_slider("round_interval", "Days between rounds", 1, 60, _BASE["round_interval"], step=1),
            ui.help_text("Horizon = rounds x interval. 6 x 30 = 180 d (default); 3 x 24 = 72 d (3 life cycles); "
                         "12 x 30 = 360 d."),
            ui.input_slider("n_farms_per_round", "Premises visited per round", 0, 100, _BASE["n_farms_per_round"], step=1),
            ui.help_text("0 = no premises visited. Set to M for a census."),
        ),
        ui.accordion_panel(
            "Initial state: leftovers at the last detection (v0.6)",
            ui.input_slider("k0_mean", "Expected undetected infested premises at the start (k0_mean)",
                            0.0, 20.0, _BASE["k0_mean"], step=0.1),
            ui.input_slider("age0_scale", "Their mean age at the start (age0_scale, d)",
                            0, 180, _BASE["age0_scale"], step=1),
            ui.help_text("The outbreak's leftovers: premises still infested and not yet found when the "
                         "clock starts. The values in the zone presets (1 premises, 14 d) are guessed "
                         "placeholders, not literature (priors_design.md). 0 = the clean start, which "
                         "reproduces v0.5 exactly. The curve starts at P_free(0) = e^-k0_mean "
                         "(0.3679 at 1, 0.0498 at 3), so the release question is about the climb."),
        ),
        ui.accordion_panel(
            "Release question",
            ui.input_slider("release_threshold", "P_free floor for releasing the zone", 0.5, 0.999,
                            _BASE["release_threshold"], step=0.001),
            ui.help_text("The APHIS 3-month rule is shown as one point on the curve; the 6- and 12-month marks "
                         "are shown but nothing is tuned to them (E6)."),
        ),
        ui.accordion_panel(
            "Cost coefficients (phase 2 placeholders)",
            ui.input_numeric("c_lost_freedom", "Cost of losing freedom", _BASE["c_lost_freedom"], min=0),
            ui.input_numeric("c_per_case", "Cost per infested premises", _BASE["c_per_case"], min=0),
            ui.input_numeric("c_per_test", "Cost per animal inspected", _BASE["c_per_test"], min=0),
            ui.input_numeric("c_per_round", "Cost per round (fixed)", _BASE["c_per_round"], min=0),
        ),
        # the initial-state group opens too: it is the v0.6 change and it moves the curve's start
        open=["Zone design", "Inspection schedule",
              "Initial state: leftovers at the last detection (v0.6)"],
        multiple=True,
    ),
    width=380,
    title=("Inputs" if os.environ.get("NWS_DEMO", "").strip().lower() not in ("", "0", "false", "no")
           else "Inputs (what-if, pinned theta)"),
)


# Demo switch: NWS_DEMO=1 in the environment hides the
# reviewer-facing prose (status line, tab notes) and shortens the titles. Nothing about the model,
# inputs or outputs changes; the default (unset) is the full review app.
DEMO = os.environ.get("NWS_DEMO", "").strip().lower() not in ("", "0", "false", "no")


def _unless_demo(*items):
    """Return the given UI items, or nothing in demo mode."""
    return () if DEMO else items

app_ui = ui.page_sidebar(
    sidebar,
    *_unless_demo(ui.markdown(STATUS_LINE)),
    ui.navset_card_tab(
        ui.nav_panel(
            "1 · The premises (mechanism)",
            *_unless_demo(ui.markdown(
                "*A freshly infested premises has nothing visible; its catchability moves with time since "
                "infestation as eggs hatch and wounds become obvious. Under the placeholder priors the "
                "larvae-bearing fraction **decays** from the seed (subcritical course) rather than building.*"
            )),
            ui.output_plot("premises_plot", height="640px"),
        ),
        ui.nav_panel(
            "2 · The zone (outcomes)",
            ui.output_ui("release_readout"),
            ui.output_ui("scorecard"),
            ui.output_plot("pfree_plot", height="400px"),
            *_unless_demo(ui.markdown(
                "*`P_free(t)` is the **running estimate**: the probability the zone has no infested premises "
                "given every inspection round through `t` came back clean and no owner report was confirmed. "
                "**The dense curve is exact**: every plotted point, between rounds too, is the running "
                "estimate read off one full-horizon evaluation of the model (`eval_times`), not a "
                "re-evaluation at a shorter horizon. "
                "The faint dashed line is the **unconditional** `P(K=0)(t) = P_free * (1 - P_lost)`; with "
                "clearance on it settles toward a plateau instead of decaying. Vertical marks are the APHIS "
                "freedom milestones, shown, never tuned to. Since v0.6 the curve starts at "
                "`P_free(0) = e^-k0_mean`, the chance the last detection left nothing behind, so with "
                "leftovers the first stretch is the climb the release rule waits out.*"
            ),
            ui.markdown(
                "*Scorecard burden and cost figures are **design-time projections at t = 0** over the full "
                "(K, D) mixture including the post-detection response branch.*"
            )),
        ),
    ),
    title=("Zone freedom from New World screwworm" if DEMO
           else "Zone freedom from New World screwworm: review app (what-if, placeholder priors)"),
    fillable=False,
)


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
            M=input.M(), design_prevalence=input.design_prevalence(), intro_rate=input.intro_rate(),
            spread_on=input.spread_on(), doubling_time=input.doubling_time(),
            clearance_on=input.clearance_on(), clearance_mean=input.clearance_mean(),
            beta_lockdown_multiplier=input.beta_lockdown_multiplier(),
            n_rounds=input.n_rounds(), round_interval=input.round_interval(),
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
        out = evaluate(p)
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
    @render.plot
    def premises_plot():
        tau, wf = premises_curves()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
        ax1.plot(tau, wf["p_shed"], color="#1f77b4", lw=2, label=r"$p_{shed}(\tau)$ larvae-bearing fraction")
        ax1.plot(tau, wf["p_clinical"], color="#d62728", lw=2, label=r"$p_{clinical}(\tau)$ visible-wound fraction")
        ax1.set_ylabel("within-premises fraction")
        ax1.set_title("Within-premises course vs premises age")
        ymax = max(0.02, float(np.nanmax(wf["p_shed"])) * 1.15) if len(wf["p_shed"]) else 1.0
        ax1.set_ylim(0, min(1.02, ymax) if ymax < 0.5 else 1.02)
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(alpha=0.25)
        ax2.plot(tau, wf["HSe"], color="#2ca02c", lw=2, label=r"$HSe(\tau)$ inspection (per round)")
        ax2.plot(tau, wf["h_pass"], color="#9467bd", lw=2, label=r"$h_{pass}(\tau)$ owner-reporting hazard (/d)")
        ax2.set_xlabel(r"premises age $\tau$ (days since infestation)")
        ax2.set_ylabel("detectability")
        ax2.set_title("Detectability vs premises age")
        ax2.set_ylim(bottom=0)
        _w = float(tau[-1]) if len(tau) else 1.0
        ax2.set_xticks([_w * k / 4.0 for k in range(5)])
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------- panel 2 plot
    @render.plot
    def pfree_plot():
        tr = pfree_trajectory()
        ts, pfree, puncond, rounds = tr["t"], tr["pfree"], tr["puncond"], tr["rounds"]
        thr = float(input.release_threshold())
        horizon = max(ts) if ts else 0.0
        fig, ax = plt.subplots(figsize=(9, 4.0))
        ax.plot(ts, pfree, color="#1f77b4", lw=2, label=r"$P_{free}(t)$ (given no confirmed detection yet)")
        ax.plot(ts, puncond, color="#1f77b4", lw=1.3, ls="--", alpha=0.5,
                label=r"$P(K{=}0)(t)=P_{free}\cdot(1-P_{lost})$ (unconditional)")
        for t in rounds:
            ax.axvline(t, color="0.6", lw=0.9, ls=":")
        for t in rounds:
            idx = min(range(len(ts)), key=lambda i: abs(ts[i] - t))
            ax.plot(ts[idx], pfree[idx], "o", color="#1f77b4", ms=5)
        # APHIS freedom marks: shown, never tuned to (E6)
        for tm, label in MARKS:
            if tm <= horizon + 1e-9:
                ax.axvline(tm, color="#ff7f0e", lw=1.2, ls="-.", alpha=0.8)
                ax.text(tm, 0.03, f" {label}", color="#ff7f0e", fontsize=8, rotation=90, va="bottom", ha="right")
        ax.axhline(thr, color="#2ca02c", lw=1.0, ls="--", alpha=0.7)
        ax.text(0.5, thr + 0.01, f"release floor {thr:.3f}", color="#2ca02c", fontsize=8)
        ax.set_xlabel("time since the last detection, t (days)")
        ax.set_ylabel(r"$P_{free}(t)$")
        ax.set_title("Zone freedom trajectory with the APHIS marks and the release floor")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0.0] + list(rounds))
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    # (The v0.6 "dense points sit low under aged leftovers" caption that stood here was removed on
    # 2026-09-15: the single-call route above makes every dense point exact, so it no longer applies.)

    # ---------------------------------------------------------------- release readout
    @render.ui
    def release_readout():
        r = zone_outputs()
        out = r["out"]
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
        marks = ", ".join(f"{lab}: `{pf_by_t[tm]:.4f}`" for tm, lab in MARKS if tm in pf_by_t)
        text = (f"**Release question.** The curve starts at `P_free(0) = {p_free_0:.4f}` "
                f"(the leftovers at the last detection). Floor `P_free >= {thr:.3f}`: {first_txt}. "
                f"{aphis}." + (f" Marks on rounds: {marks}." if marks else ""))
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
        boxes = ui.layout_columns(
            ui.value_box("P_free at start (t = 0)", f"{s['p_free_by_time'][0.0]:.3f}",
                         theme="bg-gradient-indigo-blue"),
            ui.value_box(f"t_release (floor {s['release_threshold']:.3f})",
                         "not reached" if t_rel is None else f"{float(t_rel):g} d",
                         theme="bg-gradient-blue-purple"),
            ui.value_box("P_free (final round)", f"{s['P_free']:.3f}", theme="bg-gradient-blue-purple"),
            ui.value_box("P_lost_freedom (confirmed detection over horizon)", f"{s['P_lost_freedom']:.3f}",
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
