"""
Regional Disease Freedom: Discrete-K Dynamic Model (v0.5) -- CLOSED-FORM lane.

Closed-form implementation of the model specification. No Monte Carlo: everything below is a deterministic
integration of the model.

METHOD
------
1. Within-farm latent->infectious->recovered ODE integrated in age tau to get
   p_shed(tau)=pi_I, f_clinical(tau), p_clinical(tau); from these the exact active-round
   herd sensitivity HSe(tau) and the continuous passive hazard h_pass(tau).
2. Regional (K,D) master equation: K(t) birth-death chain (0..M) x detection flag D(t),
   propagated by matrix exponential (expm_multiply) per time step with frozen coefficients.
   Both surveillance channels FILTER the tracked (K,D) distribution:
     - active rounds: exact hypergeometric all-negative likelihood L_r(K) applied as a
       Bayes update (mass moves D=0 -> D=1),
     - passive: continuous per-farm no-alarm hazard folded into the D=0 -> D=1 generator.
   P(free|theta) = P(K<=K* | no detection through t) = P0[:K*+1].sum()/sum(P0),
   K* = floor(design_prevalence*M) (default 0 => P(K=0)), COMPUTED deterministically.
3. A parallel no-secondary sub-mass Q[K,D] (same generator + an extra "a secondary happened"
   absorbing leak) yields P_R_gt0 without an E[exp]!=exp[E] Jensen collapse.
4. An age-cohort field (reconditioned on no-detection survival) supplies the age-averaged
   shape quantities <p_shed>(t) (-> beta_eff), <h_pass>(t), and abar_r (round likelihood).

v0.6-nws PRE-SEEDED START (2026-09-09; edited in place under stage 2, model_design.md
"Initial state", changes/2026-09-07-preseeded-start/spec_delta.md "Both lanes")
------------------------------------------------------------------------------------
5. The (K,D) distribution starts from K(0) ~ Poisson(k0_mean) truncated to {0..M} and
   renormalised, all mass in D=0; the age field starts with the expected initial-premises
   count k0_mean spread over ages by Exponential(mean=age0_scale) truncated to [0, horizon].
   Their within-premises course is the standard course at tau = tau_0 + t, so the age curves
   are extended to 2*horizon (by continuing the ODE, see within_farm_curves). New outputs:
   p_free_by_time[0.0] = P(K_0 <= K*) and t_release (first round time with P_free >=
   release_threshold). Cumulative incidence and the no-secondary Q chain count only events
   after t = 0. NORMATIVE reduction: k0_mean = 0 reproduces the v0.5-nws lane to machine
   precision (self-test: 19 anchor cells to 1e-12) -- the initial state adds no operation
   to the k0_mean = 0 path (same array lengths, same arithmetic).

EVALUATION-TIME OUTPUT (2026-09-15)
------------------------------------------------------------------------------------
6. evaluate() accepts an OPTIONAL `eval_times` grid (default None). At each listed time the
   running P(K <= K* | no alarm so far) is read off the SINGLE full-horizon march into
   p_free_by_time, after any round update at that same instant, and P(no alarm so far) =
   P0.sum() into the new output p_noalarm_by_time. Pure addition: the (K, D) march, the age
   field, the Bayes updates, cumulative incidence, P_lost_freedom, P_R_gt0, t_release and the
   design-time outputs are byte-for-byte what they were (the record only READS P0), and with
   eval_times = None no line of the march changes. It replaces truncated-horizon re-evaluation
   for the app's dense curve (one lane call per curve instead of about 30) and removes the
   between-round bias under aged leftovers (the initial-age law is truncated at the evaluation
   horizon, so a re-evaluation at t < about 2 age0_scale sat low). Self-test: check_eval_times().

HAZARD CHECKLIST honoured inline; see
assert_invariants() and comments flagged `# HAZARD:`.
"""

from __future__ import annotations

import math
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammaln, xlogy
from scipy.stats import hypergeom
from scipy.sparse.linalg import expm_multiply


# --------------------------------------------------------------------------------------
# Degenerate-limit hardening (normative, model_design.md "Degenerate limits").
# Harden by FLOORING THE TIME-SCALE (mean clamped to a small epsilon so 1/mean is a large
# finite cap), NEVER by zeroing the rate. mean=Inf -> rate 0 ("never"); mean=0 -> rate cap
# ("instantaneous"). Continuity f(0) ~ f(eps) is guaranteed because both the exact-0 input
# and the near-degenerate twin (e.g. 1e-6) floor to the same MIN_TIMESCALE.
# --------------------------------------------------------------------------------------
MIN_TIMESCALE = 1e-3      # days; floor for mean durations -> rate cap 1e3 / day
MIN_DOUBLING = 0.25       # days; floor for doubling times (keeps spread finite/instant-fast)


def rate_from_mean(mean, min_ts: float = MIN_TIMESCALE) -> float:
    """1/mean_time. Inf -> 0. 0 or tiny -> floor the *time-scale*, giving a large finite rate."""
    if mean is None:
        return 0.0
    m = float(mean)
    if not np.isfinite(m):
        return 0.0
    if m <= min_ts:
        return 1.0 / min_ts
    return 1.0 / m


def rate_from_doubling(dt, min_dt: float = MIN_DOUBLING) -> float:
    """log(2)/doubling_time. Inf -> 0. 0 or tiny -> floor the doubling time."""
    if dt is None:
        return 0.0
    d = float(dt)
    if not np.isfinite(d):
        return 0.0
    if d <= min_dt:
        return math.log(2.0) / min_dt
    return math.log(2.0) / d


# --------------------------------------------------------------------------------------
# Defaults: FMD (cattle) preset + the v1 validation scenario (validation_design.md).
# The grid overrides any subset of these; absent columns fall back here.
# --------------------------------------------------------------------------------------
DEFAULTS = dict(
    # regional
    M=100,
    mean_intro_interval=50.0,        # -> lambda_region = 1/interval (REGIONAL, not per-M)
    doubling_time=7.0,               # -> beta_max
    mean_recovery_time=float("inf"), # regional clearance; default Inf = none
    beta_lockdown_multiplier=0.1,
    design_prevalence=0.0,           # regional freedom threshold: "free" = K <= floor(design_prevalence*M)
    # within-farm (FMD cattle)
    N_herd=100,
    doubling_time_within=0.9,        # -> beta_within
    mean_latent_time=3.6,            # -> gamma_L
    mean_recovery_time_within=5.0,   # -> gamma_R
    p_latent_seed=0.01,
    p_infectious_seed=0.0,
    # detection / clinical
    Se_animal=0.95,
    n_animals_per_round=30,          # animals sampled per farm
    f_clinical_max=0.33,
    mean_clinical_onset_time=1.5,
    rho_a=0.89,
    rho_r=0.70,
    # surveillance schedule
    round_times=(30.0, 60.0, 90.0, 120.0),
    n_farms_per_round=20,            # farms sampled per round (scalar; symmetric policy)
    horizon=120.0,
    # initial state (v0.6-nws, model_design.md "Initial state"); design inputs of the release
    # question, not lit-derived priors. The defaults are the clean start: k0_mean = 0 is the
    # NORMATIVE reduction to the v0.5-nws lane (identical outputs to machine precision).
    k0_mean=0.0,                     # E[K(0)]: K(0) ~ Poisson(k0_mean) truncated to {0..M}
    age0_scale=0.0,                  # mean initial age (days) of the K(0) premises, Exponential
                                     # truncated to [0, horizon]; 0 = all fresh (tau_0 = 0),
                                     # Inf = uniform over [0, horizon] (the truncated limit)
    release_threshold=0.95,          # P_free floor for the t_release output
    # evaluation grid (2026-09-15): NOT a model parameter. Optional list of times (each a
    # non-negative multiple of dt at or below the horizon) at which the running P_free and
    # P(no alarm so far) are read off the single march; None = round times only (v0.6 output).
    eval_times=None,
    # cost weights (NOT lit-derived; placeholders per priors_design.md note)
    c_lost_freedom=1.0,
    c_per_case=1.0,
    c_per_test=1.0,
    c_per_round=1.0,
    # numerics
    dt=0.05,                         # time step (days); round_times must be multiples.
                                     # 0.05, not 0.5: the propagation freezes coefficients
                                     # across the step and the age grid advects by one cell
                                     # per step, so the scheme is FIRST ORDER in dt and its
                                     # error is one-signed. At 0.5 that error carried 97.6%
                                     # of the grid-wide cf<sim bias (2026-08-03 attribution).
                                     # Grid cost is nearly flat in dt (expm_multiply work
                                     # scales with ||h.A||); interactive cost is not.
)


# --------------------------------------------------------------------------------------
# Within-farm trajectory + derived per-age detection curves.
# --------------------------------------------------------------------------------------
def within_farm_curves(p: dict, age_grid: np.ndarray, age_grid_ext: np.ndarray | None = None) -> dict:
    """Integrate the within-farm ODE in tau and return p_shed, f_clinical, p_clinical,
    HSe and h_pass on `age_grid` (followed by `age_grid_ext`, if given).

    `age_grid_ext` (v0.6): extra ages BEYOND age_grid[-1], needed when the zone starts with
    premises of initial age tau_0 in [0, horizon], which reach tau_0 + t <= 2*horizon within
    the program. The extension CONTINUES the same trajectory from its state at age_grid[-1]
    with a second solve; the first solve is byte-for-byte the v0.5 call (same interval, same
    t_eval, same max_step), so the values on `age_grid` do not move.
    # HAZARD (degenerate limit k0_mean = 0, normative reduction): never re-solve over the longer
    # interval instead -- LSODA's initial step and step sequence depend on the interval, and
    # the [0, horizon] values would shift at the 1e-10 level, breaking the 1e-12 anchors.
    """
    beta_w = rate_from_doubling(p["doubling_time_within"])
    gamma_L = rate_from_mean(p["mean_latent_time"])
    gamma_R = rate_from_mean(p["mean_recovery_time_within"])
    piL0 = float(p["p_latent_seed"])
    piI0 = float(p["p_infectious_seed"])

    def rhs(tau, y):
        piL, piI, piR = y
        piS = 1.0 - piL - piI - piR
        dL = beta_w * piI * piS - gamma_L * piL
        dI = gamma_L * piL - gamma_R * piI
        dR = gamma_R * piI
        return [dL, dI, dR]

    tmax = float(age_grid[-1])
    extend = age_grid_ext is not None and len(age_grid_ext) > 0
    grid = np.concatenate([age_grid, age_grid_ext]) if extend else age_grid
    if piL0 + piI0 <= 0.0:
        # No seeding -> no within-farm epidemic ever.
        pi_I = np.zeros_like(grid)
    else:
        sol = solve_ivp(
            rhs, (0.0, tmax), [piL0, piI0, 0.0],
            t_eval=age_grid, method="LSODA", rtol=1e-8, atol=1e-11, max_step=tmax,
        )
        if not sol.success:
            raise RuntimeError(f"within-farm ODE failed: {sol.message}")
        pi_I = sol.y[1]
        if extend:
            # v0.6: continue the SAME course past the horizon from its end state (pi_L, pi_I,
            # pi_R at tau = horizon), so an initial premises' course at tau_0 + t is the
            # standard course, not a clamp at p_shed(horizon).
            tmax2 = float(age_grid_ext[-1])
            sol2 = solve_ivp(
                rhs, (tmax, tmax2), list(sol.y[:, -1]),
                t_eval=age_grid_ext, method="LSODA", rtol=1e-8, atol=1e-11, max_step=tmax2 - tmax,
            )
            if not sol2.success:
                raise RuntimeError(f"within-farm ODE (age extension) failed: {sol2.message}")
            pi_I = np.concatenate([pi_I, sol2.y[1]])
        pi_I = np.clip(pi_I, 0.0, 1.0)

    p_shed = pi_I  # shedding prevalence drives transmission AND active detection
    onset = p["mean_clinical_onset_time"]
    if onset is None or not np.isfinite(float(onset)) or float(onset) <= 0:
        # onset -> 0 : clinical fraction saturates immediately (continuity with tiny onset).
        f_clin = np.full_like(grid, float(p["f_clinical_max"]))
    else:
        f_clin = float(p["f_clinical_max"]) * (1.0 - np.exp(-grid / float(onset)))
    p_clin = f_clin * pi_I

    # Active herd sensitivity (EXACT, not the factorized approximation).
    # X ~ Bin(N, p_shed); each shedding animal tests positive iid w.p. Se, so an animal is
    # "would-test-positive" iid w.p. q = p_shed*Se. All-negative for a without-replacement
    # sample of n_animals = P(none of the n sampled is would-test-positive) = (1-q)^n
    # exactly (the hypergeometric averages out over the iid animal statuses). => closed form.
    Se = float(p["Se_animal"])
    N_herd = int(round(p["N_herd"]))
    # The sample is drawn WITHOUT replacement from the herd, so it cannot exceed the herd:
    # the spec's J ~ Hyp(N_herd, X, n_animals_per_round) is only defined for n <= N_herd.
    # (2026-08-03: the lane tested 30 animals out of a herd of 10; the simulator already
    # capped, so this is a correctness + lane-parity fix. 4 grid cells, no bias effect.)
    n_an = min(int(round(p["n_animals_per_round"])), N_herd)
    q = np.clip(p_shed * Se, 0.0, 1.0)
    HSe = 1.0 - (1.0 - q) ** n_an

    # Passive hazard (per-farm clinical-reporting hazard), model_design.md.
    rho_a = float(p["rho_a"])
    rho_r = float(p["rho_r"])
    h_pass = rho_r * (1.0 - (1.0 - rho_a * np.clip(p_clin, 0.0, 1.0)) ** N_herd)

    return dict(p_shed=p_shed, f_clinical=f_clin, p_clinical=p_clin, HSe=HSe, h_pass=h_pass)


# --------------------------------------------------------------------------------------
# (K,D) master-equation generators.
# --------------------------------------------------------------------------------------
def _bd_block(birth_in: np.ndarray, birth_out: np.ndarray, d: np.ndarray,
              extra_loss: np.ndarray) -> np.ndarray:
    """Birth-death generator block. dz/dt = A z.
    birth_in[K-1]  : inflow rate K-1 -> K (sub-diagonal)
    birth_out[K]   : total birth outflow leaving state K (diagonal loss)
    d[K]           : death rate K -> K-1
    extra_loss[K]  : additional loss out of this block (e.g. passive detection K,0 -> K,1)."""
    N = len(d)
    A = np.zeros((N, N))
    idx = np.arange(N)
    A[idx, idx] = -(birth_out + d + extra_loss)
    A[np.arange(1, N), np.arange(0, N - 1)] = birth_in[:N - 1]   # birth inflow (sub-diag)
    A[np.arange(0, N - 1), np.arange(1, N)] = d[1:]              # death inflow (super-diag)
    return A


def _build_generator(N: int, bint: np.ndarray, b0: np.ndarray, b1: np.ndarray,
                     d: np.ndarray, hpassK: np.ndarray, no_secondary: bool) -> np.ndarray:
    """Full 2N x 2N generator for [branch D=0 ; branch D=1].

    P-chain (no_secondary=False): birth inflow uses the *total* birth (intro + secondary).
    Q-chain (no_secondary=True):  birth inflow uses INTRO ONLY; secondary births are an
      absorbing leak (mass exits the tracked no-secondary set), captured by keeping the
      diagonal loss = total birth b (so b - bint = secondary rate leaves the system).
    In both, passive detection couples D=0 -> D=1 (hpassK)."""
    G = np.zeros((2 * N, 2 * N))
    if no_secondary:
        blk0 = _bd_block(bint, b0, d, hpassK)   # D=0: intro inflow, total-birth outflow, +passive loss
        blk1 = _bd_block(bint, b1, d, np.zeros(N))
    else:
        blk0 = _bd_block(b0, b0, d, hpassK)
        blk1 = _bd_block(b1, b1, d, np.zeros(N))
    G[0:N, 0:N] = blk0
    G[N:2 * N, N:2 * N] = blk1
    G[N:2 * N, 0:N] += np.diag(hpassK)          # passive detection: D=0 -> D=1
    return G


def _round_survival(abar: float, M: int, n_farms: int, N: int) -> np.ndarray:
    """L_r(K) = E_J[ abar^J ], J ~ Hypergeometric(M, K, n_farms), abar = mean per-infected-farm
    all-negative probability. EXACT hypergeometric term (avoids the rare-positive factorization
    hazard); iid-age closure supplies abar. L_r(0)=1."""
    n = min(int(n_farms), M)
    abar = float(np.clip(abar, 0.0, 1.0))
    surv = np.ones(N)
    for K in range(N):
        jmax = min(n, K)
        if jmax == 0:
            surv[K] = 1.0
            continue
        j = np.arange(jmax + 1)
        pmf = hypergeom.pmf(j, M, K, n)
        surv[K] = float(np.sum(pmf * abar ** j))
    return np.clip(surv, 0.0, 1.0)


def _truncated_poisson(mu: float, N: int) -> np.ndarray:
    """P(K_0 = k) for k = 0..N-1: Poisson(mu) truncated to {0..N-1} and renormalised
    (model_design.md "Initial state", v0.6).

    Branch-free at mu = 0: xlogy(0, 0) = 0 and gammaln(1) = 0 give exp(0) = 1.0 exactly at
    k = 0 and xlogy(k > 0, 0) = -inf gives 0.0 elsewhere, so the sum is exactly 1.0 and the
    renormalisation is exact: the point mass at K = 0 to the bit (the normative reduction).
    log-pmf <= 0 always, so exp() cannot overflow at large mu (the OAT extreme is M/2)."""
    k = np.arange(N, dtype=float)
    logpmf = xlogy(k, mu) - mu - gammaln(k + 1.0)
    pmf = np.exp(logpmf)
    return pmf / pmf.sum()


# --------------------------------------------------------------------------------------
# Main evaluation.
# --------------------------------------------------------------------------------------
def evaluate(params: dict) -> dict:
    """Evaluate the model at a pinned parameter point theta. Returns P(free|theta) at each
    round time plus the design-time (t=0) projection outputs. Deterministic (no MC)."""
    p = dict(DEFAULTS)
    p.update({k: v for k, v in params.items() if v is not None})

    M = int(round(p["M"]))
    N = M + 1
    K_star = int(np.floor(float(p["design_prevalence"]) * M))   # effective-freedom threshold; K<=K* counts "free" (K*=0 => P(K=0))
    h = float(p["dt"])
    horizon = float(p["horizon"])
    round_times = sorted(float(t) for t in p["round_times"] if float(t) <= horizon + 1e-9)
    n_farms = int(round(p["n_farms_per_round"]))
    n_farms = min(n_farms, M)

    lam = rate_from_mean(p["mean_intro_interval"])          # REGIONAL intro rate (not x M)
    beta_max = rate_from_doubling(p["doubling_time"])       # between-farm
    gamma_reg = rate_from_mean(p["mean_recovery_time"])     # regional clearance (default 0)
    lock = float(p["beta_lockdown_multiplier"])

    # initial state (v0.6-nws)
    k0_mean = float(p["k0_mean"])                           # E[K(0)] before truncation
    if not (k0_mean >= 0.0):
        raise ValueError(f"k0_mean must be >= 0, got {p['k0_mean']}")
    if p["age0_scale"] is not None and float(p["age0_scale"]) < 0.0:
        raise ValueError(f"age0_scale must be >= 0, got {p['age0_scale']}")
    release_threshold = float(p["release_threshold"])
    if not (0.0 < release_threshold <= 1.0):
        raise ValueError(f"release_threshold must be in (0, 1], got {p['release_threshold']}")

    nT = int(round(horizon / h))
    age_grid = np.arange(nT + 1) * h                        # ages 0, h, ..., horizon
    # v0.6: an initial premises of age tau_0 in [0, horizon] reaches tau_0 + t <= 2*horizon
    # within the program, so its curves are needed on (horizon, 2*horizon] as well. The
    # extension exists ONLY when there is an initial cohort: with k0_mean = 0 every array
    # keeps its v0.5 length and the reductions below sum the same numbers in the same order
    # (numpy's pairwise summation depends on the length, so padding with zeros would not be
    # bit-identical).
    n_ext = nT if k0_mean > 0.0 else 0
    age_grid_ext = np.arange(nT + 1, nT + n_ext + 1) * h if n_ext > 0 else None
    n_age = nT + n_ext + 1
    curves = within_farm_curves(p, age_grid, age_grid_ext)
    p_shed_age = curves["p_shed"]
    h_pass_age = curves["h_pass"]
    HSe_age = curves["HSe"]

    # round-time -> time index
    round_idx = {}
    for tr in round_times:
        i = int(round(tr / h))
        if abs(i * h - tr) > 1e-6:
            raise ValueError(f"round time {tr} is not a multiple of dt={h}")
        round_idx[i] = tr

    # evaluation times (2026-09-15, option A of changes/2026-09-09-eval-times/proposal.md): an
    # OPTIONAL grid at which the running P(K <= K* | no alarm so far) and P(no alarm so far) are
    # read off the single full-horizon march below. Not a model parameter: the (K, D) march, the
    # age field, the round updates and every existing output are untouched by it (the record only
    # READS P0). Validated like the round times (a multiple of dt), plus: non-negative and at or
    # below the horizon, else ValueError (a round time beyond the horizon is dropped silently for
    # the schedule's sake; an eval time beyond it is a caller error). eval_idx: step index -> key.
    eval_idx = {}
    if p["eval_times"] is not None:
        for te in p["eval_times"]:
            te = float(te)
            if not np.isfinite(te) or te < 0.0:
                raise ValueError(f"eval time {te} must be finite and >= 0")
            if te > horizon + 1e-9:
                raise ValueError(f"eval time {te} is beyond the horizon {horizon}")
            i = int(round(te / h))
            if abs(i * h - te) > 1e-6:
                raise ValueError(f"eval time {te} is not a multiple of dt={h}")
            # One entry per instant: an eval time on a round's step index keys the round's own
            # entry (the round's float), and t = 0 keys the prior's entry 0.0.
            eval_idx[i] = 0.0 if i == 0 else round_idx.get(i, te)

    Karr = np.arange(N, dtype=float)
    frac = (M - Karr) / M                                    # (M-K)/M
    dK = gamma_reg * Karr

    # distributions (v0.6: K(0) ~ Poisson(k0_mean) truncated to {0..M}, D(0) = 0; with
    # k0_mean = 0 this is the point mass at K = 0 exactly, see _truncated_poisson)
    P0 = _truncated_poisson(k0_mean, N)                      # D=0 branch carries all mass
    P1 = np.zeros(N)
    Q0 = P0.copy()                                           # no-secondary sub-mass: the initial
    Q1 = np.zeros(N)                                         # premises are not secondaries
    p_k0_zero = float(P0[0])                                 # P(K_0 = 0), for the alarm ceiling
    field = np.zeros(n_age)                                  # age-cohort weights | D=0
    if k0_mean > 0.0:
        # Initial cohort: expected count k0_mean spread over ages 0..horizon by the density of
        # Exponential(mean = age0_scale) truncated to [0, horizon], times the age step, then
        # renormalised on the grid so the field's mass is exactly k0_mean. Nothing sits on the
        # extension ages (tau_0 <= horizon by the truncation).
        # HAZARD (degenerate limit age0_scale = 0): harden by FLOORING THE TIME-SCALE through
        # rate_from_mean (rate 1/MIN_TIMESCALE = 1e3/d), never by special-casing the value:
        # exp(-1e3 * h) = exp(-50) ~ 2e-22 leaves 1 - 2e-22 of the mass at age 0, the "all
        # fresh" limit, and age0_scale = 0 and 1e-6 floor to the same weights (continuity
        # f(0) ~ f(eps)). age0_scale = Inf gives rate 0 -> uniform over [0, horizon], the
        # truncated exponential's own limit, so the boundary is total.
        # HAZARD (joint count-and-age conditioning): the field carries ONE age mix for all K
        # (the parent's iid-age closure, the open (K, age) item). The initial cohort inherits
        # it: its mass is E[K_0] and its ages are the unconditioned tau_0 law; after a clean
        # round the survivors are reweighted young by the same factor for every K, not per
        # (K, age) jointly. The Poisson-thinning oracle is blind to this (no detection), so
        # the stage-3 comparison at the many_fresh_leftovers_x_high_detection corner is what
        # tests it.
        rate0 = rate_from_mean(p["age0_scale"])
        w0 = np.exp(-rate0 * age_grid)                       # truncated-exponential density x h,
        w0 = w0 / w0.sum()                                   # renormalised on the grid
        field[:nT + 1] = k0_mean * w0
    cum_inc = 0.0                                            # counts births after t = 0 only

    # P_free(0) = P(K_0 <= K*): the truncated-Poisson mass at or below K* (e^{-k0_mean} at
    # K* = 0 up to the truncation); 1.0 exactly when k0_mean = 0.
    p_free_by_time = {0.0: float(P0[:K_star + 1].sum() / max(P0.sum(), 1e-300))}
    # P(no alarm so far) = P0.sum(), recorded at the eval times only (2026-09-15); empty when
    # eval_times is None. At t = 0, D(0) = 0 carries all the mass (1.0 up to the truncated
    # Poisson's renormalisation rounding).
    p_noalarm_by_time = {}
    if 0 in eval_idx:
        p_noalarm_by_time[eval_idx[0]] = float(P0.sum())

    surv_step_age = math.exp(-gamma_reg * h) * np.exp(-h_pass_age * h)  # per-cohort step survival

    for i in range(nT):
        W = field.sum()
        if W > 0:
            mean_pshed = float((field * p_shed_age).sum() / W)
            mean_hpass = float((field * h_pass_age).sum() / W)
        else:
            mean_pshed = 0.0
            mean_hpass = 0.0

        beta0 = beta_max * mean_pshed
        beta1 = lock * beta_max * mean_pshed
        bint = frac * lam
        b0 = frac * (lam + beta0 * Karr)
        b1 = frac * (lam + beta1 * Karr)
        hpassK = Karr * mean_hpass

        # cumulative incidence (design-time; over the JOINT (K,D), so lockdown lowers it)
        inc_rate = float((b0 * P0).sum() + (b1 * P1).sum())
        cum_inc += inc_rate * h

        # advance (K,D) by matrix exponential over the step (frozen coefficients).
        GP = _build_generator(N, bint, b0, b1, dK, hpassK, no_secondary=False)
        z = expm_multiply(h * GP, np.concatenate([P0, P1]))
        P0, P1 = z[:N], z[N:]
        GQ = _build_generator(N, bint, b0, b1, dK, hpassK, no_secondary=True)
        zq = expm_multiply(h * GQ, np.concatenate([Q0, Q1]))
        Q0, Q1 = zq[:N], zq[N:]

        # age-cohort field: born at age 0, age + clearance + passive survival (reconditions
        # the survivors YOUNG, i.e. joint (count, age) conditioning on no-detection).
        denomD0 = max(P0.sum(), 1e-300)
        B_field = float((b0 * P0).sum() / denomD0)           # E[new-infection rate | D=0]
        new_field = np.zeros(n_age)
        new_field[1:] = field[:-1] * surv_step_age[1:]
        new_field[0] = B_field * h
        field = new_field

        if (i + 1) in round_idx:
            W = field.sum()
            abar = float((field * (1.0 - HSe_age)).sum() / W) if W > 0 else 1.0
            surv = _round_survival(abar, M, n_farms, N)      # L_r(K), all-negative likelihood
            # Bayes filter on (K,D): detected mass moves D=0 -> D=1.
            P1 = P1 + P0 * (1.0 - surv)
            P0 = P0 * surv
            Q1 = Q1 + Q0 * (1.0 - surv)
            Q0 = Q0 * surv
            # recondition the age field on the round (sampled + caught farms removed young-bias)
            field = field * (1.0 - (n_farms / M) * HSe_age)
            p_free_by_time[round_idx[i + 1]] = float(P0[:K_star + 1].sum() / max(P0.sum(), 1e-300))

        if eval_idx and (i + 1) in eval_idx:
            # Evaluation-time record (2026-09-15): READ-ONLY on P0, no arithmetic feeds forward.
            # HAZARD (ordering at coincident times; "Conditioning and state", the running
            # estimate is filtered and causal): this block sits AFTER the round block on purpose.
            # At an instant that is both a round and an eval time the round has already moved the
            # caught mass D=0 -> D=1 and written the post-round value under the same key; this line
            # rewrites the identical number (same expression on the same arrays), so the entry is
            # the post-round value, which is what a round time has always recorded. Recording
            # before the round would put the pre-round value under the round's key and silently
            # move t_release.
            te = eval_idx[i + 1]
            p_free_by_time[te] = float(P0[:K_star + 1].sum() / max(P0.sum(), 1e-300))
            p_noalarm_by_time[te] = float(P0.sum())

    # ---- outputs -------------------------------------------------------------------
    P_D0 = float(P0.sum())
    P_lost_freedom = float(P1.sum())                         # P(>=1 alarm over horizon)
    P_free_final = float(P0[:K_star + 1].sum() / max(P_D0, 1e-300))
    P_R_gt0 = float(1.0 - (Q0.sum() + Q1.sum()))             # P(>=1 secondary infection)
    n_rounds = len(round_times)
    tests = n_rounds * n_farms * int(round(p["n_animals_per_round"]))
    expected_cost = (
        float(p["c_lost_freedom"]) * P_lost_freedom
        + float(p["c_per_case"]) * cum_inc
        + float(p["c_per_test"]) * tests
        + float(p["c_per_round"]) * n_rounds
    )
    cost_share_per_farm = expected_cost / M

    # t_release (v0.6): the first ROUND time t_r (r >= 1; t = 0 is the prior, not a round) at
    # which the running P_free is at least release_threshold, else None.
    t_release = None
    for tr in round_times:
        if p_free_by_time[tr] >= release_threshold:
            t_release = tr
            break

    out = dict(
        p_free_by_time=p_free_by_time,
        P_free=P_free_final,                # canonical: running estimate at the final round
        P_lost_freedom=P_lost_freedom,
        cumulative_incidence=cum_inc,
        P_R_gt0=P_R_gt0,
        expected_cost=expected_cost,
        cost_share_per_farm=cost_share_per_farm,
        t_release=t_release,
        release_threshold=release_threshold,
        # evaluation-time output (2026-09-15): the grid echoed back (canonical keys, sorted) or
        # None, and P(no alarm so far) at those times (empty dict when eval_times is None)
        eval_times=sorted(eval_idx.values()) if eval_idx else None,
        p_noalarm_by_time=p_noalarm_by_time,
        # diagnostics
        _P_D0=P_D0,
        _P_K0_zero=p_k0_zero,
        _lambda=lam,
        _beta_max=beta_max,
        _horizon=horizon,
    )
    assert_invariants(out, p, lam, horizon, p_k0_zero)
    return out


def assert_monotonicity(intro_interval: float = 20.0, **backdrop) -> None:
    """Evidence-strength monotonicity, over the three dials that sharpen a round.

    Deliberately NOT called from assert_invariants(): that runs on every evaluate(), and a
    monotonicity check needs several evaluations, so calling it there would recurse. It is a
    self_test()-level invariant instead.

    P_free is non-decreasing and P(no alarm) is non-increasing in each of n_animals_per_round,
    n_farms_per_round and Se_animal. Both follow from the same fact -- a sharper test is harder
    to pass, and worth more once passed -- so they fail together if the conditioning is ever
    inverted.

    `backdrop` (v0.6): extra pinned parameters, e.g. k0_mean=2.0 to run the same sweeps with
    initial premises present (the defaults, k0_mean = 0, are the v0.5 check unchanged).
    """
    sweeps = {
        "n_animals_per_round": [5, 15, 30, 60, 100],
        "n_farms_per_round": [2, 5, 10, 20, 50],
        "Se_animal": [0.2, 0.4, 0.6, 0.8, 0.95],
    }
    tol = 1e-9                      # deterministic lane: only float noise to absorb
    for dial, levels in sweeps.items():
        pf, nd = [], []
        for v in levels:
            o = evaluate({"mean_intro_interval": intro_interval, **backdrop, dial: v})
            pf.append(float(o["P_free"]))
            nd.append(float(o["_P_D0"]))
        for i in range(1, len(levels)):
            assert pf[i] >= pf[i - 1] - tol, (
                f"P_free must not fall as {dial} rises: {levels[i-1]} -> {levels[i]} gave "
                f"{pf[i-1]:.6f} -> {pf[i]:.6f}. P_free conditions on no alarm, so more "
                f"searching evidence that still comes back clean can only raise it.")
            assert nd[i] <= nd[i - 1] + tol, (
                f"P(no alarm) must not rise as {dial} rises: {levels[i-1]} -> {levels[i]} gave "
                f"{nd[i-1]:.6f} -> {nd[i]:.6f}. A sharper test is harder to pass.")


def assert_invariants(out: dict, p: dict, lam: float, horizon: float,
                      p_k0_zero: float = 1.0) -> None:
    """Logical bounds + known monotonicities as invariants (cheapest spec-error detector).

    `p_k0_zero` = P(K_0 = 0) from the truncated Poisson (1.0 at k0_mean = 0, v0.6)."""
    # probabilities in [0,1]
    for t, v in out["p_free_by_time"].items():
        assert -1e-9 <= v <= 1 + 1e-9, f"P_free out of [0,1] at t={t}: {v}"
    for key in ("P_free", "P_lost_freedom", "P_R_gt0", "_P_D0"):
        assert -1e-9 <= out[key] <= 1 + 1e-9, f"{key} out of [0,1]: {out[key]}"
    # total (K,D) mass conserved (no mass removed by detection; it moves branches)
    assert abs(out["_P_D0"] + out["P_lost_freedom"] - 1.0) < 1e-6, \
        f"(K,D) mass not conserved: {out['_P_D0']} + {out['P_lost_freedom']}"
    # HAZARD (v0.5 saturation; v0.6 initial state): P_lost cannot exceed P(K_0 > 0 or >= 1
    # introduction). Every infestation that begins after t = 0 (incl. secondary) traces to an
    # introduction or to an initial premises, and the instantaneous intro rate is <= lam, so
    # P(no alarm) >= P(K_0 = 0) * exp(-lam * horizon). At k0_mean = 0 this is the v0.5 ceiling
    # unchanged (P(K_0 = 0) = 1).
    intro_ceiling = 1.0 - p_k0_zero * math.exp(-lam * horizon)
    assert out["P_lost_freedom"] <= intro_ceiling + 1e-6, \
        f"P_lost {out['P_lost_freedom']} exceeds alarm ceiling {intro_ceiling}"
    # cumulative incidence non-negative
    assert out["cumulative_incidence"] >= -1e-9, "negative cumulative incidence"
    # v0.6: P_free(0) is the truncated-Poisson mass at or below K* (e^{-k0_mean} at K* = 0
    # up to the truncation): equals p_k0_zero when K* = 0.
    M = int(round(p["M"]))
    K_star = int(np.floor(float(p["design_prevalence"]) * M))
    if K_star == 0:
        assert abs(out["p_free_by_time"][0.0] - p_k0_zero) < 1e-12, \
            f"P_free(0) {out['p_free_by_time'][0.0]} != P(K_0 = 0) {p_k0_zero}"
    # v0.6: t_release is the FIRST round time at or above the threshold, else None.
    thr = float(out["release_threshold"])
    # The ROUND times come from the schedule, not from the keys of p_free_by_time: since the
    # eval-times output (2026-09-15) that dict also carries interior evaluation times, which are
    # not rounds and must not enter the t_release rule. With eval_times = None the two readings
    # are the same set (every round time at or below the horizon is recorded under its own float).
    rounds = sorted(float(t) for t in p["round_times"] if 0.0 < float(t) <= horizon + 1e-9)
    tr = out["t_release"]
    if tr is None:
        assert all(out["p_free_by_time"][t] < thr for t in rounds), \
            f"t_release is None but some round reaches {thr}"
    else:
        assert tr in rounds and out["p_free_by_time"][tr] >= thr, \
            f"t_release {tr} is not a round at or above {thr}"
        assert all(out["p_free_by_time"][t] < thr for t in rounds if t < tr), \
            f"a round before t_release {tr} already reaches {thr}"


# --------------------------------------------------------------------------------------
# v0.6 self-test helpers: the reduction anchor cells and the Poisson-thinning oracle
# (validation_design.md "Pre-seeded-start axes"; both NORMATIVE for this lane).
# --------------------------------------------------------------------------------------
ANCHOR_CSV_ENV = "NWS_ANCHOR_CSV"       # override the anchor file location (e.g. a sandbox run)
ANCHOR_CSV_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "changes", "2026-09-07-preseeded-start", "reduction_anchor_cells.csv")
ANCHOR_TOL = 1e-12                      # closed form at k0_mean = 0: machine precision
ORACLE_TOL = 1e-9                       # closed form vs exp(-k0_mean e^{-g t})


def _parse_cell(s) -> float | None:
    """Grid-CSV scalar -> float. 'Inf' spellings -> inf ('never'); blank/NA -> None (default)."""
    t = str(s).strip()
    if t == "" or t.lower() in ("na", "nan", "none"):
        return None
    if t.lower() in ("inf", "+inf", "infinity"):
        return float("inf")
    return float(t)


def round_times_from(horizon, round_interval) -> tuple:
    """The grid's schedule: interval, 2*interval, ... <= horizon."""
    horizon = float(horizon)
    interval = float(round_interval)
    if not (np.isfinite(interval) and interval > 0.0):
        raise ValueError(f"round_interval must be finite and > 0, got {round_interval}")
    n = int(math.floor(horizon / interval + 1e-9))
    return tuple(float(k) * interval for k in range(1, n + 1))


def load_anchor_cells(path: str | None = None) -> list:
    """The reduction-anchor cells (reduction_anchor_cells.csv): one dict per row with the
    params (every model column the row carries; k0_mean/age0_scale absent -> defaults 0),
    the round schedule from its horizon/round_interval, the eval time and the expected value."""
    import csv
    path = path or os.environ.get(ANCHOR_CSV_ENV) or ANCHOR_CSV_DEFAULT
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"reduction anchor file not found: {path} (set ${ANCHOR_CSV_ENV} to point at "
            f"changes/2026-09-07-preseeded-start/reduction_anchor_cells.csv)")
    cells = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            params = {}
            for k, v in row.items():
                if k in DEFAULTS and k not in ("round_times", "dt"):
                    val = _parse_cell(v)
                    if val is not None:
                        params[k] = val
            params["round_times"] = round_times_from(row["horizon"], row["round_interval"])
            cells.append(dict(
                params=params,
                eval_time=float(row["eval_time"]),
                expected=float(row["expected_p_free_at_eval"]),
                label=f"row {row['row_index']:>3} {row['arm']:<11} {row['targets']}",
            ))
    return cells


def check_reduction_anchors(path: str | None = None, verbose: bool = True) -> float:
    """NORMATIVE: at k0_mean = 0 every anchor cell reproduces the 2026-09-08 lane to 1e-12."""
    cells = load_anchor_cells(path)
    if verbose:
        # say which file and how many cells ran (a sandbox run may point $NWS_ANCHOR_CSV at a subset)
        print(f"   {len(cells)} anchor cells from {path or os.environ.get(ANCHOR_CSV_ENV) or ANCHOR_CSV_DEFAULT}")
    worst = 0.0
    for c in cells:
        got = evaluate(c["params"])["p_free_by_time"][c["eval_time"]]
        diff = abs(got - c["expected"])
        worst = max(worst, diff)
        if verbose:
            print(f"   {c['label']:<58} t={c['eval_time']:5.0f}  expected={c['expected']:.16f}"
                  f"  got={got:.16f}  |diff|={diff:.2e}")
        assert diff < ANCHOR_TOL, (
            f"reduction anchor broken at {c['label']} t={c['eval_time']}: got {got!r}, "
            f"expected {c['expected']!r} (|diff| {diff:.2e} >= {ANCHOR_TOL})")
    return worst


def check_poisson_thinning_oracle(verbose: bool = True) -> float:
    """NORMATIVE oracle (validation_design.md `oracle_poisson_thinning`): k0_mean = 3, all
    fresh, no introductions, no spread, no detection, clearance on at g = 1/21: each initial
    premises clears independently by t w.p. 1 - e^{-g t}, the survivors are
    Poisson(3 e^{-g t}), so P_free(t) = exp(-3 e^{-g t}) at every round and e^{-3} at t = 0."""
    k0, g = 3.0, 1.0 / 21.0
    rounds = (30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
    out = evaluate(dict(
        k0_mean=k0, age0_scale=0.0, mean_intro_interval=float("inf"),
        doubling_time=float("inf"), Se_animal=0.0, rho_r=0.0, rho_a=0.0,
        mean_recovery_time=21.0, horizon=180.0, round_times=rounds,
    ))
    worst = 0.0
    for t in sorted(out["p_free_by_time"]):
        expect = math.exp(-k0 * math.exp(-g * t))
        got = out["p_free_by_time"][t]
        diff = abs(got - expect)
        worst = max(worst, diff)
        if verbose:
            print(f"   t={t:5.0f}  P_free={got:.12f}  exp(-3 e^(-t/21))={expect:.12f}  |diff|={diff:.2e}")
        assert diff < ORACLE_TOL, \
            f"Poisson-thinning oracle broken at t={t}: got {got!r}, expected {expect!r}"
    # t_release follows from the same formula: the first round with exp(-3 e^{-g t}) >= 0.95.
    thr = out["release_threshold"]
    t_expect = next((t for t in rounds if math.exp(-k0 * math.exp(-g * t)) >= thr), None)
    assert out["t_release"] == t_expect, \
        f"t_release {out['t_release']} != analytic {t_expect} at threshold {thr}"
    if verbose:
        print(f"   t_release={out['t_release']} (analytic: first round with exp(-3 e^(-t/21)) "
              f">= {thr}: {t_expect})")
    return worst


# --------------------------------------------------------------------------------------
# Evaluation-time self-test (2026-09-15, changes/2026-09-09-eval-times; NORMATIVE for this lane):
# the eval_times output is a pure addition, checked three ways on the presets the app runs.
# --------------------------------------------------------------------------------------
PRESETS_PY_ENV = "NWS_PRESETS_PY"       # override the preset table location (e.g. a sandbox run)
PRESETS_PY_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "presets.py")
EVAL_ROUND_TOL = 1e-12                  # eval-times run vs plain run: round values and scalars
EVAL_INTERIOR_TOL = 1e-9                # eval-times run vs truncated re-evaluation, clean start
EVAL_AGED_TOL = 1e-4                    # the two routes from 45 d on at the zone preset


def _load_presets(path: str | None = None):
    """The app's preset table (apps/review_app/presets.py), loaded by path for the self-test only:
    the lane does not import the app; the self-test borrows its five screwworm presets."""
    import importlib.util
    path = path or os.environ.get(PRESETS_PY_ENV) or PRESETS_PY_DEFAULT
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"preset table not found: {path} (set ${PRESETS_PY_ENV} to point at "
            f"apps/review_app/presets.py)")
    spec = importlib.util.spec_from_file_location("nws_presets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dense_eval_grid(round_times, horizon, h, interior: int = 3) -> list:
    """The app's dense grid (apps/review_app/app.py, pfree_trajectory): `interior` equally spaced
    points per inter-round interval snapped to the dt grid, the t - dt point before each round,
    the rounds themselves, and 0. Sorted, every entry a multiple of h."""
    rounds = sorted(float(t) for t in round_times if float(t) <= float(horizon) + 1e-9)
    pts = {0.0}
    prev = 0.0
    for tr in rounds:
        for j in range(1, interior + 1):
            t = round(round((prev + (tr - prev) * j / (interior + 1)) / h) * h, 6)
            if prev < t < tr:
                pts.add(t)
        pre = round(tr - h, 6)
        if pre > prev:
            pts.add(pre)
        pts.add(tr)
        prev = tr
    return sorted(x for x in pts if 0.0 <= x <= float(horizon) + 1e-9)


def _truncated(p: dict, t: float, rounds: list) -> dict:
    """The pre-2026-09-15 dense route: re-evaluate with the horizon cut at t."""
    return evaluate(dict(p, horizon=float(t), round_times=[r for r in rounds if r <= t + 1e-9]))


def check_eval_times(verbose: bool = True) -> dict:
    """NORMATIVE (2026-09-15). (a) Round-time identity: on the five screwworm presets an eval_times
    run with three interior points per interval (plus the t - dt pre-round points and the rounds)
    reports the same p_free_by_time at 0 and every round, and the same P_free, P_lost_freedom,
    cumulative_incidence, P_R_gt0 and t_release, as the plain run, to 1e-12. (b) Interior identity
    at a clean start (k0_mean = 0, age0_scale = 0): each interior value equals truncated-horizon
    re-evaluation to 1e-9, on six_months and zone_last_detection. (c) Aged start: on the zone
    preset (k0_mean = 1, age0_scale = 14) the eval-times value is the higher one at 3.75, 7.5 and
    15 d (the truncated route forces the leftovers young), not lower at 22.5 and 29.95 d, and the
    two routes agree within 1e-4 from 45 d on (the proposal's table)."""
    presets = _load_presets()
    h = float(DEFAULTS["dt"])

    def P(name):
        return presets.preset_to_params(presets.SCENARIOS[name], h)

    worst = dict(round=0.0, scalar=0.0, interior=0.0, interior_noalarm=0.0)
    scalars = ("P_free", "P_lost_freedom", "cumulative_incidence", "P_R_gt0")

    # (a) round-time identity
    if verbose:
        print("   (a) round-time identity, eval_times run vs plain run, five screwworm presets:")
    for name in ("six_months", "zone_last_detection", "three_cycles", "twelve_months", "early_warning"):
        p = P(name)
        rounds = sorted(float(t) for t in p["round_times"])
        grid = dense_eval_grid(rounds, p["horizon"], h, interior=3)
        plain = evaluate(p)
        ev = evaluate(dict(p, eval_times=grid))
        assert plain["eval_times"] is None and plain["p_noalarm_by_time"] == {}, \
            "plain run must echo eval_times=None and an empty p_noalarm_by_time"
        d_round = max(abs(ev["p_free_by_time"][t] - plain["p_free_by_time"][t]) for t in [0.0] + rounds)
        d_scalar = max(abs(float(ev[k]) - float(plain[k])) for k in scalars)
        same_release = ev["t_release"] == plain["t_release"]
        assert ev["eval_times"] == grid, f"{name}: eval_times not echoed back as given"
        assert set(ev["p_free_by_time"]) == set(grid) and len(ev["p_free_by_time"]) == len(grid), \
            f"{name}: p_free_by_time keys must be exactly the grid (one entry per instant)"
        assert set(ev["p_noalarm_by_time"]) == set(grid), f"{name}: p_noalarm_by_time keys != grid"
        assert all(-1e-12 <= v <= 1 + 1e-12 for v in ev["p_noalarm_by_time"].values())
        n_int = len(grid) - 1 - len(rounds)
        if verbose:
            print(f"      {name:<20} horizon {p['horizon']:5.0f} d, {len(rounds):2d} rounds, {n_int:2d} interior pts: "
                  f"max|dP_free| at 0+rounds = {d_round:.1e}, max|d scalar| = {d_scalar:.1e}, "
                  f"t_release {plain['t_release']} == {ev['t_release']}: {same_release}")
        worst["round"] = max(worst["round"], d_round)
        worst["scalar"] = max(worst["scalar"], d_scalar)
        assert d_round < EVAL_ROUND_TOL, f"{name}: round-time identity broken ({d_round:.2e})"
        assert d_scalar < EVAL_ROUND_TOL, f"{name}: scalar identity broken ({d_scalar:.2e})"
        assert same_release, f"{name}: t_release moved ({plain['t_release']} -> {ev['t_release']})"

    # (b) interior identity at a clean start
    if verbose:
        print("   (b) interior identity at a clean start (k0_mean = 0, age0_scale = 0), eval_times run vs "
              "truncated re-evaluation:")
    for name in ("six_months", "zone_last_detection"):
        p = dict(P(name), k0_mean=0.0, age0_scale=0.0)
        rounds = sorted(float(t) for t in p["round_times"])
        grid = dense_eval_grid(rounds, p["horizon"], h, interior=3)
        ev = evaluate(dict(p, eval_times=grid))
        d_max, dn_max, n = 0.0, 0.0, 0
        for t in grid:
            if t <= 0.0 or t in rounds:
                continue
            tr = _truncated(p, t, rounds)
            d = abs(ev["p_free_by_time"][t] - float(tr["P_free"]))
            dn = abs(ev["p_noalarm_by_time"][t] - (1.0 - float(tr["P_lost_freedom"])))
            d_max, dn_max, n = max(d_max, d), max(dn_max, dn), n + 1
        if verbose:
            print(f"      {name:<20} {n:2d} interior times: max|dP_free| = {d_max:.1e}, "
                  f"max|P_noalarm - (1 - P_lost)| = {dn_max:.1e}")
        worst["interior"] = max(worst["interior"], d_max)
        worst["interior_noalarm"] = max(worst["interior_noalarm"], dn_max)
        assert d_max < EVAL_INTERIOR_TOL, f"{name}: interior identity broken at a clean start ({d_max:.2e})"
        assert dn_max < 1e-6, f"{name}: P(no alarm) disagrees with 1 - P_lost beyond mass-conservation noise"

    # (c) aged start: the documented difference
    p = P("zone_last_detection")
    assert float(p["k0_mean"]) == 1.0 and float(p["age0_scale"]) == 14.0, \
        "the zone preset no longer carries k0_mean = 1, age0_scale = 14; update check (c)"
    rounds = sorted(float(t) for t in p["round_times"])
    times = [3.75, 7.5, 15.0, 22.5, 29.95, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5, 90.0]
    ev = evaluate(dict(p, eval_times=times))
    rows = []
    for t in times:
        tr = _truncated(p, t, rounds)
        rows.append((t, ev["p_free_by_time"][t], float(tr["P_free"])))
    if verbose:
        print("   (c) aged start, zone preset (k0_mean = 1, age0_scale = 14): eval_times value vs "
              "truncated re-evaluation")
        print("      t (d)   eval_times   truncated    eval - trunc")
        for t, a, b in rows:
            print(f"      {t:6.2f}   {a:.6f}     {b:.6f}    {a - b:+.6f}")
    early = {t: a - b for t, a, b in rows if t < 30.0}
    late = {t: abs(a - b) for t, a, b in rows if t >= 45.0}
    for t in (3.75, 7.5, 15.0):
        assert early[t] > 1e-3, f"eval-times value not clearly above the truncated one at {t} d ({early[t]:+.2e})"
    for t in (22.5, 29.95):
        assert early[t] >= -1e-12, f"eval-times value below the truncated one at {t} d ({early[t]:+.2e})"
    assert max(late.values()) < EVAL_AGED_TOL, f"routes differ by {max(late.values()):.2e} from 45 d on"
    worst["aged_late"] = max(late.values())
    worst["aged_early"] = early
    return worst


if __name__ == "__main__":
    print("=== self-test / smoke run (closed-form lane) ===\n")

    # (1) Flagship FMD baseline (validation scenario).
    base = evaluate({})
    print("FMD baseline P_free by round time:")
    for t in sorted(base["p_free_by_time"]):
        print(f"   t={t:6.1f} d  P_free={base['p_free_by_time'][t]:.4f}")
    print(f"   P_lost_freedom={base['P_lost_freedom']:.4f}  cum_incidence={base['cumulative_incidence']:.4f}")
    print(f"   P_R_gt0={base['P_R_gt0']:.4f}  expected_cost={base['expected_cost']:.2f}"
          f"  cost/farm={base['cost_share_per_farm']:.4f}\n")

    # (2) ORACLE: no introductions -> P_free == 1 at all times.
    o1 = evaluate({"mean_intro_interval": float("inf")})
    m1 = max(abs(v - 1.0) for v in o1["p_free_by_time"].values())
    print(f"[oracle] no-intro  max|P_free-1| = {m1:.2e}  (expect ~0)")

    # (3) ORACLE: no detection (Se=0, rho_r=0, no spread) -> P_free == P(K=0) == exp(-lam t).
    o2 = evaluate({"Se_animal": 0.0, "rho_r": 0.0, "doubling_time": float("inf")})
    lam = 1.0 / 50.0
    worst = 0.0
    for t, v in o2["p_free_by_time"].items():
        if t == 0:
            continue
        worst = max(worst, abs(v - math.exp(-lam * t)))
    print(f"[oracle] no-detect max|P_free-exp(-lam t)| = {worst:.2e}  (expect small)")

    # (4) CONTINUITY GATE: latent=0 vs latent=1e-6 must agree (inverted-limit detector).
    z0 = evaluate({"mean_latent_time": 0.0})
    ze = evaluate({"mean_latent_time": 1e-6})
    cont = max(abs(z0["p_free_by_time"][t] - ze["p_free_by_time"][t]) for t in z0["p_free_by_time"])
    ag = np.arange(int(round(120 / 0.5)) + 1) * 0.5
    ps0 = within_farm_curves(dict(DEFAULTS, mean_latent_time=0.0), ag)["p_shed"]
    pse = within_farm_curves(dict(DEFAULTS, mean_latent_time=1e-6), ag)["p_shed"]
    print(f"[continuity] latent 0 vs 1e-6: max|dP_free|={cont:.2e}, "
          f"max|dp_shed|={np.max(np.abs(ps0 - pse)):.2e}  (expect ~0)")
    print(f"             p_shed(latent=0) at tau=30d = {ps0[60]:.3f}  (should be large, NOT ~0.01)")

    # (5) EVIDENCE-STRENGTH MONOTONICITY.
    #
    #   P_free  = P(K <= K* | NO ALARM)  -- conditional. RISES with more testing, because
    #             surviving a more searching test is stronger evidence of freedom.
    #   _P_D0   = P(no alarm)            -- unconditional. FALLS with more testing, because
    #             a harder test is harder to pass.
    #
    # Both are correct simultaneously; they answer different questions. The old comment's
    # intuition belongs to _P_D0.
    assert_monotonicity()
    print("[monotone] P_free rises and P(no alarm) falls with more testing (both asserted)")

    # ---- v0.6-nws pre-seeded start ---------------------------------------------------
    # (6) REDUCTION ANCHOR (normative, model_design.md "Degenerate limits"): at k0_mean = 0
    #     the lane must reproduce the 2026-09-08 closed-form smoke run to machine precision.
    #     19 named cells, each row carrying every parameter and its own 180-d schedule.
    print(f"\n[reduction anchor] k0_mean = 0 vs the 2026-09-08 lane, tol {ANCHOR_TOL:.0e}:")
    worst_anchor = check_reduction_anchors()
    print(f"   max |P_free - expected| over the anchor cells = {worst_anchor:.2e}  (asserted < {ANCHOR_TOL:.0e})")

    # (7) POISSON-THINNING ORACLE (normative, validation_design.md `oracle_poisson_thinning`):
    #     P_free(t) = exp(-3 e^{-t/21}) at every round, e^{-3} at t = 0, t_release from it.
    print(f"\n[oracle] Poisson thinning (k0_mean=3, all fresh, no intro/spread/detection, "
          f"clearance 21 d), tol {ORACLE_TOL:.0e}:")
    worst_oracle = check_poisson_thinning_oracle()
    print(f"   max |P_free - exp(-3 e^(-t/21))| = {worst_oracle:.2e}  (asserted < {ORACLE_TOL:.0e})")

    # (8) CONTINUITY GATES at the new degenerate boundaries (f(0) ~ f(eps)).
    #     k0_mean 0 vs 1e-9: the initial cohort's whole effect is O(1e-9), so the reduction is
    #     the limit, not a special case. age0_scale 0 vs 1e-6 (at k0_mean = 2): both floor to
    #     the same time-scale, so the two runs must coincide.
    c0 = evaluate({"k0_mean": 0.0})
    ce = evaluate({"k0_mean": 1e-9})
    d_k0 = max(abs(c0["p_free_by_time"][t] - ce["p_free_by_time"][t]) for t in c0["p_free_by_time"])
    a0 = evaluate({"k0_mean": 2.0, "age0_scale": 0.0})
    ae = evaluate({"k0_mean": 2.0, "age0_scale": 1e-6})
    d_a0 = max(abs(a0["p_free_by_time"][t] - ae["p_free_by_time"][t]) for t in a0["p_free_by_time"])
    print(f"\n[continuity] k0_mean 0 vs 1e-9: max|dP_free|={d_k0:.2e} (expect ~1e-9); "
          f"age0_scale 0 vs 1e-6 at k0_mean=2: max|dP_free|={d_a0:.2e} (expect 0)")
    assert d_k0 < 1e-6, f"k0_mean boundary is not continuous: {d_k0}"
    assert d_a0 < 1e-9, f"age0_scale boundary is not continuous: {d_a0}"

    # (9) EVIDENCE-STRENGTH MONOTONICITY with initial premises present (k0_mean = 2,
    #     age0_scale = 14): the same three sweeps as (5) on the pre-seeded start.
    assert_monotonicity(k0_mean=2.0, age0_scale=14.0)
    print("[monotone] ... and with initial premises present (k0_mean=2, age0_scale=14; asserted)")

    # (10) THE RELEASE TRANSIENT on the reviewed screwworm backdrop (anchor row 0: the OAT
    #      star centre, 6 monthly rounds to 180 d) with the zone-preset placeholders
    #      k0_mean = 1, age0_scale = 14 (priors_design.md): P_free starts at e^{-1} and climbs;
    #      t_release reads off the crossing. Then the age0_scale direction (scenarios.md #18,
    #      asserted there, REPORTED here): P_free at a fixed t vs the leftovers' mean age.
    backdrop = load_anchor_cells()[0]["params"]
    zone = evaluate(dict(backdrop, k0_mean=1.0, age0_scale=14.0))
    print("\n[transient] reviewed backdrop + k0_mean=1, age0_scale=14 (zone-preset placeholders):")
    for t in sorted(zone["p_free_by_time"]):
        print(f"   t={t:6.1f} d  P_free={zone['p_free_by_time'][t]:.6f}")
    print(f"   t_release={zone['t_release']} at threshold {zone['release_threshold']};  "
          f"P_lost_freedom={zone['P_lost_freedom']:.4f}  cum_incidence={zone['cumulative_incidence']:.4f}"
          f"  P_R_gt0={zone['P_R_gt0']:.4f}")
    assert abs(zone["p_free_by_time"][0.0] - math.exp(-1.0)) < 1e-12, "P_free(0) != e^{-k0_mean}"
    print("   P_free(0) = e^{-1} to 1e-12 (asserted; truncation at M=100 is far below that)")
    ages = [0.0, 3.0, 14.0, 60.0, 180.0]
    marks = [30.0, 90.0, 180.0]
    rows = []
    for a in ages:
        o = evaluate(dict(backdrop, k0_mean=2.0, age0_scale=a))
        rows.append([o["p_free_by_time"][t] for t in marks])
        print(f"   k0_mean=2, age0_scale={a:5.0f}: " +
              "  ".join(f"P_free({t:.0f})={v:.6f}" for t, v in zip(marks, rows[-1])) +
              f"  t_release={o['t_release']}")
    for j, t in enumerate(marks):
        col = [r[j] for r in rows]
        mono = all(col[i] >= col[i - 1] - 1e-9 for i in range(1, len(col)))
        print(f"   P_free({t:.0f}) non-decreasing in age0_scale: {'yes' if mono else 'NO'}"
              f"  (scenarios.md #18, reported not asserted)")

    # (11) EVALUATION TIMES (2026-09-15, changes/2026-09-09-eval-times, normative): a pure addition.
    #      (a) round values and scalars identical to the plain run on the five screwworm presets,
    #      (b) interior values equal truncated re-evaluation at a clean start, (c) the aged-start
    #      difference is the documented one (higher early, same from 45 d).
    print(f"\n[eval times] pure addition, tol round/scalar {EVAL_ROUND_TOL:.0e}, interior {EVAL_INTERIOR_TOL:.0e}, "
          f"aged from 45 d {EVAL_AGED_TOL:.0e}:")
    w = check_eval_times()
    print(f"   worst: round {w['round']:.1e}, scalar {w['scalar']:.1e}, interior {w['interior']:.1e}, "
          f"from 45 d {w['aged_late']:.1e}  (all asserted)")

    print("\nall invariants passed.")
