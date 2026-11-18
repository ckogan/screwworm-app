"""Traps and wildlife findings as evidence: the arithmetic of a PROPOSED design. Illustration only.

This module is not part of the model. No lane reads it and nothing here changes a model result.
It evaluates the formulas of the proposal written in model_design.md, "Additional evidence
streams" (proposed 2026-11-19, not accepted, not built), for the review app's illustration tab.

For one infested premises and one stream s (fly traps, or wildlife and companion-animal findings):

    S_s(tau) = 1 - c_s + c_s * exp(-q_s * max(tau - d_s, 0))

is the chance the stream has NOT found the premises by infestation age tau, with coverage c_s,
rate q_s per day and delay d_s days. Streams act independently, so the chance that no stream in
use has found it is the product S(tau). In the simplest case (m expected hidden infested premises
present from day 0, nothing else acting) the chance the zone is free given no finding by day t is
exp(-m * S(t)).

Standard library only, so that the browser build of the app can import it.
Self-check:  python streams_illustration.py
"""
from __future__ import annotations

import math

# Proposed starting values and ranges. Copied by hand from priors_design.md, section
# "Proposed rows: additional evidence streams (2026-11-19; not accepted; not in force)",
# columns "Central" and "Range to explore". That table carries long free-text cells, so it is
# copied here and not parsed at import. If the proposal changes, change these numbers with it.
# The trap delay is the one value marked derived there. The other five are marked guessed.
STREAMS = {
    "trap": {
        "label": "Fly traps",
        "short": "traps",
        "coverage": {"value": 0.5, "low": 0.1, "high": 1.0, "step": 0.05},      # trap_coverage
        "rate": {"value": 0.005, "low": 0.001, "high": 0.07, "step": 0.001},    # trap_catch_rate, per day
        "delay": {"value": 14.0, "low": 10.0, "high": 35.0, "step": 1.0},       # trap_delay, days
    },
    "wild": {
        "label": "Wildlife and companion-animal findings",
        "short": "wildlife",
        "coverage": {"value": 0.3, "low": 0.05, "high": 0.8, "step": 0.05},     # wild_coverage
        "rate": {"value": 0.01, "low": 0.002, "high": 0.1, "step": 0.001},      # wild_find_rate, per day
        "delay": {"value": 7.0, "low": 3.0, "high": 30.0, "step": 1.0},         # wild_delay, days
    },
}
STREAM_KEYS = ("trap", "wild")

# Expected hidden infested premises at day 0, for the simplest case on the tab.
HIDDEN = {"value": 1.0, "low": 0.2, "high": 3.0, "step": 0.1}

TABLE_DAYS_FOUND = (30, 90, 180)
TABLE_DAYS_CREDIT = (90, 180)


def not_found(tau: float, coverage: float, rate: float, delay: float) -> float:
    """S_s(tau): chance one infested premises has not been found by one stream by age tau."""
    u = max(float(tau) - float(delay), 0.0)
    return 1.0 - coverage + coverage * math.exp(-rate * u)


def not_found_all(tau: float, streams: list) -> float:
    """S(tau): chance no stream in use has found it. `streams` is a list of (c, q, d). Empty gives 1."""
    s = 1.0
    for c, q, d in streams:
        s *= not_found(tau, c, q, d)
    return s


def found(tau: float, streams: list) -> float:
    """Chance one infested premises has been found by at least one stream in use by age tau."""
    return 1.0 - not_found_all(tau, streams)


def ceiling_found(streams: list) -> float:
    """The most the streams in use could ever find: 1 minus the product of (1 - coverage)."""
    miss = 1.0
    for c, q, _ in streams:
        miss *= 1.0 - (c if q > 0.0 else 0.0)
    return 1.0 - miss


def p_free(t: float, m: float, streams: list) -> float:
    """Simplest case: chance the zone is free given no finding by day t, exp(-m * S(t))."""
    return math.exp(-m * not_found_all(t, streams))


def p_free_none(m: float) -> float:
    """Simplest case with no traps and no wildlife checks: exp(-m), flat."""
    return math.exp(-m)


def p_free_ceiling(m: float, streams: list) -> float:
    """Simplest case: the level the curve approaches and never passes."""
    return math.exp(-m * (1.0 - ceiling_found(streams)))


def credit(t: float, m: float, streams: list) -> float:
    """Rise in the chance of freedom, in the simplest case, from empty results up to day t."""
    return p_free(t, m, streams) - p_free_none(m)


def per_week(rate: float) -> float:
    """The chance per week equivalent to a rate per day."""
    return 1.0 - math.exp(-7.0 * rate)


def setting(key: str, which: str) -> tuple:
    """(coverage, rate, delay) of one stream: 'value' is the proposed starting point, 'low' the end
    of each range that earns least (low coverage, low rate, long delay), 'high' the end that earns most."""
    s = STREAMS[key]
    if which == "value":
        return (s["coverage"]["value"], s["rate"]["value"], s["delay"]["value"])
    if which == "low":
        return (s["coverage"]["low"], s["rate"]["low"], s["delay"]["high"])
    if which == "high":
        return (s["coverage"]["high"], s["rate"]["high"], s["delay"]["low"])
    raise ValueError(which)


def summary(m: float, streams: list) -> dict:
    """The table numbers for one set of streams."""
    return {
        "found": {d: found(d, streams) for d in TABLE_DAYS_FOUND},
        "credit": {d: credit(d, m, streams) for d in TABLE_DAYS_CREDIT},
        "none": p_free_none(m),
        "ceiling": p_free_ceiling(m, streams),
    }


def self_check() -> list:
    """Compare with the worked numbers written in the proposed text. Returns the lines checked."""
    lines = []

    def near(name, got, want, tol):
        assert abs(got - want) <= tol, f"{name}: got {got!r}, want {want!r}"
        lines.append(f"ok  {name}: {got:.4f} (text {want})")

    trap, wild = setting("trap", "value"), setting("wild", "value")
    # priors_design.md, "Where the data is thin": figures printed to two decimals
    for day, want in zip((30, 90, 180), (0.04, 0.16, 0.28)):
        near(f"traps, found by day {day}", found(day, [trap]), want, 0.005)
    for day, want in zip((30, 90, 180), (0.06, 0.17, 0.25)):
        near(f"wildlife, found by day {day}", found(day, [wild]), want, 0.005)
    near("trap rate as a chance per week", per_week(0.005), 0.034, 0.0005)
    # scenarios.md, proposed check 21: three expected leftovers, coverage 0.5, rate 0.05, delay 14
    oracle = [(0.5, 0.05, 14.0)]
    days = (0, 14, 30, 60, 90, 120, 150, 180)
    for day, want in zip(days, (0.0498, 0.0498, 0.1137, 0.1920, 0.2158, 0.2215, 0.2228, 0.2230)):
        near(f"check 21, chance free at day {day}", p_free(day, 3.0, oracle), want, 0.00006)
    for day, want in zip(days, (1.0, 1.0, 0.4378, 0.2593, 0.2307, 0.2248, 0.2235, 0.2232)):
        near(f"check 21, chance of no finding at day {day}",
             math.exp(-3.0 * found(day, oracle)), want, 0.00006)
    near("check 21, ceiling", p_free_ceiling(3.0, oracle), 0.2231, 0.00006)
    assert all(p_free(t, 3.0, oracle) < p_free_ceiling(3.0, oracle) for t in range(0, 361))
    assert all(p_free(t, 3.0, oracle) <= p_free_ceiling(3.0, oracle) for t in range(0, 100001, 50))
    lines.append("ok  check 21: the curve stays under the ceiling to day 360 and never passes it later")
    # full coverage: exp(-3 * exp(-0.05 * (t - 14))) from day 14
    near("check 21, full coverage at day 60", p_free(60, 3.0, [(1.0, 0.05, 14.0)]),
         math.exp(-3.0 * math.exp(-0.05 * 46.0)), 1e-12)
    # switched off: nothing is earned
    assert found(180, []) == 0.0 and credit(180, 1.0, []) == 0.0
    assert found(180, [(0.5, 0.0, 14.0)]) == 0.0 and found(180, [(0.0, 0.05, 14.0)]) == 0.0
    assert found(14, [trap]) == 0.0 and found(7, [wild]) == 0.0
    lines.append("ok  streams off, zero rate, zero coverage, or before the delay: nothing found, no credit")
    return lines


if __name__ == "__main__":
    for line in self_check():
        print(line)
    print("self-check passed")
