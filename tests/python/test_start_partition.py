"""The three-bucket start partition: reset | MANDATORY | OTHER.

Why it exists. The legacy partition is reset | rungs | one waypoint group. A rung
pool is only ever filled on a FRUIT pickup, so on a one-fruit level exactly one rung
pool exists ("just past the fruit") -- and as its own top-level candidate it took
34.6% of L4 v2's starts, re-practising ground already at 93%, while every waypoint
including the stuck frontier shared the remaining third at ~2% each.

A rung pool is also not a distinct SITUATION: "3 mandatory targets done" on L4 means
"standing just past Fr2", which the Fr2 waypoint pool already holds with a known
position. So rungs are mandatory starts without a name and belong in the same bucket
as the mandatory waypoints. On L1/L2, where mandatory targets ARE the fruits and
there are no waypoints, that bucket is exactly the old rung set -- the change
unifies rather than adding a concept.

These tests use L4 v2's measured goal_scores so the numbers mean something.
"""

import collections
import importlib.util
import pathlib
import random
import sys

import pytest

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "train_checkpoint_curriculum.py"
)

# Measured on yeti_curriculum_l4_v2_15m at ~14M.
WP_GOAL_SCORES = {
    "Lfruit_top": 0.549,
    "Lfruit_bot": 0.550,
    "Fr1": 0.552,
    "Fr1_launch": 0.552,
    "Fr2_launch": 0.556,
    "Fr2": 0.568,
    "Lascent_top": 0.600,
    "Rope1": 0.616,
    "Lclimb1_top": 0.620,
    "Rope1_launch": 0.630,
    "Spring_launch": 0.634,
    "Lclimb2_top": 0.635,
    "Spring": 0.636,
    "Step_launch": 0.637,
    "Step": 0.643,
}
# L4's mandatory position waypoints (from build_targets), plus graph aliases.
# One frozenset per route STEP. A jump landing carries two names (the curriculum's
# `Fr1`, the graph's `J2_3_b`) and they are the same step, so they share a group --
# see test_progress_rungs_are_groups.py for why counting ids instead double-counted.
MANDATORY_GROUPS = [
    frozenset({"Lfruit_top"}),
    frozenset({"Fr1", "J2_3_b"}),
    frozenset({"Lascent_top"}),
    frozenset({"Lclimb1_top"}),
    frozenset({"Rope1", "J6_7_b"}),
    frozenset({"Lclimb2_top"}),
    frozenset({"Spring", "J8_9_b"}),
    frozenset({"Step", "J9_10_b"}),
    frozenset({"Lclimb3_top"}),
    frozenset({"Lhi_down_bot"}),
    frozenset({"Lprincess_top"}),
]
MANDATORY = {name for g in MANDATORY_GROUPS for name in g}


@pytest.fixture(scope="module")
def tcc():
    spec = importlib.util.spec_from_file_location("tcc_partition", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tcc_partition"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native/ML deps absent
        # SKIP, do not ERROR. The trainer imports stable_baselines3, and CI's
        # python-test job installs only numpy/pytest/gymnasium/pyyaml -- so without
        # this the whole module errors at fixture setup and turns the build red.
        # This file was added 2026-08-17 and CI had not run since 2026-06-30, so it
        # never once passed there; the breakage surfaced the next time anything was
        # pushed. Every other test module that loads the trainer already guards this
        # way, which is why only this one failed.
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


def _mgr(tcc, split, rung3_gs=0.55, reset_gs=0.55):
    m = tcc.CheckpointManager(
        max_states_per_checkpoint=100,
        min_states_to_advance=20,
        reset_fraction=0.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        reach_threshold=0.15,
        n_rungs=13,
        mandatory_groups=MANDATORY_GROUPS,
        gate_waypoints=True,
        split_mandatory=split,
    )
    # Only rung 0 and rung 3 have pools, as measured on L4 v2.
    for lvl, gs in ((0, reset_gs), (3, rung3_gs)):
        for i in range(100):
            m.save_checkpoint(lvl, b"s%d" % i, source_cp=0, bonus=0)
        m.checkpoints[lvl].goal_score = gs
        m.reset_reach_ema[lvl] = 0.9
    for w, gs in WP_GOAL_SCORES.items():
        m.save_waypoint(w, b"x", 100, True, source_cp=0, bonus=0)
        m.waypoints[w].goal_score = gs
        m.wp_reach_ema[w] = 0.9  # all past the 0.15 gate
    return m


def _shares(m, n=40000, seed=7):
    random.seed(seed)
    c = collections.Counter()
    for _ in range(n):
        key, *_ = m.pick_start()
        c[key if isinstance(key, str) else f"rung{key}"] += 1
    return {k: v / n for k, v in c.items()}


def test_legacy_gives_the_lone_rung_pool_a_third_of_all_starts(tcc):
    s = _shares(_mgr(tcc, split=False))
    assert s["rung3"] > 0.25, f"expected the legacy rung sink, got {s.get('rung3')}"
    assert s["rung0"] > 0.25


def test_legacy_starves_the_frontier(tcc):
    s = _shares(_mgr(tcc, split=False))
    assert (
        s.get("Step", 0) < 0.03
    ), f"Step should be starved in legacy, got {s.get('Step')}"


def test_split_helps_the_frontier_but_does_not_fix_it(tcc):
    """Measured: Step goes 1.86% -> 3.12%, i.e. ~1.7x, not more.

    That ceiling is the point. The partition removes the duplicate rung sink, but
    WITHIN the mandatory bucket the weights are still 1 - goal_score, and
    goal_score is absolute depth reached, so a deep start scores high for free and
    gets the smallest slice. Step remains the worst-served mandatory start. Fixing
    that needs the second change: score what the episode ADDED over what it had
    left, not how deep it ended up.
    """
    legacy = _shares(_mgr(tcc, split=False))
    split = _shares(_mgr(tcc, split=True))
    assert (
        split["Step"] > 1.5 * legacy["Step"]
    ), f"Step {legacy['Step']:.4f} -> {split['Step']:.4f}, expected >1.5x"
    assert split["Step"] < 0.06, (
        "if this now passes comfortably, the goal_score fix probably landed too "
        "and this test is no longer isolating the partition"
    )


def test_split_keeps_reset_protected(tcc):
    """The whole point of grouping (v8) was that adding starts must not crowd out
    reset. Buckets use the MEAN, so reset keeps roughly a third."""
    s = _shares(_mgr(tcc, split=True))
    assert 0.25 < s["rung0"] < 0.45, s["rung0"]


def test_split_puts_the_rung_pool_in_the_mandatory_bucket(tcc):
    """It stops being its own third and competes with the mandatory waypoints."""
    legacy = _shares(_mgr(tcc, split=False))
    split = _shares(_mgr(tcc, split=True))
    assert (
        split["rung3"] < legacy["rung3"] / 2
    ), f"rung3 {legacy['rung3']:.3f} -> {split['rung3']:.3f}"


def test_split_routes_launch_pads_into_the_other_bucket(tcc):
    """Launch pads are not mandatory, so they must not dilute the spine."""
    s = _shares(_mgr(tcc, split=True))
    mand = sum(v for k, v in s.items() if k in MANDATORY or k == "rung3")
    other = sum(v for k, v in s.items() if k.endswith("_launch") or k == "Lfruit_bot")
    assert mand > other, f"mandatory {mand:.3f} vs other {other:.3f}"


def test_default_is_legacy(tcc):
    m = tcc.CheckpointManager(
        max_states_per_checkpoint=10,
        min_states_to_advance=2,
        reset_fraction=0.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        n_rungs=4,
        mandatory_groups=[],
    )
    assert m.split_mandatory is False
