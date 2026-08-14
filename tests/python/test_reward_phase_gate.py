"""The waypoint sum must honour the fruit phase, like the base potential does.

Context. The base potential is already phase-aware: it sums distance to REMAINING
FRUITS and only switches to the princess once none are left. The mandatory-waypoint
sum never inherited that rule, so on L4 every one of the 12 groups was active from
step 0 — and because L4's fruit sits at the opposite end of the map from the
princess, the sum was minimised by walking AWAY from the fruit. Measured on
yeti_curriculum_l4_v1: Lfruit_top reached 0.1%, Lascent_top 99%, and ZERO fruits
collected across 3000 reset-origin episodes. The fruit is mandatory to finish, so
the level was unwinnable by construction, not merely hard.

These tests pin the direction of the shaping gradient, which is the property that
actually broke. They do not need an emulator.
"""

import pytest
from retro_ai.training.rewards import RewardContext
from retro_ai.training.rewards import create as create_reward
from retro_ai.training.yeti_map import get_level_map

# Grounded walking pose, alive, on the L4 ground floor (y=182).
GROUND_Y = 182
PARAMS = {
    "scale": 0.01,
    "fruit_scale": 0.01,
    "princess_scale": 0.05,
    "level": 4,
    "gamma": 1.0,
    "defer_fruit_credit": True,
    "waypoint_reward_tol": 2,
    "ladder_segment_shaping": True,
}


def _ctx(x_ram, fruits_present=(True,), **kw):
    return RewardContext(
        prev_fruits=1,
        curr_fruits=1,
        prev_bonus=1000,
        curr_bonus=1000,
        prev_score=0,
        curr_score=0,
        prev_lives=6,
        curr_lives=6,
        step_count=kw.pop("step", 10),
        curr_x=x_ram,
        curr_y=kw.pop("y", GROUND_Y),
        fruits_present=fruits_present,
        pose=0,
        **kw,
    )


def _walk(fn, xs, fruits_present=(True,), y=GROUND_Y):
    """Total shaping reward for walking through ``xs`` (RAM x units)."""
    total = 0.0
    for i, x in enumerate(xs):
        total += fn(_ctx(x, fruits_present, step=i + 1, y=y))
    return total


def _fresh(**overrides):
    p = dict(PARAMS)
    p.update(overrides)
    return create_reward("fruit_bonus_path_progress_pbrs_grounded", p)


# Agent starts at x_ram 42 (px 176, col 22). The fruit's ladder is at x_ram 50
# (px 208, col 26); the ascent ladder is at x_ram 16 (px 72, col 9).
START, TOWARD_FRUIT, TOWARD_ASCENT = 42, 50, 16


def test_l4_declares_a_post_fruit_phase():
    lvl = get_level_map(4)
    assert lvl.waypoints_after_fruit, "L4 must defer the ascent groups"
    # The two groups on the way to the fruit must NOT be deferred.
    assert "Lfruit_top" not in lvl.waypoints_after_fruit
    assert "J2_3_b" not in lvl.waypoints_after_fruit
    # The princess approach must be.
    assert "Lprincess_top" in lvl.waypoints_after_fruit


def test_walking_toward_the_fruit_pays_more_than_walking_away():
    """The property whose absence made v1 unwinnable."""
    to_fruit = _walk(_fresh(), list(range(START, TOWARD_FRUIT + 1)))
    to_ascent = _walk(_fresh(), list(range(START, TOWARD_ASCENT - 1, -1)))
    assert to_fruit > to_ascent, (
        f"shaping still favours the ascent while the fruit is uncollected: "
        f"toward fruit {to_fruit:.4f} vs toward ascent {to_ascent:.4f}"
    )


def test_without_the_gate_the_gradient_would_point_the_wrong_way(monkeypatch):
    """Characterise the BUG, so a regression cannot pass silently.

    With the phase list emptied, the L4 sum reproduces v1's behaviour: walking
    away from the fruit pays better than walking toward it.
    """
    import dataclasses

    import retro_ai.training.yeti_map as ym

    broken = dataclasses.replace(ym.LEVEL4, waypoints_after_fruit=None)
    monkeypatch.setitem(ym.LEVELS, 4, broken)
    to_fruit = _walk(_fresh(), list(range(START, TOWARD_FRUIT + 1)))
    to_ascent = _walk(_fresh(), list(range(START, TOWARD_ASCENT - 1, -1)))
    assert to_ascent > to_fruit, (
        "expected the ungated sum to favour the ascent (the v1 bug); if this "
        "fails the bug characterisation is stale"
    )


def test_after_the_fruit_the_ascent_becomes_attractive():
    """Once the fruit is gone the deferred groups switch on, so heading back
    left (toward the ascent and the princess) must pay."""
    collected = (False,)
    to_ascent = _walk(_fresh(), list(range(START, TOWARD_ASCENT - 1, -1)), collected)
    to_fruit = _walk(_fresh(), list(range(START, TOWARD_FRUIT + 1)), collected)
    assert to_ascent > to_fruit, (
        f"after the fruit the pull should reverse: toward ascent {to_ascent:.4f} "
        f"vs toward the empty fruit corner {to_fruit:.4f}"
    )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_lower_levels_declare_no_phase_gate(level):
    """L1-L3 must be untouched: their targets are co-directional, which is the
    unstated assumption that made the ungated sum work there."""
    assert not get_level_map(level).waypoints_after_fruit
