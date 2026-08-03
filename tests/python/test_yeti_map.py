"""Tests for the hand-coded Yeti navigation graph."""

from __future__ import annotations

import pytest
from retro_ai.training.yeti_map import (
    agent_floor_from_pixel_y,
    build_navigation_map,
)


@pytest.fixture(scope="module")
def nav():
    return build_navigation_map()


def test_fixed_nodes_present(nav):
    """All expected nodes are in the graph."""
    for fid in (1, 2, 3, 4):
        assert f"F{fid}" in nav.node_by_ident
    for ladder in ("L12a", "L12b", "L23", "L34", "L45"):
        assert f"{ladder}_bot" in nav.node_by_ident
        assert f"{ladder}_top" in nav.node_by_ident
    assert "princess" in nav.node_by_ident


def test_same_floor_distance_is_absolute_x(nav):
    """Two nodes on the same floor: cost = |dx|."""
    f1 = nav.nodes[nav.node_by_ident["F1"]]
    l12a_bot = nav.nodes[nav.node_by_ident["L12a_bot"]]
    assert f1.floor == l12a_bot.floor == 1
    d = nav.dist[nav.node_by_ident["F1"]][nav.node_by_ident["L12a_bot"]]
    assert d == abs(f1.x - l12a_bot.x) == 64


def test_adjacent_floor_via_ladder(nav):
    """Fruit 2 to fruit 1: walk to L12a_top (40 px), climb (32 px),
    walk to F1 (64 px) = 136 px."""
    d = nav.dist[nav.node_by_ident["F2"]][nav.node_by_ident["F1"]]
    assert d == 136


def test_far_apart_fruits(nav):
    """Fruit 1 (floor 1) to fruit 4 (floor 4): must traverse multiple
    ladders. Expected path cost: 392 px."""
    d = nav.dist[nav.node_by_ident["F1"]][nav.node_by_ident["F4"]]
    assert d == 392


def test_princess_from_spawn_fruit(nav):
    """F1 -> princess should be 464 px (all the way up and across)."""
    d = nav.dist[nav.node_by_ident["F1"]][nav.node_by_ident["princess"]]
    assert d == 464


def test_distance_symmetric(nav):
    """Graph edges are bidirectional; distances symmetric."""
    a = nav.node_by_ident["F1"]
    b = nav.node_by_ident["F4"]
    assert nav.dist[a][b] == nav.dist[b][a]


def test_path_distance_from_agent_at_spawn(nav):
    """Agent at (floor=1, x=0) to F1 (x=184) is a 184px walk."""
    d = nav.path_distance_from_agent(1, 0, "F1")
    assert d == 184


def test_path_distance_uses_better_ladder(nav):
    """Agent on floor 1 at x=280 (right by L12b). To F2 (x=80 on floor
    2). Should go up L12b, then walk left: 0 + 32 + 200 = 232."""
    d = nav.path_distance_from_agent(1, 280, "F2")
    assert d == 232


def test_path_distance_agent_at_target(nav):
    """Agent at the target fruit gives distance 0."""
    d = nav.path_distance_from_agent(1, 184, "F1")
    assert d == 0


def test_agent_floor_from_pixel_y_standing():
    """Standing y (within tolerance) resolves to correct floor."""
    assert agent_floor_from_pixel_y(184) == 1
    assert agent_floor_from_pixel_y(152) == 2
    assert agent_floor_from_pixel_y(120) == 3
    assert agent_floor_from_pixel_y(88) == 4
    assert agent_floor_from_pixel_y(56) == 5


def test_agent_floor_from_pixel_y_tolerance():
    """8px tolerance around each floor top."""
    assert agent_floor_from_pixel_y(180) == 1
    assert agent_floor_from_pixel_y(192) == 1


def test_agent_floor_from_pixel_y_mid_air_returns_none():
    """A y clearly between floors resolves to None."""
    # Between floor 1 (y=184) and floor 2 (y=152): midpoint 168 is
    # outside the 8px tolerance of either, so None.
    assert agent_floor_from_pixel_y(168) is None
    # Death animation region.
    assert agent_floor_from_pixel_y(16) is None


# ---------------------------------------------------------------------------
# X-aware floor resolution (agent_floor_from_pixel_xy). On L1/L2 (no platform
# extents) it must be byte-identical to the y-only rule; on L3 it disambiguates
# stacked/overlapping platforms and returns None on a tall ladder passing
# between platforms (so shaping freezes rather than pulling to a wrong
# same-height structure — the goat-ladder bug).
# ---------------------------------------------------------------------------


def test_xaware_is_yonly_on_l1_l2():
    from retro_ai.training.yeti_map import (
        agent_floor_from_pixel_xy,
        agent_floor_from_pixel_y,
    )

    for level in (1, 2):
        for y in range(0, 220):
            for x in (0, 50, 150, 300):
                assert agent_floor_from_pixel_xy(
                    x, y, level
                ) == agent_floor_from_pixel_y(y, level)


def test_xaware_disambiguates_l3_y_collisions():
    from retro_ai.training.yeti_map import agent_floor_from_pixel_xy as xy

    # y=80 is within 8px of BOTH floor 3 (y72, fruit platform x0-48) and floor
    # 4 (y80, goat platform x56-136). x decides which.
    assert xy(72, 80, 3) == 4  # goat platform
    assert xy(24, 72, 3) == 3  # fruit platform
    # y=152 snowball platform (x208-312) vs the goat ladder passing through.
    assert xy(280, 152, 3) == 7  # on the snowball platform
    assert xy(72, 152, 3) is None  # on the goat ladder -> no platform -> freeze


def test_xaware_goat_climb_never_resolves_to_snowball():
    from retro_ai.training.yeti_map import agent_floor_from_pixel_xy as xy

    # Climbing the goat ladder (x_px=72) from the 2-ladder platform (floor 8,
    # y168) up to the goat platform (floor 4, y80): the mid-climb heights
    # (152/128/104 = snowball floors 7/6/5) must NOT resolve to those floors.
    assert xy(72, 168, 3) == 8
    for y in (152, 128, 104):
        assert xy(72, y, 3) is None
    assert xy(72, 80, 3) == 4


def test_l3_goat_climb_reward_not_penalised():
    """Regression guard for the L3 v2 bug: climbing toward the goat platform
    must not net negative (previously ~ -4.8 from the snowball mis-pull)."""
    from retro_ai.training.rewards import RewardContext, create

    def ctx(x, y, pose):
        return RewardContext(
            prev_fruits=1,
            curr_fruits=1,
            prev_bonus=1000,
            curr_bonus=1000,
            prev_score=0,
            curr_score=0,
            prev_lives=5,
            curr_lives=5,
            step_count=0,
            curr_x=x,
            curr_y=y,
            fruits_present=(True,),
            pose=pose,
            died=False,
        )

    p = {
        "scale": 0.01,
        "fruit_scale": 0.01,
        "princess_scale": 0.05,
        "level": 3,
        "gamma": 1.0,
        "defer_fruit_credit": True,
        "waypoint_reward_tol": 2,
    }
    fn = create("fruit_bonus_path_progress_pbrs_grounded", p)
    fn.reset()
    traj = [
        (16, 168, 0),
        (16, 160, 8),
        (16, 152, 8),
        (16, 128, 8),
        (16, 104, 8),
        (16, 88, 8),
        (16, 80, 0),
    ]
    total = sum(fn(ctx(x, y, pose)) for (x, y, pose) in traj)
    assert total >= 0.0
