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

    # y=62 is shared by A3 (floor 13, px 96-136) and A4 (floor 14, px 56-80);
    # x decides which.
    assert xy(64, 62, 3) == 14  # A4 (fruit platform)
    assert xy(100, 62, 3) == 13  # A3
    # y=158 is shared by STEP (2, px48-64), ELAND (5, px160-192), BR (7, px224-320).
    assert xy(56, 158, 3) == 2
    assert xy(176, 158, 3) == 5
    assert xy(280, 158, 3) == 7


def test_xaware_goat_climb_never_resolves_to_snowball():
    from retro_ai.training.yeti_map import agent_floor_from_pixel_xy as xy

    # Climbing the goat ladder (x_px=80) from the 2-ladder platform (floor 3,
    # y150) up to the goat platform (floor 4, y94): mid-climb heights are on
    # no platform (x=80 is far left of every snowball platform) -> None
    # (frozen), never a snowball floor.
    assert xy(80, 150, 3) == 3  # 2-ladder platform
    for y in (140, 125, 110):  # >8px from both 2LAD(150) and GOAT(94)
        assert xy(80, y, 3) is None
    assert xy(80, 94, 3) == 4  # goat platform


def test_l3_escalator_ladder_connects_route():
    """The escalator is modelled as a ladder (GOAT y94 <-> ELAND y158), so the
    route is graph-connected across it (no INF gap) and the reward has a
    gradient off the goat platform toward the jump-off point. The final
    ascending-platform jumps stay INF (sparse by design)."""
    from retro_ai.training.yeti_map import build_navigation_map

    nav = build_navigation_map(3)
    INF = 10**8

    def d(a, b):
        return nav.dist[nav.node_by_ident[a]][nav.node_by_ident[b]]

    assert d("Lesc_top", "Lesc_bot") == 64  # the vertical descent (|94-158|)
    assert d("Lgoat_a_top", "Ldown_bot") < INF  # goat -> across escalator -> bottom
    assert d("Lgoat_a_top", "Lsc4_top") < INF  # goat -> snowball top, all finite
    # The A1-A5 ascent is now modelled as jump_edges (was INF/sparse), so the
    # whole route SN3 -> fruit -> princess is graph-connected and shaped.
    assert d("Lsc4_top", "F1") < INF  # SN3 -> fruit (A4) via ascent jumps
    assert d("Lsc4_top", "Lprincess_top") < INF  # SN3 -> princess, finite now
    # goat-platform gradient: distance to the escalator top decreases moving right
    left = nav.path_distance_from_agent(4, 18 * 4 + 8, "Lesc_top")
    right = nav.path_distance_from_agent(4, 27 * 4 + 8, "Lesc_top")
    assert right < left
    # Ascent gradient: distance to the fruit (A4) decreases monotonically as the
    # agent climbs SN3 -> A1 -> A2 -> A3 -> A4 (the jump_edges shaping).
    ascent = [(10, 288), (11, 176), (12, 152), (13, 116), (14, 64)]
    dists = [nav.path_distance_from_agent(fl, x, "F1") for fl, x in ascent]
    assert dists == sorted(dists, reverse=True) and dists[-1] == 0
    # START->2LAD is now graph-connected too (was INF; START/STEP had no nodes).
    assert nav.path_distance_from_agent(1, 20, "Lgoat_a_top") < INF


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
    # walk on the 2-ladder platform toward the goat ladder (x_px=80 = ram18),
    # climb it (mid-heights freeze), land on the goat platform (y94).
    traj = [
        (18, 150, 0),  # 2-ladder platform (floor 3)
        (18, 140, 8),  # climbing (frozen)
        (18, 120, 8),
        (18, 100, 8),
        (18, 94, 0),  # goat platform (floor 4)
    ]
    total = sum(fn(ctx(x, y, pose)) for (x, y, pose) in traj)
    assert total >= 0.0
