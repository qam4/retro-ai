"""Tests for retro_ai.games.yeti.waypoints (pure, tilemap-derived)."""

from __future__ import annotations

from retro_ai.games import yeti


def test_waypoints_level2_has_top_and_bottom_per_ladder():
    wps = yeti.waypoints(2)
    # L2 ladders: L12a, L12b, L23a, L23b, L34, L45a, L45b, L56, and the two down to
    # the ground (L67a / L67b, added 2026-10-24 with the princess's real floor) -> 10
    # ladders, each contributing a _top and a _bot waypoint.
    assert len(wps) == 20
    for name in (
        "L12a",
        "L12b",
        "L23a",
        "L23b",
        "L34",
        "L45a",
        "L45b",
        "L56",
        "L67a",
        "L67b",
    ):
        assert f"{name}_top" in wps
        assert f"{name}_bot" in wps


def test_level2_princess_is_on_the_ground_where_l2_v10_touches_her():
    """level2_map.json (RAM) puts her at agent_x 72, agent_y 182 -- the implicit
    ground below floor 6 -- and l2_v10 descends L67b at x_ram 56 and touches her at
    x_ram 72. The map had her on floor 6 at y 150 with no way down."""
    from retro_ai.training.yeti_map import agent_floor_from_pixel_y, get_level_map

    m = get_level_map(2)
    assert m.princess_floor == 7 and m.floor_top_y[7] == 182
    assert m.princess_centre_px == (72 * 4 + 8, 182)
    assert agent_floor_from_pixel_y(182, level=2) == 7
    assert yeti.waypoints(2)["L67b_top"] == (56, 150, 6)
    assert yeti.waypoints(2)["L67b_bot"] == (56, 182, 7)


def test_waypoints_match_observed_landings():
    """The pixel<->RAM equation is confirmed by observed agent landings:
    F2 landing RAM x=18 (=L12a px80), F3 landing RAM x=46 (=L23b px192),
    and the F3->F4 goat/L34 at RAM x=32 (px136)."""
    wps = yeti.waypoints(2)
    # L12a bottom: floor 2 (y=54), x_ram = (80-8)//4 = 18.
    assert wps["L12a_bot"] == (18, 54, 2)
    # L23b bottom: floor 3 (y=78), x_ram = (192-8)//4 = 46.
    assert wps["L23b_bot"] == (46, 78, 3)
    # L34 bottom: floor 4 (y=102), x_ram = (136-8)//4 = 32.
    assert wps["L34_bot"] == (32, 102, 4)
    # L34 top is on floor 3 (same x), the goat chokepoint side.
    assert wps["L34_top"] == (32, 78, 3)


def test_waypoints_level1_present():
    wps = yeti.waypoints(1)
    assert wps  # level 1 has ladders too
    for wid, (x_ram, y, floor) in wps.items():
        assert x_ram >= 0
        assert 1 <= floor <= 6
