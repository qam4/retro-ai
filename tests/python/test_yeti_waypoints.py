"""Tests for retro_ai.games.yeti.waypoints (pure, tilemap-derived)."""

from __future__ import annotations

from retro_ai.games import yeti


def test_waypoints_level2_has_top_and_bottom_per_ladder():
    wps = yeti.waypoints(2)
    # L2 ladders: L12a, L12b, L23a, L23b, L34, L45a, L45b, L56 -> 8 ladders,
    # each contributing a _top and a _bot waypoint.
    assert len(wps) == 16
    for name in ("L12a", "L12b", "L23a", "L23b", "L34", "L45a", "L45b", "L56"):
        assert f"{name}_top" in wps
        assert f"{name}_bot" in wps


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
