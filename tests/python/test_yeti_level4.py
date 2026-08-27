"""Lock LEVEL4's hand-authored geometry.

LEVEL4 is ~200 lines of coordinates transcribed from a RAM tilemap and a verbal
route description. A single transposed digit would not crash anything — it would
put a reward target slightly off the route and surface weeks later as an
unexplained training ceiling, which is precisely how L3 cost us eight versions.
These assertions encode the invariants that a transcription error breaks.

No emulator and no ROM: everything here is static map data.
"""

import math

import pytest
from retro_ai.games import yeti
from retro_ai.training.targets import build_targets
from retro_ai.training.yeti_map import (
    build_navigation_map,
    get_level_map,
    jump_waypoints,
)

INF_SENTINEL = 10**6
# The agent's verified start: ground floor, x_ram 42 -> px 176, y 182.
START_FLOOR, START_X, START_Y = 1, 176, 182


@pytest.fixture(scope="module")
def lvl():
    return get_level_map(4)


def test_level4_is_registered(lvl):
    assert lvl is not None
    assert len(lvl.floor_top_y) == 24  # 23 tile platforms + implicit ground
    assert len(lvl.ladders) == 8
    assert len(lvl.jump_edges) == 12
    assert len(lvl.platforms) == 24


def test_standing_y_follows_the_measured_row_rule(lvl):
    """standing_y = tile_row*8 - 18, measured by climbing onto row 22 (y=158).

    Verifying via the inverse: every floor_top_y must land on an integer tile row.
    """
    for floor, y in lvl.floor_top_y.items():
        row = (y + 18) / 8
        assert float(row).is_integer(), f"floor {floor} y={y} -> non-integer row {row}"
        assert 0 <= row <= 25, f"floor {floor} row {row} off screen"


def test_ground_floor_matches_the_measured_start(lvl):
    assert lvl.floor_top_y[START_FLOOR] == START_Y


def test_every_floor_has_exactly_one_platform(lvl):
    floors = sorted(lvl.floor_top_y)
    assert sorted(p.floor for p in lvl.platforms) == floors
    for p in lvl.platforms:
        assert p.y == lvl.floor_top_y[p.floor], f"platform {p.floor} y mismatch"
        assert p.x_min < p.x_max, f"platform {p.floor} has empty extent"
        assert 0 <= p.x_min and p.x_max <= 320


def test_ladders_are_ordered_top_first(lvl):
    """LevelMap requires (name, top_floor, bot_floor, x) with top = smaller y."""
    for name, top, bot, _x in lvl.ladders:
        assert lvl.floor_top_y[top] < lvl.floor_top_y[bot], f"{name} not top-first"


def test_ladder_x_lies_within_both_platforms(lvl):
    """A ladder must actually touch the platforms it claims to connect."""
    pf = {p.floor: p for p in lvl.platforms}
    for name, top, bot, x in lvl.ladders:
        for floor in (top, bot):
            p = pf[floor]
            assert p.x_min <= x <= p.x_max, f"{name} x={x} outside floor {floor}"


def test_jump_edge_landing_is_always_the_larger_floor_id(lvl):
    """jump_waypoints() names edges by max(floor), so route order must ascend."""
    for a, b in lvl.jump_edges:
        assert b > a, f"jump edge ({a},{b}) has its landing as the smaller id"


def test_named_jump_waypoints_cover_every_landing_we_track(lvl):
    landings = {max(a, b) for a, b in lvl.jump_edges}
    assert set(lvl.jump_waypoint_names) <= landings
    jw = jump_waypoints(lvl)
    skipped = set(lvl.jump_waypoint_skip or ())
    for name in lvl.jump_waypoint_names.values():
        assert name in jw, f"{name} missing arrival waypoint"
        # A launch pad may be deliberately dropped when the same platform already
        # carries a waypoint that marks correctly -- otherwise it is a second, wider,
        # misplaced box for the same traversal. See LevelMap.jump_waypoint_skip.
        if f"{name}_launch" not in skipped:
            assert f"{name}_launch" in jw, f"{name} missing launch pad"
        else:
            assert (
                f"{name}_launch" not in jw
            ), f"{name}_launch skipped but still emitted"


def test_route_order_matches_the_waypoint_universe(lvl):
    """route_order is display-only, but a name that does not exist hides a point
    from the log table, and a waypoint left out of it is silently unordered."""
    wps = set(yeti.waypoints(4))
    assert set(lvl.route_order) == wps


def test_fruit_and_princess_sit_on_their_platforms(lvl):
    pf = {p.floor: p for p in lvl.platforms}
    for fid, (fx, fy) in lvl.fruit_centre_px.items():
        floor = lvl.fruit_floor[fid]
        assert fy == lvl.floor_top_y[floor]
        assert pf[floor].x_min <= fx <= pf[floor].x_max
    px, py = lvl.princess_centre_px
    assert py == lvl.floor_top_y[lvl.princess_floor]
    assert pf[lvl.princess_floor].x_min <= px <= pf[lvl.princess_floor].x_max


def test_single_fruit_uses_the_counter_as_its_presence_byte():
    """L4 has one fruit, so the remaining-counter satisfies the presence
    contract (non-zero = still on the map) with no dedicated byte."""
    addrs = yeti.fruit_presence_addrs(4)
    assert addrs == {1: yeti.FRUITS_ADDR}


def test_mandatory_targets_are_the_forced_route_only():
    """Neither branch may be mandatory: putting reward on a path the agent will
    not take is the v14 mistake."""
    ts = {t.id: t for t in build_targets(4)}
    branch_only = [
        "Low1",
        "Hi1",
        "Hi2",
        "Hi3",
        "Hi4",
        "Hi5",
        "Lhi_up_top",
    ]
    for tid in branch_only:
        assert not ts[tid].mandatory, f"{tid} must not be a reward target"
    for tid in ("F1", "princess", "Lprincess_top", "Rope1", "Spring"):
        assert ts[tid].mandatory, f"{tid} must be a reward target"


def test_launch_pads_are_never_mandatory():
    """Launch pads exist to SEED a jump, not to be rewarded for standing on."""
    for t in build_targets(4):
        if t.id.endswith("_launch"):
            assert not t.mandatory
            assert t.seedable


def test_fruit_platform_has_no_duplicate_milestone():
    """Fr2 lands where F1 sits; rewarding both double-counts one rung (the
    reason L3 omits A4)."""
    ts = {t.id: t for t in build_targets(4)}
    assert ts["F1"].mandatory
    assert not ts["Fr2"].mandatory


def test_every_mandatory_target_is_reachable_from_the_start():
    """An unreachable target contributes an infinite term to the potential, so
    the reward gives no gradient toward it — the failure mode jump_edges exist
    to prevent."""
    g = build_navigation_map(4)
    for t in build_targets(4):
        if not t.mandatory:
            continue
        assert t.node_ident in g.node_by_ident, f"{t.id}: no graph node"
        d = g.path_distance_from_pos(START_FLOOR, None, START_X, START_Y, t.node_ident)
        assert (
            d is not None and math.isfinite(d) and d < INF_SENTINEL
        ), f"{t.id} ({t.node_ident}) unreachable from the start: {d}"


def test_lower_levels_are_untouched():
    """L4 must not perturb the levels we already have champions for."""
    assert len(get_level_map(1).floor_top_y) == 5
    assert len(get_level_map(3).floor_top_y) == 16
    assert get_level_map(3).princess_centre_px == (16, 30)
    assert yeti.fruit_presence_addrs(3) == {1: yeti.FRUITS_ADDR}
    assert yeti.fruit_presence_addrs(2) == {1: 11950, 2: 11975}
