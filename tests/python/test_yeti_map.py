"""Tests for the hand-coded Yeti navigation graph."""

from __future__ import annotations

import pytest
from retro_ai.training.yeti_map import (
    LEVELS,
    agent_floor_from_pixel_y,
    build_navigation_map,
    get_level_map,
)


@pytest.fixture(scope="module")
def nav():
    return build_navigation_map()


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_fruit_and_princess_stand_on_their_floor(level):
    """A fruit's or the princess's y is the standing Y of its floor, like an anchor.

    L1 and L2 had no single convention (y = standing +0, +2, +6, +10 or +18), which
    drew their markers off the floor on the level maps. Only the drawing reads this y.
    """
    lvl = get_level_map(level)
    for fid, (_x, y) in lvl.fruit_centre_px.items():
        assert y == lvl.floor_top_y[lvl.fruit_floor[fid]], f"L{level} F{fid}"
    assert lvl.princess_centre_px[1] == lvl.floor_top_y[lvl.princess_floor]


# The princess's sprite in each level-start frame (pink and cyan pixels, measured
# 2026-10-29): px 306-317 on L1/L5 and L2, px 2-13 on L3 and L4. The agent touches her
# 16 px before the centres meet (champions' 300-episode results: L1 296, L2 296, L3 24,
# L4 24), so the touch point depends on the side the agent comes from.
@pytest.mark.parametrize("level,centre", [(1, 312), (2, 312), (3, 8), (4, 8)])
def test_princess_is_her_sprite_centre(level, centre):
    """One convention on every level: her sprite centre. L2 used the touch point (296)
    and L3 neither (16), which drew their princess boxes beside her on the maps."""
    assert get_level_map(level).princess_centre_px[0] == centre


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
    """Measured standing y (Y RAM byte, grounded) resolves to the correct floor."""
    assert agent_floor_from_pixel_y(182) == 1
    assert agent_floor_from_pixel_y(150) == 2
    assert agent_floor_from_pixel_y(118) == 3
    assert agent_floor_from_pixel_y(86) == 4
    assert agent_floor_from_pixel_y(54) == 5


def test_agent_floor_from_pixel_y_tolerance():
    """8px tolerance around each floor's standing y (floor 1 stands at 182)."""
    assert agent_floor_from_pixel_y(174) == 1
    assert agent_floor_from_pixel_y(190) == 1
    assert agent_floor_from_pixel_y(173) is None
    assert agent_floor_from_pixel_y(191) is None


def test_agent_floor_from_pixel_y_mid_air_returns_none():
    """A y clearly between floors resolves to None."""
    # Between floor 1 (y=182) and floor 2 (y=150): 168 is outside the 8px
    # tolerance of either, so None.
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


def test_l3_jump_waypoints_on_ascent():
    """v8: A1..A5 jump-edge landing waypoints exist on L3 (for curriculum
    seed/reach), placed on each arrival platform's route-facing edge, and match
    the reward graph's `_b` endpoint x. They are SEED-only (NOT reward targets),
    so reward_waypoints is unchanged. L1/L2 emit none (byte-identical)."""
    from retro_ai.games import yeti
    from retro_ai.training.yeti_map import get_level_map, jump_waypoints

    w3 = yeti.waypoints(3)
    # One waypoint per named landing floor (A1..A5), on the expected floor.
    expected_floor = {"A1": 11, "A2": 12, "A3": 13, "A4": 14, "A5": 15}
    for name, floor in expected_floor.items():
        assert name in w3, f"missing jump waypoint {name}"
        x_ram, y_px, f = w3[name]
        assert f == floor
        assert y_px == get_level_map(3).floor_top_y[floor]
    # Landing x matches the reward graph's `_b` (arrival) endpoint exactly, in
    # RAM units — so the seeder and the shaping agree on the jump-off geometry.
    from retro_ai.training.yeti_map import build_fixed_nodes

    # A measured override (A3, see test_l3_a3_anchor_is_where_the_crossing_lands) is
    # deliberately off the node: the node stays on the edge, the anchor moves.
    by_ident = {nd.ident: nd for nd in build_fixed_nodes(get_level_map(3))}
    overridden = set(get_level_map(3).jump_waypoint_pos or {})
    for fa, fb in [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]:
        name = {11: "A1", 12: "A2", 13: "A3", 14: "A4", 15: "A5"}[fb]
        if name in overridden:
            continue
        assert w3[name][0] == (by_ident[f"J{fa}_{fb}_b"].x - 8) // 4

    # LAUNCH pads: one per named edge, on the DEPARTURE (lower) platform. The
    # critical one is A1_launch = SN3's left edge (floor 10) — capturable, so it
    # can seed the ascent. Each launch sits on the platform below its arrival.
    launch_floor = {
        "A1_launch": 10,
        "A2_launch": 11,
        "A3_launch": 12,
        "A4_launch": 13,
        "A5_launch": 14,
    }
    for name, floor in launch_floor.items():
        assert name in w3, f"missing launch waypoint {name}"
        assert w3[name][2] == floor
        assert w3[name][1] == get_level_map(3).floor_top_y[floor]
    # A1_launch is on the SN3 LEFT edge (smaller x than the Lsc4_top ladder WP
    # on the right), i.e. the safe jump-off side.
    assert w3["A1_launch"][0] < w3["Lsc4_top"][0]

    # SEED-only: A-names / launches are NOT among the reward waypoint targets.
    reward_targets = {
        ident for grp in (get_level_map(3).reward_waypoints or []) for ident in grp
    }
    assert not ((set(expected_floor) | set(launch_floor)) & reward_targets)

    # L1/L2 have no jump_waypoint_names -> no jump waypoints (byte-identical).
    assert jump_waypoints(get_level_map(1)) == {}
    assert jump_waypoints(get_level_map(2)) == {}
    assert not any(k.startswith("A") for k in yeti.waypoints(1))
    assert not any(k.startswith("A") for k in yeti.waypoints(2))


def test_l3_a3_anchor_is_where_the_crossing_lands():
    """A3's measured anchor (LEVEL3.jump_waypoint_pos). The A2 -> A3 crossing of
    l3_v15, identical in 40/40 princess episodes: floor 12's edge (x 33, y 70), one
    airborne frame (x 32, y 66), landing (x 30, y 62), gone two frames later. The
    derived anchor, x 32, is covered by none of those frames under sprite reach; x 31 is
    covered by the landing and by standing on the one safe spot, x 32."""
    from retro_ai.games import yeti
    from retro_ai.training.targets import (
        build_targets,
        progress_ladder,
        sprite_overlaps,
    )

    crossing = [(33, 70), (32, 66), (30, 62)]
    assert not any(sprite_overlaps((32, 62), x, y) for x, y in crossing)
    assert yeti.waypoints(3)["A3"] == (31, 62, 13)
    a3 = (31, 62)
    assert sprite_overlaps(a3, 30, 62), "the landing the policy makes"
    assert sprite_overlaps(a3, 32, 62), "the safe spot (x 30 / 31 die in place)"
    # Standing on floor 12's edge, before the jump, must not read as arriving.
    assert not sprite_overlaps(a3, 33, 70) and not sprite_overlaps(a3, 34, 70)
    # The reward marks the A3 milestone at the moved anchor, still as the mandatory
    # J12_13_b step, and the progress ladder is unchanged.
    t = {x.id: x for x in build_targets(3)}["A3"]
    assert (t.pos, t.node_ident, t.mandatory) == ((31, 62), "J12_13_b", True)
    assert progress_ladder(3)[1] == 12


def test_l3_lesc_top_is_where_the_ride_starts():
    """Lesc_top's measured anchor (LEVEL3.ladder_waypoint_pos). l3_v15 boarding the
    escalator, seed 0 (all 20 measured episodes board the same way): steps 83-90 the
    jump off the goat platform, 91 on the ride (pose 13) at x_ram 32. The derived
    anchor (33, 94) is covered on 2 frames, the jump's last, and never while riding;
    (32, 98) on the jump's last 3 and the first 2 ride frames."""
    from retro_ai.games import yeti
    from retro_ai.training.targets import build_targets, sprite_overlaps

    boarding = {
        83: (27, 90),
        84: (28, 86),
        85: (29, 84),
        86: (30, 84),
        87: (30, 84),
        88: (31, 84),
        89: (32, 86),
        90: (32, 86),
        91: (32, 98),
        92: (32, 98),
        93: (32, 102),
        94: (32, 102),
        95: (32, 102),
        96: (32, 106),
    }

    def hits(anchor):
        return [t for t, (x, y) in boarding.items() if sprite_overlaps(anchor, x, y)]

    assert hits((33, 94)) == [89, 90]
    assert yeti.waypoints(3)["Lesc_top"] == (32, 98, 4)
    assert hits((32, 98)) == [88, 89, 90, 91, 92]
    new = (32, 98)
    # Not while still walking on the goat platform (floor 4, x_ram 18-27, y 94).
    assert not any(sprite_overlaps(new, x, 94) for x in range(18, 28))
    # Still the same target; only its position moved. The graph keeps the ladder.
    t = {x.id: x for x in build_targets(3)}["Lesc_top"]
    assert (t.pos, t.node_ident) == ((32, 98), "Lesc_top")


def test_ladder_waypoint_override_rejects_unknown_names(monkeypatch):
    import dataclasses

    from retro_ai.games import yeti
    from retro_ai.training import yeti_map

    bad = dataclasses.replace(yeti_map.LEVEL3, ladder_waypoint_pos={"Nope_top": (1, 2)})
    monkeypatch.setitem(yeti_map.LEVELS, 3, bad)
    with pytest.raises(KeyError, match="Nope_top"):
        yeti.waypoints(3)


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


def test_jump_node_inset_stays_off():
    """Jump-graph nodes sit ON the tile edge. `edge_inset` is 0 everywhere, and stays 0.

    THIS IS A NEGATIVE RESULT, PINNED. Setting L4's `edge_inset` to 4 looks obviously
    right -- the node is what the PBRS potential aims at, and the tile edge is a pixel
    the agent falls off -- and it is wrong. The node marks where the agent JUMPS FROM,
    and a jump departs the edge in motion. You cannot stand on the edge; you must leave
    from it.

    Measured on every L4 jump edge at inset 4: the last step onto the departure edge
    went from +0.04 to -0.04, all 12 flipped, so the shaping punished the departure of
    every jump on the level -- including rope 1, which the agent crosses ~53% of the
    time. Run v25 against v24 (mean `Low2_launch` rate 0.195 vs 0.349) was first read as
    confirming it, but the v28a/b/c replicates of one config span 0.149-0.239, so that
    comparison shows nothing. The sign flip is the reason.

    `test_departing_a_jump_edge_pays` pins the mechanism. This pins the setting.
    """
    from retro_ai.training.yeti_map import build_navigation_map, get_level_map

    for level in (3, 4):
        lvl = get_level_map(level)
        assert lvl.edge_inset == 0, (
            f"L{level} edge_inset is {lvl.edge_inset}; 4 was measured harmful "
            f"(see this test's docstring and LEVEL4's edge_inset comment)"
        )
        pf = {p.floor: p for p in lvl.platforms}
        for n in build_navigation_map(level).nodes:
            if n.kind != "jump":
                continue
            p = pf[n.floor]
            assert n.x in (p.x_min, p.x_max) or p.x_min < n.x < p.x_max, (
                f"{n.ident} at px {n.x} is outside floor {n.floor} "
                f"[{p.x_min}, {p.x_max}]"
            )


def test_departing_a_jump_edge_pays():
    """Walking ONTO a jump's departure edge must shorten the path, at every L4 jump.

    This is the property `edge_inset=4` broke. The agent has to reach the edge to jump,
    so the shaping must pay for getting there; insetting the node makes the final step
    onto the edge cost the same amount it used to pay. Fails if the inset returns.
    """
    import dataclasses

    from retro_ai.training.yeti_map import LEVELS, build_navigation_map, get_level_map

    lvl = get_level_map(4)
    pf = {p.floor: p for p in lvl.platforms}

    def step_onto_edge(nav, fa, fb):
        """Distance saved by the last step onto f{fa}'s edge facing f{fb}."""
        pa, pb = pf[fa], pf[fb]
        cb = (pb.x_min + pb.x_max) / 2
        edge = pa.x_min if cb <= pa.x_min else (pa.x_max if cb >= pa.x_max else None)
        if edge is None:
            return None
        inner = edge + 4 if edge == pa.x_min else edge - 4
        tgt = f"J{fa}_{fb}_b"
        return nav.path_distance_from_agent(
            fa, inner, tgt
        ) - nav.path_distance_from_agent(fa, edge, tgt)

    nav = build_navigation_map(4)
    edges = [(a, b) for a, b in lvl.jump_edges if step_onto_edge(nav, a, b) is not None]
    assert len(edges) == 12
    for a, b in edges:
        assert step_onto_edge(nav, a, b) > 0, (
            f"stepping onto f{a}'s departure edge for the {a}->{b} jump does not pay; "
            f"the agent must reach the edge to jump from it"
        )

    # Show the test discriminates: at inset 4 every one of the 12 flips sign.
    key = max(LEVELS) + 1
    LEVELS[key] = dataclasses.replace(lvl, edge_inset=4)
    try:
        bad = build_navigation_map(key)
        flipped = sum(1 for a, b in edges if step_onto_edge(bad, a, b) < 0)
        assert flipped == 12, f"expected all 12 to flip at inset 4, got {flipped}"
    finally:
        del LEVELS[key]
