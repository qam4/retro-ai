"""Static map model for Yeti (Thomson MO5 / Crayon).

Encodes floors, ladders, fruits, and the princess in pixel coordinates
so we can compute path-distance reward shaping without the agent
having to rediscover navigation from scratch.

Agent-to-pixel mapping (verified via debug/cp0_reference_grid2.png):
  pixel_x = ram_x * 4
  pixel_y = ram_y
Agent sprite is 16x16; centre is (ram_x * 4 + 8, ram_y + 8).

Multi-level support
-------------------
Level geometry lives in a :class:`LevelMap` per level, selected by
``build_navigation_map(level)`` / ``agent_floor_from_pixel_y(y, level)``.
Level 1 is the original hand-mapped climb-up layout (5 floors, 4 fruits,
princess top). Level 2 is the descending layout read straight from RAM
(see experiments/003-yeti/ram_map_re.md): 6 floors, 10 ladders, 2 fruits
on floor 5, princess bottom-right. The module-level FLOOR_TOP_Y / etc.
constants remain bound to level 1 for backward compatibility.

The graph structure is identical across levels: fruit/ladder/princess
nodes, horizontal same-floor edges (cost |dx|), vertical ladder edges
(cost floor_height * floors-spanned). Gaps in a floor are NOT modelled
as separate nodes — same-floor distance is still |dx|, so the shaping
pulls the agent across a gap; learning to *jump* it (vs walk off and
die) is left to the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Per-level geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Platform:
    """A walkable segment: the floor it realises, its standing pixel-y, and
    its horizontal pixel extent [x_min, x_max].

    A platform is a LOGICAL walkable line and MAY contain jumpable gaps —
    it is NOT a contiguous tile run. The whole extent uses |dx| distance, so
    the shaping still pulls the agent across an internal gap (the behaviour
    that taught L2 to jump gaps). Platforms exist to disambiguate levels
    where several distinct structures share a pixel-y (L3): an (x, y) is
    attributed to the platform whose extent contains x, so a point on a tall
    ladder is NOT mistaken for a same-height platform elsewhere on screen.
    """

    floor: int
    y: int
    x_min: int
    x_max: int


@dataclass(frozen=True)
class LevelMap:
    """All static geometry for one level (pixel coordinates)."""

    floor_top_y: Dict[int, int]  # floor -> agent standing pixel-y
    floor_height: int  # pixels between adjacent floors
    fruit_centre_px: Dict[int, Tuple[int, int]]
    fruit_floor: Dict[int, int]
    # (name, top_floor, bot_floor, centre_x). ALWAYS ordered top-first, where
    # "top" = higher on screen = smaller floor_top_y. This positional rule is
    # shared with yeti.waypoints() so ladder node idents ("<name>_top"/"_bot")
    # agree between the reward navigation graph and the waypoint seeder.
    ladders: List[Tuple[str, int, int, int]]
    princess_centre_px: Tuple[int, int]
    princess_floor: int
    # Optional per-ladder waypoint placement: name -> "top" | "bot" | "both".
    # Default (None / unspecified ladder) = "both" ends, matching L1/L2. Used
    # to expose only the end the agent ARRIVES at on a level whose route
    # direction is known (L3), so seeds start right before each hard segment
    # instead of re-climbing.
    waypoint_ends: Optional[Dict[str, str]] = None
    # Optional MANDATORY-waypoint reward targets: an (unordered) list of
    # OR-groups, each a list of nav-node idents (e.g. ["Ldown_bot"] or
    # ["Lgoat_a_top", "Lgoat_b_top"] for a branch reached either way). The
    # path-progress reward sums distance to these (min over an OR-group's
    # members), exactly like it sums distance to remaining fruits — so they
    # are mandatory but unordered. Reaching any member of a group (within
    # tol) marks that group done; unreachable groups drop out of the sum.
    # None (L1/L2) => no WP reward targets => reward unchanged.
    reward_waypoints: Optional[List[List[str]]] = None
    # Optional PHASE GATE for reward_waypoints: idents whose group only enters
    # the potential once EVERY fruit is collected. Any member ident names its
    # group; unlisted groups are always active (L1/L2/L3 behaviour unchanged).
    #
    # Why this exists. The base potential is already phase-aware: it sums
    # distance to REMAINING FRUITS, and only switches to the princess once none
    # are left. The waypoint sum never inherited that rule, and on L4 it cost a
    # whole run. L4's fruit is at the opposite end of the map from the princess,
    # so with all 12 groups active from step 0 the sum is minimised by walking
    # AWAY from the fruit: 9 of the terms shrink going left while only 3 grow.
    # Measured on yeti_curriculum_l4_v1: Lfruit_top reached 0.1%, Lascent_top
    # 99%, and 0 fruits collected in 3000 reset-origin episodes — the level was
    # unwinnable by construction.
    #
    # A sum of distances cannot express "do A, then B" when A and B lie in
    # opposite directions; the terms simply fight. L1-L3 never exposed that
    # because their targets are roughly co-directional, an assumption we had
    # relied on without stating it. Trimming the mandatory set does NOT fix it
    # (with only F1 + princess the potential is flat toward the fruit, then
    # adverse) — the missing notion is ORDER.
    waypoints_after_fruit: Optional[List[str]] = None
    # Optional JUMP edges: (floor_a, floor_b) platform pairs the agent
    # traverses by JUMPING (not walking/climbing) — e.g. START->STEP->2LAD and
    # the SN3->A1->..->A5 ascent on L3. Without them those platforms are graph-
    # disconnected (INF), so the path-progress reward gives no gradient across
    # them (they'd be learned by raw exploration only). Modelled as graph edges
    # (like the escalator's Lesc ladder), they reconnect the route so the
    # reward shapes across the jump and WP capture can seed the far side. Each
    # endpoint node is placed at the departing platform's EDGE nearest the
    # landing platform (so the gradient pulls toward the jump-off, not the
    # platform centre); cost = |Δx| + |Δy|. Verified topology mirrors
    # scripts/mo5/yeti/annotate_level3_map.py (JUMPS + _edge_pt). None (L1/L2)
    # => no jump edges => graph/reward unchanged, byte-identical.
    jump_edges: Optional[List[Tuple[int, int]]] = None
    # Optional explicit walkable-segment extents. When set (L3), the agent's
    # floor is resolved X-AWARELY: (x, y) resolves to a floor only if x lies
    # within that floor's platform extent (and y within tol). This stops the
    # y-only lookup from attributing a point on a tall ladder to a same-height
    # platform elsewhere on screen (the L3 goat-ladder bug). When None (L1/L2)
    # each floor is implicitly one full-width platform, so resolution is the
    # pure y-only rule and behaviour is byte-identical.
    platforms: Optional[List[Platform]] = None
    # Optional JUMP-EDGE waypoints: landing_floor -> waypoint name. For each
    # jump_edge, the ARRIVAL platform (the higher route-order / larger floor id
    # endpoint) can carry a curriculum waypoint at its landing edge, so the
    # otherwise-unseedable jump ascent (L3 A1..A5) becomes capture/seed/reach-
    # tracked — WITHOUT adding a reward target (these are NOT reward_waypoints,
    # so shaping stays byte-identical; the jump_edges already give the
    # gradient). OPT-IN per landing floor: only floors listed here get a WP, so
    # the bottom hops (START->STEP->2LAD) stay WP-free unless named. None
    # (L1/L2, no jump_edges) => no jump waypoints.
    jump_waypoint_names: Optional[Dict[int, str]] = None
    # Optional MEASURED anchor overrides for jump waypoints, ``name -> (x_ram, y_px)``.
    #
    # ``jump_waypoints`` derives every anchor from a platform EDGE, which is a
    # geometric guess: the agent takes off and lands 3-4 units INSIDE the platform, so
    # an edge anchor can sit somewhere it never stands. Measured consequence on L4
    # ``Fr1`` (edge anchor x_ram 60 on floor 3): the agent occupies only 64..68 there,
    # so the REWARD's tol-2 box never fired and that milestone was never marked --
    # leaving a permanent distance term in the potential at ~12x the base route
    # gradient. See experiments/003-yeti/level4_notes.md.
    #
    # An override replaces the derived anchor for detection ONLY. It does NOT move the
    # navigation-graph node, so path-distance shaping geometry is untouched.
    # None (L1/L2/L3) => derived anchors, unchanged.
    jump_waypoint_pos: Optional[Dict[str, Tuple[int, int]]] = None
    # Optional list of jump-waypoint names NOT to emit. Use for launch pads that are
    # redundant because the same platform already carries a waypoint that marks
    # correctly -- e.g. on L4 floor 11, `Low1_launch` and `Lclimb3_top` are the same
    # platform, and the ladder anchor is exact while the launch pad's wide box also
    # caught a stalled climb 24 px away and reported it as an arrival.
    jump_waypoint_skip: Optional[List[str]] = None
    # Px to pull each JUMP-GRAPH NODE inside its platform's tile edge, so the PBRS
    # potential aims at a position the agent can stand on. See EDGE_INSET_PX for the
    # measurement and `_jump_graph` for why only the nodes move, never the curriculum
    # anchors.
    #
    # PER-LEVEL AND DEFAULT 0 ON PURPOSE. The mechanic is universal -- 8x8 tiles, 4-px
    # steps -- so the right value is EDGE_INSET_PX everywhere. The opt-in is about blast
    # radius, not about geometry: this moves a reward target, and L3 is the only level
    # this project finishes reliably (14 of 16 runs reach the princess). Perturbing L3's
    # shaping to fix L4's rope-2 pad would put the one working level at risk for no
    # measured benefit, and it would break the seeder/shaping agreement that
    # test_l3_jump_waypoints_on_ascent pins there. Turn it on for L3 only with its own
    # control arm, after L4 has read out.
    edge_inset: int = 0
    # Optional DISPLAY-ONLY route order: route-point ids bottom-of-route first.
    # Carries NO semantics — the reward still sums over ALL not-yet-reached
    # targets, unordered (settled decision #5 in curriculum_cp_wp_model.md). It began
    # as display order for the route TABLE in the training log, instead of it being
    # sorted three different ways by three different metrics. Levels that branch (L2 has
    # two ladders per floor) can list any one representative order, or omit it —
    # rendering falls back to a stable arbitrary order.
    #
    # NO LONGER DISPLAY-ONLY, AND THIS COMMENT USED TO SAY IT WAS. Two consumers now
    # depend on the ORDER and on MEMBERSHIP:
    #
    #   * `train_checkpoint_curriculum._wp_predecessor` / `_wp_eligible` — the start
    #     gate's predecessor rule. A pool absent from this list has no predecessor, and
    #     the `prev is None` branch reads that as "not on the route" and refuses it.
    #   * `keep_best_sweep._frontier` and its scoring — the frontier is the deepest
    #     listed point still reached, and `(index + 1) / len(route_order)` is the depth
    #     term in the champion score. Inserting or removing an entry therefore rescales
    #     every stored score and breaks comparability across runs.
    #
    # The stale "nothing gates on this" is what made it look safe to add a start pool
    # whose id is absent here. Measured cost on L4: the `F1` fruit pool held 100 seeds,
    # logged 10439 captures and was sampled ZERO times in a 6M run, against 925 starts
    # for the same seeds a run earlier. Fixed at the reach-table call site, NOT by
    # editing this list, precisely to keep the evaluator's scores comparable — see
    # tests/python/test_start_pool_reach_universe.py.
    route_order: Optional[List[str]] = None


# One 4-px step: the distance from a platform's tile edge to the last sprite centre
# that can actually STAND there. Measured 2026-09-24 and uniform, which it has to be --
# tiles are 8x8 and the agent moves in 4-px units, so the collision mechanic cannot vary
# by platform. Evidence, on L4:
#
#   * The tilemap (40x25 tile-ids at 0x2C27, floor ids 5-8) confirms all 24 declared
#     `Platform` extents EXACTLY, once ladder-through-floor tiles (ids 3/4) are counted
#     as floor. Nothing is mis-transcribed, so the tile edge is the real edge.
#   * Reload-and-hold at the tile edge FELL on 6 of 6 floors whose right edge a walk
#     reached, and 4 of 5 on the left, with the limit one 4-px step inside each time.
#   * The one apparent exception, floor 10's px 248 reading standable while floor 11's
#     px 248 fell, was a MEASUREMENT error: both rows carry tile id 7 (LEFT-end) at
#     column 31, and a direct NOOP trace at px 248 on floor 11 is pose 11 (falling) from
#     frame 0, dropping 4 px/frame straight THROUGH floor 10's standing y without
#     landing. So px 248 holds on neither, and the mechanic is uniform after all.
#
# Earlier per-floor insets of +8/+16/+20/+24 in `standable_span`'s docstring and in
# l4_platform_audit output are walk artifacts -- a walk that stops short reports where
# it gave up, not where the platform ends. Do NOT reintroduce a per-floor table: read
# the extent from the TILEMAP and apply this constant.
EDGE_INSET_PX = 4


# Level 1 — original climb-up layout (floor 1 = bottom/spawn, 5 = princess).
LEVEL1 = LevelMap(
    floor_top_y={1: 184, 2: 152, 3: 120, 4: 88, 5: 56},
    floor_height=32,
    fruit_centre_px={1: (184, 184), 2: (80, 150), 3: (144, 120), 4: (272, 88)},
    fruit_floor={1: 1, 2: 2, 3: 3, 4: 4},
    # Ordered (name, TOP_floor, BOT_floor, x): top = higher on screen (smaller
    # y). L1 climbs up, so the top floor has the larger number here.
    ladders=[
        ("L12a", 2, 1, 120),
        ("L12b", 2, 1, 280),
        ("L23", 3, 2, 240),
        ("L34", 4, 3, 176),
        ("L45", 5, 4, 208),
    ],
    princess_centre_px=(312, 60),
    princess_floor=5,
)

# Level 2 — descending layout, read from RAM (ram_map_re.md). Floors numbered
# top->bottom: F1 (start) .. F6 (bottom), 24 px apart. floor_top_y is the
# agent's STANDING RAM-y (sprite upper-left), which is ~18 px ABOVE the floor
# *tile* row (the 16px sprite stands on top of the tile): F1 tile y48 -> agent
# stands at y30 (verified: the agent rests at y=30 at spawn). So standing-y =
# tile_y - 18 = {48,72,96,120,144,168} - 18. Both fruits on floor 5; ladder
# centre_x = extracted UL x_px + 8. Princess on floor 6 at her RAM x (288).
LEVEL2 = LevelMap(
    floor_top_y={1: 30, 2: 54, 3: 78, 4: 102, 5: 126, 6: 150},
    floor_height=24,
    fruit_centre_px={1: (64, 136), 2: (264, 136)},
    fruit_floor={1: 5, 2: 5},
    ladders=[
        ("L12a", 1, 2, 80),
        ("L12b", 1, 2, 304),
        ("L23a", 2, 3, 16),
        ("L23b", 2, 3, 192),
        ("L34", 3, 4, 136),
        ("L45a", 4, 5, 40),
        ("L45b", 4, 5, 296),
        ("L56", 5, 6, 120),
    ],
    princess_centre_px=(288, 168),
    princess_floor=6,
)

# Level 3 — fragmented multi-platform layout, REBUILT from the raw tilemap
# (output/mo5/yeti/level3/level3_map.json + per-cell RAM read) and verified
# tile-by-tile with the user against a rendered frame (see
# scripts/mo5/yeti/annotate_level3_map.py). 1 fruit, princess top-left, player
# starts bottom-left.
#
# KEY CALIBRATION: agent standing-y = tile_row*8 - 18 (the 16px sprite stands
# ~18px above the tile row; same offset as L2). The earlier L3 map omitted the
# -18 and merged gappy tile rows, so its floors were ~18px off and its "goat
# platform" was mid-ladder. Confirmed against ground truth: start=y166 (row23),
# 2-ladder platform=y150 (row21), goat platform=y94 (row14, top of the goat
# ladders — NOT the higher r10 tiles), princess=y30.
#
# MODEL: each PLATFORM is its own floor-id (so the same-floor graph logic ==
# same-PLATFORM walkability). Several platforms share a standing-y (e.g. STEP/
# ELAND/BR at y158) — fine, because the agent's floor is resolved X-AWARELY
# (agent_floor_from_pixel_xy + the `platforms` extents), never by y alone.
# Ladders are graph edges (cost = |Δy|). The ESCALATOR (moving wall) and the
# jump-only transitions (START->STEP->2LAD, and the ascending climb
# SN3->A1->..->A5) are NOT edges -> INF in the graph -> sparse/"learned by
# exploration", exactly as designed. COMPRESSOR / SNOWBALLS are moving sprites,
# curriculum-carried, not modelled.
#
# floor-id : platform (standing_y, ram_x range) — role
#   1 START (166,  0-9)   start ledge (raised, bottom-left)
#   2 STEP  (158, 12-15)  step up from start
#   3 2LAD  (150, 18-27)  2-ladder platform (goat-ladder bottoms)
#   4 GOAT  ( 94, 18-27)  goat platform (goat-ladder tops)  -> escalator
#   5 ELAND (158, 40-47)  escalator landing
#   6 BOTTOM(182,  0-79)  bottom floor (full width, screen bottom)
#   7 BR    (158, 56-79)  bottom-right
#   8 SN1   (134, 52-77)  snowball climb 1
#   9 SN2   (110, 54-79)  snowball climb 2
#  10 SN3   ( 86, 50-77)  snowball climb 3 (top)
#  11 A1    ( 78, 42-45)  ascending 1
#  12 A2    ( 70, 36-39)  ascending 2
#  13 A3    ( 62, 24-33)  ascending 3 (compressor)
#  14 A4    ( 62, 14-19)  ascending 4 (FRUIT)
#  15 A5    ( 54,  0-11)  ascending 5 -> Lprincess
#  16 PRIN  ( 30,  0-17)  princess platform
LEVEL3 = LevelMap(
    floor_top_y={
        1: 166,
        2: 158,
        3: 150,
        4: 94,
        5: 158,
        6: 182,
        7: 158,
        8: 134,
        9: 110,
        10: 86,
        11: 78,
        12: 70,
        13: 62,
        14: 62,
        15: 54,
        16: 30,
    },
    floor_height=24,  # nominal; unused for L3 costs (ladder cost = |Δy|)
    fruit_centre_px={1: (64, 62)},  # on A4 (floor 14); x = ram14*4+8
    fruit_floor={1: 14},
    # (name, TOP_floor, BOT_floor, centre_x_px); top = smaller standing-y.
    # centre_x_px = ladder_ram*4 + 8 (agent standing x at the ladder).
    ladders=[
        ("Lgoat_a", 4, 3, 80),  # 2LAD <-> GOAT (left)
        ("Lgoat_b", 4, 3, 104),  # 2LAD <-> GOAT (right)
        # ESCALATOR modelled as a ladder: the descent is vertical (moving
        # platforms carry the agent from goat level y94 down to landing level
        # y158 at the fixed column ram~33, just left of the wall). Treating it
        # as a ladder gives the reward a FINITE goat->landing path (no INF
        # gap) -> a gradient off the goat platform toward the jump-off point,
        # and credit for landing across. The agent still learns the on/off
        # JUMP TIMING visually (like jumping an L2 gap); a mistimed jump falls
        # -> death gate -> no credit. Its top/bottom are capture+seed
        # waypoints. Connects GOAT (floor 4, y94) <-> ELAND (floor 5, y158).
        ("Lesc", 4, 5, 140),  # escalator descent (ram33)
        ("Ldown", 5, 6, 176),  # ELAND <-> BOTTOM (post-escalator descent)
        ("Lsc1", 7, 6, 248),  # BOTTOM <-> BR (snowball climb 1)
        ("Lsc2", 8, 7, 288),  # BR <-> SN1 (2)
        ("Lsc3", 9, 8, 240),  # SN1 <-> SN2 (3)
        ("Lsc4", 10, 9, 288),  # SN2 <-> SN3 (4)
        ("Lprincess", 16, 15, 32),  # A5 <-> PRIN
    ],
    princess_centre_px=(16, 30),
    princess_floor=16,
    # Jump-traversed platform links (mirrors annotate_level3_map.JUMPS): the
    # bottom START->STEP->2LAD hops and the SN3->A1->A2->A3->A4->A5 ascent.
    # These reconnect the route so path-progress shaping spans them (A4 = the
    # fruit, then Lprincess A5->PRIN). The escalator GOAT<->ELAND is already
    # the Lesc ladder edge; the compressor on A3 stays an unmodelled visual
    # hazard (like the snowballs).
    jump_edges=[(1, 2), (2, 3), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)],
    # Jump-edge waypoints on the A1..A5 ascent (landing floor -> name). Gives
    # the unseedable SN3->A1->..->A5 climb capture/seed/reach tracking (v8).
    # A3 (floor 13) carries the compressor; A4 (floor 14) the fruit. The bottom
    # hops (landing floors 2/3) are intentionally omitted (reliably reached
    # from reset already).
    jump_waypoint_names={11: "A1", 12: "A2", 13: "A3", 14: "A4", 15: "A5"},
    # DISPLAY-ONLY travel order (see LevelMap.route_order): start -> princess.
    route_order=[
        "Lgoat_a_top",
        "Lgoat_b_top",
        "Lesc_top",
        "Ldown_bot",
        "Lsc1_top",
        "Lsc2_top",
        "Lsc3_top",
        "Lsc4_top",
        "A1_launch",
        "A1",
        "A2_launch",
        "A2",
        "A3_launch",
        "A3",
        "A4_launch",
        "A4",
        "A5_launch",
        "A5",
        "Lprincess_top",
    ],
    waypoint_ends={
        "Lgoat_a": "top",
        "Lgoat_b": "top",
        # Escalator: capture only the BOARD point (Lesc_top, ram33/y94). With
        # pose 13 (the ride) now in the seeding allow-list, this captures the
        # on-escalator "just boarded" state as a reverse-curriculum seed. The
        # bottom (Lesc_bot) is NOT captured: it can only be reached by already
        # doing the exit jump (chicken-and-egg) and its grounded frame is a
        # 1-frame clip before free-fall (the old doomed-seed source).
        "Lesc": "top",
        "Ldown": "bot",
        "Lsc1": "top",
        "Lsc2": "top",
        "Lsc3": "top",
        "Lsc4": "top",
        "Lprincess": "top",
    },
    # Mandatory-waypoint reward targets (arrival ends; summed like fruits,
    # min over an OR-group). With the escalator modelled as a ladder the whole
    # route is graph-connected, so every target has a finite path and the
    # reward shapes the agent along it. The escalator top/bottom are
    # DELIBERATELY NOT reward targets (only seed waypoints, see waypoint_ends):
    # the ladder + the downstream Ldown_bot already give the directional pull
    # across the escalator, so we don't add an explicit "be exactly here" bonus
    # that could over-specify the jump-off. The frame-precise on/off timing is
    # never in the reward regardless -- the policy learns it from the pixels
    # (like dodging a snowball). Only INF gaps left: the final ascending jumps
    # (SN3 -> A1..A5), sparse by design.
    reward_waypoints=[
        ["Lgoat_a_top", "Lgoat_b_top"],  # GOAT (either ladder)
        ["Ldown_bot"],  # BOTTOM (post-escalator; reachable via the Lesc ladder)
        ["Lsc1_top"],  # BR
        ["Lsc2_top"],  # SN1
        ["Lsc3_top"],  # SN2
        ["Lsc4_top"],  # SN3
        # ASCENT milestones (v11). The lower chain had a mandatory waypoint per
        # rung and got mastered; the A1..A5 jump ascent had NONE (the reward
        # jumped straight from SN3 to Lprincess_top), leaving only the diffuse
        # fruit/princess distance — and it was never learned (measured: the
        # A1_launch->A1 jump is 10/10 EXECUTABLE by a scripted jump-left, yet
        # the policy only managed 5-20%). These are the jump-edge LANDING nodes
        # (same positions as the A1..A5 seed waypoints), so each completed jump
        # now banks an explicit milestone. A4 is deliberately OMITTED: the FRUIT
        # (F1) already sits on that platform and is a mandatory target, so a
        # milestone there would double-count the same rung.
        ["J10_11_b"],  # A1
        ["J11_12_b"],  # A2
        ["J12_13_b"],  # A3 (compressor)
        ["J14_15_b"],  # A5
        ["Lprincess_top"],  # PRIN
    ],
    # Walkable-segment x-extents (PIXELS: [ram_min*4, (ram_max+1)*4]) for the
    # x-aware floor resolver. One per floor-id above.
    platforms=[
        Platform(1, 166, 0, 40),
        Platform(2, 158, 48, 64),
        Platform(3, 150, 72, 112),
        Platform(4, 94, 72, 112),
        Platform(5, 158, 160, 192),
        Platform(6, 182, 0, 320),
        Platform(7, 158, 224, 320),
        Platform(8, 134, 208, 312),
        Platform(9, 110, 216, 320),
        Platform(10, 86, 200, 312),
        Platform(11, 78, 168, 184),
        Platform(12, 70, 144, 160),
        Platform(13, 62, 96, 136),
        Platform(14, 62, 56, 80),
        Platform(15, 54, 0, 48),
        Platform(16, 30, 0, 72),
    ],
)

# ---------------------------------------------------------------------------
# LEVEL 4 — kangaroos, two oscillating ropes, a spring, and a branching route
# ---------------------------------------------------------------------------
# Geometry extracted from the RAM tilemap (base 0x2C27) by
# scripts/mo5/yeti/map_builder.py and cross-checked against LEVEL3: 23 tile
# platforms plus an IMPLICIT GROUND at the screen bottom (row 25) that carries no
# tiles but is walkable and is where the agent starts.
#
# Standing y is NOT row*8. Y_ADDR is the SPRITE TOP, and the measured rule is
#   standing_y = tile_row * 8 - 18
# verified two ways: it maps all 16 of LEVEL3's hand-authored floor_top_y values
# onto integer tile rows, and climbing L4's first ladder lands on the row-22
# platform at exactly y=158 = 22*8-18.
#
# Floor ids are assigned in ROUTE ORDER (1 = ground, ascending toward the
# princess) so that every jump_edge has its LANDING as the larger id, which is
# what jump_waypoints() assumes.
#
# HAZARDS (measured with debug/l4_motion_map.py, agrees with play observation):
#   cols >= 29  kangaroos. They spawn on P5 (top right), jump left and fall
#               platform to platform, then drop the rest of the way; spawns are
#               periodic so several are on screen. The FRUIT sits inside this
#               zone, so the fruit trip is a forced timed crossing.
#   cols 0-15   snowballs, thrown by the yeti at cols 2-3 / rows 2-4 (the
#               strongest motion on screen). They run along P7 and P10 — the
#               final princess approach.
# Neither is modelled in the graph, exactly like L3's snowballs and compressor:
# they are visual hazards the policy must read from pixels.
#
# ROPES AND SPRING are plain jump_edges for now, with NO intermediate node. A
# rope carries the agent and needs a grab AND a release, and a spring is a
# bounce, so both probably need their own pose handling the way L3's escalator
# needed pose 13 — but we do not yet know those poses, and inventing nodes before
# measuring is how the escalator ended up looking free. Revisit when training
# demonstrably stalls on them.
LEVEL4 = LevelMap(
    floor_top_y={
        1: 182,  # GROUND (implicit, row 25) — start
        2: 158,  # P22, row 22
        3: 158,  # P23, row 22
        4: 150,  # P21, row 21 — FRUIT
        5: 150,  # P20, row 21
        6: 118,  # P16, row 17
        7: 118,  # P17, row 17
        8: 94,  # P13, row 14
        9: 94,  # P14, row 14
        10: 102,  # P15, row 15
        11: 78,  # P12, row 12
        12: 70,  # P11, row 11 (low route)
        13: 70,  # P10, row 11 — both routes converge here
        14: 54,  # P9,  row 9  (high route)
        15: 46,  # P8,  row 8
        16: 38,  # P6,  row 7
        17: 30,  # P4,  row 6
        18: 22,  # P2,  row 5
        19: 30,  # P3,  row 6
        20: 46,  # P7,  row 8  — PRINCESS
        21: 22,  # P1,  row 5  (off route, above the princess)
        22: 30,  # P5,  row 6  (off route, kangaroo spawn)
        23: 118,  # P18, row 17 (off route)
        24: 142,  # P19, row 20 (off route; the spring sits above it)
    },
    floor_height=24,  # nominal; ladder cost is |dy|
    # Sprite block at rows 19-20 cols 38-39 sits on P21 (row 21), so the target
    # is P21's standing y, matching how L3's fruit is placed on its platform.
    fruit_centre_px={1: (312, 150)},
    fruit_floor={1: 4},
    # (name, top_floor, bot_floor, centre_x_px), ALWAYS top-first (smaller y).
    # centre_x_px = col0*8 + 8 for a 2-column ladder; the agent must be at
    # x_ram = centre_x_px//4 - 2 to engage it (verified: ladder Lfruit only
    # climbs at x_ram exactly 50, not 49 or 51).
    ladders=[
        ("Lfruit", 2, 1, 208),  # GROUND <-> P22, the fruit trip
        ("Lascent", 5, 1, 72),  # GROUND <-> P20, start of the climb
        ("Lclimb1", 6, 5, 32),  # P20 <-> P16
        ("Lclimb2", 8, 7, 144),  # P17 <-> P13
        ("Lclimb3", 11, 10, 272),  # P15 <-> P12
        ("Lhi_up", 14, 11, 296),  # P12 <-> P9   (high route)
        ("Lhi_down", 19, 13, 104),  # P3  <-> P10  (high route descent)
        ("Lprincess", 20, 13, 40),  # P10 <-> P7   (final)
    ],
    princess_centre_px=(8, 46),
    princess_floor=20,
    # Every non-walk traversal. Landing is always the larger floor id.
    jump_edges=[
        (2, 3),  # P22 -> P23   jump  (tiles 131->132)
        (3, 4),  # P23 -> P21   jump  (tiles 135->123), fruit platform
        (6, 7),  # P16 -> P17   ROPE  (tiles 92->93)
        (8, 9),  # P13 -> P14   SPRING (tiles 74->75)
        (9, 10),  # P14 -> P15  jump  (tiles 78->79)
        (11, 12),  # P12 -> P11 jump  (tiles 60->59)   low route
        (12, 13),  # P11 -> P10 ROPE  (tiles 54->53)   low route
        (14, 15),  # P9 -> P8   jump  (tiles 32->31)   high route
        (15, 16),  # P8 -> P6   jump  (tiles 30->21)
        (16, 17),  # P6 -> P4   jump  (tiles 20->16)
        (17, 18),  # P4 -> P2   jump  (tiles 15->8)
        (18, 19),  # P2 -> P3   jump  (tiles 7->14)
    ],
    # Curriculum waypoints on jump landings (landing floor -> name). Each also
    # gets a "<name>_launch" pad on the departure platform.
    jump_waypoint_names={
        3: "Fr1",  # P23, fruit trip step 1
        4: "Fr2",  # P21, the fruit platform
        7: "Rope1",  # P17, across the first rope
        9: "Spring",  # P14, off the spring
        10: "Step",  # P15
        12: "Low1",  # P11, low route
        13: "Low2",  # P10, low route rope landing / convergence
        15: "Hi1",  # P8, high route
        16: "Hi2",  # P6
        17: "Hi3",  # P4
        18: "Hi4",  # P2
        19: "Hi5",  # P3
    },
    # DISPLAY-ONLY order (see route_order): out to the fruit, back, up the left
    # side, then either branch, then the princess.
    route_order=[
        "Lfruit_top",
        "Fr1_launch",
        "Fr1",
        "Fr2_launch",
        "Fr2",
        "Lfruit_bot",
        "Lascent_top",
        "Lclimb1_top",
        "Rope1_launch",
        "Rope1",
        "Lclimb2_top",
        "Spring",
        "Step",
        "Lclimb3_top",
        "Low1",
        # `Low2_launch` was here until 2026-10-01, when it joined jump_waypoint_skip.
        # It MUST go when the waypoint goes: with `gate_waypoints_by_predecessor`, a
        # listed point that is never emitted gets no `wp_reach_ema` row, reads 0.0, and
        # so refuses its SUCCESSOR as a start state forever. Leaving it would have made
        # `Low2` -- the rope-2 landing, the pool this change exists to fill --
        # permanently unseedable, the `F1` defect over again. `Low2`'s predecessor is
        # now `Low1`, on the same platform, which is emitted and healthy.
        "Low2",
        "Lhi_up_top",
        "Hi1_launch",
        "Hi1",
        "Hi2_launch",
        "Hi2",
        "Hi3_launch",
        "Hi3",
        "Hi4_launch",
        "Hi4",
        "Hi5_launch",
        "Hi5",
        "Lhi_down_bot",
        "Lprincess_top",
    ],
    # Arrival ends only, except Lfruit which is genuinely used both ways (up on
    # the way out, down on the way back).
    waypoint_ends={
        "Lfruit": "both",
        "Lascent": "top",
        "Lclimb1": "top",
        "Lclimb2": "top",
        "Lclimb3": "top",
        "Lhi_up": "top",
        "Lhi_down": "bot",
        "Lprincess": "top",
    },
    # MANDATORY reward targets: the FORCED route only. Neither branch is
    # required, so no low- or high-route landing appears here — they are
    # seedable and tracked waypoints without a reward term. This is the first
    # level where that distinction is real, and putting reward on a path the
    # agent will not take is what made v14 expensive.
    #
    # Fr2 (the fruit platform) is deliberately omitted: F1 sits on it and is
    # already mandatory as a fruit, so a milestone there would double-count the
    # same rung — the same reason L3 omits A4.
    reward_waypoints=[
        ["Lfruit_top"],  # P22
        ["J2_3_b"],  # P23
        ["Lascent_top"],  # P20
        ["Lclimb1_top"],  # P16
        ["J6_7_b"],  # P17, across the rope
        ["Lclimb2_top"],  # P13
        ["J8_9_b"],  # P14, off the spring
        ["J9_10_b"],  # P15
        ["Lclimb3_top"],  # P12
        # P10 is forced but reachable EITHER way: low route lands via the second
        # rope, high route arrives down Lhi_down. An OR-group is exactly the
        # construct for that (as L3 uses for the two goat ladders).
        #
        # KNOWN WART, shared with L3: build_targets flags BOTH members mandatory,
        # and rung_of counts mandatory IDS rather than satisfied GROUPS, so with
        # n_rungs=13 the top rung needs both members and an episode reaches only
        # one. Effect is cosmetic — one pool that never fills and one reach entry
        # pinned at 0, while the princess sentinel (n_rungs+1) still fires — but
        # the honest fix is for rung_of to count groups. Tracked as a follow-up
        # with the other `mandatory`-overloading issues.
        ["J12_13_b", "Lhi_down_bot"],
        ["Lprincess_top"],  # P7
    ],
    # Everything from the ascent onward waits for the fruit. Only Lfruit_top and
    # J2_3_b (P22, P23) stay active from step 0, because they ARE the way to the
    # fruit. Without this the run is unwinnable — see waypoints_after_fruit.
    waypoints_after_fruit=[
        "Lascent_top",
        "Lclimb1_top",
        "J6_7_b",
        "Lclimb2_top",
        "J8_9_b",
        "J9_10_b",
        "Lclimb3_top",
        "J12_13_b",
        "Lhi_down_bot",
        "Lprincess_top",
    ],
    # Walkable extents in PIXELS, [col0*8, (col1+1)*8]. One per floor id above.
    platforms=[
        Platform(1, 182, 0, 320),  # GROUND, full width
        Platform(2, 158, 184, 232),  # P22
        Platform(3, 158, 248, 280),  # P23
        Platform(4, 150, 296, 320),  # P21, fruit
        Platform(5, 150, 8, 120),  # P20
        Platform(6, 118, 0, 56),  # P16
        Platform(7, 118, 104, 168),  # P17
        Platform(8, 94, 120, 168),  # P13
        Platform(9, 94, 200, 232),  # P14
        Platform(10, 102, 248, 304),  # P15
        Platform(11, 78, 248, 320),  # P12
        Platform(12, 70, 184, 232),  # P11
        Platform(13, 70, 0, 128),  # P10
        Platform(14, 54, 272, 320),  # P9
        Platform(15, 46, 240, 256),  # P8
        Platform(16, 38, 208, 224),  # P6
        Platform(17, 30, 176, 192),  # P4
        Platform(18, 22, 144, 160),  # P2
        Platform(19, 30, 80, 128),  # P3
        Platform(20, 46, 0, 64),  # P7, princess
        Platform(21, 22, 0, 48),  # P1
        Platform(22, 30, 296, 320),  # P5, kangaroo spawn
        Platform(23, 118, 200, 232),  # P18
        Platform(24, 142, 168, 200),  # P19, spring above
    ],
    # MEASURED anchors, replacing edge-derived guesses that never marked.
    # Both sit on their floor's standing y and on positions the agent demonstrably
    # occupies (debug/yeti_validate_targets.py --level 4 --measure).
    #   Fr1   edge 60 -> 64. Floor 3's standing run is 60..68 and the agent occupies
    #         only 64..68; 64 was the landing in all 15 observed 2->3 crossings.
    #   Rope1 edge 24 -> 27. Floor 7's standing y is 118; (27,118) is the rope landing,
    #         visually confirmed. The alternative the tool first proposed, (34,110),
    #         is pose 8 -- the agent CLIMBING the ladder off that platform, i.e. a
    #         different event -- so it was rejected.
    #   Low2_launch  edge 44 -> 45. px 184 -> 188. APPLIED (see the entry below). Floor
    #         12's tile edge is 184 but the agent CANNOT stand there: px 184 reads as
    #         grounded (y still 70, walk pose) and then falls on the next step. Measured
    #         left limit is 188 = x_min + 4 (debug/l4_edge_limit.py, which confirms a
    #         stance by reloading it and holding NOOP). This is why all 100 Low2_launch
    #         seeds were doomed -- the capture box was centred one step past the edge,
    #         so
    #         the pool taught falling instead of the rope-2 crossing, and the agent
    #         never
    #         attempted the jump.
    #         NOTE this paragraph asserted the fix in the past tense for weeks while the
    #         code did NOT contain it: e667206 applied it and was reverted wholesale
    #         over
    #         an unrelated `Rope1` regression. If you change an anchor, change this
    #         comment and the dict in the same edit.
    #   Low2  edge 30 -> 29. px 128 -> 124. NOT APPLIED -- floor 13 is reached ~24
    #   times in
    #         the project's history so there is no seed evidence to check it against.
    #         Same
    #         defect mirrored in principle: floor 13's tile edge is 128 and px 128
    #         falls on
    #         NOOP; the last standable centre is 124.
    # ANCHORS COME FROM A MEASURED GROUNDED CENSUS, NOT FROM PLATFORM GEOMETRY.
    #
    # A geometric derivation was tried here and measured harmful. Both the idea and the
    # measurement that killed it are recorded so it is not attempted again.
    #
    # THE FAILED IDEA. Place every jump anchor against the edge facing the other
    # platform, pulled inside far enough that its detection box fits within the
    # standable span (`standable_span`), narrowing the tolerance per platform to make it
    # fit. That made all 30 L4 boxes "correct" by that invariant: violations 21 -> 0.
    #
    # WHY IT IS WRONG. A jump's landing position depends on the POLICY that jumped.
    # Measured on floor 7, same seed pool, two policies:
    #
    #     v6 champion   lands at px 116                     box 112..128 covered 83%
    #     v11 policy    lands at px 108, jumps to 136..148  same box covered 1.8%
    #
    # A box sized to one policy's landing misses another's. Worse, the agent is grounded
    # for only TWO FRAMES at px 108 and is then airborne (pose 9) all the way to the
    # ladder at px 144 -- and detection is pose-gated, so those frames cannot fire. So a
    # jump platform offers one narrow grounded window, and an anchor 4 px off it detects
    # nothing. Anchor 28 (px 120) sat on the jump arc and scored 0.00.
    #
    # Narrowing reproduced a documented failure verbatim, including its number: `Rope1`
    # read 1.8% while `Lclimb2_top` -- reachable only THROUGH Rope1's platform -- read
    # 0.71. That divergence is the diagnostic: a waypoint far below its own downstream
    # neighbour means detection is blind, not that the agent stopped going there.
    #
    # HOW THESE WERE CHOSEN INSTEAD (debug/l4_anchor_recommend.py). Census the positions
    # the agent is actually grounded at, across at least TWO policies from different
    # runs, score per EPISODE, and pick by the WORST policy's score. Single-policy
    # validation is what produced the landmine: anchor 28 scored 1.00 against v6's
    # champion and 0.00 against a later one. Tolerance stays flat (jump 6, ladder 2);
    # the reward channel's tolerance is a fixed 2 regardless, so the anchor alone
    # decides whether a MANDATORY milestone can ever be marked.
    #
    # THE DISTINCTION THAT WAS MISSED. A box extending into the VOID is harmless --
    # detection is pose-gated, so there are no grounded frames out there. A box covering
    # a platform's LETHAL EDGE, where the agent reads grounded and falls on the next
    # step, is what poisoned `Low2_launch`'s pool. Those are different problems, and
    # narrowing detection fixed the second by breaking the first. The right shape is
    # wide DETECTION plus a CAPTURE filter that rejects unrecoverable seeds -- which is
    # `admit_requires_survival`, a direct test, not a pose proxy.
    #
    # `standable_span` and `waypoint_tolerance` remain as MEASUREMENT tools (used by
    # debug/l4_edge_limit.py and debug/yeti_standable_audit.py) but are deliberately NOT
    # wired into detection.
    jump_waypoint_pos={
        "Fr1": (64, 158),
        # x_ram 27 -> 25 (px 116 -> 112 -> 108). Both 25 and 27 score 1.00 per episode,
        # but 25 puts the MODAL landing at the box CENTRE instead of its edge. Anchor 28
        # scored 1.00 against v6's champion and 0.00 against a later policy that landed
        # 4 px further left, so leftward margin is the thing that was missing.
        "Rope1": (25, 118),
        # x_ram 48 -> 51. Mandatory, and its tol-2 reward box at 192..208 fired on 0.12
        # of episodes for the worse of two policies; the agent's modal grounded position
        # on floor 9 is px 212. New box 196..212.
        "Spring": (51, 94),
        # x_ram 60 -> 66. Mandatory, and its tol-2 reward box at 240..256 fired on
        # 0.00 of episodes -- the agent stands at px 256/272 on floor 10, so the box
        # never contained it. Unmarked since v4, leaving its distance term switched
        # on for every episode (the `Fr1` defect). New box 256..288, worst policy 0.92.
        "Step": (66, 102),
        # x_ram 44 -> 45 (px 184 -> 188). RE-LANDED 2026-09-09. This exact fix shipped
        # in
        # e667206 and was lost when that commit was reverted WHOLESALE over the `Rope1`
        # regression -- the comment above kept asserting it in the past tense while the
        # code had no entry here at all, so the file read as already fixed for weeks.
        #
        # px 184 is floor 12's tile edge and the agent CANNOT stand on it: walk there,
        # hold NOOP, and it falls 0/8 (px 188 stands 8/8). Independently measured twice,
        # weeks apart, by debug/l4_edge_limit.py and again by a full-platform sweep, and
        # the two agree. Approach direction does NOT matter: 8/8 both walking left and
        # walking right at every usable px, tested because the foot row is asymmetric
        # (centre-6..centre+2) so a mirrored sprite might plausibly have changed it.
        #
        # THIS MOVES THE POTENTIAL'S TARGET ONLY. The shaping was paying +1.08 -- the
        # largest single payment in a from-reset trace -- for the grounded step onto the
        # lethal pixel. It does NOT stop px-184 states entering the pool: capture is
        # tol 6, so the box is 164..212 after the move and still spans 184. Cleaning the
        # pool needs the survival gate (see H-AW): 81/100 of this pool's seeds sit on
        # px 184 and fall on load, yet all 100 pass the >=30-step gate because the
        # trampoline below keeps them alive a median of 83 steps.
        "Low2_launch": (45, 70),
        # x_ram 56 -> 54 (px 232 -> 224). px 232 is floor 12's x_max and falls 0/8; 224
        # stands 8/8. Harmless in practice -- this pool is healthy (96 seeds at px 220,
        # 4 at 224) because the agent falls at 228 BEFORE it can reach 232, so captures
        # land on safe ground. Kept anyway because the potential should not aim at a
        # pixel the agent can never occupy. Worth remembering as the counter-example: a
        # lethal anchor can be completely inert, so "this anchor is wrong" does not
        # imply
        # "this is what is blocking us".
        "Low1": (54, 70),
    },
    # Redundant launch pads: each shares a platform with a waypoint that already marks
    # correctly, so they added a second, wider, misplaced box for the same traversal.
    #   Spring_launch  floor 8  -> Lclimb2_top is at the very position the measured
    #                             proposal for Spring_launch resolved to (34, 94)
    #   Step_launch    floor 9  -> Spring marks fine there
    #   Low1_launch    floor 11 -> Lclimb3_top is exact; the launch pad's tol-6 box
    #                             also caught the y82 ladder stall 24 px away
    #   Low2_launch    floor 12 -> Low1 is on the SAME platform (anchor px 224) and
    #                             marks correctly. Added 2026-10-01; this one is a
    #                             LEVER, not a tidy-up, see below.
    #
    # WHY Low2_launch GOES (2026-10-01). Its pool is mostly states you cannot play from.
    # Measured with `pool_revalidate.py --only Low2_launch --window 30` on v23's pool,
    # applying the `admit_requires_grounded` criterion:
    #
    #     Low2_launch  100 -> 39   dropped 61   died_in_window 0
    #        dropped by px {184: 60, 188: 1}    end_pose {17: 61}
    #
    # 60 of the 100 sit on px 184, which dies at step 82 from rest under a NOOP hold
    # (px 188 and 192 survive 120+). `died_in_window 0` is why the survival gate never
    # caught them: the spring below the gap keeps every one alive past
    # `min_survival_steps` 30, and they end the window in pose 17, the trampoline rise.
    # So the pool meant to
    # teach the rope-2 crossing has been teaching the fall-bounce loop.
    #
    # AND THE CROSSING IS EXECUTABLE FROM `Low1` INSTEAD: 26 of 136 scripted plans cross
    # from a px-220 `Low1` seed -- run-up 11..13 left steps to px 188/192, then wait
    # 16..24, then hold jump-left. (`Low1`'s pool is healthy: 95 of 100 seeds at px 220,
    # on safe ground.) An earlier sweep recorded 0 crossings from `Low1` and that number
    # is retracted -- it swept run-up length but not PHASE, and its N=0..14 could not
    # reach px 188 anyway: the leftward walk cycle stalls, so 12 steps are needed.
    #
    # ALSO REMOVED FROM `route_order`, AND IT HAS TO BE. The first attempt kept the
    # entry, to preserve champion-score comparability: depth is
    # `(index + 1) / len(route_order)` in `keep_best_sweep`, so changing the length
    # rescales every stored score. `test_route_order_matches_the_waypoint_universe`
    # rejected that, correctly. With `gate_waypoints_by_predecessor` a listed point that
    # is never emitted gets no `wp_reach_ema` row, reads 0.0, and refuses its SUCCESSOR
    # forever -- so `Low2`, the rope-2 landing and the pool this whole change exists to
    # fill, would have been permanently unseedable. That is the `F1` defect again.
    #
    # The price is real but small: champion `score` is not comparable across this commit
    # (denominator 30 -> 29). Cross-run comparison here uses `mean_rung` and the
    # frontier rate, not `score`, and `score` only picks the best snapshot WITHIN a run.
    #
    # The `jump_waypoint_pos` entry above does stay: the skip loop pops before the
    # override loop, which is guarded by `if nm in out`, so it is inert, and the anchor
    # is measured data worth keeping.
    #
    # CONTROL ARM: v28a/b/c at commit 723993e -- three replicates of the otherwise
    # identical config, `mean_rung` 3.998 / 5.009 / 4.782. Judge this on whether `Low2`
    # reach leaves 0.000, which it has been in all seven runs to date (~420 evals); do
    # NOT judge it on a `mean_rung` shift, which needs 3 runs an arm to see 1.2 rungs.
    jump_waypoint_skip=[
        "Spring_launch",
        "Step_launch",
        "Low1_launch",
        "Low2_launch",
    ],
    # 0 = nodes stay ON the tile edge. MEASURED HARMFUL AT 4, DO NOT RAISE IT.
    #
    # The idea was that a node on the tile edge aims the potential at a pixel the agent
    # falls off, so pulling it one step in would stop the shaping paying to step onto
    # L4 floor 12's px 184 (which kills 12/12 agents that walk there and stop, and was
    # the pad's only positive shaped reward at +0.12). That reasoning confuses two
    # opposite things: the node marks where the agent JUMPS FROM, and a jump departs the
    # edge in motion. You cannot stand on the edge; you can and must leave from it.
    #
    # WHAT 4 ACTUALLY DID, measured on every L4 jump edge: the last step onto the
    # departure edge went from +0.04 to -0.04. All 12 flipped sign, so the shaping
    # punished the departure of every jump on the level, including rope 1 (which the
    # agent crosses ~53% of the time) and including rope 2 itself -- the notes record a
    # real crossing that departs px 184 with leftward momentum.
    #
    # Run v25 (6M, warm from v23, empty pools, evaluator at n=30 over 60 snapshots)
    # against v24 as control:
    #
    #     mean Low2_launch rate   0.195 vs 0.349      median 0.17 vs 0.37
    #     frontier collapsed in   23/60 vs 8/56 evals
    #     per-waypoint from-reset reach: flat -0.12 to -0.13 from Lclimb1_top all the
    #     way to Low2_launch -- a constant offset, i.e. loss incurred EARLY and
    #     inherited downstream, not a rope-2 effect
    #
    # v25 bundled this with the rewards.SURFACE_POSES fix, so the split is not proven;
    # but only this change has a mechanism that predicts a uniform whole-route offset.
    # Kept as a field rather than deleted so the arm is reproducible from data, not from
    # a git checkout, per the control-arm rule in experiments/003-yeti-training.md.
    # test_jump_node_inset_stays_off pins it.
    edge_inset=0,
)

LEVELS: Dict[int, LevelMap] = {1: LEVEL1, 2: LEVEL2, 3: LEVEL3, 4: LEVEL4}


def get_level_map(level: int = 1) -> LevelMap:
    if level not in LEVELS:
        raise ValueError(f"No Yeti level map for level {level}; have {sorted(LEVELS)}")
    return LEVELS[level]


# ---------------------------------------------------------------------------
# Backward-compatible module-level constants (level 1).
# ---------------------------------------------------------------------------

FLOOR_TOP_Y: Dict[int, int] = LEVEL1.floor_top_y
FLOOR_HEIGHT = LEVEL1.floor_height
FRUIT_CENTRE_PX: Dict[int, Tuple[int, int]] = LEVEL1.fruit_centre_px
FRUIT_FLOOR: Dict[int, int] = LEVEL1.fruit_floor
LADDERS: List[Tuple[str, int, int, int]] = LEVEL1.ladders
PRINCESS_CENTRE_PX: Tuple[int, int] = LEVEL1.princess_centre_px
PRINCESS_FLOOR = LEVEL1.princess_floor


@dataclass(frozen=True)
class Node:
    """A fixed node in the navigation graph.

    ``kind`` is one of {"fruit", "ladder_bot", "ladder_top", "princess"}
    for debug readability. ``ident`` is a human label.
    """

    floor: int
    x: int
    kind: str
    ident: str


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _edge_px(p: "Platform", toward_x: float, inset: int = 0) -> int:
    """The x (px) on platform ``p``'s edge nearest ``toward_x`` — the jump-off /
    landing point. Mirrors annotate_level3_map._edge_pt (platform extents here
    are already pixels, so no ram->px scaling).

    ``inset`` pulls the result that many px INSIDE the platform, for callers that need a
    position the agent can stand on rather than the tile boundary. Default 0 keeps every
    existing caller byte-identical; only `_jump_graph` passes it (see EDGE_INSET_PX).

    WHICH SIDE the other platform is on is decided against the REAL extent, so the
    inset moves the returned point without changing the geometry of the choice. The
    third branch (the platforms overlap in x, so the nearest point is not on an edge at
    all) clamps rather than insets: there is no edge to step back from. That branch
    never fires on L4 -- all 24 endpoints hit a left or right clamp -- so the inset
    applies uniformly there.
    """
    lo, hi = p.x_min + inset, p.x_max - inset
    if lo > hi:  # platform narrower than 2*inset: collapse to its centre
        lo = hi = (p.x_min + p.x_max) // 2
    if toward_x <= p.x_min:
        return lo
    if toward_x >= p.x_max:
        return hi
    return min(max(int(toward_x), lo), hi)


def _jump_graph(lvl: LevelMap):
    """Nodes + edges contributed by ``lvl.jump_edges``.

    Returns (nodes, edge_specs) where edge_specs are (identA, identB, cost).
    Each jump-edge (fa, fb) gets an endpoint node on each platform placed
    ``EDGE_INSET_PX`` inside that platform's edge nearest the other (so shaping pulls
    toward the jump-off point on wide platforms), joined by a jump edge of cost
    |Δx| + |Δy|. Empty unless the level defines both jump_edges and platforms.

    WHY INSET, AND WHY ONLY HERE (2026-09-24). These nodes are what the PBRS potential
    measures distance to, so their position decides which pixel the shaping walks the
    agent to. Placed on the tile boundary they name a pixel the agent falls off, which
    contradicts this docstring's own word "jump-off point". On L4 floor 12 that was not
    academic: `J12_13_a` sat on px 184, the step 188 -> 184 was the ONLY positive shaped
    reward anywhere on the rope-2 pad (+0.12 over three live targets), and px 184 kills
    12/12 agents that walk there and stop. Inset, the same step pays about -0.12.

    `jump_waypoints` deliberately does NOT inset, even though it derives its defaults
    from the same function. Those feed DETECTION and CAPTURE, a different subsystem with
    its own history -- e667206 moved anchors, regressed `Rope1`, and was reverted
    wholesale, taking a correct `Low2_launch` fix with it. Moving them is a separate
    lever with a separate control arm; `test_jump_nodes_inset_curriculum_anchors_not`
    pins the split.
    """
    if not lvl.jump_edges or not lvl.platforms:
        return [], []
    pf = {p.floor: p for p in lvl.platforms}
    nodes: List[Node] = []
    edge_specs: List[Tuple[str, str, int]] = []
    for fa, fb in lvl.jump_edges:
        pa, pb = pf[fa], pf[fb]
        ca = (pa.x_min + pa.x_max) / 2.0
        cb = (pb.x_min + pb.x_max) / 2.0
        xa = _edge_px(pa, cb, inset=lvl.edge_inset)
        xb = _edge_px(pb, ca, inset=lvl.edge_inset)
        ia, ib = f"J{fa}_{fb}_a", f"J{fa}_{fb}_b"
        nodes.append(Node(floor=fa, x=xa, kind="jump", ident=ia))
        nodes.append(Node(floor=fb, x=xb, kind="jump", ident=ib))
        cost = abs(xa - xb) + abs(lvl.floor_top_y[fa] - lvl.floor_top_y[fb])
        edge_specs.append((ia, ib, cost))
    return nodes, edge_specs


def jump_waypoints(lvl: LevelMap) -> Dict[str, Tuple[int, int, int]]:
    """Curriculum waypoints for the jump-edge ascent, keyed by name, as
    ``{name: (x_ram, y_px, floor)}``.

    Two waypoints per NAMED edge (opt-in via ``lvl.jump_waypoint_names``, keyed
    by landing floor):
      - ARRIVAL (``name``): on the landing platform (larger route-order / floor
        id) at its edge nearest the departure platform — identical to the
        ``_b`` endpoint x that ``_jump_graph`` puts in the reward graph.
      - LAUNCH (``f"{name}_launch"``): on the DEPARTURE platform at its edge
        nearest the landing platform — the jump-off pad.
    Both use the agent's X RAM units (``(px-8)//4``, same rule as ladder
    waypoints). The launch pad matters because it sits on the LOWER platform
    (which the agent already reaches), so it is capturable and can SEED the
    otherwise-unbootstrappable jump — e.g. A1's launch is SN3's SAFE left edge,
    away from the ladder-side snowball, so seeding there lets the agent practise
    SN3->A1 directly instead of dying on SN3's right side. Empty unless the
    level defines ``jump_edges``, ``platforms`` AND ``jump_waypoint_names`` (so
    L1/L2 => {}).
    """
    names = lvl.jump_waypoint_names
    if not (lvl.jump_edges and lvl.platforms and names):
        return {}
    pf = {p.floor: p for p in lvl.platforms}
    out: Dict[str, Tuple[int, int, int]] = {}

    def _wp(p, toward_centre, floor):
        return (
            (int(_edge_px(p, toward_centre)) - 8) // 4,
            lvl.floor_top_y[floor],
            floor,
        )

    for fa, fb in lvl.jump_edges:
        land, other = max(fa, fb), min(fa, fb)
        name = names.get(land)
        if name is None:
            continue
        p_land, p_other = pf[land], pf[other]
        c_land = (p_land.x_min + p_land.x_max) / 2.0
        c_other = (p_other.x_min + p_other.x_max) / 2.0
        out[name] = _wp(p_land, c_other, land)  # arrival (landing edge)
        out[f"{name}_launch"] = _wp(p_other, c_land, other)  # jump-off pad
    # Measured overrides and removals (see the LevelMap fields). Applied after
    # derivation so the derived value stays the default and only named entries change.
    for nm in lvl.jump_waypoint_skip or ():
        out.pop(nm, None)
    for nm, xy in (lvl.jump_waypoint_pos or {}).items():
        if nm in out:
            out[nm] = (int(xy[0]), int(xy[1]), out[nm][2])
    return out


def standable_span(lvl: LevelMap, floor: int) -> Optional[Tuple[int, int]]:
    """CONSERVATIVE range of sprite-centre px the agent can stand on, on ``floor``.

    ``[x_min + 8, x_max - 8]``. Inclusive; None if the floor has no platform entry, and
    ``lo > hi`` for a platform too narrow to have a safe centre at all.

    WHERE THE 8 COMES FROM, AND WHY IT IS CONSERVATIVE RATHER THAN EXACT. ``x_max`` is
    EXCLUSIVE (extents are ``[col0*8, (col1+1)*8]``) and the sprite CENTRE is
    ``x_ram*4 + 8`` with the sprite 14 px wide and its feet ~9 px across. The true
    limits were measured per floor by walking to the edge, saving each candidate,
    reloading it and holding NOOP -- which is necessary because a frame at the edge
    reads as GROUNDED (y still at the standing value, walk pose, not 11) while already
    committed to a fall, so any test that does not confirm survival believes it:

        floor 12  [184..232)   measured  188 .. 224     = x_min+4, x_max-8
        floor 13  [  0..128)   measured    ? .. 124     =          x_max-4
        floor  3  [248..280)   measured  256 .. 276     = x_min+8, x_max-4
        floor  9  [200..232)   measured  208 .. 232     = x_min+8, x_max+0

    Those do not reduce to one formula, so this deliberately does NOT try to be exact.
    ``[x_min+8, x_max-8]`` is inside every span measured above, so a box that fits in
    here is safe on all of them -- and being provably-inside is what the callers need,
    not tightness. Tighten a specific floor only with a fresh NOOP measurement.
    """
    if not lvl.platforms:
        return None
    for p in lvl.platforms:
        if p.floor == floor:
            return (p.x_min + 8, p.x_max - 8)
    return None


def waypoint_tolerance(lvl: LevelMap, floor: int, x_ram: int, requested: int) -> int:
    """Largest tolerance <= ``requested`` whose box fits inside the floor's safe span.

    THE INVARIANT: a detection box must lie entirely within the standable span. Every
    waypoint defect found on this game violated it, three times over the same shape:

    * ``Low2_launch`` was anchored at floor 12's tile edge and its box straddled the
      brink, so the seed pool filled with states that read as grounded and fell one step
      later -- all 100 of them, and the pool meant to teach the rope-2 crossing taught
      falling instead.
    * ``Low1_launch``'s +-24 px box reached a ladder 24 px away and reported a stalled
      climb as an arrival (reach 0.36 against 0.03 for the same event).
    * ``Fr1`` sat on a platform extremity the agent never occupies, so the reward could
      never mark it and its distance term never switched off.

    Why a fixed number cannot work: the jump tolerance was 6 (+-24 px), which needs a
    48 px span, and span width on L4 ranges from 0 to 300 px. Measured against
    ``[x_min+8, x_max-8]``, 21 of 30 L4 boxes and 13 of 19 L3 boxes overflowed -- and
    ALL 21 L4 failures were jump waypoints while all nine ladder waypoints passed. The
    Hi-chain platforms are 16 px wide, so their safe span is a SINGLE centre and no
    tolerance above 0 can fit. The constraint is per-platform, so the number must be
    derived per platform.

    ``requested`` is the caller's ceiling (the curriculum's ladder/jump tolerance), so
    this only ever narrows. Returns 0 when nothing wider fits, which still detects: x
    advances one 4-px unit per step, so the agent cannot skip an exact value.
    """
    span = standable_span(lvl, floor)
    if span is None:
        return int(requested)
    lo, hi = span
    centre = x_ram * 4 + 8
    # Room on each side, in whole 4-px units; the box must fit BOTH ways.
    room = min((centre - lo) // 4, (hi - centre) // 4)
    return max(0, min(int(requested), int(room)))


def build_fixed_nodes(lvl: LevelMap = LEVEL1) -> List[Node]:
    """Build the list of fixed nodes (fruits + ladder endpoints + princess,
    plus jump-edge endpoints for levels that define them)."""
    nodes: List[Node] = []
    for f_id, (x, _y) in lvl.fruit_centre_px.items():
        nodes.append(
            Node(floor=lvl.fruit_floor[f_id], x=x, kind="fruit", ident=f"F{f_id}")
        )
    for name, top_floor, bot_floor, x in lvl.ladders:
        # Ladder tuples are ordered (name, TOP_floor, BOT_floor, x) on every
        # level (top = higher on screen = smaller floor_top_y). Same positional
        # rule as yeti.waypoints(), so node idents agree between the reward
        # graph and the waypoint seeder.
        nodes.append(Node(floor=top_floor, x=x, kind="ladder_top", ident=f"{name}_top"))
        nodes.append(Node(floor=bot_floor, x=x, kind="ladder_bot", ident=f"{name}_bot"))
    nodes.append(
        Node(
            floor=lvl.princess_floor,
            x=lvl.princess_centre_px[0],
            kind="princess",
            ident="princess",
        )
    )
    nodes.extend(_jump_graph(lvl)[0])
    return nodes


def build_edges(
    nodes: List[Node], lvl: LevelMap = LEVEL1
) -> List[Tuple[int, int, int]]:
    """Return edges as (src_idx, dst_idx, cost) triples.

    - Horizontal: between any two nodes on the same floor (cost = |dx|).
    - Vertical: between a ladder's bottom and its top (cost = floor_height
      per floor the ladder spans).
    """
    edges: List[Tuple[int, int, int]] = []

    by_floor: Dict[int, List[int]] = {}
    for i, node in enumerate(nodes):
        by_floor.setdefault(node.floor, []).append(i)
    for _floor, idxs in by_floor.items():
        for i in idxs:
            for j in idxs:
                if i == j:
                    continue
                cost = abs(nodes[i].x - nodes[j].x)
                edges.append((i, j, cost))

    for name, top_floor, bot_floor, x in lvl.ladders:
        bot_ident = f"{name}_bot"
        top_ident = f"{name}_top"
        bot_idx = next(i for i, nd in enumerate(nodes) if nd.ident == bot_ident)
        top_idx = next(i for i, nd in enumerate(nodes) if nd.ident == top_ident)
        # Ladder cost = actual vertical climb in px (|Δ standing_y|). For L1/L2
        # (evenly spaced floors) this equals floor_height*|Δfloor_id| exactly,
        # so distances are byte-identical; for L3 (irregular, non-y-ordered
        # floor ids where several platforms share a y) it is the correct cost.
        cost = abs(lvl.floor_top_y[bot_floor] - lvl.floor_top_y[top_floor])
        edges.append((bot_idx, top_idx, cost))
        edges.append((top_idx, bot_idx, cost))

    # Jump edges (bidirectional): reconnect platforms the agent reaches by
    # jumping (START->STEP->2LAD, SN3->A1..A5). See LevelMap.jump_edges.
    ident_to_idx = {nd.ident: i for i, nd in enumerate(nodes)}
    for ia, ib, cost in _jump_graph(lvl)[1]:
        a, b = ident_to_idx[ia], ident_to_idx[ib]
        edges.append((a, b, cost))
        edges.append((b, a, cost))
    return edges


def floyd_warshall(n: int, edges: List[Tuple[int, int, int]]) -> List[List[int]]:
    """All-pairs shortest path. n small (~15), so O(n^3) is fine."""
    INF = 10**9
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            di = dist[i]
            dik = di[k]
            if dik >= INF:
                continue
            for j in range(n):
                via = dik + dk[j]
                if via < di[j]:
                    di[j] = via
    return dist


@dataclass
class NavigationMap:
    """Convenient bundle: nodes, edges, all-pairs distances."""

    nodes: List[Node]
    dist: List[List[int]]
    node_by_ident: Dict[str, int]

    def fruit_node_idx(self, fruit_id: int) -> int:
        return self.node_by_ident[f"F{fruit_id}"]

    def princess_node_idx(self) -> int:
        return self.node_by_ident["princess"]

    def path_distance_from_agent(
        self,
        agent_floor: int,
        agent_x: int,
        target_ident: str,
    ) -> int:
        """Shortest-path distance from (agent_floor, agent_x) to the
        named target node.

        The agent is a transient node: distance through any ladder
        endpoint on the agent's floor is ``|agent_x - endpoint.x|``
        plus that endpoint's precomputed distance to the target. Same
        for any fruit or princess on the agent's floor (walk directly).
        """
        target_idx = self.node_by_ident[target_ident]
        if self.nodes[target_idx].floor == agent_floor:
            direct = abs(agent_x - self.nodes[target_idx].x)
        else:
            direct = 10**9
        best = direct
        for i, node in enumerate(self.nodes):
            if node.floor != agent_floor:
                continue
            via = abs(agent_x - node.x) + self.dist[i][target_idx]
            if via < best:
                best = via
        return best

    def path_distance_from_ladder(
        self, ladder_name: str, pixel_y: int, y_top: int, y_bot: int, target_ident: str
    ) -> int:
        """Shortest-path distance from a point at height ``pixel_y`` ON a
        vertical ladder edge to the target.

        The agent is a transient point on the edge: distance to each endpoint
        is the vertical gap ``|pixel_y - endpoint_y|`` (the same |Δy| metric
        the ladder edge cost uses), plus that endpoint's precomputed graph
        distance to the target. This is what lets a ladder/escalator DESCENT
        earn continuous progress instead of a lump on arrival.
        """
        ti = self.node_by_ident[target_ident]
        top_i = self.node_by_ident[f"{ladder_name}_top"]
        bot_i = self.node_by_ident[f"{ladder_name}_bot"]
        return min(
            abs(pixel_y - y_top) + self.dist[top_i][ti],
            abs(pixel_y - y_bot) + self.dist[bot_i][ti],
        )

    def path_distance_from_pos(
        self, floor, ladder, agent_x: int, pixel_y: int, target_ident: str
    ) -> int:
        """Segment-aware distance: from a horizontal ``floor`` (as today) or,
        if ``floor`` is None and ``ladder`` = (name, y_top, y_bot) is given,
        from a point on that vertical edge. Returns the INF sentinel if
        neither resolves."""
        if floor is not None:
            return self.path_distance_from_agent(floor, agent_x, target_ident)
        if ladder is not None:
            return self.path_distance_from_ladder(
                ladder[0], pixel_y, ladder[1], ladder[2], target_ident
            )
        return 10**9


def build_navigation_map(level: int = 1) -> NavigationMap:
    """Assemble the NavigationMap for ``level``; cheap, so call per env."""
    lvl = get_level_map(level)
    nodes = build_fixed_nodes(lvl)
    edges = build_edges(nodes, lvl)
    dist = floyd_warshall(len(nodes), edges)
    node_by_ident = {nd.ident: i for i, nd in enumerate(nodes)}
    return NavigationMap(nodes=nodes, dist=dist, node_by_ident=node_by_ident)


# ---------------------------------------------------------------------------
# Helper: agent floor from pixel y
# ---------------------------------------------------------------------------


def agent_floor_from_pixel_y(pixel_y: int, level: int = 1) -> Optional[int]:
    """Return the nearest floor the agent is "standing on", or None if
    the agent is mid-jump / off-floor / in the death-animation zone.

    Uses a tolerance around each floor's standing-y. Y-ONLY: valid on levels
    where each floor is a single full-width surface (L1/L2). For levels with
    stacked/short platforms (L3) use :func:`agent_floor_from_pixel_xy`.
    """
    for f_id, ftop in get_level_map(level).floor_top_y.items():
        if abs(pixel_y - ftop) <= 8:
            return f_id
    return None


def agent_floor_from_pixel_xy(
    pixel_x: int, pixel_y: int, level: int = 1
) -> Optional[int]:
    """X-aware floor resolution.

    On a level WITHOUT explicit platforms (L1/L2) this is exactly
    :func:`agent_floor_from_pixel_y` — the pixel_x is ignored and each floor
    is treated as one full-width surface, so results are byte-identical.

    On a level WITH platforms (L3) the agent resolves to a floor only if its
    pixel_x lies within that floor's platform extent (and pixel_y within the
    ±8 tolerance); the nearest-y match wins. If no platform contains (x, y)
    — e.g. the agent is on a tall ladder passing between platforms, or in a
    gap — this returns None (shaping is then frozen upstream, never pulled
    toward a wrong same-height structure).
    """
    lvl = get_level_map(level)
    if lvl.platforms is None:
        return agent_floor_from_pixel_y(pixel_y, level)
    best_floor: Optional[int] = None
    best_dy = 9
    for p in lvl.platforms:
        if p.x_min <= pixel_x <= p.x_max and abs(pixel_y - p.y) <= 8:
            dy = abs(pixel_y - p.y)
            if dy < best_dy:
                best_dy = dy
                best_floor = p.floor
    return best_floor


def agent_ladder_from_pixel_xy(
    pixel_x: int, pixel_y: int, level: int = 1, x_tol: int = 6
) -> Optional[Tuple[str, int, int]]:
    """Resolve the VERTICAL ladder edge the agent is on, or None.

    Returns ``(ladder_name, y_top, y_bot)`` when ``pixel_x`` is within
    ``x_tol`` of a ladder's centre AND ``pixel_y`` is STRICTLY between its
    endpoints (endpoints belong to the floor, resolved by
    :func:`agent_floor_from_pixel_xy` which callers try first). Nearest-x
    ladder wins. Measured: a real climb sits at the ladder centre exactly
    (offset 0); the escalator rides ~4px off the Lesc centre, so a small
    ``x_tol`` suffices and keeps floor/ladder resolution unambiguous.
    """
    m = get_level_map(level)
    best = None
    best_dx = x_tol + 1
    for name, top_floor, bot_floor, cx in m.ladders:
        y_top = m.floor_top_y[top_floor]
        y_bot = m.floor_top_y[bot_floor]
        lo, hi = min(y_top, y_bot), max(y_top, y_bot)
        dx = abs(pixel_x - cx)
        if dx <= x_tol and lo < pixel_y < hi and dx < best_dx:
            best_dx = dx
            best = (name, y_top, y_bot)
    return best


__all__ = [
    "LevelMap",
    "Platform",
    "LEVELS",
    "get_level_map",
    "jump_waypoints",
    "agent_ladder_from_pixel_xy",
    "FLOOR_TOP_Y",
    "FLOOR_HEIGHT",
    "FRUIT_CENTRE_PX",
    "FRUIT_FLOOR",
    "LADDERS",
    "PRINCESS_CENTRE_PX",
    "PRINCESS_FLOOR",
    "Node",
    "NavigationMap",
    "build_fixed_nodes",
    "build_edges",
    "floyd_warshall",
    "build_navigation_map",
    "agent_floor_from_pixel_y",
    "agent_floor_from_pixel_xy",
]
