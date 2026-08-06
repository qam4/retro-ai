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

LEVELS: Dict[int, LevelMap] = {1: LEVEL1, 2: LEVEL2, 3: LEVEL3}


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


def _edge_px(p: "Platform", toward_x: float) -> int:
    """The x (px) on platform ``p``'s edge nearest ``toward_x`` — the jump-off /
    landing point. Mirrors annotate_level3_map._edge_pt (platform extents here
    are already pixels, so no ram->px scaling)."""
    if toward_x <= p.x_min:
        return p.x_min
    if toward_x >= p.x_max:
        return p.x_max
    return int(toward_x)


def _jump_graph(lvl: LevelMap):
    """Nodes + edges contributed by ``lvl.jump_edges``.

    Returns (nodes, edge_specs) where edge_specs are (identA, identB, cost).
    Each jump-edge (fa, fb) gets an endpoint node on each platform placed at
    that platform's edge nearest the other (so shaping pulls toward the
    jump-off point on wide platforms), joined by a jump edge of cost
    |Δx| + |Δy|. Empty unless the level defines both jump_edges and platforms.
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
        xa, xb = _edge_px(pa, cb), _edge_px(pb, ca)
        ia, ib = f"J{fa}_{fb}_a", f"J{fa}_{fb}_b"
        nodes.append(Node(floor=fa, x=xa, kind="jump", ident=ia))
        nodes.append(Node(floor=fb, x=xb, kind="jump", ident=ib))
        cost = abs(xa - xb) + abs(lvl.floor_top_y[fa] - lvl.floor_top_y[fb])
        edge_specs.append((ia, ib, cost))
    return nodes, edge_specs


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
