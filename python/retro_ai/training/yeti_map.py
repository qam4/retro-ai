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

# Level 3 — fragmented multi-platform layout, read from RAM (level3_start.sav)
# with the route confirmed by play. 1 fruit (top-left area), princess top-left,
# player starts bottom-left. Floors are keyed by their platform standing-y.
#
# CAVEATS (L3-specific):
#  - Platforms are as close as 8px in y (e.g. 40/48, 72/80), so the ±8
#    agent_floor_from_pixel_y mapping is ambiguous here -> the floor-based
#    PATH-PROGRESS SHAPING is unreliable on L3 and needs a rethink (the
#    escalator already forces that). The LADDER-ENDPOINT WAYPOINTS below are
#    still valid (positions only).
#  - The ESCALATOR (moving platforms circulating CCW around a wall), the
#    COMPRESSOR (periodic crush on climb-platform #3), and SNOWBALLS are
#    MOVING SPRITES — not in this static tilemap. They are curriculum-carried,
#    not waypoint/graph-modelled.
#
# Ladder route roles (confirmed with the user): the two ladders up to the goat
# platform (Lgoat_a left / Lgoat_b right), the post-escalator DOWN ladder
# (Ldown), the snowball climb bottom->top (Lsc1 -> Lsc2 -> Lsc3 -> Lsc4), and
# the final ladder up toward the princess (Lprincess).
LEVEL3 = LevelMap(
    # Floors keyed by platform standing-y. Ladder connections are by
    # X-ALIGNMENT (a ladder joins the platform whose column-range spans its x,
    # above and below) so the graph chains correctly. Verified: the snowball
    # climb (Ldown_bot -> Lsc1 -> Lsc2 -> Lsc3 -> Lsc4) is graph-connected; the
    # only graph GAPS are the true JUMP segments — the escalator (goat ->
    # Ldown) and the final 5-platform climb (Lsc4 -> Lprincess) — which are
    # sparse ("keep 0") by design.
    floor_top_y={
        2: 48,  # Lprincess top / princess platform (upper-left)
        3: 72,  # Lprincess bottom / fruit platform
        4: 80,  # goat platform (Lgoat tops)
        5: 104,  # Lsc4 top
        6: 128,  # Lsc4 bot / Lsc3 top
        7: 152,  # Lsc3 bot / Lsc2 top
        8: 168,  # Lgoat bottoms (2-ladder platform)
        9: 176,  # Lsc2 bot / Lsc1 top / Ldown top (big platform, row22)
        10: 184,  # Lsc1 bot / Ldown bot (bottom-right)
    },
    floor_height=24,  # nominal; the real layout is irregular
    fruit_centre_px={1: (56, 64)},
    fruit_floor={1: 3},  # upper-left, on/near the Lprincess platform
    ladders=[
        # (name, upper_floor, lower_floor, centre_x_px). upper = smaller y.
        # Connections are X-ALIGNED (platform col-range spans the ladder x).
        ("Lprincess", 2, 3, 24),  # final ladder up toward the princess
        ("Lgoat_a", 4, 8, 72),  # left of the two ladders to the goat platform
        ("Lgoat_b", 4, 8, 96),  # right of the two
        ("Ldown", 9, 10, 168),  # post-escalator down ladder
        ("Lsc1", 9, 10, 240),  # snowball climb, 1st (bottom)
        ("Lsc2", 7, 9, 280),  # snowball climb, 2nd
        ("Lsc3", 6, 7, 232),  # snowball climb, 3rd
        ("Lsc4", 5, 6, 280),  # snowball climb, 4th (top)
    ],
    princess_centre_px=(16, 30),
    princess_floor=2,  # on the Lprincess-top platform (so it's graph-connected)
    # One waypoint per ladder, at the end the agent ARRIVES at along the
    # route: the down-ladder (Ldown) -> its BOTTOM; every climb-ladder -> its
    # TOP. This seeds each episode just before the next hard segment (jump /
    # dodge / escalator) rather than re-climbing. The escalator (no ladder) is
    # then bracketed: Lgoat tops before it, Ldown bottom after it.
    waypoint_ends={
        "Lgoat_a": "top",
        "Lgoat_b": "top",
        "Ldown": "bot",
        "Lsc1": "top",
        "Lsc2": "top",
        "Lsc3": "top",
        "Lsc4": "top",
        "Lprincess": "top",
    },
    # Mandatory-waypoint reward targets (unordered, summed like fruits). The
    # goat platform is one target reached via EITHER ladder (OR-group); the
    # rest are single chokepoints. Across the escalator the post-escalator
    # ones are unreachable and drop out until the agent crosses.
    reward_waypoints=[
        ["Lgoat_a_top", "Lgoat_b_top"],  # goat platform (branch: either ladder)
        ["Ldown_bot"],  # post-escalator
        ["Lsc1_top"],
        ["Lsc2_top"],
        ["Lsc3_top"],
        ["Lsc4_top"],
        ["Lprincess_top"],
    ],
    # Walkable-segment x-extents (pixels), read from the tilemap
    # (output/mo5/yeti/level3/level3_map.json "floors"). These disambiguate
    # L3's stacked/overlapping platforms so the agent's floor is resolved by
    # x AND y — e.g. a point on the goat ladder (x~72, y~152) no longer
    # resolves to the snowball platform that merely shares y=152 (x 208-312);
    # x is out of every platform there, so it resolves to None (shaping
    # frozen mid-climb) instead of being pulled to the wrong escalator side.
    # NOTE (post-escalator TODO): floors 3 and 10's segments do not yet cover
    # all their graph nodes (F1 at x56; Ldown/Lsc1 bottoms at x168/240) — a
    # known inconsistency in the bottom-right hand-map to re-derive when we
    # tackle the post-escalator climb. Pre-escalator (2, 4, 8, 9) is exact.
    platforms=[
        Platform(2, 48, 0, 72),
        Platform(3, 72, 0, 48),
        Platform(4, 80, 56, 136),  # goat platform
        Platform(5, 104, 200, 312),
        Platform(6, 128, 216, 320),
        Platform(7, 152, 208, 312),
        Platform(8, 168, 72, 112),  # 2-ladder platform (goat-ladder bottoms)
        Platform(9, 176, 48, 320),  # big post-escalator platform
        Platform(10, 184, 0, 40),
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


def build_fixed_nodes(lvl: LevelMap = LEVEL1) -> List[Node]:
    """Build the list of fixed nodes (fruits + ladder endpoints + princess)."""
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
        cost = lvl.floor_height * abs(bot_floor - top_floor)
        edges.append((bot_idx, top_idx, cost))
        edges.append((top_idx, bot_idx, cost))
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


__all__ = [
    "LevelMap",
    "Platform",
    "LEVELS",
    "get_level_map",
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
