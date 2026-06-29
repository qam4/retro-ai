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
class LevelMap:
    """All static geometry for one level (pixel coordinates)."""

    floor_top_y: Dict[int, int]  # floor -> agent standing pixel-y
    floor_height: int  # pixels between adjacent floors
    fruit_centre_px: Dict[int, Tuple[int, int]]
    fruit_floor: Dict[int, int]
    ladders: List[Tuple[str, int, int, int]]  # (name, from_floor, to_floor, centre_x)
    princess_centre_px: Tuple[int, int]
    princess_floor: int


# Level 1 — original climb-up layout (floor 1 = bottom/spawn, 5 = princess).
LEVEL1 = LevelMap(
    floor_top_y={1: 184, 2: 152, 3: 120, 4: 88, 5: 56},
    floor_height=32,
    fruit_centre_px={1: (184, 184), 2: (80, 150), 3: (144, 120), 4: (272, 88)},
    fruit_floor={1: 1, 2: 2, 3: 3, 4: 4},
    ladders=[
        ("L12a", 1, 2, 120),
        ("L12b", 1, 2, 280),
        ("L23", 2, 3, 240),
        ("L34", 3, 4, 176),
        ("L45", 4, 5, 208),
    ],
    princess_centre_px=(312, 60),
    princess_floor=5,
)

# Level 2 — descending layout, read from RAM (ram_map_re.md). Floors numbered
# top->bottom: F1 (start, y48) .. F6 (bottom, y168), 24 px apart. Both fruits
# on floor 5 (y144). Ladder centre_x = extracted UL x_px + 8 (16 px wide).
# Princess modelled on floor 6 at her RAM x (288); she actually sits ~14 px
# below F6 (RAM y=182) reachable via the floor-6 descending ladders (x56,
# x232, omitted here) — a deliberate approximation: the shaping only needs to
# pull the agent down-and-right; the final step is learned from the sparse
# princess bonus. Refine to a floor 7 if the agent stalls just above her.
LEVEL2 = LevelMap(
    floor_top_y={1: 48, 2: 72, 3: 96, 4: 120, 5: 144, 6: 168},
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

LEVELS: Dict[int, LevelMap] = {1: LEVEL1, 2: LEVEL2}


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
    for name, fr, to, x in lvl.ladders:
        nodes.append(Node(floor=fr, x=x, kind="ladder_bot", ident=f"{name}_bot"))
        nodes.append(Node(floor=to, x=x, kind="ladder_top", ident=f"{name}_top"))
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

    for name, fr, to, x in lvl.ladders:
        bot_ident = f"{name}_bot"
        top_ident = f"{name}_top"
        bot_idx = next(i for i, nd in enumerate(nodes) if nd.ident == bot_ident)
        top_idx = next(i for i, nd in enumerate(nodes) if nd.ident == top_ident)
        cost = lvl.floor_height * abs(to - fr)
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

    Uses a tolerance around each floor's standing-y.
    """
    for f_id, ftop in get_level_map(level).floor_top_y.items():
        if abs(pixel_y - ftop) <= 8:
            return f_id
    return None


__all__ = [
    "LevelMap",
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
]
