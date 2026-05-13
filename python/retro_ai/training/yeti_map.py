"""Static map model for Yeti (Thomson MO5 / Crayon).

Encodes floors, ladders, fruits, and the princess in pixel coordinates
so we can compute path-distance reward shaping without the agent
having to rediscover navigation from scratch.

Agent-to-pixel mapping (verified via debug/cp0_reference_grid2.png):
  pixel_x = ram_x * 4
  pixel_y = ram_y
Agent sprite is 16x16; centre is (ram_x * 4 + 8, ram_y + 8).

Map data (verified via debug/cp0_ladders_annotated.png, approach 16.5):

Floors, by top-of-tile pixel y (where an agent sprite's UL sits when
the sprite is standing on the floor):

  floor 1 (spawn):  y = 184   (floor tile spans y=200..207 roughly)
  floor 2:          y = 152
  floor 3:          y = 120
  floor 4:          y =  88
  floor 5:          y =  56   (princess floor)

Fruit pixel CENTRES (sprite is 16x16):
  fruit 1: (184, 184)  floor 1
  fruit 2: ( 80, 150)  floor 2
  fruit 3: (144, 120)  floor 3
  fruit 4: (272,  88)  floor 4

Ladders (UL pixel x, 16 px wide). Each ladder spans exactly one
floor's worth of vertical distance (32 px between consecutive floor
tops). Represented by the x of the ladder's centre line:

  L12a: x=120 (UL=112), connects floor 1 (top) <-> floor 2 (top)
  L12b: x=280 (UL=272), connects floor 1 <-> floor 2
  L23:  x=240 (UL=232), connects floor 2 <-> floor 3
  L34:  x=176 (UL=168), connects floor 3 <-> floor 4
  L45:  x=208 (UL=200), connects floor 4 <-> floor 5

Princess: 16x24 sprite at UL (304, 48), centre (312, 60).

Graph model
-----------

Nodes: (floor, x) pairs for each ladder bottom and top, plus each
fruit and the princess. We add the agent's current position as a
transient node when computing distances.

Edges:
  - Horizontal edge between any two nodes on the same floor, cost
    equal to |dx|. (Agent can walk freely along a floor.)
  - Vertical edge between a ladder's bottom and its top, cost equal
    to the floor height (32 px).

We precompute the pairwise distances between fixed nodes; the agent
is a transient node whose distances to the fixed nodes only require
one horizontal-edge lookup per fixed node on the agent's floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLOOR_TOP_Y: Dict[int, int] = {1: 184, 2: 152, 3: 120, 4: 88, 5: 56}
FLOOR_HEIGHT = 32  # pixels between floors

FRUIT_CENTRE_PX: Dict[int, Tuple[int, int]] = {
    1: (184, 184),
    2: (80, 150),
    3: (144, 120),
    4: (272, 88),
}
FRUIT_FLOOR: Dict[int, int] = {1: 1, 2: 2, 3: 3, 4: 4}

# Ladder centre x (UL + 8 since ladder is 16 px wide).
LADDERS: List[Tuple[str, int, int, int]] = [
    # (name, from_floor, to_floor, centre_x)
    ("L12a", 1, 2, 120),
    ("L12b", 1, 2, 280),
    ("L23", 2, 3, 240),
    ("L34", 3, 4, 176),
    ("L45", 4, 5, 208),
]

PRINCESS_CENTRE_PX: Tuple[int, int] = (312, 60)
PRINCESS_FLOOR = 5


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


def build_fixed_nodes() -> List[Node]:
    """Build the list of fixed nodes (fruits + ladder endpoints + princess)."""
    nodes: List[Node] = []
    for f_id, (x, _y) in FRUIT_CENTRE_PX.items():
        nodes.append(Node(floor=FRUIT_FLOOR[f_id], x=x, kind="fruit", ident=f"F{f_id}"))
    for name, fr, to, x in LADDERS:
        nodes.append(Node(floor=fr, x=x, kind="ladder_bot", ident=f"{name}_bot"))
        nodes.append(Node(floor=to, x=x, kind="ladder_top", ident=f"{name}_top"))
    nodes.append(
        Node(
            floor=PRINCESS_FLOOR,
            x=PRINCESS_CENTRE_PX[0],
            kind="princess",
            ident="princess",
        )
    )
    return nodes


def build_edges(nodes: List[Node]) -> List[Tuple[int, int, int]]:
    """Return edges as (src_idx, dst_idx, cost) triples.

    - Horizontal: between any two nodes on the same floor (cost = |dx|).
    - Vertical: between a ladder's bottom and its top (cost = FLOOR_HEIGHT
      per floor the ladder spans; always 1 floor here).
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

    for name, fr, to, x in LADDERS:
        bot_ident = f"{name}_bot"
        top_ident = f"{name}_top"
        bot_idx = next(i for i, nd in enumerate(nodes) if nd.ident == bot_ident)
        top_idx = next(i for i, nd in enumerate(nodes) if nd.ident == top_ident)
        cost = FLOOR_HEIGHT * abs(to - fr)
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


def build_navigation_map() -> NavigationMap:
    """Assemble the NavigationMap once; cheap, so call per env."""
    nodes = build_fixed_nodes()
    edges = build_edges(nodes)
    dist = floyd_warshall(len(nodes), edges)
    node_by_ident = {nd.ident: i for i, nd in enumerate(nodes)}
    return NavigationMap(nodes=nodes, dist=dist, node_by_ident=node_by_ident)


# ---------------------------------------------------------------------------
# Helper: agent floor from pixel y
# ---------------------------------------------------------------------------


def agent_floor_from_pixel_y(pixel_y: int) -> Optional[int]:
    """Return the nearest floor the agent is "standing on", or None if
    the agent is mid-jump / off-floor / in the death-animation zone.

    Uses a tolerance around each FLOOR_TOP_Y entry.
    """
    for f_id, ftop in FLOOR_TOP_Y.items():
        if abs(pixel_y - ftop) <= 8:
            return f_id
    return None


__all__ = [
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
