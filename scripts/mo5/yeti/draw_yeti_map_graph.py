#!/usr/bin/env python3
"""Render the yeti navigation graph as an overlay on a CP0 screenshot.

Shows:
  - Fixed nodes (fruits, ladder endpoints, princess) as dots with labels.
  - Graph edges (horizontal walks + ladder climbs) as lines.
  - Shortest path from a probe starting position to each remaining
    fruit, highlighted.

Also prints the all-pairs distance matrix and a few worked examples.
"""
from __future__ import annotations

import pickle

import numpy as np
from PIL import Image, ImageDraw
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.yeti_map import (
    FLOOR_TOP_Y,
    FRUIT_CENTRE_PX,
    build_navigation_map,
)

OUT = "debug/cp0_nav_graph.png"


with open("output/mo5/yeti/go_explore_v9/archive.pkl", "rb") as f:
    archive = pickle.load(f)
cp0 = None
for key, entry in archive.items():
    v = key[2] if len(key) >= 3 else None
    if isinstance(v, (frozenset, set, list, tuple)) and len(v) == 0:
        cp0 = bytes(entry["state"])
        break
env_cfg = EnvConfig(
    profile="yeti_fruit",
    action_mode="joystick",
    max_steps=1_000_000,
    stall_threshold=1_000_000,
    resize=(84, 84),
)
stack = build_training_env("yeti_fruit", env_cfg)
base = stack.base
base.reset(seed=0)
base._interface.load_state(cp0)
for _ in range(5):
    _ = base.step([0, 0, 0])
img = np.asarray(base._last_raw_obs, dtype=np.uint8)
pil = Image.fromarray(img).convert("RGB")
draw = ImageDraw.Draw(pil)

nav = build_navigation_map()

# Position helpers: a node at (floor, x) renders at (x, FLOOR_TOP_Y[floor] + 8)
# so the dot sits at the middle of the sprite's standing zone.


def node_pixel(nd):
    return (nd.x, FLOOR_TOP_Y[nd.floor] + 8)


# Draw edges first (so nodes paint over them).
EDGE_COLOUR = (120, 120, 255)
LADDER_COLOUR = (255, 0, 255)
for u, _du in enumerate(nav.dist):
    for v, cost in enumerate(_du):
        if u == v or cost >= 10**9:
            continue
        nd_u = nav.nodes[u]
        nd_v = nav.nodes[v]
        # Only draw direct edges (not via others). Crude filter: skip
        # if the cost equals a two-hop combination of other edges.
        # Simpler: only draw edges that are actually in the original
        # edge list — horizontal same-floor neighbours OR ladder
        # bot<->top pairs.
        same_floor_neighbours = nd_u.floor == nd_v.floor
        ladder_pair = (
            nd_u.kind.startswith("ladder")
            and nd_v.kind.startswith("ladder")
            and nd_u.ident.split("_")[0] == nd_v.ident.split("_")[0]
            and nd_u.floor != nd_v.floor
        )
        if not (same_floor_neighbours or ladder_pair):
            continue
        x0, y0 = node_pixel(nd_u)
        x1, y1 = node_pixel(nd_v)
        if ladder_pair:
            draw.line([(x0, y0), (x1, y1)], fill=LADDER_COLOUR, width=2)
        else:
            draw.line([(x0, y0), (x1, y1)], fill=EDGE_COLOUR, width=1)


KIND_COLOUR = {
    "fruit": (255, 192, 0),
    "ladder_bot": (255, 0, 255),
    "ladder_top": (255, 0, 255),
    "princess": (255, 255, 255),
}
for node in nav.nodes:
    x, y = node_pixel(node)
    c = KIND_COLOUR[node.kind]
    r = 4
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=c, width=2)
    draw.text((x + r + 1, y - 6), node.ident, fill=c)

# Also mark fruit pickup centres explicitly (smaller inner dots).
for fid, (fx, fy) in FRUIT_CENTRE_PX.items():
    draw.rectangle([(fx - 2, fy - 2), (fx + 2, fy + 2)], fill=(255, 255, 0))

# Print the distance matrix.
print("Fixed nodes:")
for i, nd in enumerate(nav.nodes):
    print(f"  [{i}] floor={nd.floor} x={nd.x:>3} {nd.ident}")
print()
print("All-pairs distances (cost = pixels travelled):")
idents = [nd.ident for nd in nav.nodes]
w = max(len(x) for x in idents) + 1
header = " " * w + "".join(f"{x:>7}" for x in idents)
print(header)
for i, row in enumerate(nav.dist):
    print(f"{idents[i]:<{w}}" + "".join(f"{v:>7}" for v in row))

# Worked examples.
print("\nPath distance from a few sample agent positions:")
for floor, x in [(1, 0), (1, 184), (2, 80), (4, 272), (4, 16), (3, 144)]:
    print(f"  agent at floor {floor}, x={x}:")
    for fid in (1, 2, 3, 4):
        d = nav.path_distance_from_agent(floor, x, f"F{fid}")
        print(f"    to F{fid}: {d}")
    d = nav.path_distance_from_agent(floor, x, "princess")
    print(f"    to princess: {d}")

pil.save(OUT)
print(f"\nWrote {OUT}")
