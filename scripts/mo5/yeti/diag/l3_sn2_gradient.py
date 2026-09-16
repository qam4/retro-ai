#!/usr/bin/env python3
"""Does the reward gradient pull UP at SN2? Compute graph path-distance from
SN1/SN2/SN3 positions to the deep reward targets. If distance decreases going
up (SN1>SN2>SN3), the shaping points up and the stall is policy/navigation,
not a reward bug.
"""
from __future__ import annotations

from retro_ai.training.yeti_map import build_navigation_map, get_level_map

nav = build_navigation_map(3)
m = get_level_map(3)

# Reward waypoint idents (targets the potential sums over), deep ones first.
targets = ["Lsc4_top", "Lprincess_top", "princess"]
print("available node idents:", sorted(nav.node_by_ident.keys()))

# (label, floor, agent_pix_x)
positions = [
    ("SN1 mid  (f8,x64)", 8, 64 * 4 + 8),
    ("SN2 left (f9,x58)", 9, 58 * 4 + 8),
    ("SN2 ladder(f9,x70)", 9, 70 * 4 + 8),
    ("SN3 top  (f10,x70)", 10, 70 * 4 + 8),
    ("A1       (f11,x44)", 11, 44 * 4 + 8),
]

for tgt in targets:
    if tgt not in nav.node_by_ident:
        print(f"\n[target {tgt} not a node]")
        continue
    print(f"\n=== path-distance to {tgt} ===")
    for label, floor, px in positions:
        d = nav.path_distance_from_agent(floor, px, tgt)
        print(f"  {label:20s} d={d}")
