#!/usr/bin/env python3
"""Does the shaping gradient keep decreasing up the A1->A5 ascent (incentive
all the way to fruit/princess), or does it flatten past SN3?"""
from __future__ import annotations

from retro_ai.training.yeti_map import build_navigation_map, get_level_map

nav = build_navigation_map(3)
m = get_level_map(3)
# platform centres (floor -> standing_y, ram range) from LEVEL3
pos = [
    ("SN3 (f10,x64)", 10, 64 * 4 + 8),
    ("A1  (f11,x43)", 11, 43 * 4 + 8),
    ("A2  (f12,x37)", 12, 37 * 4 + 8),
    ("A3  (f13,x30) compressor", 13, 30 * 4 + 8),
    ("A4  (f14,x17) FRUIT", 14, 17 * 4 + 8),
    ("A5  (f15,x5)", 15, 5 * 4 + 8),
]
for tgt in ("princess", "Lprincess_top"):
    print(f"=== path-distance to {tgt} ===")
    for label, f, px in pos:
        print(f"  {label:26s} d={nav.path_distance_from_agent(f, px, tgt)}")
