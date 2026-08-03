#!/usr/bin/env python3
"""Render the L3 start frame with the RAW per-cell tilemap overlaid.

Draws every solid tile (floor / ladder / sprite) from RAM so gaps and small
platforms the lossy extractor missed are visible, plus the agent standing-y
line (tile_row*8 - 18) per row. Lets a human verify the true L3 geometry
before we rebuild the LEVEL3 map.

pixel_x = col*8 = ram_x*4 ; pixel_y = row*8 ; agent standing-y = row*8 - 18.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

OUT = "debug/l3_annotated_map.png"
STATE = "output/mo5/yeti/level3/level3_start.sav"
SCALE = 4
OFF = 18
BASE = 0x2C27
W, H = 40, 25
FRUIT_TILE = (7, 8)  # (col,row) real L3 fruit, presence 11630


def build():
    cfg = EnvConfig(
        profile="yeti_fruit_level3", action_mode="joystick", max_steps=400,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    ifc = stack.base._interface
    ifc.load_state(open(STATE, "rb").read())
    stack.preprocessed.notify_state_loaded()
    for _ in range(30):
        stack.gym.step([0, 0, 0])
    rgb = np.asarray(stack.base._last_raw_obs, dtype=np.uint8)
    grid = [[ifc.read_ram_byte(BASE + r * W + c) for c in range(W)] for r in range(H)]
    return rgb, grid, ifc


def main():
    rgb, grid, ifc = build()
    img = Image.fromarray(rgb).resize(
        (rgb.shape[1] * SCALE, rgb.shape[0] * SCALE), Image.NEAREST
    )
    dr = ImageDraw.Draw(img)

    def R(c, r):  # tile rect in scaled px
        return [c * 8 * SCALE, r * 8 * SCALE, (c + 1) * 8 * SCALE, (r + 1) * 8 * SCALE]

    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == 0:
                continue
            if 5 <= v <= 8:  # floor
                dr.rectangle(R(c, r), outline=(0, 255, 0), width=2)
            elif 1 <= v <= 4:  # ladder
                dr.rectangle(R(c, r), outline=(0, 200, 255), width=2)
            elif v >= 10:  # sprite
                dr.rectangle(R(c, r), outline=(255, 90, 0), width=1)
                dr.text((c * 8 * SCALE + 1, r * 8 * SCALE + 1), str(v), fill=(255, 90, 0))

    # Label each contiguous PLATFORM run on the platform itself. A run spans
    # floor tiles AND embedded ladder tiles (the top of a ladder is walkable,
    # so a ladder passing through a platform does NOT split it) but breaks on
    # empty tiles (un-crossable gaps). A run must contain >=1 floor tile (a
    # pure vertical ladder segment is not a platform).
    def is_solid(v):
        return 1 <= v <= 8  # floor (5-8) or ladder (1-4)

    for r in range(H):
        c = 0
        while c < W:
            if is_solid(grid[r][c]):
                s = c
                while c < W and is_solid(grid[r][c]):
                    c += 1
                e = c - 1
                if any(5 <= grid[r][cc] <= 8 for cc in range(s, e + 1)):
                    sy = r * 8 - OFF
                    dr.text(
                        (s * 8 * SCALE + 1, r * 8 * SCALE - 11),
                        f"y{sy} x{s*2}-{e*2+1}",
                        fill=(230, 230, 0),
                    )
            else:
                c += 1

    # real fruit (16x16 = 2x2 tiles from the top-left tile)
    fc, fr = FRUIT_TILE
    dr.rectangle(
        [fc * 8 * SCALE, fr * 8 * SCALE, (fc + 2) * 8 * SCALE, (fr + 2) * 8 * SCALE],
        outline=(255, 255, 0), width=3,
    )
    dr.text((fc * 8 * SCALE + 2, fr * 8 * SCALE - 12), "FRUIT", fill=(255, 255, 0))

    # princess entity (Y@0x2B00, X@0x2B01 in 4px units). The X byte is the
    # sprite's RIGHT edge, so the 16x16 sprite occupies [x*4 - 16, x*4].
    py = ifc.read_ram_byte(0x2B00)
    px_right = ifc.read_ram_byte(0x2B01) * 4
    px = px_right - 16
    dr.rectangle(
        [px * SCALE, py * SCALE, (px + 16) * SCALE, (py + 16) * SCALE],
        outline=(255, 0, 255), width=2,
    )
    dr.text((px * SCALE, py * SCALE - 12), "PRINCESS", fill=(255, 0, 255))

    # agent 16x16
    ax, ay = ifc.read_ram_byte(11090) * 4, ifc.read_ram_byte(11089)
    dr.rectangle(
        [ax * SCALE, ay * SCALE, (ax + 16) * SCALE, (ay + 16) * SCALE],
        outline=(255, 0, 0), width=2,
    )
    dr.text((ax * SCALE, ay * SCALE - 12), "AGENT", fill=(255, 0, 0))

    _draw_graph(dr, SCALE)

    os.makedirs("debug", exist_ok=True)
    img.save(OUT)
    print(f"saved {OUT} ({img.size[0]}x{img.size[1]})")


# --- proposed LEVEL3 nav graph (the structure to be encoded) ---------------
# platform id -> (standing_y, x_min_ram, x_max_ram, role)
PLATFORMS = {
    "START": (166, 0, 9, "start ledge (raised, bottom-left)"),
    "BOTTOM": (182, 0, 79, "bottom floor (full width, screen bottom)"),
    "STEP": (158, 12, 15, "step up from start"),
    "2LAD": (150, 18, 27, "2-ladder platform"),
    "GOAT": (94, 18, 27, "goat platform"),
    "ELAND": (158, 40, 47, "escalator landing"),
    "BR": (158, 56, 79, "bottom-right"),
    "SN1": (134, 52, 77, "snowball 1"),
    "SN2": (110, 54, 79, "snowball 2"),
    "SN3": (86, 50, 77, "snowball 3 (top)"),
    "A1": (78, 42, 45, "ascend 1"),
    "A2": (70, 36, 39, "ascend 2"),
    "A3": (62, 24, 33, "ascend 3 (compressor)"),
    "A4": (62, 14, 19, "ascend 4 (FRUIT)"),
    "A5": (54, 0, 11, "ascend 5"),
    "PRIN": (30, 0, 17, "princess platform"),
}
# Ladders: name -> (x_ram, top_platform, bot_platform). The graph NODES are
# the ladder endpoints, located at (x_ram, platform_standing_y).
LADDERS = {
    "Lgoat_a": (18, "GOAT", "2LAD"),
    "Lgoat_b": (24, "GOAT", "2LAD"),
    "Ldown": (42, "ELAND", "BOTTOM"),  # escalator landing down to bottom floor
    "Lsc1": (60, "BR", "BOTTOM"),      # 1st snowball climb: bottom -> BR
    "Lsc2": (70, "SN1", "BR"),         # 2nd
    "Lsc3": (58, "SN2", "SN1"),        # 3rd
    "Lsc4": (70, "SN3", "SN2"),        # 4th (top)
    "Lprincess": (6, "PRIN", "A5"),
}
# escalator special link (learned; INF in shaping): from goat platform right
# edge to the escalator landing.
ESCALATOR = ("GOAT", "ELAND")
# jump links between platforms (sparse/learned) drawn edge-to-edge.
JUMPS = [
    ("START", "STEP"), ("STEP", "2LAD"),
    ("SN3", "A1"), ("A1", "A2"), ("A2", "A3"), ("A3", "A4"), ("A4", "A5"),
]
# mandatory reward waypoints = ladder ARRIVAL ends (climb->top, down->bottom).
# name -> which end. Drawn as rings at that ladder endpoint.
WP_ARRIVAL = {
    "Lgoat_a": "top", "Lgoat_b": "top", "Ldown": "bot",
    "Lsc1": "top", "Lsc2": "top", "Lsc3": "top", "Lsc4": "top", "Lprincess": "top",
}


def _edge_pt(pid, toward_x_ram, S):
    """Point on platform pid's edge nearest toward_x_ram (for jump lines).

    Right edge is (x_max+1)*4 px (x_max is the last 4px ram cell; its far side
    is the true platform edge); left edge is x_min*4 px.
    """
    y, x0, x1, _ = PLATFORMS[pid]
    if toward_x_ram <= x0:
        xpx = x0 * 4
    elif toward_x_ram >= x1:
        xpx = (x1 + 1) * 4
    else:
        xpx = toward_x_ram * 4
    return (xpx * S, y * S)


def _draw_graph(dr, S):
    # platform extents (white line only; no center node)
    for pid, (y, x0, x1, role) in PLATFORMS.items():
        dr.line([(x0 * 4 * S, y * S), (x1 * 4 * S + 4 * S, y * S)],
                fill=(255, 255, 255), width=2)
        dr.text((x0 * 4 * S, y * S + 2), pid, fill=(200, 200, 200))
    # ladders (green) with endpoint dots at (x_ram, platform_y)
    for name, (lx, top, bot) in LADDERS.items():
        yt, yb = PLATFORMS[top][0], PLATFORMS[bot][0]
        px = lx * 4 * S
        dr.line([(px, yt * S), (px, yb * S)], fill=(0, 255, 0), width=3)
        for yy in (yt, yb):
            dr.ellipse([px - 4, yy * S - 4, px + 4, yy * S + 4], fill=(0, 255, 0))
    # jumps (orange) edge-to-edge
    for a, b in JUMPS:
        ya, xa0, xa1, _ = PLATFORMS[a]
        yb, xb0, xb1, _ = PLATFORMS[b]
        bc, ac = (xb0 + xb1) / 2, (xa0 + xa1) / 2
        dr.line([_edge_pt(a, bc, S), _edge_pt(b, ac, S)], fill=(255, 140, 0), width=2)
    # escalator special link (magenta): goat RIGHT edge -> landing LEFT edge
    ga = PLATFORMS[ESCALATOR[0]]
    la = PLATFORMS[ESCALATOR[1]]
    dr.line([((ga[2] + 1) * 4 * S, ga[0] * S), (la[1] * 4 * S, la[0] * S)],
            fill=(255, 0, 255), width=3)
    # mandatory waypoints = ladder arrival ends (yellow rings + order #)
    order = ["Lgoat_a", "Ldown", "Lsc1", "Lsc2", "Lsc3", "Lsc4", "Lprincess"]
    for i, name in enumerate(order, 1):
        lx, top, bot = LADDERS[name]
        end = WP_ARRIVAL[name]
        y = PLATFORMS[top][0] if end == "top" else PLATFORMS[bot][0]
        cx, cy = lx * 4 * S, y * S
        dr.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=(255, 255, 0), width=3)
        dr.text((cx - 3, cy - 20), str(i), fill=(255, 255, 0))


if __name__ == "__main__":
    main()
