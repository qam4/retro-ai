#!/usr/bin/env python3
"""Ground-truth platform extents, read from the screen tilemap in RAM.

Every span in this project so far was measured by WALKING to an edge, which
conflates the collision mechanic with wherever the walk happened to stop. The
tilemap is the actual geometry: a 40x25 grid of tile ids at 0x2C27, where 5-8
are floor (body / alt-body / LEFT-END / RIGHT-END) and pixel_x = col*8.

So the declared `Platform(floor, y, x_min, x_max)` extents can be checked
directly against the tiles they were hand-transcribed from, and the question
"do floors 10 and 11 really have different left edges" gets a yes/no instead of
an inference from two walks that stopped in different places.

A platform's visible surface row = (standing_y + 18) // 8.

Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import EnvConfig  # noqa: E402
from retro_ai.training.yeti_map import get_level_map  # noqa: E402

BASE = 0x2C27
W = 40
# 5-8 are floor (5/6 body, 7 LEFT-end, 8 RIGHT-end). 3 and 4 are the
# LADDER-THROUGH-FLOOR join -- a column where a ladder passes through a platform, so the
# platform IS there and the agent walks across it. Counting only 5-8 splits every
# platform at its ladder and makes eight of L4's 24 extents look over-declared; each
# phantom gap sits exactly on a known ladder x (floor 2 -> Lfruit 208, floor 11 ->
# Lclimb3 272, ...). With 3/4 included, all 24 declared extents match EXACTLY.
WALKABLE_IDS = {3, 4, 5, 6, 7, 8}
SURFACE_DY = 18


def runs(cols):
    """Contiguous column runs in a sorted column list."""
    out = []
    for c in sorted(cols):
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [tuple(r) for r in out]


def main() -> int:
    cfg = EnvConfig(
        profile="yeti_fruit_level4",
        action_mode="joystick",
        max_steps=10**6,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level4", cfg)
    ifc = stack.base._interface
    stack.gym.reset()
    ifc.load_state(Path("output/mo5/yeti/level4/level4_start.sav").read_bytes())
    stack.preprocessed.notify_state_loaded()
    stack.gym.step([0, 0, 0])
    # read_ram() wraps to a 0-d array under np.asarray; go via bytes.
    ram = np.frombuffer(bytes(ifc.read_ram()), dtype=np.uint8)

    def row_tiles(row):
        off = BASE + row * W
        return [int(ram[off + c]) for c in range(W)]

    lvl = get_level_map(4)
    print(
        "floor  standing_y  row   declared        tiles (px)        ends      verdict"
    )
    for p in sorted(lvl.platforms, key=lambda p: p.floor):
        row = (p.y + SURFACE_DY) // 8
        ids = row_tiles(row)
        cols = [c for c, t in enumerate(ids) if t in WALKABLE_IDS]
        rr = runs(cols)
        # the run that overlaps the declared extent
        best, ov = None, 0
        for c0, c1 in rr:
            a, b = max(c0 * 8, p.x_min), min((c1 + 1) * 8, p.x_max)
            if b - a > ov:
                best, ov = (c0, c1), b - a
        if best is None:
            print(
                f"{p.floor:>5} {p.y:>11} {row:>4}   {str((p.x_min,p.x_max)):<14} "
                f"{'no floor tiles in this row':<18}"
            )
            continue
        c0, c1 = best
        tx0, tx1 = c0 * 8, (c1 + 1) * 8
        ends = f"{ids[c0]}..{ids[c1]}"
        ok = (
            "MATCH"
            if (tx0, tx1) == (p.x_min, p.x_max)
            else (
                "declared WIDER"
                if p.x_min < tx0 or p.x_max > tx1
                else "declared NARROWER"
            )
        )
        print(
            f"{p.floor:>5} {p.y:>11} {row:>4}   {str((p.x_min,p.x_max)):<14} "
            f"{str((tx0,tx1)):<18} {ends:<9} {ok}"
        )
        if len(rr) > 1:
            print(
                f"{'':>34}   other runs on this row: "
                f"{[(a*8,(b+1)*8) for a,b in rr if (a,b) != best]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
