#!/usr/bin/env python3
"""Inspect the entity/sprite table (player, princess, goats) in user RAM.

The player position is at RAM 0x2B51 (Y) / 0x2B52 (X). The level-diff found
a level-specific cluster around 0x2B00-0x2B74, so entity records likely live
there. We print that region for a RAM snapshot and flag bytes whose value
matches a target coordinate (e.g. the princess at bottom-right), in several
encodings, to find the princess record.

Player X is in 4px units (x_px = X*4); Y is in px. Princess bottom-right is
~ x_px 288-304 (X ~ 72-76), y_px ~ 168-176 (bottom floor).
"""
from __future__ import annotations
import argparse
import numpy as np

PLAYER_Y, PLAYER_X = 0x2B51, 0x2B52


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="output/mo5/yeti/level2/ram/ram_l2a.npy")
    ap.add_argument("--lo", type=lambda s: int(s, 0), default=0x2B00)
    ap.add_argument("--hi", type=lambda s: int(s, 0), default=0x2B90)
    args = ap.parse_args()

    ram = np.load(args.npy)
    py, px = int(ram[PLAYER_Y]), int(ram[PLAYER_X])
    print(f"player: X(0x2B52)={px} (x_px={px*4})  Y(0x2B51)={py} (y_px={py})")

    # Candidate princess coords (bottom-right) in various encodings.
    targets = {
        "X*4=288 -> 72": 72,
        "X*4=296 -> 74": 74,
        "X*4=304 -> 76": 76,
        "x_px 288": 288 & 0xFF,
        "x_px 304": 304 & 0xFF,
        "y_px 168": 168,
        "y_px 176": 176,
        "y_px 160": 160,
        "col 36": 36,
        "col 38": 38,
        "row 21": 21,
        "row 23": 23,
    }
    print(f"\nregion 0x{args.lo:04X}..0x{args.hi:04X}:")
    for a in range(args.lo, args.hi + 1):
        v = int(ram[a])
        tags = [name for name, t in targets.items() if v == t]
        mark = ""
        if a == PLAYER_X:
            mark = "  <== player X"
        elif a == PLAYER_Y:
            mark = "  <== player Y"
        elif tags:
            mark = "  <-- " + " | ".join(tags)
        print(f"  0x{a:04X} ({a}) = {v:3d} (0x{v:02X}){mark}")


if __name__ == "__main__":
    main()
