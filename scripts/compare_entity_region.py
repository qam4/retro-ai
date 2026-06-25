#!/usr/bin/env python3
"""Compare the global/entity region across L1, L2-start, L2-moved.

Goal: find the princess record. The princess is static within a level but
differs between levels (top-right in L1, bottom-right in L2). The player and
goats MOVE within a level (differ L2a vs L2b). So:
  princess byte = differs L1 vs L2a  AND  same L2a vs L2b
  moving entity = same/var, differs L2a vs L2b

Prints L1 / L2a / L2b side by side for the region, tagging each byte.
"""
from __future__ import annotations
import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/mo5/yeti/level2/ram")
    ap.add_argument("--lo", type=lambda s: int(s, 0), default=0x2B00)
    ap.add_argument("--hi", type=lambda s: int(s, 0), default=0x2B58)
    args = ap.parse_args()
    l1 = np.load(f"{args.dir}/ram_l1.npy")
    a = np.load(f"{args.dir}/ram_l2a.npy")
    b = np.load(f"{args.dir}/ram_l2b.npy")

    print(f"  addr      L1   L2a  L2b   notes")
    for ad in range(args.lo, args.hi + 1):
        v1, va, vb = int(l1[ad]), int(a[ad]), int(b[ad])
        notes = []
        if v1 != va and va == vb:
            notes.append("LEVEL-STATIC (princess?)")
        if va != vb:
            notes.append("moves-in-L2 (player/goat)")
        # interpret as possible x_px if it's an X in 4px units
        tag = " | ".join(notes)
        print(f"  0x{ad:04X}  {v1:4d} {va:4d} {vb:4d}   {tag}")


if __name__ == "__main__":
    main()
