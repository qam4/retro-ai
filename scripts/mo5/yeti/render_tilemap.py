#!/usr/bin/env python3
"""Render the level tilemap straight from RAM (not the video buffer).

Calibrated against level 1's 4 known fruit tiles, the screen tilemap is a
40x25 grid of 1-byte tile-ids in user RAM, row-major, base 0x2C27:

    tile(col, row) = RAM[0x2C27 + row*40 + col]

This reads that grid from a RAM snapshot (.npy from find_map_in_ram.py) or a
live save-state, prints an occupancy view (empty vs solid) and a tile-id
view, and lists the distinct tile-ids with counts.
"""
from __future__ import annotations
import argparse
import numpy as np
from collections import Counter

BASE = 0x2C27
W, H = 40, 25


def load_ram(args) -> np.ndarray:
    if args.npy:
        return np.load(args.npy)
    from go_explore import make_env

    env = make_env("yeti_fruit")
    env.reset()
    if args.state:
        env.load_state(open(args.state, "rb").read())
    for _ in range(args.settle):
        env.step([0, 0, 0])
    return np.frombuffer(bytes(env._interface.read_ram()), dtype=np.uint8).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default=None, help="RAM snapshot .npy")
    ap.add_argument("--state", default=None, help="live save-state (if no --npy)")
    ap.add_argument("--settle", type=int, default=6)
    ap.add_argument("--base", type=lambda s: int(s, 0), default=BASE)
    args = ap.parse_args()

    ram = load_ram(args)
    grid = np.array(
        [[ram[args.base + r * W + c] for c in range(W)] for r in range(H)],
        dtype=np.uint8,
    )

    vals = Counter(int(v) for v in grid.flatten())
    print(f"base=0x{args.base:04X}  grid {W}x{H}")
    print("distinct tile-ids (value: count):")
    for v, n in sorted(vals.items(), key=lambda kv: -kv[1]):
        print(f"  {v:3d} (0x{v:02X}): {n}")

    # Occupancy view: 0 -> '.', nonzero -> '#'
    print("\n=== occupancy (0=empty '.', nonzero='#') ===")
    print("    " + "".join(str((i // 10) % 10) for i in range(W)))
    print("    " + "".join(str(i % 10) for i in range(W)))
    for r in range(H):
        line = "".join("." if grid[r, c] == 0 else "#" for c in range(W))
        print(f"{r:2d}  {line}")

    # Tile-id view: map each distinct value to a stable char.
    palette = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@%&*+=<>/\\"
    )
    charmap = {0: " "}
    for i, (v, _) in enumerate(sorted(vals.items())):
        if v == 0:
            continue
        charmap[v] = palette[(i - (0 in vals)) % len(palette)] if v != 0 else " "
    # simpler: assign by sorted nonzero value order
    charmap = {0: " "}
    nz = [v for v in sorted(vals) if v != 0]
    for i, v in enumerate(nz):
        charmap[v] = palette[i % len(palette)]
    print("\n=== tile-ids ===")
    print("    " + "".join(str((i // 10) % 10) for i in range(W)))
    print("    " + "".join(str(i % 10) for i in range(W)))
    for r in range(H):
        print(f"{r:2d}  " + "".join(charmap[int(grid[r, c])] for c in range(W)))
    print("\nchar legend (char = tile-id):")
    for v in nz:
        print(f"  '{charmap[v]}' = {v} (0x{v:02X})")


if __name__ == "__main__":
    main()
