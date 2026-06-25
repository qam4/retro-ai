#!/usr/bin/env python3
"""Locate a static level-layout ("map") table in RAM.

Hypothesis (user): level 2's geometry (floors/ladders/gaps/fruits/princess)
is stored as an 8x8-tile map somewhere in RAM, not just rendered to the
screen. If so we can READ the whole map instead of driving the agent around
to discover it (which Go-Explore can't do past floor 2's gap jumps).

Strategy -- diff RAM across three conditions:
  L1   : level 1 start (env.reset)
  L2a  : level 2 start (load level2_start.sav)
  L2b  : level 2 after the player moves/descends a bit

A static map table is:
  - DIFFERENT between L1 and L2a   (level-specific geometry)
  - SAME between L2a and L2b       (doesn't change as player/sprites move)

So candidate = (L1 != L2a) AND (L2a == L2b). We print the contiguous byte
ranges that satisfy this, which are the prime suspects for the map table.

Also dumps the three RAM snapshots to <out>/ for offline inspection.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
from go_explore import make_env, read_state

X, Y, LIVES = 11090, 11089, 11095


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="output/mo5/yeti/level2/level2_start.sav")
    ap.add_argument("--out", default="output/mo5/yeti/level2/ram")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = make_env("yeti_fruit")
    iface = env._interface

    def ram() -> np.ndarray:
        return np.frombuffer(bytes(iface.read_ram()), dtype=np.uint8).copy()

    def rb(a):
        return iface.read_ram_byte(a)

    # L1: level 1 start
    env.reset()
    for _ in range(10):
        env.step([0, 0, 0])
    ram_l1 = ram()
    print(f"RAM size: {len(ram_l1)} bytes (0x{len(ram_l1):04X})")
    print(f"L1 (level1 start): player x={rb(X)} y={rb(Y)} lives={rb(LIVES)}")

    # L2a: level 2 start
    env.reset()
    env.load_state(open(args.state, "rb").read())
    for _ in range(10):
        env.step([0, 0, 0])
    ram_l2a = ram()
    print(f"L2a (level2 start): player x={rb(X)} y={rb(Y)} lives={rb(LIVES)}")

    # L2b: move within level 2 (walk right a while, then climb down).
    # We don't care if we die at the end -- we snapshot before that. Take
    # the snapshot right after some real movement so the player pos and
    # scroll have changed but we're still in level 2.
    for _ in range(30):
        env.step([0, 1, 0])  # walk right
    for _ in range(40):
        env.step([2, 0, 0])  # try to climb/go down
    ram_l2b = ram()
    print(f"L2b (level2 moved): player x={rb(X)} y={rb(Y)} lives={rb(LIVES)}")

    n = min(len(ram_l1), len(ram_l2a), len(ram_l2b))
    l1, l2a, l2b = ram_l1[:n], ram_l2a[:n], ram_l2b[:n]

    diff_levels = l1 != l2a  # level-specific
    stable_in_l2 = l2a == l2b  # unchanged as player moves
    changed_in_l2 = l2a != l2b  # sprites / scroll / player
    candidate = diff_levels & stable_in_l2

    print(f"\nbytes differing L1 vs L2a (level-specific): {diff_levels.sum()}")
    print(f"bytes stable within L2 (L2a==L2b):          {stable_in_l2.sum()}")
    print(f"bytes changed within L2 (sprites/scroll):   {changed_in_l2.sum()}")
    print(f"MAP CANDIDATES (level-specific & static):   {candidate.sum()}")

    # Contiguous candidate ranges (merge gaps <= 4 bytes).
    idx = np.where(candidate)[0]
    print("\ncontiguous candidate ranges (addr_start..addr_end  len):")
    if len(idx):
        runs = []
        start = prev = idx[0]
        for i in idx[1:]:
            if i - prev <= 4:
                prev = i
            else:
                runs.append((start, prev))
                start = prev = i
        runs.append((start, prev))
        runs.sort(key=lambda r: (r[1] - r[0]), reverse=True)
        for s, e in runs[:40]:
            ln = e - s + 1
            print(f"  0x{s:04X}..0x{e:04X}  ({s}..{e})  len={ln}")
        print(f"\ntotal runs: {len(runs)}")

    np.save(os.path.join(args.out, "ram_l1.npy"), l1)
    np.save(os.path.join(args.out, "ram_l2a.npy"), l2a)
    np.save(os.path.join(args.out, "ram_l2b.npy"), l2b)
    print(f"\nsaved snapshots to {args.out}/ram_l1.npy ram_l2a.npy ram_l2b.npy")


if __name__ == "__main__":
    main()
