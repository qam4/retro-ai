#!/usr/bin/env python3
"""Sweep probe-length: for every cell in an archive, report when done
fires under noops after load+settle.

Output columns (CSV, stdout):
  cell_idx, cp_level, first_done_frame (or -1 if survived)

Also prints a summary histogram.

Usage:
  env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
    python3 scripts/probe_archive_done_frames.py \\
      --archive output/mo5/yeti/go_explore_v9/archive.pkl \\
      --settle 5 --probe 120
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig


def _cp_for_cell(cell_key):
    if len(cell_key) < 3:
        return None
    v = cell_key[2]
    if isinstance(v, (frozenset, set, list, tuple)):
        return len(v)
    if isinstance(v, int) and 0 <= v <= 4:
        return 4 - v
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--settle", type=int, default=5)
    p.add_argument("--probe", type=int, default=120)
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--out", default=None, help="Optional CSV output path")
    args = p.parse_args()

    with open(args.archive, "rb") as f:
        archive = pickle.load(f)

    # Build cell list with cp
    cells = []
    for cell_key, entry in archive.items():
        cp = _cp_for_cell(cell_key)
        if cp is None:
            continue
        cells.append((cell_key, cp, bytes(entry["state"])))
    print(f"archive: {args.archive}")
    print(f"cells: {len(cells)}")
    print(f"by cp: {Counter(c[1] for c in cells)}")
    print(f"settle={args.settle} probe={args.probe}")
    print()

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=1_000_000,
        stall_threshold=1_000_000,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    base = stack.base
    base.reset(seed=0)

    rows = []
    for i, (cell_key, cp, state) in enumerate(cells):
        base._interface.load_state(state)
        # Settle
        for _ in range(args.settle):
            base.step([0, 0, 0])

        first_done = -1
        for f in range(1, args.probe + 1):
            obs, r, done, trunc, info = base.step([0, 0, 0])
            if done:
                first_done = f
                break
        rows.append((i, cp, first_done))

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(cells)}", flush=True)

    # Histogram
    print()
    print("=== first_done_frame histogram, by cp ===")
    for cp in sorted(set(r[1] for r in rows)):
        cp_rows = [r for r in rows if r[1] == cp]
        done_frames = [r[2] for r in cp_rows if r[2] > 0]
        survived = sum(1 for r in cp_rows if r[2] == -1)
        print(f"\nCP{cp}: n={len(cp_rows)}, survived_all={survived}")
        if not done_frames:
            continue
        # Buckets
        buckets = [
            (0, 10),
            (11, 20),
            (21, 30),
            (31, 50),
            (51, 80),
            (81, 120),
        ]
        for lo, hi in buckets:
            n = sum(1 for f in done_frames if lo <= f <= hi)
            if n:
                print(f"  frames {lo:>3}-{hi:>3}: {n:>4}")

    # Cumulative %
    print()
    print("=== cumulative reject rate vs probe length (all cps) ===")
    total = len(rows)
    for cutoff in [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120]:
        n = sum(1 for r in rows if 0 < r[2] <= cutoff)
        pct = 100 * n / total
        print(f"  probe={cutoff:>3} frames -> {n:>4}/{total} rejected ({pct:.1f}%)")

    # Per-cp cumulative
    print()
    print("=== cumulative reject rate vs probe length, per cp ===")
    print(
        f"  {'cp':>3}  {'n':>4}  "
        + "  ".join(f"{c:>4}" for c in [10, 15, 20, 30, 60, 120])
    )
    for cp in sorted(set(r[1] for r in rows)):
        cp_rows = [r for r in rows if r[1] == cp]
        n = len(cp_rows)
        vals = []
        for cutoff in [10, 15, 20, 30, 60, 120]:
            rej = sum(1 for r in cp_rows if 0 < r[2] <= cutoff)
            vals.append(f"{rej:>4}")
        print(f"  CP{cp}  {n:>4}  " + "  ".join(vals))

    if args.out:
        with open(args.out, "w") as f:
            f.write("cell_idx,cp,first_done_frame\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]}\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
