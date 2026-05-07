#!/usr/bin/env python3
"""Per-cell noop-probe videos, specifically for cells in higher
``first_done_frame`` buckets where snapshots aren't enough (balls may
be mid-flight and invisible in a still frame).

Reads the first_done_frame result from a CSV produced by
``probe_archive_done_frames.py``, filters to a user-specified probe
bucket, and writes one 50 fps MP4 per cell.

Videos cover load + settle + probe frames, so you see the state from
load through to death (or the full probe window if it survived).

Usage:
  env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
    python3 scripts/dump_probe_videos.py \\
      --archive output/mo5/yeti/go_explore_v9/archive.pkl \\
      --csv debug/probe_v9_sweep_500.csv \\
      --out debug/v9_probe_videos \\
      --bucket survived  # or: 201_300, 301_plus, 121_200, etc.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle

import imageio.v2 as imageio
import numpy as np

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


def _in_bucket(first_done: int, bucket: str) -> bool:
    if bucket == "survived":
        return first_done == -1
    if bucket == "0_10":
        return 0 < first_done <= 10
    if bucket == "11_20":
        return 10 < first_done <= 20
    if bucket == "21_30":
        return 20 < first_done <= 30
    if bucket == "31_60":
        return 30 < first_done <= 60
    if bucket == "61_120":
        return 60 < first_done <= 120
    if bucket == "121_200":
        return 120 < first_done <= 200
    if bucket == "201_300":
        return 200 < first_done <= 300
    if bucket == "301_plus":
        return first_done > 300
    raise ValueError(f"unknown bucket: {bucket}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument(
        "--csv", required=True, help="Output of probe_archive_done_frames.py"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--settle", type=int, default=5)
    p.add_argument("--max-probe-frames", type=int, default=500)
    p.add_argument(
        "--tail-frames",
        type=int,
        default=50,
        help="Frames to keep running after done for context.",
    )
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="Limit cells per bucket (0 = all).",
    )
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    matching = [
        int(r["cell_idx"])
        for r in rows
        if _in_bucket(int(r["first_done_frame"]), args.bucket)
    ]
    if not matching:
        print(f"No cells in bucket '{args.bucket}'")
        return
    print(f"Bucket {args.bucket}: {len(matching)} cells")
    if args.max_cells > 0:
        matching = matching[: args.max_cells]

    with open(args.archive, "rb") as f:
        archive = pickle.load(f)
    all_cells = []
    for cell_key, entry in archive.items():
        cp = _cp_for_cell(cell_key)
        if cp is None:
            continue
        all_cells.append((cell_key, cp, bytes(entry["state"])))
    want = set(matching)
    targets = [
        (i, cp, state) for i, (cell_key, cp, state) in enumerate(all_cells) if i in want
    ]

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

    os.makedirs(args.out, exist_ok=True)
    by_idx = {int(r["cell_idx"]): int(r["first_done_frame"]) for r in rows}

    for i, cp, state in targets:
        first_done = by_idx[i]
        base._interface.load_state(state)
        frames = []
        last_obs = None
        for _ in range(args.settle):
            obs, r, d, t, info = base.step([0, 0, 0])
            frames.append(np.asarray(obs, dtype=np.uint8).copy())
            last_obs = obs

        if first_done > 0:
            total = first_done + args.tail_frames
        else:
            total = args.max_probe_frames

        for f in range(total):
            obs, r, done, trunc, info = base.step([0, 0, 0])
            frames.append(np.asarray(obs, dtype=np.uint8).copy())

        tag = f"done{first_done:03d}" if first_done > 0 else "SURVIVED"
        out_name = f"cp{cp}_i{i:04d}_{tag}.mp4"
        out_path = os.path.join(args.out, out_name)
        imageio.mimsave(out_path, frames, fps=50)
        print(f"  wrote {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
