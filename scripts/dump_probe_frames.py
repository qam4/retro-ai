#!/usr/bin/env python3
"""Dump per-cell load-snapshot + done-frame snapshot, bucketed by when
done fires under noops. Lets a human eyeball the states at different
validator-cutoff boundaries and pick a threshold.

For each cell in the archive:
  1. load + settle_frames
  2. save the 320x200 RGB framebuffer as frame_00_loaded.png
  3. step noops up to probe_frames, watching for done
  4. if done fires at frame F, save frame as frame_F_done.png
  5. else save frame at probe_frames as frame_SURVIVED.png

Output layout:
  out_dir/
    done_000_010/  # cells where done fires within first 10 frames
      cp2_i042_done07.png      # load snapshot
      cp2_i042_done07_diedon.png  # the frame done fired
      cp3_i123_done04.png
      ...
    done_011_020/
    done_021_030/
    done_031_060/
    done_061_120/
    survived/

Usage:
  env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
    python3 scripts/dump_probe_frames.py \\
      --archive output/mo5/yeti/go_explore_v9/archive.pkl \\
      --out debug/v9_probe_frames \\
      --settle 5 --probe 120
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from PIL import Image

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


def _bucket(first_done: int) -> str:
    if first_done == -1:
        return "survived"
    if first_done <= 10:
        return "done_000_010"
    if first_done <= 20:
        return "done_011_020"
    if first_done <= 30:
        return "done_021_030"
    if first_done <= 60:
        return "done_031_060"
    if first_done <= 120:
        return "done_061_120"
    if first_done <= 200:
        return "done_121_200"
    if first_done <= 300:
        return "done_201_300"
    return "done_301_plus"


def _frame_from_obs(obs) -> np.ndarray:
    """Return RGB uint8 ndarray from a BaseEnv observation."""
    a = np.asarray(obs, dtype=np.uint8)
    return a.copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--settle", type=int, default=5)
    p.add_argument("--probe", type=int, default=120)
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument(
        "--max-per-bucket",
        type=int,
        default=0,
        help="If >0, only dump this many cells per bucket (for quick review).",
    )
    args = p.parse_args()

    with open(args.archive, "rb") as f:
        archive = pickle.load(f)
    cells = []
    for cell_key, entry in archive.items():
        cp = _cp_for_cell(cell_key)
        if cp is None:
            continue
        cells.append((cell_key, cp, bytes(entry["state"])))
    print(f"archive: {args.archive} ({len(cells)} cells)")

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
    per_bucket_count: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}

    for i, (cell_key, cp, state) in enumerate(cells):
        base._interface.load_state(state)
        last_obs = None
        for _ in range(args.settle):
            obs, r, d, t, info = base.step([0, 0, 0])
            last_obs = obs

        loaded_frame = _frame_from_obs(last_obs)

        first_done = -1
        done_frame = None
        for f in range(1, args.probe + 1):
            obs, r, done, trunc, info = base.step([0, 0, 0])
            last_obs = obs
            if done:
                first_done = f
                done_frame = _frame_from_obs(obs)
                break
        if first_done == -1:
            done_frame = _frame_from_obs(last_obs)

        bucket = _bucket(first_done)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if args.max_per_bucket > 0:
            n_so_far = per_bucket_count.get(bucket, 0)
            if n_so_far >= args.max_per_bucket:
                continue
            per_bucket_count[bucket] = n_so_far + 1

        bucket_dir = os.path.join(args.out, bucket)
        os.makedirs(bucket_dir, exist_ok=True)

        tag_done = f"done{first_done:03d}" if first_done > 0 else "NEVER"
        fname_base = f"cp{cp}_i{i:03d}_{tag_done}"

        Image.fromarray(loaded_frame).save(
            os.path.join(bucket_dir, f"{fname_base}_loaded.png")
        )
        if first_done > 0:
            Image.fromarray(done_frame).save(
                os.path.join(bucket_dir, f"{fname_base}_diedon.png")
            )

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cells)}", flush=True)

    print("\nbucket counts:")
    for bucket in [
        "done_000_010",
        "done_011_020",
        "done_021_030",
        "done_031_060",
        "done_061_120",
        "done_121_200",
        "done_201_300",
        "done_301_plus",
        "survived",
    ]:
        n = bucket_counts.get(bucket, 0)
        print(f"  {bucket}: {n}")


if __name__ == "__main__":
    main()
