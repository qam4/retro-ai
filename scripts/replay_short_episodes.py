#!/usr/bin/env python3
"""Find short/failure episodes in a run and replay their starting state.

Workflow:
  1. Read ``episodes.csv``, pick episodes matching a filter (short length,
     specific start/reached level).
  2. For each, look up the starting save-state by matching the logged
     ``start_state_hash`` against a seed archive's hashes.
  3. Call :mod:`scripts.play_state` on each match to dump frames.

This only finds episodes whose start state is in the provided seed
archive. Episodes where the curriculum or frontier produced a new
state mid-training won't match — we don't persist those.

Usage
-----

Replay the first 3 short-failure CP2-start episodes from
segment_2to3_v2, using v9 seeds::

    python scripts/replay_short_episodes.py \\
        output/.../segment_2to3_v2/episodes.csv \\
        --seeds output/mo5/yeti/seeds/v9_checkpoints.pkl \\
        --start-level 2 \\
        --max-length 40 \\
        --limit 3 \\
        --out debug/short_cp2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import subprocess
import sys
from typing import Dict, List


def _hash(state_bytes: bytes) -> str:
    return hashlib.blake2b(state_bytes, digest_size=8).hexdigest()


def _load_seeds_by_hash(path: str) -> Dict[str, bytes]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    states: List[bytes] = []
    if isinstance(data, dict) and "checkpoints" in data:
        for bucket in data["checkpoints"]:
            for s in bucket:
                states.append(bytes(s))
    elif isinstance(data, dict):
        for entry in data.values():
            states.append(bytes(entry["state"]))
    else:
        raise SystemExit(f"unknown seed archive format: {path}")
    return {_hash(s): s for s in states}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("episodes_csv")
    p.add_argument("--seeds", required=True)
    p.add_argument("--start-level", type=int, default=None)
    p.add_argument("--reached-level-max", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--out", required=True, help="Base output directory.")
    p.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Frames to replay per state (default 120).",
    )
    p.add_argument(
        "--profile",
        default="yeti_fruit",
        help="Game profile for the playback env.",
    )
    args = p.parse_args()

    with open(args.episodes_csv) as f:
        rows = list(csv.DictReader(f))

    seeds_by_hash = _load_seeds_by_hash(args.seeds)
    print(f"Loaded {len(seeds_by_hash)} distinct seed hashes from {args.seeds}")

    def matches(row: Dict[str, str]) -> bool:
        if args.start_level is not None and int(row["start_level"]) != args.start_level:
            return False
        if (
            args.reached_level_max is not None
            and int(row["reached_level"]) > args.reached_level_max
        ):
            return False
        if args.max_length is not None and int(row["length"]) > args.max_length:
            return False
        return True

    candidates = [r for r in rows if matches(r)]
    print(
        f"{len(candidates)} episodes matched filters "
        f"(start_level={args.start_level} reached_level_max={args.reached_level_max} "
        f"max_length={args.max_length}). Taking first {args.limit}."
    )

    os.makedirs(args.out, exist_ok=True)
    hits = 0
    for idx, row in enumerate(candidates):
        if hits >= args.limit:
            break
        h = row["start_state_hash"]
        if h not in seeds_by_hash:
            continue
        # Write the state to a tiny pickle so play_state.py can load it.
        state_path = os.path.join(args.out, f"episode_{idx}_state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(seeds_by_hash[h], f)
        out_dir = os.path.join(args.out, f"episode_{idx}_frames")
        print(
            f"\n  episode #{idx}: start={row['start_level']} reached="
            f"{row['reached_level']} length={row['length']} end={row['end_reason']}"
        )
        print(f"    hash={h}  replay_dir={out_dir}")
        subprocess.run(
            [
                sys.executable,
                "scripts/play_state.py",
                state_path,
                "--profile",
                args.profile,
                "--frames",
                str(args.frames),
                "--out",
                out_dir,
            ],
            check=True,
        )
        hits += 1

    print(f"\nReplayed {hits} episodes.")


if __name__ == "__main__":
    main()
