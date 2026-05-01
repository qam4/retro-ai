#!/usr/bin/env python3
"""Replay ``episodes.csv`` into a TensorBoard event file.

One-shot tool for runs that finished before ``EpisodeMetricsCallback``
was added (or any time you want to re-aggregate an existing csv without
re-running training). Reads the per-episode log, slides a window across
the ``global_step`` axis, and writes the same tags the live callback
produces into a new TB log dir.

Usage
-----

::

    python scripts/episodes_to_tb.py \\
        output/.../ablation_C_mix/episodes.csv \\
        --out output/.../ablation_C_mix/tb_replay \\
        --window 100000

Then point TensorBoard at the parent directory and pick the new run.
The live callback and this tool share a single aggregator module
(:mod:`retro_ai.training.episode_metrics`) so their tag schemes cannot
drift.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence


def _load(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _as_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def _bucketize(
    rows: Sequence[Dict[str, str]], step_points: Sequence[int], window: int
) -> List[List[Dict[str, str]]]:
    """For each point P, return rows with ``global_step`` in ``(P - window, P]``."""
    rows_sorted = sorted(rows, key=lambda r: _as_int(r["global_step"]))
    steps = [_as_int(r["global_step"]) for r in rows_sorted]
    out: List[List[Dict[str, str]]] = []
    for point in step_points:
        lo = point - window
        bucket = [r for r, s in zip(rows_sorted, steps) if lo < s <= point]
        out.append(bucket)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv_path", help="Path to episodes.csv.")
    p.add_argument(
        "--out",
        default=None,
        help="Output TB log dir. Defaults to <csv_dir>/tb_replay.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=100_000,
        help="Sliding window size in global_steps (default 100k).",
    )
    p.add_argument(
        "--points",
        type=int,
        default=50,
        help="Number of x-axis points to emit (default 50).",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=5,
        help="Skip a tag for a window if fewer than this many matching episodes.",
    )
    args = p.parse_args()

    # Delay imports so --help stays cheap and torch only loads when needed.
    from retro_ai.training.episode_metrics import aggregate, infer_max_level
    from torch.utils.tensorboard import SummaryWriter  # type: ignore

    rows = _load(args.csv_path)
    if not rows:
        print("No rows in csv; nothing to emit.")
        return

    steps = [_as_int(r["global_step"]) for r in rows]
    lo, hi = min(steps), max(steps)
    if hi == lo:
        step_points = [hi]
    else:
        stride = (hi - lo) / max(1, args.points - 1)
        step_points = [int(lo + i * stride) for i in range(args.points)]

    max_level = infer_max_level(rows)
    out_dir = args.out or os.path.join(os.path.dirname(args.csv_path), "tb_replay")
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir)

    buckets = _bucketize(rows, step_points, args.window)
    for step, bucket in zip(step_points, buckets):
        metrics = aggregate(bucket, max_level=max_level, min_n=args.min_n)
        for tag, value in metrics.items():
            writer.add_scalar(tag, value, step)

    writer.flush()
    writer.close()
    print(f"Wrote {len(step_points)} windows to {out_dir}")
    print(f"  step range: {lo} .. {hi}")
    print(f"  window: {args.window:,}")
    print(f"  point TensorBoard at: {os.path.dirname(out_dir) or '.'}")


if __name__ == "__main__":
    main()
