#!/usr/bin/env python3
"""One row per training run: configuration, outcome, and logged training health.

Why this exists. Decisions on this project have repeatedly been made from one or
two runs, and several were later withdrawn as noise. There are ~180 run
directories on disk carrying `run.yaml` (configuration), `curriculum_diag.csv`
(per-step reach rates including the princess) and TensorBoard scalars
(entropy, clip fraction, approx KL, explained variance). Nothing has ever read
them together, so no claim of the form "X predicts finishing a level" has been
checked against the runs that DID finish one.

This collects every run into a single table so such a claim can be tested. It
reads only; it launches nothing.

Outcome column is `princess`: the maximum princess-reach rate the run ever
logged. That is the end goal of a level, and it is the only outcome that does
not depend on level-specific route bookkeeping.

Example::

    env PYTHONPATH=python:build/ci-linux python3 \\
      scripts/mo5/yeti/diag/run_retrospective.py \\
        --root output/mo5/yeti/training --out .kiro/tmp/retrospective.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional

MAX_H_JOYSTICK = math.log(3) * 2 + math.log(2)

TB_TAGS = [
    "train/entropy_loss",
    "train/clip_fraction",
    "train/approx_kl",
    "train/explained_variance",
    "train/value_loss",
    "rollout/ep_rew_mean",
]


def _load_yaml(path: str) -> Optional[dict]:
    try:
        import yaml

        with open(path) as fh:
            return yaml.safe_load(fh)
    except Exception:
        return None


def _cfg_row(run_dir: str) -> Dict[str, Any]:
    """Configuration fields, from run.yaml when present."""
    y = _load_yaml(os.path.join(run_dir, "run.yaml"))
    row: Dict[str, Any] = {
        "level": None,
        "timesteps": None,
        "seed": None,
        "warm": None,
        "resume": None,
        "n_steps": None,
        "target_kl": None,
        "ent_coef": None,
        "lr": None,
        "num_envs": None,
        "reward": None,
        "profile": None,
    }
    if not y:
        return row
    ex = y.get("extras") or {}
    tr = ex.get("training") or {}
    env = ex.get("env") or {}
    ppo = ex.get("ppo") or {}
    rew = ex.get("reward") or {}
    params = rew.get("params") or {}
    prof = env.get("profile")
    lvl = params.get("level")
    if lvl is None and isinstance(prof, str):
        # profile names carry the level: yeti_fruit_level4 -> 4, yeti_fruit -> 1
        lvl = 1
        for k in (2, 3, 4, 5):
            if prof.endswith(f"level{k}"):
                lvl = k
    row.update(
        level=lvl,
        timesteps=tr.get("timesteps"),
        seed=tr.get("seed"),
        resume=tr.get("resume"),
        warm=bool(tr.get("resume")),
        n_steps=ppo.get("n_steps"),
        target_kl=ppo.get("target_kl"),
        ent_coef=ppo.get("ent_coef"),
        lr=ppo.get("learning_rate"),
        num_envs=tr.get("num_envs"),
        reward=rew.get("name"),
        profile=prof,
    )
    return row


def _outcome(run_dir: str) -> Dict[str, Any]:
    """Best princess rate and deepest reach the run ever logged."""
    out: Dict[str, Any] = {
        "princess": None,
        "deepest_reach": None,
        "deepest_reach_col": None,
        "last_step": None,
        "diag_rows": 0,
    }
    path = os.path.join(run_dir, "curriculum_diag.csv")
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames or []
        reach_cols = [c for c in cols if c.startswith("reach") and c[5:].isdigit()]
        best = {c: 0.0 for c in reach_cols}
        princess = 0.0
        last = 0
        n = 0
        for rec in rdr:
            n += 1
            try:
                last = int(float(rec["step"]))
            except (KeyError, TypeError, ValueError):
                pass
            for c in reach_cols:
                try:
                    v = float(rec[c])
                except (TypeError, ValueError):
                    continue
                if v > best[c]:
                    best[c] = v
            try:
                princess = max(princess, float(rec.get("reach_princess") or 0.0))
            except (TypeError, ValueError):
                pass
    deepest = None
    deepest_col = None
    for c in sorted(reach_cols, key=lambda s: int(s[5:])):
        if best[c] > 0.10:
            deepest = best[c]
            deepest_col = int(c[5:])
    out.update(
        princess=princess,
        deepest_reach=deepest,
        deepest_reach_col=deepest_col,
        last_step=last,
        diag_rows=n,
    )
    return out


def _tb(run_dir: str) -> Dict[str, Any]:
    """First/last/min/max of the training-health scalars.

    Streams the event file and keeps only ``TB_TAGS``. ``EventAccumulator``
    builds every series in the file, and these runs log ~250 reach/length tags,
    which made a full scan of the run directory take over half an hour.
    """
    res: Dict[str, Any] = {}
    files = sorted(glob.glob(os.path.join(run_dir, "tb", "*", "events*")))
    if not files:
        return res
    try:
        from tensorboard.backend.event_processing.event_file_loader import (
            EventFileLoader,
        )
    except Exception:
        return res
    try:
        from tensorboard.util import tensor_util
    except Exception:
        tensor_util = None
    wanted = set(TB_TAGS)
    series: Dict[str, List[float]] = {t: [] for t in TB_TAGS}

    def _scalar(val) -> Optional[float]:
        # SB3 writes scalars as rank-0 TENSORS, not simple_value. Reading
        # simple_value silently yields 0.0 for every point.
        if val.HasField("simple_value"):
            return float(val.simple_value)
        if val.HasField("tensor"):
            if val.tensor.float_val:
                return float(val.tensor.float_val[0])
            if tensor_util is not None:
                try:
                    return float(tensor_util.make_ndarray(val.tensor))
                except Exception:
                    return None
        return None

    try:
        for ev in EventFileLoader(files[-1]).Load():
            if not ev.summary.value:
                continue
            for val in ev.summary.value:
                if val.tag not in wanted:
                    continue
                x = _scalar(val)
                if x is not None:
                    series[val.tag].append(x)
    except Exception:
        pass
    for tag, vals in series.items():
        if not vals:
            continue
        short = tag.split("/")[-1]
        n = len(vals)
        head = sum(vals[: max(1, n // 10)]) / max(1, n // 10)
        tail = sum(vals[n - max(1, n // 10) :]) / max(1, n // 10)
        res[f"{short}_head"] = head
        res[f"{short}_tail"] = tail
        res[f"{short}_min"] = min(vals)
        res[f"{short}_max"] = max(vals)
        res[f"{short}_n"] = n
    # entropy_loss is NEGATIVE entropy; convert to a fraction of the maximum
    # attainable entropy so runs with different action spaces compare.
    if "entropy_loss_head" in res:
        res["H_head_frac"] = -res["entropy_loss_head"] / MAX_H_JOYSTICK
        res["H_tail_frac"] = -res["entropy_loss_tail"] / MAX_H_JOYSTICK
        res["H_min_frac"] = -res["entropy_loss_max"] / MAX_H_JOYSTICK
        res["H_max_frac"] = -res["entropy_loss_min"] / MAX_H_JOYSTICK
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="output/mo5/yeti/training")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="skip runs whose diag ends below this step (smoke tests)",
    )
    p.add_argument("--no-tb", action="store_true", help="skip TensorBoard (faster)")
    args = p.parse_args()

    dirs = sorted(
        d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)
    )
    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(dirs, 1):
        row: Dict[str, Any] = {"run": os.path.basename(d)}
        row.update(_cfg_row(d))
        row.update(_outcome(d))
        if not args.no_tb:
            row.update(_tb(d))
        rows.append(row)
        print(f"  [{i}/{len(dirs)}] {row['run']}", flush=True)
        # Write after every run. A scan of the full directory takes over an
        # hour; a crash or a timeout at row 150 must not lose the first 149.
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump({"root": args.root, "rows": rows}, fh, indent=1)

    keep = [
        r
        for r in rows
        if (r.get("last_step") or 0) >= args.min_steps and r.get("diag_rows")
    ]
    print(
        f"\n{len(rows)} dirs, {len(keep)} with a curriculum diag "
        f"reaching >= {args.min_steps} steps\n"
    )

    hdr = (
        f"{'run':<46} {'lvl':>3} {'warm':>4} {'steps':>9} {'nstp':>5} "
        f"{'tkl':>5} {'ent':>5} {'princess':>8} {'deep':>5} "
        f"{'H_head':>6} {'H_tail':>6} {'clipf':>6}"
    )
    print(hdr)
    for r in sorted(
        keep,
        key=lambda x: (-(x.get("princess") or 0.0), x.get("level") or 0, x["run"]),
    ):
        print(
            f"{r['run'][:46]:<46} {str(r.get('level') or '-'):>3} "
            f"{('W' if r.get('warm') else 'C'):>4} "
            f"{(r.get('last_step') or 0):>9} "
            f"{str(r.get('n_steps') or '-'):>5} "
            f"{str(r.get('target_kl') or '-'):>5} "
            f"{str(r.get('ent_coef') or '-'):>5} "
            f"{(r.get('princess') or 0.0):>8.3f} "
            f"{str(r.get('deepest_reach_col') or '-'):>5} "
            f"{(r.get('H_head_frac') or float('nan')):>6.2f} "
            f"{(r.get('H_tail_frac') or float('nan')):>6.2f} "
            f"{(r.get('clip_fraction_tail') or float('nan')):>6.2f}"
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"root": args.root, "rows": rows}, fh, indent=1)
        print(f"\nwrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
