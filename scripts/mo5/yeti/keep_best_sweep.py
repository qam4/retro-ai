#!/usr/bin/env python3
"""Keep-best capture: eval training snapshots from reset and retain the
best policy (by princess-from-reset rate).

Why this exists (experiment 003, H-U): a single PPO policy on Yeti
oscillates — reach-4/princess swing the full range across snapshots and
the *final* model is usually degraded. The good policy is a transient
peak (v14 hit 58%% princess at 12.75M while the snapshots on either side
read ~0%%). Training metrics can't identify it (most training episodes
don't start from reset; the live signal is a noisy EMA), so we must eval
frozen snapshots from reset and keep the best.

This runs each eval as a SEPARATE PROCESS (scripts/mo5/yeti/eval_from_reset.py).
The Crayon emulator keeps in-process global state, so eval must not share
a process with training; a subprocess is fully isolated. Defaults to CPU
so it is safe to run *alongside* a GPU training job (pass --device gpu to
use the GPU when nothing else is training).

Usage
-----
One-shot (eval every snapshot present, keep the best)::

    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
      python scripts/mo5/yeti/keep_best_sweep.py \\
        --snapshots-dir output/mo5/yeti/training/<run>/snapshots

Watch a live run (poll for new snapshots, stop after idle)::

    ... keep_best_sweep.py --snapshots-dir <run>/snapshots --watch

The best policy is copied to ``<best-dir>/best_model.zip`` with
``best_meta.json``; per-snapshot results accumulate in
``<best-dir>/sweep_state.json`` so re-runs skip already-eval'd snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

_STEP_RE = re.compile(r"model_(\d+)_steps\.zip$")


def _snapshots(snap_dir):
    """Return [(step, path)] sorted by step."""
    out = []
    for name in os.listdir(snap_dir):
        m = _STEP_RE.search(name)
        if m:
            out.append((int(m.group(1)), os.path.join(snap_dir, name)))
    out.sort()
    return out


def _load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"evaluated": {}, "best": None}


def _save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _eval_snapshot(
    model_path,
    episodes,
    device,
    tmp_json,
    *,
    profile,
    fruits_total,
    start_state,
    stall_threshold,
    max_steps,
    level=None,
):
    """Run eval_from_reset.py in a subprocess; return (princess, reach_top, mean_rung,
    n_rungs).

    ``mean_rung`` is the average number of MANDATORY route targets an episode gets
    behind it -- the same quantity training reports as ``reset_reach``. It exists
    because princess and reach_top cannot rank snapshots on a single-fruit level: see
    the scoring note in main().
    """
    cmd = [
        sys.executable,
        "scripts/mo5/yeti/eval_from_reset.py",
        "--model",
        model_path,
        "--episodes",
        str(episodes),
        "--stochastic",
        "--out",
        tmp_json,
        "--profile",
        profile,
        "--fruits-total",
        str(fruits_total),
        "--stall-threshold",
        str(stall_threshold),
        "--max-steps",
        str(max_steps),
    ]
    if level is not None:
        cmd += ["--level", str(level)]
    if start_state:
        cmd += ["--start-state", start_state]
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, env=env, capture_output=True, timeout=3600, check=True)
    with open(tmp_json) as f:
        data = json.load(f)
    n = data["episodes"]
    princess = data["princess_touches"] / n
    reach_top = (
        sum(v for k, v in data["max_cp_counts"].items() if int(k) >= fruits_total) / n
    )
    return princess, reach_top, data.get("mean_rung", 0.0), data.get("n_rungs", 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshots-dir", required=True)
    p.add_argument(
        "--best-dir",
        default=None,
        help="where to keep best_model.zip (default: " "<snapshots-dir>/../best)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="episodes per eval (cheap trigger; re-eval the "
        "winner with more for a precise number)",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu (safe alongside training) or gpu",
    )
    p.add_argument(
        "--watch", action="store_true", help="poll for new snapshots until idle"
    )
    p.add_argument("--poll-sec", type=int, default=120)
    p.add_argument("--max-idle-min", type=float, default=30.0)
    # Level awareness (defaults = level 1). For level 2 pass:
    #   --profile yeti_fruit_level2 --fruits-total 2 --stall-threshold 40
    #   --start-state output/mo5/yeti/level2/level2_start.sav
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--fruits-total", type=int, default=4)
    p.add_argument(
        "--level",
        type=int,
        default=None,
        help="level geometry; REQUIRED for route-depth scoring to be meaningful",
    )
    p.add_argument("--start-state", default=None)
    p.add_argument("--stall-threshold", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=1000)
    args = p.parse_args()

    snap_dir = args.snapshots_dir
    best_dir = args.best_dir or os.path.join(os.path.dirname(snap_dir), "best")
    os.makedirs(best_dir, exist_ok=True)
    state_path = os.path.join(best_dir, "sweep_state.json")
    tmp_json = os.path.join(best_dir, "_eval.json")
    state = _load_state(state_path)

    def best_score():
        return state["best"]["score"] if state["best"] else -1.0

    last_new = time.time()
    while True:
        snaps = _snapshots(snap_dir)
        new = [
            (s, path)
            for s, path in snaps
            if os.path.basename(path) not in state["evaluated"]
        ]
        if new:
            last_new = time.time()
        for step, path in new:
            name = os.path.basename(path)
            try:
                princess, reach_top, mean_rung, n_rungs = _eval_snapshot(
                    path,
                    args.episodes,
                    args.device,
                    tmp_json,
                    profile=args.profile,
                    fruits_total=args.fruits_total,
                    start_state=args.start_state,
                    stall_threshold=args.stall_threshold,
                    max_steps=args.max_steps,
                    level=args.level,
                )
            except Exception as e:
                print(f"[keep-best] {name}: eval FAILED ({e})", flush=True)
                continue
            # SCORING, in strict priority: princess, then ROUTE DEPTH, then fruit.
            #
            # It used to be `princess + 1e-3 * reach_top`, which cannot rank snapshots
            # on a single-fruit level: princess is uniformly 0 and reach_top just means
            # "collected the fruit", which saturates at 1.0. Ties then went to whichever
            # snapshot was seen FIRST, so every L4 champion ever picked was arbitrary --
            # v4 kept its 100k snapshot, v5 its 1M, v6 its 200k, while the run's actual
            # depth peak was elsewhere entirely (v6's was ~8.76M).
            #
            # `mean_rung / n_rungs` is the fraction of mandatory route targets an
            # episode gets behind it on average -- the same quantity the training route
            # table reports as reset_reach. Weighted below princess so a single princess
            # touch still outranks any amount of depth, and above reach_top so the fruit
            # only breaks ties between equally deep policies.
            depth = (mean_rung / n_rungs) if n_rungs else 0.0
            score = princess + 1e-2 * depth + 1e-5 * reach_top
            state["evaluated"][name] = {
                "step": step,
                "princess": princess,
                "reach_top": reach_top,
                "mean_rung": mean_rung,
                "n_rungs": n_rungs,
                "depth": depth,
                "score": score,
                "n_eval": args.episodes,
            }
            msg = (
                f"[keep-best] step {step}: princess={princess:.3f} "
                f"rung={mean_rung:.2f}/{n_rungs} reach_top={reach_top:.3f} "
                f"(best={best_score():.4f})"
            )
            if score > best_score():
                shutil.copyfile(path, os.path.join(best_dir, "best_model.zip"))
                state["best"] = {
                    "model": name,
                    "step": step,
                    "score": score,
                    "princess": princess,
                    "reach_top": reach_top,
                    "mean_rung": mean_rung,
                    "n_rungs": n_rungs,
                    "n_eval": args.episodes,
                }
                with open(os.path.join(best_dir, "best_meta.json"), "w") as f:
                    json.dump(state["best"], f, indent=2)
                msg += "  -> NEW BEST (saved)"
            print(msg, flush=True)
            _save_state(state_path, state)

        if not args.watch:
            break
        idle_min = (time.time() - last_new) / 60.0
        if idle_min >= args.max_idle_min:
            print(f"[keep-best] idle {idle_min:.0f} min, stopping.", flush=True)
            break
        time.sleep(args.poll_sec)

    b = state["best"]
    if b:
        print(
            f"\nBest: {b['model']} (step {b['step']}) "
            f"princess={b['princess']:.3f} "
            f"rung={b.get('mean_rung', 0):.2f}/{b.get('n_rungs', 0)} "
            f"reach_top={b.get('reach_top', 0):.3f}  "
            f"-> {os.path.join(best_dir, 'best_model.zip')}"
        )
        print("Re-eval the winner with more episodes for a precise number, e.g.:")
        print(
            f"  python scripts/mo5/yeti/eval_from_reset.py --model "
            f"{os.path.join(best_dir, 'best_model.zip')} --episodes 300 --stochastic"
        )
    else:
        print("No snapshots evaluated.")


if __name__ == "__main__":
    main()
