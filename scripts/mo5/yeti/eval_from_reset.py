#!/usr/bin/env python3
"""Evaluate a single trained policy from game reset (CP0).

Rolls out N episodes from a clean reset (no save-state loading), under the
SAME episode termination as training (death / stall / max_steps / princess
touch), and reports the distribution of the deepest checkpoint reached and
the princess-touch rate.

This measures the North Star directly: P(reach CP_k | start = reset) for a
single policy. Defaults to a deterministic policy; pass --stochastic to
sample actions instead.

Example::

    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
      python scripts/mo5/yeti/eval_from_reset.py \\
        --model output/mo5/yeti/warmstart/v2_clean/final_model.zip \\
        --episodes 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from retro_ai.games.yeti_rollout import rollout_episode
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--stall-threshold", type=int, default=15)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out", default=None)
    # Level awareness (defaults = level 1). For level 2 pass e.g.:
    #   --profile yeti_fruit_level2 --fruits-total 2 --stall-threshold 40
    #   --start-state output/mo5/yeti/level2/level2_start.sav
    p.add_argument(
        "--fruits-total",
        type=int,
        default=4,
        help="collectible fruits in the level (L1=4, L2=2). Sets CP indexing "
        "and the princess goal index (= fruits_total + 1).",
    )
    p.add_argument(
        "--start-state",
        default=None,
        help="save-state to load on each reset (level 2 starts from a save, "
        "not a game reset). Omit for level 1 (game reset).",
    )
    p.add_argument(
        "--level",
        type=int,
        default=None,
        help="level geometry to use. Previously INFERRED as '2 if --start-state else "
        "1', so every L3/L4 eval silently used level-2 geometry. Defaults to that old "
        "behaviour when omitted so existing invocations are unchanged; pass it "
        "explicitly (and you must, for route-depth scoring to mean anything).",
    )
    p.add_argument(
        "--no-depth",
        action="store_true",
        help="skip route-depth tracking (max_rung). Depth is what lets a sweep rank "
        "snapshots on a level where princess is uniformly 0 and the fruit count "
        "saturates.",
    )
    p.add_argument(
        "--waypoint-tolerance",
        type=int,
        default=2,
        help="ladder tolerance, used by --reach-mode box only",
    )
    p.add_argument(
        "--jump-waypoint-tolerance",
        type=int,
        default=6,
        help="jump-landing tolerance, used by --reach-mode box only",
    )
    p.add_argument(
        "--reach-mode",
        default="sprite",
        choices=["box", "sprite"],
        help="how a route point counts as reached. MUST match the run's "
        "curriculum.waypoint_reach_mode (default sprite since de21939). 'box' is kept "
        "to reproduce evals made before 2026-10-07, which used it whatever the run "
        "trained with -- and so read L4's rope-2 landing as never reached.",
    )
    p.add_argument(
        "--resize-mode",
        default="nearest",
        choices=["nearest", "max"],
        help="MUST match the mode the model was trained with (env.resize_mode). A "
        "mismatch does not error, it just scores the policy on a picture it never saw: "
        "the L4 v29 champion reads mean rung 8.17 under its own mode and 0.02 under "
        "the other.",
    )
    args = p.parse_args()

    deterministic = not args.stochastic
    fruits_total = args.fruits_total
    princess_cp = fruits_total + 1
    start_state_bytes = None
    if args.start_state:
        with open(args.start_state, "rb") as f:
            start_state_bytes = f.read()

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.stall_threshold,
        resize=(84, 84),
        resize_mode=args.resize_mode,
    )
    stack = build_training_env(args.profile, env_cfg)

    model = PPO.load(args.model, device="auto")

    level = args.level if args.level is not None else (2 if start_state_bytes else 1)
    track = not args.no_depth

    rows = []
    max_cp_counts: Counter[int] = Counter()
    rung_counts: Counter[int] = Counter()
    princess_touches = 0
    n_rungs = 0

    for ep in range(args.episodes):
        # Shared rollout harness: identical termination (princess -> death via
        # 0x2AFC -> stall -> env done -> max_steps) and CP tracking for all
        # eval/analysis scripts. Level 2 boots from a save-state.
        result = rollout_episode(
            stack,
            model,
            level=level,
            fruits_total=fruits_total,
            start_state=start_state_bytes,
            max_steps=args.max_steps,
            stall_threshold=args.stall_threshold,
            deterministic=deterministic,
            track_waypoints=track,
            wp_tol=args.waypoint_tolerance,
            wp_jump_tol=args.jump_waypoint_tolerance,
            reach_mode=args.reach_mode,
        )
        max_cp_counts[result.max_cp] += 1
        rung_counts[result.max_rung] += 1
        n_rungs = result.n_rungs or n_rungs
        if result.princess_touched:
            princess_touches += 1
        rows.append(
            {
                "ep": ep,
                "max_cp": result.max_cp,
                "max_rung": result.max_rung,
                "steps": result.length,
                "end_reason": result.end_reason,
                "final_x": result.final_x,
                "final_y": result.final_y,
                "reached_points": sorted(result.reached_points),
            }
        )
        if (ep + 1) % 25 == 0:
            reach1 = sum(v for k, v in max_cp_counts.items() if k >= 1)
            reach_top = sum(v for k, v in max_cp_counts.items() if k >= fruits_total)
            print(
                f"  {ep + 1}/{args.episodes} episodes "
                f"(reach1={reach1}, reach{fruits_total}={reach_top}, "
                f"princess={princess_touches})",
                flush=True,
            )

    n = args.episodes
    print(f"\n=== from-reset eval: {args.model} ===")
    print(f"episodes={n}  policy={'deterministic' if deterministic else 'stochastic'}")
    print("\nDeepest checkpoint reached (cumulative):")
    cum = 0
    for cp in range(princess_cp, -1, -1):
        cum += max_cp_counts.get(cp, 0)
        label = "princess" if cp == princess_cp else f"{cp} fruits"
        print(
            f"  reached >= {label:>10}: {cum:>4}/{n}  ({100*cum/n:5.1f}%)"
            + (
                f"   [exactly {cp}: {max_cp_counts.get(cp,0)}]"
                if max_cp_counts.get(cp, 0)
                else ""
            )
        )
    print(f"\nprincess touches: {princess_touches}/{n} ({100*princess_touches/n:.1f}%)")

    mean_rung = (
        sum(k * v for k, v in rung_counts.items()) / n if track and rung_counts else 0.0
    )
    if track and n_rungs:
        print(f"\nRoute depth (mandatory rungs, {n_rungs} total) -- this is what")
        print("distinguishes snapshots on a level where princess is uniformly 0:")
        cum = 0
        for r in range(n_rungs, -1, -1):
            cum += rung_counts.get(r, 0)
            if cum:
                print(
                    f"  reached >= rung {r:>2}: {cum:>4}/{n}  ({100 * cum / n:5.1f}%)"
                )
        print(f"  mean rung: {mean_rung:.2f} / {n_rungs}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "episodes": n,
                    "deterministic": deterministic,
                    "fruits_total": fruits_total,
                    "start_state": args.start_state,
                    "level": level,
                    "max_cp_counts": dict(max_cp_counts),
                    "princess_touches": princess_touches,
                    "rung_counts": dict(rung_counts),
                    "n_rungs": n_rungs,
                    "mean_rung": mean_rung,
                    # How these numbers were measured. Both silently change every
                    # route figure, and neither was recorded before 2026-10-07.
                    "reach_mode": args.reach_mode,
                    "resize_mode": args.resize_mode,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
