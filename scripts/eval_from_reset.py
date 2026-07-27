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
      python scripts/eval_from_reset.py \\
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
    )
    stack = build_training_env(args.profile, env_cfg)

    model = PPO.load(args.model, device="auto")

    rows = []
    max_cp_counts: Counter[int] = Counter()
    princess_touches = 0

    for ep in range(args.episodes):
        # Shared rollout harness: identical termination (princess -> death via
        # 0x2AFC -> stall -> env done -> max_steps) and CP tracking for all
        # eval/analysis scripts. Level 2 boots from a save-state.
        result = rollout_episode(
            stack,
            model,
            level=2 if start_state_bytes is not None else 1,
            fruits_total=fruits_total,
            start_state=start_state_bytes,
            max_steps=args.max_steps,
            stall_threshold=args.stall_threshold,
            deterministic=deterministic,
        )
        max_cp_counts[result.max_cp] += 1
        if result.princess_touched:
            princess_touches += 1
        rows.append(
            {
                "ep": ep,
                "max_cp": result.max_cp,
                "steps": result.length,
                "end_reason": result.end_reason,
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

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "episodes": n,
                    "deterministic": deterministic,
                    "fruits_total": fruits_total,
                    "start_state": args.start_state,
                    "max_cp_counts": dict(max_cp_counts),
                    "princess_touches": princess_touches,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
