#!/usr/bin/env python3
"""Profile a trained policy from reset: per-leg speed (step+bonus at each
fruit pickup) and where each leg fails (final position / outcome).

No training. Used to check whether deep legs are slow (more snowball
exposure) and whether princess failures are navigation (never reaches the
L45 ladder) vs timing (dies on the climb).

Example::

    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
      python scripts/profile_run.py --model <snapshot.zip> --episodes 150
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from retro_ai.games.yeti_rollout import rollout_episode
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--stall-threshold", type=int, default=15)
    args = p.parse_args()

    env_cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.stall_threshold,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", env_cfg)
    model = PPO.load(args.model, device="auto")

    # arrival[cp] = list of (step, bonus) at first time reaching cp this ep
    arrival_step = defaultdict(list)
    arrival_bonus = defaultdict(list)
    # final position of episodes whose max cp == n (where the n->n+1 leg failed)
    fail_pos = defaultdict(list)
    end_reasons = defaultdict(int)

    for _ep in range(args.episodes):
        result = rollout_episode(
            stack,
            model,
            level=1,
            fruits_total=4,
            max_steps=args.max_steps,
            stall_threshold=args.stall_threshold,
            deterministic=False,
        )
        for cp, (astep, abonus) in result.cp_arrival.items():
            arrival_step[cp].append(astep)
            arrival_bonus[cp].append(abonus)
        end_reasons[result.end_reason] += 1
        fail_pos[result.max_cp].append(
            (result.final_x * 4 + 8, result.final_y, result.end_reason)
        )

    n = args.episodes
    print(f"\n=== efficiency/failure profile: {args.model} ({n} eps) ===\n")
    print("Per-leg arrival (median step / median bonus among eps that reached it):")
    for cp in range(1, 6):
        if arrival_step[cp]:
            ms = int(statistics.median(arrival_step[cp]))
            mb = int(statistics.median(arrival_bonus[cp]))
            label = "princess" if cp == 5 else f"CP{cp}"
            print(
                f"  {label:>8}: reached {len(arrival_step[cp]):>3}/{n}  "
                f"median_step={ms:>4}  median_bonus={mb:>4}"
            )
    print("\nWhere episodes ended, by deepest CP reached (the failed leg):")
    for cp in sorted(fail_pos):
        positions = fail_pos[cp]
        ends = defaultdict(int)
        for _x, _y, e in positions:
            ends[e] += 1
        xs = [x for x, _y, _e in positions]
        ys = [y for _x, y, _e in positions]
        mx = int(statistics.median(xs)) if xs else 0
        my = int(statistics.median(ys)) if ys else 0
        print(
            f"  max CP{cp}: n={len(positions):>3}  median_final_px=({mx},{my})  "
            f"ends={dict(ends)}"
        )
    print(f"\nend_reasons: {dict(end_reasons)}")


if __name__ == "__main__":
    main()
