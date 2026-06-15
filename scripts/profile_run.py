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

import numpy as np
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

FRUITS_ADDR = 11055
LIVES_ADDR = 11095
BONUS_HI = 11010
BONUS_LO = 11011
X_POS = 11090
Y_POS = 11089
PRINCESS_FLAG_ADDR = 11050


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
    gym_env, iface = stack.gym, stack.base._interface
    model = PPO.load(args.model, device="auto")

    def bonus():
        return (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO)

    # arrival[cp] = list of (step, bonus) at first time reaching cp this ep
    arrival_step = defaultdict(list)
    arrival_bonus = defaultdict(list)
    # final position of episodes whose max cp == n (where the n->n+1 leg failed)
    fail_pos = defaultdict(list)
    end_reasons = defaultdict(int)

    for _ep in range(args.episodes):
        obs, _ = gym_env.reset()
        prev_fruits = iface.read_ram_byte(FRUITS_ADDR)
        prev_lives = iface.read_ram_byte(LIVES_ADDR)
        prev_bonus = bonus()
        prev_princess = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        stall = 0
        max_cp = 4 - prev_fruits
        end = "max_steps"
        step = 0
        while step < args.max_steps:
            a, _ = model.predict(np.transpose(obs, (2, 0, 1)), deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(a)
            step += 1
            fruits = iface.read_ram_byte(FRUITS_ADDR)
            lives = iface.read_ram_byte(LIVES_ADDR)
            b = bonus()
            pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
            cp = 4 - fruits
            if cp > max_cp:
                max_cp = cp
                arrival_step[cp].append(step)
                arrival_bonus[cp].append(b)
            if pr == 1 and prev_princess == 0:
                max_cp = 5
                arrival_step[5].append(step)
                arrival_bonus[5].append(b)
                end = "princess"
                break
            prev_princess = pr
            if lives < prev_lives and prev_lives > 0:
                end = "death"
                break
            prev_lives = lives
            if b == prev_bonus:
                stall += 1
            else:
                stall = 0
                prev_bonus = b
            if stall >= args.stall_threshold:
                end = "stall"
                break
            if done or trunc:
                end = "env_done"
                break
        end_reasons[end] += 1
        fx, fy = iface.read_ram_byte(X_POS) * 4 + 8, iface.read_ram_byte(Y_POS)
        fail_pos[max_cp].append((fx, fy, end))

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
