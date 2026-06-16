#!/usr/bin/env python3
"""Diagnose the CP4->princess (L45 ascent) failure mode.

Loads CP4 seed states (reset-origin, agent just collected F4 ~floor 4
x=272) from a checkpoints.pkl, rolls out a policy from each, and reports
the princess-touch rate and — for failures — where the episode ended.

Map refs: F4 (272,88) floor4; ladder L45 x=208; princess (312,60) floor5.
  - end at floor4 far-right (x~272, y~88), never near L45 -> NAVIGATION
    (won't reverse left to the ladder)
  - end at/above L45 (x~208, y between 88 and 56) -> TIMING (dies on climb)

Example::

    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
      python scripts/profile_cp4_princess.py \\
        --model <model.zip> --seeds <checkpoints.pkl> --episodes 300
"""

from __future__ import annotations

import argparse
import pickle
import random
import statistics
from collections import Counter

import numpy as np
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

FRUITS_ADDR = 11055
LIVES_ADDR = 11095
BONUS_HI, BONUS_LO = 11010, 11011
X_POS, Y_POS = 11090, 11089
PRINCESS_FLAG_ADDR = 11050


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", required=True, help="checkpoints.pkl with a CP4 pool")
    p.add_argument("--episodes", type=int, default=300)
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
    base_iface = stack.base._interface
    model = PPO.load(args.model, device="auto")

    with open(args.seeds, "rb") as f:
        data = pickle.load(f)
    cp4 = [e[-1] for e in data["checkpoints"][4]]  # state_bytes
    assert cp4, "no CP4 seeds in checkpoints.pkl"

    def bonus():
        return (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO)

    touches = 0
    fail_pos = []
    end_reasons: Counter[str] = Counter()
    gym_env.reset()

    for _ep in range(args.episodes):
        state = random.choice(cp4)
        base_iface.load_state(state)
        obs = None
        for _ in range(5):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])
        prev_lives = iface.read_ram_byte(LIVES_ADDR)
        prev_bonus = bonus()
        prev_pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        stall = 0
        end = "max_steps"
        touched = False
        for _ in range(args.max_steps):
            a, _ = model.predict(np.transpose(obs, (2, 0, 1)), deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(a)
            lives = iface.read_ram_byte(LIVES_ADDR)
            b = bonus()
            pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
            if pr == 1 and prev_pr == 0:
                touched = True
                end = "princess"
                break
            prev_pr = pr
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
        if touched:
            touches += 1
        else:
            fx = iface.read_ram_byte(X_POS) * 4 + 8
            fy = iface.read_ram_byte(Y_POS)
            fail_pos.append((fx, fy))

    n = args.episodes
    print(f"\n=== CP4->princess profile: {args.model} ===")
    print(f"seeds={args.seeds}  episodes={n}")
    print(f"\nprincess touches: {touches}/{n} ({100*touches/n:.1f}%)")
    print(f"end_reasons: {dict(end_reasons)}")
    if fail_pos:
        xs = [x for x, _ in fail_pos]
        ys = [y for _, y in fail_pos]
        print(
            f"\nfailures (n={len(fail_pos)}): final px median=("
            f"{int(statistics.median(xs))},{int(statistics.median(ys))})"
        )
        # Bucket by region relative to L45 (x=208) and floor (y: 88=fl4, 56=fl5)
        near_ladder = sum(1 for x, y in fail_pos if 190 <= x <= 226)
        right_of_f4 = sum(1 for x, y in fail_pos if x > 240)
        on_climb = sum(1 for x, y in fail_pos if 56 < y < 88)
        floor5 = sum(1 for x, y in fail_pos if y <= 64)
        print(f"  near L45 ladder x in[190,226]: {near_ladder}")
        print(f"  right of F4 x>240 (didn't reverse): {right_of_f4}")
        print(f"  mid-climb 56<y<88: {on_climb}")
        print(f"  reached floor5 y<=64: {floor5}")


if __name__ == "__main__":
    main()
