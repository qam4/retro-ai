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
      python scripts/mo5/yeti/profile_cp4_princess.py \\
        --model <model.zip> --seeds <checkpoints.pkl> --episodes 300
"""

from __future__ import annotations

import argparse
import pickle
import random
import statistics
from collections import Counter

from retro_ai.games.yeti_rollout import rollout_episode
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO


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
    model = PPO.load(args.model, device="auto")

    with open(args.seeds, "rb") as f:
        data = pickle.load(f)
    cp4 = [e[-1] for e in data["checkpoints"][4]]  # state_bytes
    assert cp4, "no CP4 seeds in checkpoints.pkl"

    touches = 0
    fail_pos = []
    end_reasons: Counter[str] = Counter()
    stack.gym.reset()  # boot once; rollout_episode(reset_env=False) load_states below

    for _ep in range(args.episodes):
        # Seed-pool rollout: a different CP4 state each episode, no per-episode
        # game reset (reset_env=False) to avoid the ~32s MO5 startup each time.
        result = rollout_episode(
            stack,
            model,
            level=1,
            fruits_total=4,
            start_state=random.choice(cp4),
            settle=5,
            max_steps=args.max_steps,
            stall_threshold=args.stall_threshold,
            deterministic=False,
            reset_env=False,
        )
        end_reasons[result.end_reason] += 1
        if result.princess_touched:
            touches += 1
        else:
            fail_pos.append((result.final_x * 4 + 8, result.final_y))

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
