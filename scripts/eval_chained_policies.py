#!/usr/bin/env python3
"""Evaluate chained per-segment policies end-to-end.

Loads two trained models — one per segment — and rolls them out from
a list of starting CP seeds. The first policy plays until it either
reaches the next CP or its episode ends. If it reached, the second
policy takes over from the resulting state and plays until the
following CP or its episode ends.

Usage example: chain v7 (CP2→CP3) and v8 (CP3→CP4) from CP2 seeds:

  env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
    python3 scripts/eval_chained_policies.py \\
      --policies output/.../segment_2to3_v7/final_model.zip \\
                 output/.../segment_3to4_v1/final_model.zip \\
      --seeds output/mo5/yeti/seeds/v9_v3_cp3enriched.pkl \\
      --start-cp 2 \\
      --episodes-per-seed 5 \\
      --max-segment-steps 1000 \\
      --out output/mo5/yeti/eval/chain_v7_v8.json

Outputs per-episode JSON lines with start CP, max CP reached, length,
which policy was active, and end reason.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

FRUITS_ADDR = 11055


def _load_seeds(path: str, cp: int):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [bytes(s) for s in data["checkpoints"][cp]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policies",
        nargs="+",
        required=True,
        help="Ordered list of policy zip files, one per segment.",
    )
    p.add_argument("--seeds", required=True)
    p.add_argument(
        "--start-cp",
        type=int,
        required=True,
        help="CP level the first policy starts from (0..4).",
    )
    p.add_argument("--episodes-per-seed", type=int, default=5)
    p.add_argument("--max-segment-steps", type=int, default=1000)
    p.add_argument("--settle-frames", type=int, default=5)
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    seeds = _load_seeds(args.seeds, args.start_cp)
    print(f"Loaded {len(seeds)} CP{args.start_cp} seeds")
    print(f"Chaining {len(args.policies)} policies")

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=args.max_segment_steps * len(args.policies) + 100,
        stall_threshold=args.max_segment_steps * len(args.policies) + 100,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    base = stack.base
    gym_env = stack.gym
    base.reset(seed=0)

    models = [PPO.load(p) for p in args.policies]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f_out = open(args.out, "w")

    n_episodes = 0
    summary = {"per_max_cp": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}
    for seed_idx, state in enumerate(seeds):
        for ep in range(args.episodes_per_seed):
            base._interface.load_state(state)
            obs = None
            for _ in range(args.settle_frames):
                obs, _, _, _, _ = gym_env.step([0, 0, 0])
            start_cp = 4 - base._interface.read_ram_byte(FRUITS_ADDR)
            current_cp = start_cp
            max_cp_reached = current_cp
            total_steps = 0
            done = False
            end_reason = None

            for seg_idx, model in enumerate(models):
                # The seg_idx-th policy targets CP_(start_cp + seg_idx)
                # -> CP_(start_cp + seg_idx + 1).
                expected_cp = start_cp + seg_idx
                if current_cp != expected_cp:
                    end_reason = (
                        f"current_cp={current_cp} but seg_idx={seg_idx} "
                        f"expects {expected_cp}"
                    )
                    break

                # When handing off to a non-first policy, the new
                # policy was trained against load_state-then-settle.
                # We mimic that here: settle a few noop frames so the
                # new policy's first observation is consistent with
                # its training distribution.
                if seg_idx > 0:
                    for _ in range(args.settle_frames):
                        obs, _, done, trunc, _ = gym_env.step([0, 0, 0])
                        if done or trunc:
                            break
                    if done or trunc:
                        end_reason = "died_during_handoff_settle"
                        break

                seg_steps = 0
                while seg_steps < args.max_segment_steps:
                    obs_chw = np.transpose(obs, (2, 0, 1))
                    action, _ = model.predict(obs_chw, deterministic=False)
                    obs, _, done, trunc, _ = gym_env.step(action)
                    seg_steps += 1
                    total_steps += 1

                    cp = 4 - base._interface.read_ram_byte(FRUITS_ADDR)
                    if cp > max_cp_reached:
                        max_cp_reached = cp
                    if cp > current_cp:
                        current_cp = cp
                        # Hand off to next policy.
                        break
                    if done or trunc:
                        end_reason = "episode_ended_in_segment"
                        break

                if done or trunc:
                    break
                if current_cp == start_cp + seg_idx:
                    end_reason = "segment_max_steps_no_advance"
                    break

            if end_reason is None:
                end_reason = "completed_all_segments"

            row = {
                "seed_idx": seed_idx,
                "ep": ep,
                "start_cp": start_cp,
                "max_cp_reached": max_cp_reached,
                "total_steps": total_steps,
                "end_reason": end_reason,
            }
            f_out.write(json.dumps(row) + "\n")
            f_out.flush()
            summary["per_max_cp"][max_cp_reached] = (
                summary["per_max_cp"].get(max_cp_reached, 0) + 1
            )
            n_episodes += 1
        if (seed_idx + 1) % 20 == 0:
            print(
                f"  processed seed {seed_idx + 1}/{len(seeds)}  "
                f"({n_episodes} episodes)",
                flush=True,
            )

    f_out.close()
    print()
    print(f"=== chained eval summary ({n_episodes} episodes) ===")
    for cp in sorted(summary["per_max_cp"]):
        n = summary["per_max_cp"][cp]
        pct = 100 * n / n_episodes if n_episodes else 0
        print(f"  max CP reached = {cp}: {n} ({pct:.1f}%)")
    cp_to_4 = summary["per_max_cp"].get(4, 0)
    print(f"\nCP{args.start_cp} -> CP4 chain rate: {100*cp_to_4/n_episodes:.2f}%")
    print(f"\nDetailed log: {args.out}")


if __name__ == "__main__":
    main()
