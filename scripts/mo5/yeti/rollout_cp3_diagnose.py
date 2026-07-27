#!/usr/bin/env python3
"""Diagnostic rollout from a CP3 seed (3 fruits collected, F4 remaining).

Loads the yeti_curriculum_v3 policy + a CP3 seed from its pool, plays
N episodes, and traces per-frame (ram_x, ram_y, floor, path-distance
to F4, action) so we can see exactly why F3->F4 fails.

Action layout (from mo5_rl.cpp apply_joystick):
  action[0]: 1=UP, 2=DOWN
  action[1]: 1=RIGHT, 2=LEFT
  action[2]: 1=fire/jump
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build" / "ci-linux"))

from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import EnvConfig  # noqa: E402
from retro_ai.training.yeti_map import (  # noqa: E402
    agent_floor_from_pixel_y,
    build_navigation_map,
)

X_ADDR = 11090
Y_ADDR = 11089
LIVES_ADDR = 11095
FRUITS_ADDR = 11055
BONUS_HI = 11010
BONUS_LO = 11011


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model",
        type=Path,
        default=ROOT / "output/mo5/yeti/training/yeti_curriculum_v3/final_model.zip",
    )
    ap.add_argument(
        "--checkpoints",
        type=Path,
        default=ROOT / "output/mo5/yeti/training/yeti_curriculum_v3/checkpoints.pkl",
    )
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=2000,
        stall_threshold=2000,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", cfg)
    base = stack.base
    gym_env = stack.gym
    base.reset(seed=0)
    iface = base._interface

    with args.checkpoints.open("rb") as f:
        cp = pickle.load(f)["checkpoints"]
    cp3 = cp[3]
    print(f"CP3 pool: {len(cp3)} states")
    model = PPO.load(args.model)
    nav = build_navigation_map()

    rng = np.random.default_rng(0)
    summaries = []
    for run in range(args.n_runs):
        _bonus, seed = cp3[int(rng.integers(0, len(cp3)))]
        iface.load_state(bytes(seed))
        obs = None
        for _ in range(5):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])

        x0 = iface.read_ram_byte(X_ADDR)
        y0 = iface.read_ram_byte(Y_ADDR)
        f0 = agent_floor_from_pixel_y(y0)
        d0 = nav.path_distance_from_agent(f0 or 3, x0 * 4 + 8, "F4")
        print(
            f"\nrun {run}: start ram=({x0},{y0}) px=({x0 * 4 + 8},{y0}) "
            f"floor={f0} path_d_to_F4={d0}"
        )
        print(
            f"{'frm':>4} {'rx':>3} {'ry':>3} {'fl':>2} {'dF4':>4} "
            f"{'lv':>2} {'fr':>2} {'UP':>2} {'LR':>2} {'J':>1}"
        )

        last_d = d0
        end_reason = "max_frames"
        prev_fruits = iface.read_ram_byte(FRUITS_ADDR)
        for fr in range(args.max_frames):
            obs_chw = np.transpose(obs, (2, 0, 1))
            action, _ = model.predict(obs_chw, deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(action)
            x = iface.read_ram_byte(X_ADDR)
            y = iface.read_ram_byte(Y_ADDR)
            lv = iface.read_ram_byte(LIVES_ADDR)
            fruits = iface.read_ram_byte(FRUITS_ADDR)
            floor = agent_floor_from_pixel_y(y)
            d = (
                nav.path_distance_from_agent(floor, x * 4 + 8, "F4")
                if floor is not None
                else None
            )
            d_str = str(d) if d is not None else "?"
            changed = fr < 6 or (d is not None and d != last_d) or x != x0 or y != y0
            if changed:
                up, lr, j = int(action[0]), int(action[1]), int(action[2])
                print(
                    f"{fr:>4} {x:>3} {y:>3} "
                    f"{(floor if floor is not None else '-'):>2} {d_str:>4} "
                    f"{lv:>2} {fruits:>2} {up:>2} {lr:>2} {j:>1}"
                )
                if d is not None:
                    last_d = d
            if fruits < prev_fruits:
                end_reason = "GOT_F4"
                print(f"  *** F4 collected at frame {fr} ***")
                break
            prev_fruits = fruits
            if done or trunc:
                end_reason = "died/done" if done else "trunc"
                print(f"  episode ended frame {fr}: {end_reason}")
                break
            if lv < 1:
                end_reason = "no_lives"
                break
        summaries.append((run, end_reason, last_d))

    print("\nsummary:")
    for run, er, d in summaries:
        print(f"  run {run}: {er}, last_path_d_to_F4={d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
