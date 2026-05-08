#!/usr/bin/env python3
"""Run a trained model from specific seed states, record videos.

Loads a ``final_model.zip`` from a train_segment run and rolls it out
starting from a picked list of CP seeds. Writes one MP4 per episode at
real-time FPS so you can watch the policy's behaviour.

Usage
-----

Rollout from every seed in the re-validated v9 archive's CP2 bucket::

    env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
      python3 scripts/rollout_policy_from_seeds.py \\
        --model output/mo5/yeti/training/segment_2to3_v3/final_model.zip \\
        --seeds output/mo5/yeti/seeds/v9_checkpoints_v2.pkl \\
        --cp 2 \\
        --episodes 1 \\
        --out debug/rollout_v3_cp2
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig


# Yeti RAM addresses (documented in scripts/trace_state.py).
FRUITS_ADDR = 11055
LIVES_ADDR = 11095
X_ADDR = 11090
Y_ADDR = 11089
BONUS_HI = 11010
BONUS_LO = 11011
FRUIT_PRESENCE = {1: 0x2FAD, 2: 0x2F00, 3: 0x2E68, 4: 0x2DD8}


def _load_cp_seeds(path: str, cp: int) -> List[bytes]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "checkpoints" in data:
        return [bytes(s) for s in data["checkpoints"][cp]]
    raise SystemExit(f"expected curriculum format, got keys {list(data.keys())}")


def _read_fruits(iface) -> Tuple[int, int, int, int]:
    return tuple(iface.read_ram_byte(FRUIT_PRESENCE[i]) != 0 for i in (1, 2, 3, 4))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--cp", type=int, default=2)
    p.add_argument("--episodes", type=int, default=1, help="Episodes per seed.")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--settle-frames", type=int, default=5)
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--out", required=True)
    p.add_argument("--seed-indices", type=int, nargs="*", default=None,
                   help="Only roll out these seed indices (default: all).")
    args = p.parse_args()

    seeds = _load_cp_seeds(args.seeds, args.cp)
    print(f"Loaded {len(seeds)} CP{args.cp} seeds from {args.seeds}")

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.max_steps,  # don't stall-truncate, let env done handle it
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    base = stack.base
    gym_env = stack.gym
    base.reset(seed=0)
    model = PPO.load(args.model)

    os.makedirs(args.out, exist_ok=True)
    indices = args.seed_indices if args.seed_indices else list(range(len(seeds)))

    summary = []
    for i in indices:
        state = seeds[i]
        for ep in range(args.episodes):
            # Load and settle.
            base._interface.load_state(state)
            for _ in range(args.settle_frames):
                gym_env.step([0, 0, 0])
            # Get gym obs for model input.
            obs = stack.preprocessed._last_obs if hasattr(stack.preprocessed, "_last_obs") else None
            # If that's not exposed, just do one more noop step:
            if obs is None:
                obs, _, _, _, _ = gym_env.step([0, 0, 0])

            # Start state snapshot.
            iface = base._interface
            start_fruits = iface.read_ram_byte(FRUITS_ADDR)
            start_x = iface.read_ram_byte(X_ADDR)
            start_y = iface.read_ram_byte(Y_ADDR)
            fp = _read_fruits(iface)

            # Collect raw RGB frames (320x200) for the video.
            frames: List[np.ndarray] = []
            raw0 = base._last_raw_obs
            if raw0 is not None:
                frames.append(np.asarray(raw0, dtype=np.uint8).copy())

            steps = 0
            done = False
            reached = start_fruits
            while not done and steps < args.max_steps:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, trunc, info = gym_env.step(action)
                steps += 1
                raw = base._last_raw_obs
                if raw is not None:
                    frames.append(np.asarray(raw, dtype=np.uint8).copy())
                if trunc:
                    break
                fr = iface.read_ram_byte(FRUITS_ADDR)
                if fr < reached:
                    reached = fr

            end_x = iface.read_ram_byte(X_ADDR)
            end_y = iface.read_ram_byte(Y_ADDR)
            end_fruits = iface.read_ram_byte(FRUITS_ADDR)
            advanced = end_fruits < start_fruits

            tag = "WIN" if advanced else "FAIL"
            fname = f"seed{i:03d}_ep{ep}_fp{''.join(str(int(b)) for b in fp)}_sf{start_fruits}_to_{end_fruits}_{tag}_len{steps}.mp4"
            out_path = os.path.join(args.out, fname)
            imageio.mimsave(out_path, frames, fps=50)
            summary.append({
                "seed": i, "ep": ep,
                "start_xy": (start_x, start_y),
                "end_xy": (end_x, end_y),
                "start_fruits_rem": start_fruits,
                "end_fruits_rem": end_fruits,
                "fruits_present": fp,
                "steps": steps,
                "advanced": advanced,
                "path": out_path,
            })
            print(f"  seed {i} ep {ep}: start_y={start_y} fp={fp} "
                  f"fruits_rem {start_fruits}->{end_fruits} steps={steps} {tag}")

    print()
    wins = sum(1 for s in summary if s["advanced"])
    print(f"=== {wins}/{len(summary)} advanced at least one CP ===")


if __name__ == "__main__":
    main()
