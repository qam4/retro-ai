#!/usr/bin/env python3
"""Render a trained policy playing Yeti FROM RESET to mp4, capturing
princess wins (and a few typical runs) so they can be watched.

Saves one mp4 per episode, named with the outcome (PRINCESS / NfruitsN).
Runs until it has captured ``--wins`` princess wins or ``--episodes``
episodes, whichever first; always keeps the first few for contrast.

Example::

    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
      python scripts/render_from_reset.py \\
        --model output/mo5/yeti/champions/v11_4750k/final_model.zip \\
        --out output/mo5/yeti/videos/v11 --episodes 60 --wins 3
"""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

FRUITS_ADDR = 11055
LIVES_ADDR = 11095
PRINCESS_FLAG_ADDR = 11050


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--wins", type=int, default=3, help="stop after this many wins")
    p.add_argument("--keep-first", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--fps", type=int, default=50)
    args = p.parse_args()

    env_cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=15,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", env_cfg)
    base, gym_env, iface = stack.base, stack.gym, stack.base._interface
    model = PPO.load(args.model, device="auto")
    os.makedirs(args.out, exist_ok=True)

    wins = 0
    saved = 0
    for ep in range(args.episodes):
        obs, _ = gym_env.reset()
        prev_lives = iface.read_ram_byte(LIVES_ADDR)
        prev_pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        frames = []
        touched = False
        max_collected = 0
        for _ in range(args.max_steps):
            a, _ = model.predict(np.transpose(obs, (2, 0, 1)), deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(a)
            raw = base._last_raw_obs
            if raw is not None:
                frames.append(np.asarray(raw, dtype=np.uint8))
            fruits = iface.read_ram_byte(FRUITS_ADDR)
            lives = iface.read_ram_byte(LIVES_ADDR)
            pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
            max_collected = max(max_collected, 4 - fruits)
            if pr == 1 and prev_pr == 0:
                touched = True
                break
            prev_pr = pr
            if lives < prev_lives and prev_lives > 0:
                break
            prev_lives = lives
            if done or trunc:
                break

        is_win = touched
        keep = is_win or ep < args.keep_first
        if keep and frames:
            tag = "PRINCESS" if is_win else f"{max_collected}fruits"
            path = os.path.join(args.out, f"ep{ep:03d}_{tag}_len{len(frames)}.mp4")
            imageio.mimsave(path, frames, fps=args.fps)
            saved += 1
            print(f"  ep {ep}: {tag} len={len(frames)} -> {path}", flush=True)
        if is_win:
            wins += 1
            if wins >= args.wins:
                break

    print(f"\nDone: {wins} princess win(s), {saved} videos in {args.out}")


if __name__ == "__main__":
    main()
