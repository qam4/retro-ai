#!/usr/bin/env python3
"""Render a trained policy playing Yeti FROM RESET to mp4, capturing
princess wins (and a few typical runs) so they can be watched.

Saves one mp4 per episode, named with the outcome (PRINCESS / NfruitsN).
Runs until it has captured ``--wins`` princess wins or ``--episodes``
episodes, whichever first; always keeps the first few for contrast.

Example::

    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
      python scripts/mo5/yeti/render_from_reset.py \\
        --model output/mo5/yeti/champions/v11_4750k/final_model.zip \\
        --out output/mo5/yeti/videos/v11 --episodes 60 --wins 3
"""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
from retro_ai.games.yeti_rollout import rollout_episode
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO


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
    model = PPO.load(args.model, device="auto")
    os.makedirs(args.out, exist_ok=True)

    wins = 0
    saved = 0
    for ep in range(args.episodes):
        result = rollout_episode(
            stack,
            model,
            level=1,
            fruits_total=4,
            max_steps=args.max_steps,
            stall_threshold=15,
            deterministic=False,
            keep_frames=True,
        )
        frames = result.frames or []
        is_win = result.princess_touched
        keep = is_win or ep < args.keep_first
        if keep and frames:
            tag = "PRINCESS" if is_win else f"{result.fruits_collected}fruits"
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
