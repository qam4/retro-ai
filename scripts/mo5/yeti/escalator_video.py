#!/usr/bin/env python3
"""Roll out a trained L3 model from the GOAT-platform waypoint seeds and save
the episode that gets furthest into the escalator as an MP4, so we can watch
how the agent gets onto/down the escalator and why the jump-off fails.

Scoring per episode (to pick the most informative one):
  - enters the escalator column (ram_x in [28,40]) -> score = max y reached
    there (how far DOWN it rode; y increases downward),
  - reaches the ELAND side (ram_x >= 40 at y~150-170, alive) -> big bonus
    (an actual crossing).
Keeps only the current-best episode's frames in memory (writes on improvement).
"""
from __future__ import annotations

import argparse
import os
import pickle

import imageio.v2 as imageio
import numpy as np
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

X, Y, DEATH = 11090, 11089, 11004


def goat_seeds(pkl):
    d = pickle.load(open(pkl, "rb"))
    wp = d.get("waypoints", {})
    out = []
    for k in ("Lgoat_a_top", "Lgoat_b_top"):
        for s in wp.get(k, [[]])[0]:
            out.append(s[2])  # state_bytes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="debug/escalator_best.mp4")
    args = ap.parse_args()

    seeds = goat_seeds(args.seeds)
    print(f"{len(seeds)} goat seeds")
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.max_steps,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    base, gym_env, ifc = stack.base, stack.gym, stack.base._interface
    base.reset(seed=0)
    model = PPO.load(args.model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    best = -1
    import random

    for ep in range(args.episodes):
        ifc.load_state(random.choice(seeds))
        stack.preprocessed.notify_state_loaded()
        obs = None
        for _ in range(5):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])
        frames = [np.asarray(base._last_raw_obs, np.uint8).copy()]
        traj = []
        crossed = False
        score = 0
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(action)
            x, y = ifc.read_ram_byte(X), ifc.read_ram_byte(Y)
            dead = ifc.read_ram_byte(DEATH) == 65
            traj.append((x, y))
            frames.append(np.asarray(base._last_raw_obs, np.uint8).copy())
            if 28 <= x <= 40:
                score = max(score, y)
            if x >= 40 and 150 <= y <= 170 and not dead:
                crossed = True
                score = 1000
            if done or trunc or dead:
                break
        maxx = max(t[0] for t in traj) if traj else 0
        maxy_col = max((t[1] for t in traj if 28 <= t[0] <= 40), default=0)
        if crossed or score > best:
            best = score
            imageio.mimsave(args.out, frames, fps=20)
            print(
                f"ep{ep}: NEW BEST score={score} crossed={crossed} "
                f"maxx={maxx} maxy_in_col={maxy_col} len={len(traj)} -> {args.out}"
            )
    print(f"done. best score={best}")


if __name__ == "__main__":
    main()
