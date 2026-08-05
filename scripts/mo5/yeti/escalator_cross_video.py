#!/usr/bin/env python3
"""Render the escalator crossing to MP4 so we can watch it.

Two clips from the same goat seed:
  crossing : run right + jump into the wall (board, pose 13), NOOP-ride down,
             then jump right near the bottom -> land on ELAND (the full cross).
  minimal  : run right + jump, then pure NOOP (no right held) -> shows the
             ride without an explicit exit.
Frames are upscaled 3x for visibility; a HUD prints frame x/y/pose/dead.
"""
from __future__ import annotations

import os
import pickle

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)
RIGHT, RJUMP, NOOP = [0, 1, 0], [0, 1, 1], [0, 0, 0]


def grab(base):
    rgb = np.asarray(base._last_raw_obs, np.uint8)
    img = Image.fromarray(rgb).resize(
        (rgb.shape[1] * 3, rgb.shape[0] * 3), Image.NEAREST
    )
    return np.asarray(img)


def main():
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=400,
        stall_threshold=400,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    base, gym_env, ifc = stack.base, stack.gym, stack.base._interface
    base.reset(seed=0)
    d = pickle.load(
        open("output/mo5/yeti/training/yeti_curriculum_l3_v4_15m/checkpoints.pkl", "rb")
    )
    seed = d["waypoints"]["Lgoat_a_top"][0][0][2]
    os.makedirs("debug", exist_ok=True)

    def reset_to_seed():
        ifc.load_state(seed)
        stack.preprocessed.notify_state_loaded()
        for _ in range(5):
            gym_env.step(NOOP)

    def step(a, frames):
        gym_env.step(a)
        frames.append(grab(base))
        return (
            ifc.read_ram_byte(X),
            ifc.read_ram_byte(Y),
            ifc.read_ram_byte(POSE),
            ifc.read_ram_byte(DEATH) == 65,
        )

    # ---- crossing: board, ride, exit near y150 -----------------------------
    reset_to_seed()
    frames = [grab(base)]
    for a in [RIGHT] * 10 + [RJUMP] * 4 + [RIGHT] * 14:
        step(a, frames)
    for _ in range(40):  # ride until near the bottom
        x, y, p, dead = step(NOOP, frames)
        if dead or y >= 150:
            break
    for a in [RJUMP] * 3 + [NOOP] * 18:  # jump right off onto eland
        x, y, p, dead = step(a, frames)
        if dead:
            break
    imageio.mimsave("debug/escalator_crossing.mp4", frames, fps=10)
    print(
        f"crossing: {len(frames)} frames, last (x{x},y{y},pose{p},dead{int(dead)}) "
        "-> debug/escalator_crossing.mp4",
        flush=True,
    )

    # ---- minimal: board then pure NOOP -------------------------------------
    reset_to_seed()
    frames = [grab(base)]
    for a in [RIGHT] * 10 + [RJUMP] * 4:
        step(a, frames)
    for _ in range(55):
        x, y, p, dead = step(NOOP, frames)
        if dead:
            break
    imageio.mimsave("debug/escalator_minimal.mp4", frames, fps=10)
    print(
        f"minimal:  {len(frames)} frames, last (x{x},y{y},pose{p},dead{int(dead)}) "
        "-> debug/escalator_minimal.mp4",
        flush=True,
    )


if __name__ == "__main__":
    main()
