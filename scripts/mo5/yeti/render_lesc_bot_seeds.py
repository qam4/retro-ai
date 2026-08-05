#!/usr/bin/env python3
"""Render the 6 Lesc_bot seed states (each loaded + ~16 no-op frames) into one
MP4 with an overlay (seed idx, agent x/y/pose, DEAD), so we can see whether the
agent is genuinely on a descending platform (rides) or just clipping/falling.
"""
from __future__ import annotations

import pickle

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SCALE = 3


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
    seeds = d["waypoints"]["Lesc_bot"][0]
    frames = []
    for i, s in enumerate(seeds):
        ifc.load_state(s[2])
        stack.preprocessed.notify_state_loaded()
        for t in range(16):
            rgb = np.asarray(base._last_raw_obs, np.uint8)
            img = Image.fromarray(rgb).resize(
                (rgb.shape[1] * SCALE, rgb.shape[0] * SCALE), Image.NEAREST
            )
            dr = ImageDraw.Draw(img)
            x, y, p = (
                ifc.read_ram_byte(X),
                ifc.read_ram_byte(Y),
                ifc.read_ram_byte(POSE),
            )
            dead = ifc.read_ram_byte(DEATH) == 65
            # mark the agent box (16x16 sprite at ram_x*4, y)
            ax = x * 4
            dr.rectangle(
                [ax * SCALE, y * SCALE, (ax + 16) * SCALE, (y + 16) * SCALE],
                outline=(255, 0, 0),
                width=2,
            )
            dr.text(
                (4, 4),
                f"Lesc_bot seed{i}  t={t}  x={x} y={y} pose={p}"
                + ("  DEAD" if dead else ""),
                fill=(255, 255, 0),
            )
            frames.append(np.asarray(img, np.uint8))
            gym_env.step([0, 0, 0])
    imageio.mimsave("debug/lesc_bot_seeds.mp4", frames, fps=6)
    print("saved debug/lesc_bot_seeds.mp4", len(frames), "frames")


if __name__ == "__main__":
    main()
