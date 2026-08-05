#!/usr/bin/env python3
"""Gridded filmstrip of a SCRIPTED jump attempt from a goat-platform seed.

Renders one frame per action of the sequence
    NOOP*delay + RIGHT*walk + RJUMP*jhold + RIGHT*drift + NOOP*ride
cropped to the escalator region, with an 8px grid + absolute ram-x / y labels
and the agent's (x,y) printed per frame. Lets us SEE, frame by frame, where the
agent's jump goes relative to the moving platforms (why every board attempt
falls short and free-falls the shaft).
"""
from __future__ import annotations

import os
import pickle

import numpy as np
from PIL import Image, ImageDraw
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
OUT = "debug/l3_jump_strip.png"
SCALE = 4
# crop (pixels): goat platform (ram18-27=px72-108) .. eland (ram40-47=px160-188)
X0, X1 = 64, 200
Y0, Y1 = 64, 192
DELAY, WALK, JHOLD, DRIFT, RIDE = 2, 3, 4, 4, 18


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
    sb = d["waypoints"]["Lgoat_a_top"][0][0][2]

    RIGHT, RJUMP, NOOP = [0, 1, 0], [0, 1, 1], [0, 0, 0]
    seq = (
        [NOOP] * DELAY
        + [RIGHT] * WALK
        + [RJUMP] * JHOLD
        + [RIGHT] * DRIFT
        + [NOOP] * RIDE
    )

    ifc.load_state(sb)
    stack.preprocessed.notify_state_loaded()
    for _ in range(5):
        gym_env.step(NOOP)

    frames = []
    for act in seq:
        gym_env.step(act)
        rgb = np.asarray(base._last_raw_obs, np.uint8)[Y0:Y1, X0:X1].copy()
        ax, ay, p = (
            ifc.read_ram_byte(X),
            ifc.read_ram_byte(Y),
            ifc.read_ram_byte(POSE),
        )
        dead = ifc.read_ram_byte(DEATH) == 65
        frames.append((rgb, ax, ay, p, dead))
        if dead:
            break

    cw, ch = (X1 - X0) * SCALE, (Y1 - Y0) * SCALE
    gap = 18
    canvas = Image.new("RGB", (cw, (ch + gap) * len(frames)), (0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    for i, (rgb, ax, ay, p, dead) in enumerate(frames):
        img = Image.fromarray(rgb).resize((cw, ch), Image.NEAREST)
        oy = i * (ch + gap)
        canvas.paste(img, (0, oy))
        for px in range(X0 - (X0 % 8), X1 + 1, 8):
            gx = (px - X0) * SCALE
            dr.line([(gx, oy), (gx, oy + ch)], fill=(50, 50, 50), width=1)
            if (px // 8) % 2 == 0:
                dr.text((gx + 1, oy + 1), f"x{px // 4}", fill=(120, 120, 255))
        for py in range(Y0 - (Y0 % 8), Y1 + 1, 8):
            gy = oy + (py - Y0) * SCALE
            dr.line([(0, gy), (cw, gy)], fill=(50, 50, 50), width=1)
            dr.text((1, gy + 1), f"y{py}", fill=(120, 255, 120))
        # mark the agent's RAM position (x is ram*4 in pixels)
        apx, apy = ax * 4, ay
        if X0 <= apx <= X1 and Y0 <= apy <= Y1:
            mx, my = (apx - X0) * SCALE, oy + (apy - Y0) * SCALE
            dr.ellipse([mx - 4, my - 4, mx + 4, my + 4], outline=(255, 0, 0), width=2)
        tag = "DEAD" if dead else f"pose{p}"
        dr.text((cw - 150, oy + 2), f"t{i:02d} ax{ax} ay{ay} {tag}", fill=(255, 255, 0))

    os.makedirs("debug", exist_ok=True)
    canvas.save(OUT)
    print(
        f"saved {OUT} ({canvas.size[0]}x{canvas.size[1]}), {len(frames)} frames",
        flush=True,
    )


if __name__ == "__main__":
    main()
