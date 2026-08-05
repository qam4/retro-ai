#!/usr/bin/env python3
"""Render a filmstrip of the L3 escalator region over ~one cycle.

The moving platforms are grid-aligned sprites (not tilemap tiles), so a gridded
render lets us read the descending-platform column + row range and place a
mid-descent waypoint. Agent is parked at spawn; the escalator animates anyway.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

OUT = "debug/l3_escalator_strip.png"
SCALE = 6
N_FRAMES = 8
STEP_EVERY = 3  # game steps between captured frames
SETTLE = 200
# escalator region crop (pixels): around the wall (cols 17-18 = px136-152)
X0, X1 = 96, 200  # ram 24-50
Y0, Y1 = 56, 192


def main():
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=3000,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    ifc = stack.base._interface
    ifc.load_state(open("output/mo5/yeti/level3/level3_start.sav", "rb").read())
    stack.preprocessed.notify_state_loaded()
    for _ in range(SETTLE):
        stack.gym.step([0, 0, 0])

    tiles = []
    for _ in range(N_FRAMES):
        rgb = np.asarray(stack.base._last_raw_obs, dtype=np.uint8)[Y0:Y1, X0:X1]
        tiles.append(rgb.copy())
        for _ in range(STEP_EVERY):
            stack.gym.step([0, 0, 0])

    cw, ch = (X1 - X0) * SCALE, (Y1 - Y0) * SCALE
    gap = 16
    canvas = Image.new("RGB", (cw, (ch + gap) * N_FRAMES), (0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    for i, rgb in enumerate(tiles):
        img = Image.fromarray(rgb).resize((cw, ch), Image.NEAREST)
        oy = i * (ch + gap)
        canvas.paste(img, (0, oy))
        # 8px tile grid + ram-x / y labels (absolute game coords)
        for px in range(X0 - (X0 % 8), X1 + 1, 8):
            gx = (px - X0) * SCALE
            dr.line([(gx, oy), (gx, oy + ch)], fill=(50, 50, 50), width=1)
            if (px // 8) % 2 == 0:
                dr.text((gx + 1, oy + 1), f"x{px // 4}", fill=(120, 120, 255))
        for py in range(Y0 - (Y0 % 8), Y1 + 1, 8):
            gy = oy + (py - Y0) * SCALE
            dr.line([(0, gy), (cw, gy)], fill=(50, 50, 50), width=1)
            dr.text((1, gy + 1), f"y{py}", fill=(120, 255, 120))
        dr.text((cw - 70, oy + 2), f"t+{i * STEP_EVERY}", fill=(255, 255, 0))

    os.makedirs("debug", exist_ok=True)
    canvas.save(OUT)
    print(f"saved {OUT} ({canvas.size[0]}x{canvas.size[1]}), {N_FRAMES} frames")


if __name__ == "__main__":
    main()
