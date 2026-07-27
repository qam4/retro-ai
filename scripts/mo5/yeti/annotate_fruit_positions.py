#!/usr/bin/env python3
"""Overlay fruit positions on the CP0 reference screenshot for verification.

User-reported fruit UPPER-LEFT pixel coordinates (sprite is 16x16):
  fruit 1: (184, 184)
  fruit 2: ( 72, 142)
  fruit 3: (136, 112)
  fruit 4: (264,  80)
"""
from __future__ import annotations

import pickle

import numpy as np
from PIL import Image, ImageDraw
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

FRUITS_UL = {
    1: (176, 176),
    2: (72, 142),
    3: (136, 112),
    4: (264, 80),
}
SPRITE_SIZE = 16
OUT = "debug/cp0_fruits_annotated.png"


with open("output/mo5/yeti/go_explore_v9/archive.pkl", "rb") as f:
    archive = pickle.load(f)
cp0 = None
for key, entry in archive.items():
    v = key[2] if len(key) >= 3 else None
    if isinstance(v, (frozenset, set, list, tuple)) and len(v) == 0:
        cp0 = bytes(entry["state"])
        break

env_cfg = EnvConfig(
    profile="yeti_fruit",
    action_mode="joystick",
    max_steps=1_000_000,
    stall_threshold=1_000_000,
    resize=(84, 84),
)
stack = build_training_env("yeti_fruit", env_cfg)
base = stack.base
base.reset(seed=0)
base._interface.load_state(cp0)
for _ in range(5):
    _ = base.step([0, 0, 0])
img = np.asarray(base._last_raw_obs, dtype=np.uint8)
pil = Image.fromarray(img).convert("RGB")
draw = ImageDraw.Draw(pil)

# 8px grid, dark every 32.
W, H = pil.size
for x in range(0, W + 1, 8):
    color = (64, 64, 64) if x % 32 else (120, 120, 120)
    draw.line([(x, 0), (x, H)], fill=color, width=1)
for y in range(0, H + 1, 8):
    color = (64, 64, 64) if y % 32 else (120, 120, 120)
    draw.line([(0, y), (W, y)], fill=color, width=1)

COLORS = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 128, 255), 4: (255, 192, 0)}
for f, (ux, uy) in FRUITS_UL.items():
    c = COLORS[f]
    cx = ux + SPRITE_SIZE // 2
    cy = uy + SPRITE_SIZE // 2
    # Draw the 16x16 sprite box.
    draw.rectangle([(ux, uy), (ux + SPRITE_SIZE, uy + SPRITE_SIZE)], outline=c, width=1)
    # Crosshair at centre.
    draw.line([(cx - 6, cy), (cx + 6, cy)], fill=c, width=1)
    draw.line([(cx, cy - 6), (cx, cy + 6)], fill=c, width=1)
    # Label.
    draw.text((ux + 2, uy - 10), f"F{f}", fill=c)
    print(f"fruit {f}: UL=({ux},{uy}) centre=({cx},{cy})")

pil.save(OUT)
print(f"\nWrote {OUT}")
