#!/usr/bin/env python3
"""Overlay fruit + ladder positions on the CP0 reference screenshot.

Based on user-reported ladder positions (x = upper-left pixel,
ladders are 16 px wide = 2 blocks, span vertically between floors):

  floor 1 (spawn): ladders at x=112 and x=272, go UP to floor 2
  floor 2:         1 ladder at x=232, goes UP to floor 3
  floor 3:         1 ladder at x=172, goes UP to floor 4
  floor 4:         1 ladder at x=200, goes UP to floor 5 (princess)
  floor 5:         Princess at x=304

Fruit pixel centres (measured in approach 15/16):
  fruit 1: (184, 184) - floor 1
  fruit 2: ( 80, 150) - floor 2
  fruit 3: (144, 120) - floor 3
  fruit 4: (272,  88) - floor 4
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
LADDERS = [
    (1, 2, 112),
    (1, 2, 272),
    (2, 3, 232),
    (3, 4, 168),
    (4, 5, 200),
]
PRINCESS_X = 304
OUT = "debug/cp0_ladders_annotated.png"


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
W, H = pil.size

for x in range(0, W + 1, 8):
    color = (64, 64, 64) if x % 32 else (120, 120, 120)
    draw.line([(x, 0), (x, H)], fill=color, width=1)
for y in range(0, H + 1, 8):
    color = (64, 64, 64) if y % 32 else (120, 120, 120)
    draw.line([(0, y), (W, y)], fill=color, width=1)
for x in range(0, W, 32):
    draw.text((x + 1, 0), str(x), fill=(255, 255, 0))
for y in range(0, H, 32):
    draw.text((0, y + 1), str(y), fill=(255, 255, 0))

F_COL = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 128, 255), 4: (255, 192, 0)}
for f, (ux, uy) in FRUITS_UL.items():
    c = F_COL[f]
    draw.rectangle([(ux, uy), (ux + 16, uy + 16)], outline=c, width=1)
    draw.text((ux + 2, uy - 10), f"F{f}", fill=c)

# Floor-top pixel y (where a standing sprite's FEET rest on the floor,
# minus 16 for a 16x16 sprite's UL). User confirmed ladders sit 16 px
# above these original anchors, so the right ranges are:
FLOOR_TOP_Y = {1: 200, 2: 168, 3: 136, 4: 104, 5: 72}
LADDER_COL = (255, 0, 255)
for fr, to, x in LADDERS:
    y_top = FLOOR_TOP_Y[to]
    y_bot = FLOOR_TOP_Y[fr]
    draw.rectangle([(x, y_top), (x + 16, y_bot)], outline=LADDER_COL, width=1)
    my = (y_top + y_bot) // 2
    draw.text((x + 1, my - 5), f"L{fr}{to}", fill=LADDER_COL)

# Princess: 16 px wide x 24 px tall. After verification: 8 px left of
# original report and 8 px below floor-5 top.
P_COL = (255, 255, 255)
px = PRINCESS_X
py = FLOOR_TOP_Y[5] - 24
draw.rectangle([(px, py), (px + 16, py + 24)], outline=P_COL, width=2)
draw.text((px, py - 10), "Princess", fill=P_COL)

pil.save(OUT)
print(f"Wrote {OUT}")
