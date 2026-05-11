#!/usr/bin/env python3
"""Dump a CP0 screenshot with grid overlay + agent position marker.

Loads the first CP0 cell from v9, settles, writes a 320x200 PNG with:
  - 8x8 pixel grid (lines every 8 px, darker every 32 px)
  - agent position marked with a red crosshair
  - column labels every 8 px across the top
  - row labels every 8 px along the left

Game coordinates on MO5:
  x in RAM ranges 0..~80 (8-px sprite width, 40-column area wide)
  y in RAM ranges ~16..200 (pixel coordinates within the 200px display)

Pixel-to-game mapping:
  pixel_x = ram_x * 4    (game x addresses half of a logical 80-wide grid?)
  pixel_y = ram_y        (game y is direct pixel y)

This script leaves the conversion as an empirical exercise: it marks
where RAM x,y puts the agent and you can compare to where the sprite
actually is on screen.
"""
from __future__ import annotations

import pickle

import numpy as np
from PIL import Image, ImageDraw
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

OUT = "debug/cp0_reference.png"
OUT_GRID = "debug/cp0_reference_grid.png"


with open("output/mo5/yeti/go_explore_v9/archive.pkl", "rb") as f:
    archive = pickle.load(f)

cp0 = None
cp2_nonzero_x = None
for key, entry in archive.items():
    v = key[2] if len(key) >= 3 else None
    if isinstance(v, (frozenset, set, list, tuple)):
        if len(v) == 0 and cp0 is None:
            cp0 = bytes(entry["state"])
        if cp2_nonzero_x is None and len(v) == 2:
            cp2_nonzero_x = bytes(entry["state"])

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
raw = base._last_raw_obs
img = np.asarray(raw, dtype=np.uint8)
Image.fromarray(img).save(OUT)
print(f"Wrote {OUT} (shape={img.shape})")

iface = base._interface
agent_x = iface.read_ram_byte(11090)
agent_y = iface.read_ram_byte(11089)
print(f"Agent RAM position: x={agent_x} y={agent_y}")

# Build grid overlay.
pil = Image.fromarray(img).convert("RGB")
draw = ImageDraw.Draw(pil)
W, H = pil.size  # 320, 200

# 8-px light grid
for x in range(0, W + 1, 8):
    color = (64, 64, 64) if x % 32 else (120, 120, 120)
    draw.line([(x, 0), (x, H)], fill=color, width=1)
for y in range(0, H + 1, 8):
    color = (64, 64, 64) if y % 32 else (120, 120, 120)
    draw.line([(0, y), (W, y)], fill=color, width=1)

# Column / row labels every 32 px (readable without clutter).
for x in range(0, W, 32):
    draw.text((x + 1, 0), str(x), fill=(255, 255, 0))
for y in range(0, H, 32):
    draw.text((0, y + 1), str(y), fill=(255, 255, 0))

# Agent RAM-coord marker. We don't know the RAM-to-pixel mapping yet,
# so mark both candidates:
#   (agent_x, agent_y)       — if RAM coords ARE pixel coords
#   (agent_x * 4, agent_y)   — if RAM x is pixel/4 (8-px-wide sprite grid)
# We'll figure out the right one by comparing to the visible sprite.
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# hypothesis A: direct pixel
ax, ay = int(agent_x), int(agent_y)
draw.line([(ax - 4, ay), (ax + 4, ay)], fill=RED, width=1)
draw.line([(ax, ay - 4), (ax, ay + 4)], fill=RED, width=1)
draw.text((ax + 5, ay), "A", fill=RED)

# hypothesis B: x*4
bx, by = int(agent_x * 4), int(agent_y)
draw.line([(bx - 4, by), (bx + 4, by)], fill=GREEN, width=1)
draw.line([(bx, by - 4), (bx, by + 4)], fill=GREEN, width=1)
draw.text((bx + 5, by), "B", fill=GREEN)

pil.save(OUT_GRID)
print(f"Wrote {OUT_GRID}")

# Also dump a second state with a non-zero agent x, so we can tell
# A vs B (at x=0 they both point to the same pixel).
base._interface.load_state(cp2_nonzero_x)
for _ in range(5):
    _ = base.step([0, 0, 0])
raw2 = np.asarray(base._last_raw_obs, dtype=np.uint8)
x2 = iface.read_ram_byte(11090)
y2 = iface.read_ram_byte(11089)
print(f"\nSecond state: agent RAM x={x2} y={y2}")
pil2 = Image.fromarray(raw2).convert("RGB")
d2 = ImageDraw.Draw(pil2)
for x in range(0, W + 1, 8):
    color = (64, 64, 64) if x % 32 else (120, 120, 120)
    d2.line([(x, 0), (x, H)], fill=color, width=1)
for y in range(0, H + 1, 8):
    color = (64, 64, 64) if y % 32 else (120, 120, 120)
    d2.line([(0, y), (W, y)], fill=color, width=1)
for x in range(0, W, 32):
    d2.text((x + 1, 0), str(x), fill=(255, 255, 0))
for y in range(0, H, 32):
    d2.text((0, y + 1), str(y), fill=(255, 255, 0))
ax2, ay2 = int(x2), int(y2)
d2.line([(ax2 - 4, ay2), (ax2 + 4, ay2)], fill=RED, width=1)
d2.line([(ax2, ay2 - 4), (ax2, ay2 + 4)], fill=RED, width=1)
d2.text((ax2 + 5, ay2), "A", fill=RED)
bx2, by2 = int(x2 * 4), int(y2)
d2.line([(bx2 - 4, by2), (bx2 + 4, by2)], fill=GREEN, width=1)
d2.line([(bx2, by2 - 4), (bx2, by2 + 4)], fill=GREEN, width=1)
d2.text((bx2 + 5, by2), "B", fill=GREEN)
pil2.save("debug/cp0_reference_grid2.png")
print("Wrote debug/cp0_reference_grid2.png")
print()
print("Key:")
print(f"  Red 'A' crosshair: RAM coords used as pixel coords ({ax}, {ay})")
print(f"  Green 'B' crosshair: RAM x*4 as pixel x ({bx}, {by})")
print("  Grid: 8-px squares, dark every 32 px")
print("  Column/row labels in yellow every 32 px")
print()
print("Open debug/cp0_reference_grid.png and tell me which crosshair")
print("is on the agent sprite. Then read off fruit pixel positions.")
