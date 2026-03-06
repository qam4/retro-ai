#!/usr/bin/env python3
"""Debug: visualize what the vision reward system sees.

Runs the emulator for a few hundred frames, saves the framebuffer as PNG,
and prints the detected score at each step.

Usage:
    source setup_env.sh
    python scripts/debug_vision_ocr.py

Requires: numpy, Pillow (pip install Pillow)
"""

import os
import sys
import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("pip install Pillow  — needed to save debug images")

import retro_ai_native as _native

ROM_DIR = os.environ.get("RETRO_AI_ROM_DIR", "roms")
BIOS = os.path.join(ROM_DIR, "videopac", "Philips C52 BIOS (19xx)(Philips)(FR).bin")
ROM = os.path.join(ROM_DIR, "videopac",
                   "Course de Voitures + Autodrome + Cryptogramme (1980)(Philips)(FR).bin")

OUT_DIR = "output/debug_vision"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Create emulator with vision reward mode ---
print(f"BIOS: {BIOS} (exists={os.path.exists(BIOS)})")
print(f"ROM:  {ROM} (exists={os.path.exists(ROM)})")

try:
    emu = _native.VideopacRLInterface(BIOS, ROM, "vision")
except Exception as e:
    print(f"Failed to create emulator with vision mode: {e}")
    print("Falling back to survival mode to inspect framebuffer...")
    emu = _native.VideopacRLInterface(BIOS, ROM, "survival")

obs_space = emu.observation_space()
W, H, C = obs_space.width, obs_space.height, obs_space.channels
print(f"Observation: {W}x{H}x{C}")

# --- Reset and run startup sequence (Key1 twice for Course Auto) ---
result = emu.reset()

# Key1 = action 11, hold 10 frames (level select)
for _ in range(10):
    result = emu.step([11])

# Post-delay: 120 NOOP frames
for _ in range(120):
    result = emu.step([0])

print("Startup complete, saving frames...")

def save_frame(obs, filename, label=""):
    """Save observation as PNG with optional label."""
    arr = np.array(obs, dtype=np.uint8).reshape(H, W, C)
    img = Image.fromarray(arr, "RGB")
    # Scale up 3x for visibility
    img = img.resize((W * 3, H * 3), Image.NEAREST)
    if label:
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), label, fill=(255, 255, 0))
    img.save(os.path.join(OUT_DIR, filename))

# Save initial gameplay frame
save_frame(result.observation, "frame_000_start.png", "After startup")

# Run 300 frames, save every 50, print reward each step
print("\nRunning 300 frames...")
print(f"{'Frame':>6} {'Reward':>8} {'Done':>5} {'Info'}")
print("-" * 60)

for i in range(1, 301):
    # Alternate between NOOP and random-ish actions
    action = [0]  # NOOP
    if i % 30 < 10:
        action = [1]  # UP
    elif i % 30 < 20:
        action = [2]  # DOWN

    result = emu.step(action)

    if i % 10 == 0 or result.reward != 0.0 or result.done:
        info_str = result.info[:80] if result.info else ""
        print(f"{i:6d} {result.reward:8.2f} {result.done!s:>5} {info_str}")

    if i % 50 == 0:
        fname = f"frame_{i:03d}.png"
        save_frame(result.observation, fname, f"Frame {i}, reward={result.reward:.2f}")

    if result.done:
        print(f"Episode ended at frame {i}")
        save_frame(result.observation, f"frame_{i:03d}_done.png", "DONE")
        break

print(f"\nDebug images saved to {OUT_DIR}/")
print("Check the PNGs to see what the emulator is rendering.")
