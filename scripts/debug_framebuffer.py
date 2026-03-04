#!/usr/bin/env python3
"""Debug: check framebuffer content at various stages of reset/play."""

import sys
import numpy as np

sys.path.insert(0, r"C:\src\retro-ai\python")
sys.path.insert(0, r"C:\src\retro-ai\build\dev-mingw")

import retro_ai_native as _native  # noqa: E402

BIOS = r"C:\src\videopac\roms\Philips C52 BIOS (19xx)(Philips)(FR).bin"
ROM = r"C:\src\videopac\roms\Satellite Attack (1981)(Philips)(EU).bin"


def dump_obs(label, obs):
    """Print framebuffer diagnostics."""
    arr = np.array(obs, dtype=np.uint8).reshape(240, 160, 3)
    unique = np.unique(arr.reshape(-1, 3), axis=0)
    print(f"\n{label}: {len(unique)} unique colors")
    for c in unique[:30]:
        count = int(np.sum(np.all(arr.reshape(-1, 3) == c, axis=1)))
        print(f"  RGB({c[0]:3d},{c[1]:3d},{c[2]:3d}) = {count:6d} px")
    # Check if any non-background pixels exist
    bg = arr[0, 0]  # assume top-left is background
    non_bg = np.sum(np.any(arr.reshape(-1, 3) != bg, axis=1))
    print(f"  Non-background pixels: {non_bg} / {240 * 160}")


# Create emulator directly via native module
emu = _native.VideopacRLInterface(BIOS, ROM, "survival")

# 1) Right after construction (before reset)
print("=" * 60)
print("STEP 1: Before reset (just after construction)")
result = emu.step([0])  # single NOOP to get a frame
dump_obs("Pre-reset NOOP", result.observation)

# 2) After reset
print("\n" + "=" * 60)
print("STEP 2: After reset (includes warmup + Key1 + post-warmup)")
result = emu.reset()
dump_obs("Post-reset", result.observation)

# 3) After 60 NOOPs
print("\n" + "=" * 60)
print("STEP 3: After 60 NOOPs")
for _ in range(60):
    result = emu.step([0])
dump_obs("60 NOOPs", result.observation)

# 4) After 300 more NOOPs
print("\n" + "=" * 60)
print("STEP 4: After 300 more NOOPs (360 total)")
for _ in range(300):
    result = emu.step([0])
dump_obs("360 NOOPs", result.observation)

# 5) Try pressing fire repeatedly
print("\n" + "=" * 60)
print("STEP 5: After 60 Fire presses")
for _ in range(60):
    result = emu.step([5])
dump_obs("60 Fires", result.observation)

# 6) Try a fresh reset with more warmup
print("\n" + "=" * 60)
print("STEP 6: Fresh reset, then 600 NOOPs")
result = emu.reset()
for _ in range(600):
    result = emu.step([0])
dump_obs("600 NOOPs after fresh reset", result.observation)

print("\nDone.")
