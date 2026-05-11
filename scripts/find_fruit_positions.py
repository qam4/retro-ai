#!/usr/bin/env python3
"""Discover fruit (x, y) positions by examining RAM near the presence bytes.

Yeti has 4 fruits with known presence bytes at
  fruit 1: 0x2FAD, fruit 2: 0x2F00, fruit 3: 0x2E68, fruit 4: 0x2DD8
non-zero means "sprite is present", zero means "collected".

For each fruit, look at RAM near its presence byte across several CP0
save-states (where all 4 fruits are present). Bytes that are:
  - consistent across states (same value every time, since the map
    is fixed),
  - in the expected game-coordinate range (x in 0..80 roughly,
    y in 16..182 roughly),
are candidate position bytes.

Usage:
  env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
    python3 scripts/find_fruit_positions.py
"""
from __future__ import annotations

import pickle

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

PRESENCE = {1: 0x2FAD, 2: 0x2F00, 3: 0x2E68, 4: 0x2DD8}
WINDOW = 16  # bytes on each side of the presence byte


def main() -> None:
    with open("output/mo5/yeti/go_explore_v9/archive.pkl", "rb") as f:
        archive = pickle.load(f)

    # CP0 cells = no fruits collected = fruits_remaining==4
    cp0 = []
    for key, entry in archive.items():
        if len(key) < 3:
            continue
        v = key[2]
        if isinstance(v, (frozenset, set, list, tuple)) and len(v) == 0:
            cp0.append(bytes(entry["state"]))
    print(f"Found {len(cp0)} CP0 cells in v9")
    if not cp0:
        return

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

    # For each fruit, sample RAM in a window around the presence byte
    # across all CP0 states. Look for bytes that are the same across
    # every sample — those are position bytes (fixed per fruit).
    for fruit, addr in PRESENCE.items():
        print(f"\n=== Fruit {fruit} (presence byte at 0x{addr:04X} = {addr}) ===")
        samples = []
        for state in cp0:
            base._interface.load_state(state)
            # Settle 5 frames to be safe.
            for _ in range(5):
                base.step([0, 0, 0])
            row = []
            for offset in range(-WINDOW, WINDOW + 1):
                row.append(base._interface.read_ram_byte(addr + offset))
            samples.append(row)

        # Find bytes that are consistent across all samples.
        print(f"  Sampled {len(samples)} CP0 states")
        print("  offset  value  (decimal)")
        for i, offset in enumerate(range(-WINDOW, WINDOW + 1)):
            vals = [s[i] for s in samples]
            if len(set(vals)) == 1:
                v = vals[0]
                # Flag ranges plausible as game coordinates.
                is_x = 0 <= v <= 80
                is_y = 16 <= v <= 200
                tag = ""
                if offset == 0:
                    tag = " <- presence"
                elif is_x and is_y:
                    tag = " (possible x or y)"
                elif is_x:
                    tag = " (possible x)"
                elif is_y:
                    tag = " (possible y)"
                print(f"  {offset:+4d}   0x{v:02X} ({v:3d}){tag}")
            else:
                # varying — skip unless it's the presence
                if offset == 0:
                    print(f"  {offset:+4d}   VARIES  {sorted(set(vals))}  <- presence")

    # Sanity: check also v9 for CP4 cells and verify the presence bytes
    # actually go to zero there (confirms we have the right bytes).
    cp4 = []
    for key, entry in archive.items():
        v = key[2] if len(key) >= 3 else None
        if isinstance(v, (frozenset, set, list, tuple)) and len(v) == 4:
            cp4.append(bytes(entry["state"]))
    if cp4:
        print(f"\n=== Sanity: {len(cp4)} CP4 cells should have presence = 0 ===")
        state = cp4[0]
        base._interface.load_state(state)
        for _ in range(5):
            base.step([0, 0, 0])
        for fruit, addr in PRESENCE.items():
            v = base._interface.read_ram_byte(addr)
            print(f"  fruit {fruit}: presence 0x{v:02X} ({v})")


if __name__ == "__main__":
    main()
