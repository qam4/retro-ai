#!/usr/bin/env python3
"""Manually walk a CP4 state up to the princess and dump per-frame
RAM. Used to verify the princess-touch detection rule before we
implement reward shaping for it.
"""
from __future__ import annotations

import pickle

import numpy as np
from PIL import Image
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

FRUITS_ADDR = 11055
LIVES_ADDR = 11095
BONUS_HI = 11010
BONUS_LO = 11011
SCORE_HI = 11093
SCORE_LO = 11094
X_ADDR = 11090
Y_ADDR = 11089

cfg = EnvConfig(
    profile="yeti_fruit",
    action_mode="joystick",
    max_steps=5000,
    stall_threshold=5000,
    resize=(84, 84),
)
stack = build_training_env("yeti_fruit", cfg)
base = stack.base
base.reset(seed=0)


def read_state():
    iface = base._interface
    return {
        "x": iface.read_ram_byte(X_ADDR),
        "y": iface.read_ram_byte(Y_ADDR),
        "fruits_remaining": iface.read_ram_byte(FRUITS_ADDR),
        "lives": iface.read_ram_byte(LIVES_ADDR),
        "bonus": (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO),
        "score": (iface.read_ram_byte(SCORE_HI) << 8) | iface.read_ram_byte(SCORE_LO),
    }


# Pick a CP4 state from v8's collected_states (they all start at
# (20, 146) in our sample).
with open("output/mo5/yeti/training/segment_3to4_v1/collected_states.pkl", "rb") as f:
    cp4_states = pickle.load(f)["states"]
state = cp4_states[0]
base._interface.load_state(state)
for _ in range(5):
    base.step([0, 0, 0])

print("=== CP4 starting state ===")
print(read_state())
print()

# Plan: from (ram_x=20, y=146) (floor 2):
#   1. walk right to L23 (UL pix x=232 = ram_x=58, but allowed ram_x range
#      we don't know — we observed L34 was 1-pixel-wide for descent;
#      hopefully UP is wider).
#   2. press UP to climb to floor 3.
#   3. walk left to L34 (ram_x=42).
#   4. press UP to climb to floor 4.
#   5. walk right to L45 (ram_x=50).
#   6. press UP to climb to floor 5.
#   7. walk right to princess (ram_x=78).
#   8. princess touch -> game should reset fruits to 4, bonus to 1000.


# The exact ladder x ranges for UP we haven't measured. Let me just
# attempt the journey. If we get stuck somewhere, we'll see in the trace.
def walk_to(ram_x_target, max_frames=400):
    for _ in range(max_frames):
        cur = base._interface.read_ram_byte(X_ADDR)
        cy = base._interface.read_ram_byte(Y_ADDR)
        if cy < 30:
            return  # death animation
        if cur < ram_x_target:
            base.step([0, 1, 0])  # right
        elif cur > ram_x_target:
            base.step([0, 2, 0])  # left
        else:
            return


def climb(target_y, max_frames=400):
    for _ in range(max_frames):
        cy = base._interface.read_ram_byte(Y_ADDR)
        if cy <= target_y:
            return
        base.step([1, 0, 0])  # up


print("Step 1: walk right to ram_x=58 (~L23):")
walk_to(58)
print(f"  arrived at {read_state()}")

print("Step 2: try UP to climb floor 2 -> 3:")
climb(target_y=120)
print(f"  arrived at {read_state()}")

# If we made it, now at floor 3.
if base._interface.read_ram_byte(Y_ADDR) <= 120:
    print("Step 3: walk left to ram_x=42 (~L34):")
    walk_to(42)
    print(f"  arrived at {read_state()}")

    print("Step 4: try UP to climb floor 3 -> 4:")
    climb(target_y=88)
    print(f"  arrived at {read_state()}")

    if base._interface.read_ram_byte(Y_ADDR) <= 90:
        print("Step 5: walk right to ram_x=50 (~L45):")
        walk_to(50)
        print(f"  arrived at {read_state()}")

        print("Step 6: try UP to climb floor 4 -> 5:")
        climb(target_y=56)
        print(f"  arrived at {read_state()}")

        if base._interface.read_ram_byte(Y_ADDR) <= 58:
            print("Step 7: walk right to ram_x=78 (princess):")
            # Per-frame trace this time so we capture princess touch
            print()
            print(
                f"{'f':>4} {'x':>3} {'y':>3} {'fr':>2} {'lv':>2} {'bonus':>5} {'score':>5}"
            )
            prev = read_state()
            print(
                f"  start  {prev['x']:>3} {prev['y']:>3} {prev['fruits_remaining']:>2} {prev['lives']:>2} {prev['bonus']:>5} {prev['score']:>5}"
            )
            for f in range(1, 200):
                base.step([0, 1, 0])  # right
                cur = read_state()
                if cur != prev or cur["fruits_remaining"] != prev["fruits_remaining"]:
                    print(
                        f"  {f:>4}   {cur['x']:>3} {cur['y']:>3} {cur['fruits_remaining']:>2} {cur['lives']:>2} {cur['bonus']:>5} {cur['score']:>5}"
                    )
                if (
                    cur["fruits_remaining"] != prev["fruits_remaining"]
                    and cur["fruits_remaining"] > prev["fruits_remaining"]
                ):
                    print(
                        f"  *** PRINCESS TOUCH at frame {f}: fruits {prev['fruits_remaining']} -> {cur['fruits_remaining']}, lives {prev['lives']} -> {cur['lives']}, bonus {prev['bonus']} -> {cur['bonus']} ***"
                    )
                    # Run a few more frames to see what happens.
                    for ff in range(20):
                        base.step([0, 0, 0])
                        cur2 = read_state()
                        print(
                            f"  +{ff+1:>3}   {cur2['x']:>3} {cur2['y']:>3} {cur2['fruits_remaining']:>2} {cur2['lives']:>2} {cur2['bonus']:>5} {cur2['score']:>5}"
                        )
                    break
                prev = cur

# Save final framebuffer for visual inspection.
raw = base._last_raw_obs
if raw is not None:
    Image.fromarray(np.asarray(raw, dtype=np.uint8)).save(
        "debug/walk_to_princess_final.png"
    )
    print("\nSaved debug/walk_to_princess_final.png")
