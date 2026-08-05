#!/usr/bin/env python3
"""Does the agent need to HOLD RIGHT after boarding the escalator?

Board = run right + jump into the wall at x32 (pose 13). We vary how many RIGHT
presses are held AFTER the jump (hold_r) before switching to pure NOOP, and
report: did it enter pose 13, how far down it rode ALIVE, and whether pure NOOP
sustains the ride to the bottom (y>=155) or it free-falls (pose 11) and dies.
"""
from __future__ import annotations

import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)
RIDE_POSE = 13


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
    seed = d["waypoints"]["Lgoat_a_top"][0][0][2]
    RIGHT, RJUMP, NOOP = [0, 1, 0], [0, 1, 1], [0, 0, 0]

    print("hold_r = RIGHT presses held AFTER the jump, then pure NOOP:", flush=True)
    for hold_r in range(0, 17):
        ifc.load_state(seed)
        stack.preprocessed.notify_state_loaded()
        for _ in range(5):
            gym_env.step(NOOP)
        seq = [RIGHT] * 10 + [RJUMP] * 4 + [RIGHT] * hold_r + [NOOP] * 55
        entered13 = False
        max_alive_y = 0
        died_y = None
        rode_bottom = False
        for a in seq:
            gym_env.step(a)
            y, p = ifc.read_ram_byte(Y), ifc.read_ram_byte(POSE)
            dead = ifc.read_ram_byte(DEATH) == 65
            if dead:
                died_y = y
                break
            if p == RIDE_POSE:
                entered13 = True
            max_alive_y = max(max_alive_y, y)
            if entered13 and y >= 155:
                rode_bottom = True
        print(
            f"  hold_r={hold_r:2d} entered_pose13={entered13} "
            f"rode_to_bottom={rode_bottom} max_alive_y={max_alive_y} "
            f"died_y={died_y}",
            flush=True,
        )


if __name__ == "__main__":
    main()
