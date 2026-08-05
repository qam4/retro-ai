#!/usr/bin/env python3
"""Measure the agent's actual (x,y) while on a ladder / escalator, vs the node x.

Answers: does the agent's pixel-x sit ON the ladder's node x during a climb (so
a TIGHT x-tolerance suffices), or does it wander (needing a loose tol that
causes the floor/ladder ambiguity we saw)? Drives the tolerance choice for the
segment resolver.
"""
from __future__ import annotations

import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.yeti_map import get_level_map

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004


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
    m = get_level_map(3)
    lgoat_a_cx = [cx for (n, _t, _b, cx) in m.ladders if n == "Lgoat_a"][0]
    print(
        f"Lgoat_a node cx={lgoat_a_cx}px (ram {(lgoat_a_cx - 8) // 4}); "
        "agent pixel_x = ram*4+8",
        flush=True,
    )

    NOOP = [0, 0, 0]
    for label, act in [("dim0=1", [1, 0, 0]), ("dim0=2", [2, 0, 0])]:
        ifc.load_state(seed)
        stack.preprocessed.notify_state_loaded()
        for _ in range(5):
            gym_env.step(NOOP)
        x0, y0, p0 = (
            ifc.read_ram_byte(X),
            ifc.read_ram_byte(Y),
            ifc.read_ram_byte(POSE),
        )
        print(f"\n[{label}] rest: ax={x0} ay={y0} pose={p0}", flush=True)
        for i in range(16):
            gym_env.step(act)
            x, y, p = (
                ifc.read_ram_byte(X),
                ifc.read_ram_byte(Y),
                ifc.read_ram_byte(POSE),
            )
            pix = x * 4 + 8
            print(
                f"  t{i:02d} ax={x} pix_x={pix} (|pix-cx|={abs(pix - lgoat_a_cx)}) "
                f"ay={y} pose={p}",
                flush=True,
            )
            if ifc.read_ram_byte(DEATH) == 65:
                print("  (DEAD)", flush=True)
                break


if __name__ == "__main__":
    main()
