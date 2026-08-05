#!/usr/bin/env python3
"""Measure the agent's max rightward reach off the goat platform.

Tests standstill vs RUNNING jumps: does keeping RIGHT held through the jump
(momentum) carry the agent further right than walk->stop->jump? For each
(run, jhold, air_right) print the max ax reached and the final (x,y,pose,dead).
"""
from __future__ import annotations

import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)


def run_seq(gym_env, ifc, seq):
    maxx = 0
    last = None
    for a in seq:
        gym_env.step(a)
        x, y, p = ifc.read_ram_byte(X), ifc.read_ram_byte(Y), ifc.read_ram_byte(POSE)
        dead = ifc.read_ram_byte(DEATH) == 65
        maxx = max(maxx, x)
        last = (x, y, p, int(dead))
        if dead:
            break
    return maxx, last


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

    print("RUN-then-JUMP holding RIGHT in the air (momentum test):", flush=True)
    for run in (4, 6, 8, 10, 12):
        for jhold in (3, 4, 5, 6):
            for air_right in (0, 6, 12):
                ifc.load_state(seed)
                stack.preprocessed.notify_state_loaded()
                for _ in range(5):
                    gym_env.step(NOOP)
                seq = (
                    [RIGHT] * run + [RJUMP] * jhold + [RIGHT] * air_right + [NOOP] * 20
                )
                maxx, last = run_seq(gym_env, ifc, seq)
                print(
                    f"  run={run:2d} jhold={jhold} air_right={air_right:2d} "
                    f"-> maxx={maxx} last(x,y,pose,dead)={last}",
                    flush=True,
                )

    # Follow the pose-13 wall-slide with a LONG ride: land alive or die?
    print("\nLONG-RIDE trace: run=10 RJUMP=4 air_right=14 then NOOP*60", flush=True)
    ifc.load_state(seed)
    stack.preprocessed.notify_state_loaded()
    for _ in range(5):
        gym_env.step(NOOP)
    seq = [RIGHT] * 10 + [RJUMP] * 4 + [RIGHT] * 14 + [NOOP] * 60
    for i, a in enumerate(seq):
        gym_env.step(a)
        x, y, p = ifc.read_ram_byte(X), ifc.read_ram_byte(Y), ifc.read_ram_byte(POSE)
        dead = ifc.read_ram_byte(DEATH) == 65
        g = "G" if p in SURF else "."
        print(f"  t{i:02d} x={x} y={y} pose={p}{g} dead={int(dead)}", flush=True)
        if dead:
            break


if __name__ == "__main__":
    main()
