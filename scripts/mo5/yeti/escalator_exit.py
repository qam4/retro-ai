#!/usr/bin/env python3
"""Phase-2 search: exit the pose-13 escalator ride onto the ELAND platform.

Boarding is easy (run right + jump into the wall at x32 -> pose 13 ride down
y94->158). The hard part is jumping RIGHT off at the right moment to land on
ELAND (ram_x 40-47, y158) before overshooting into free-fall (death at y182).

Entry (fixed): RIGHT*10 + RJUMP*4 + RIGHT*14  -> agent riding at x32, pose 13.
Then for each frame of the ride we branch: save state, try RJUMP*ej + NOOP*tail
and check for a landing GROUNDED + ALIVE on the eland side (x>=36, y in
[150,172]). Reports which ride-y and jump-hold produce a clean crossing and
saves the successful landing states.
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
    entry = [RIGHT] * 10 + [RJUMP] * 4 + [RIGHT] * 14

    # Collect the ride: replay entry, then step NOOP capturing a save-state at
    # every pose-13 frame (the moments we could choose to jump off).
    ifc.load_state(seed)
    stack.preprocessed.notify_state_loaded()
    for _ in range(5):
        gym_env.step(NOOP)
    for a in entry:
        gym_env.step(a)
    ride = []  # (y, state)
    for _ in range(50):
        gym_env.step(NOOP)
        y, p = ifc.read_ram_byte(Y), ifc.read_ram_byte(POSE)
        dead = ifc.read_ram_byte(DEATH) == 65
        if dead:
            break
        if p == RIDE_POSE:
            ride.append((y, ifc.save_state()))
    print(
        (
            f"ride frames captured (pose13): {len(ride)}; "
            f"y range {ride[0][0]}..{ride[-1][0]}"
            if ride
            else "no ride"
        ),
        flush=True,
    )

    crossings = []  # (ride_y, ej, land_x, land_y)
    land_states = []
    for ride_y, st in ride:
        for ej in (2, 3, 4, 5, 6):
            ifc.load_state(st)
            stack.preprocessed.notify_state_loaded()
            seq = [RJUMP] * ej + [NOOP] * 22
            landed = None
            for a in seq:
                gym_env.step(a)
                x, y, p = (
                    ifc.read_ram_byte(X),
                    ifc.read_ram_byte(Y),
                    ifc.read_ram_byte(POSE),
                )
                dead = ifc.read_ram_byte(DEATH) == 65
                if dead:
                    break
                if p in SURF and x >= 36 and 150 <= y <= 172:
                    landed = (x, y)
                    break
            if landed:
                crossings.append((ride_y, ej, landed[0], landed[1]))
                land_states.append(ifc.save_state())

    print(f"\nRESULT: crossings={len(crossings)}", flush=True)
    for c in crossings[:40]:
        print(
            f"  ride_y={c[0]} jumphold={c[1]} -> land(x={c[2]}, y={c[3]})", flush=True
        )
    if land_states:
        pickle.dump(
            {"eland_landings": land_states}, open("debug/eland_landings.pkl", "wb")
        )
        print(f"saved {len(land_states)} eland landing states", flush=True)
    else:
        print("NO clean exit found from the pose-13 ride.", flush=True)


if __name__ == "__main__":
    main()
