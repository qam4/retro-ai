#!/usr/bin/env python3
"""Which x positions can the agent actually STAND on, per platform edge?

Why this is not obvious. Reading "the last position where pose is a surface pose and
y equals the floor's standing y" does NOT answer it: walking off an edge produces a
frame where y is still the standing value and the pose is still a walk pose, so the
agent reads as grounded while already committed to a fall. That is exactly the state
L4's `Low2_launch` seed pool is full of. Sampling at frame_skip 4 then makes the
apparent limit depend on where the walk started, which is why a first pass measured
left limits of +0, +4 and +8 px on different floors -- an artifact, since the collision
mechanic cannot vary by floor.

This probes it properly:

1. walk toward the edge one gym step at a time, saving the emulator state at each
   distinct x;
2. reload each saved state and hold NOOP for ``--settle`` steps;
3. a position is STANDABLE only if the agent is still at the floor's standing y in a
   surface pose afterwards.

Usage::

    env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms python3 \
      debug/l4_edge_limit.py --pool Low1 --floor 12
"""
from __future__ import annotations

import argparse
import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.yeti_map import get_level_map

NOOP = [0, 0, 0]
LEFT = [0, 2, 0]
RIGHT = [0, 1, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", default="output/mo5/yeti/training/yeti_curriculum_l4_v6_anchorfix_15m"
    )
    ap.add_argument("--pool", required=True)
    ap.add_argument("--floor", type=int, required=True)
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--profile", default="yeti_fruit_level4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--walk", type=int, default=30, help="gym steps of walking")
    ap.add_argument("--settle", type=int, default=12, help="NOOP steps used to confirm")
    args = ap.parse_args()

    lvl = get_level_map(args.level)
    plat = {p.floor: p for p in lvl.platforms}[args.floor]
    stand_y = lvl.floor_top_y[args.floor]

    with open(f"{args.run}/checkpoints.pkl", "rb") as f:
        pools = pickle.load(f)["waypoints"]
    seeds = pools[args.pool][0][: args.seeds]

    cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=10**6,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, cfg)
    ifc = stack.base._interface
    stack.gym.reset()

    print(
        f"floor {args.floor}: tiles px "
        f"[{plat.x_min}..{plat.x_max})  standing y {stand_y}\n"
        f"pool {args.pool}, {len(seeds)} seed(s), "
        f"confirm with {args.settle} NOOP steps\n"
    )

    for direction, act in (("LEFT", LEFT), ("RIGHT", RIGHT)):
        # collect candidate states along the walk
        cand: dict[int, bytes] = {}
        for s in seeds:
            ifc.load_state(bytes(s[2]))
            stack.preprocessed.notify_state_loaded()
            stack.gym.step(NOOP)
            for _ in range(args.walk):
                stack.gym.step(act)
                x, y = yeti.read_pos(ifc)
                if y == stand_y and yeti.read_pose(ifc) in yeti.SURFACE_POSES:
                    cand.setdefault(x * 4 + 8, ifc.save_state())
                if yeti.read_pose(ifc) == 11 or yeti.is_dead(ifc):
                    break
        # verify each candidate really holds
        rows = []
        for centre in sorted(cand):
            ifc.load_state(bytes(cand[centre]))
            stack.preprocessed.notify_state_loaded()
            ok = True
            for _ in range(args.settle):
                stack.gym.step(NOOP)
                x, y = yeti.read_pos(ifc)
                if y != stand_y or yeti.read_pose(ifc) not in yeti.SURFACE_POSES:
                    ok = False
                    break
            rows.append((centre, ok))
        good = [c for c, ok in rows if ok]
        bad = [c for c, ok in rows if not ok]
        print(f"  walking {direction:<5} reached centres {sorted(cand)}")
        print(f"     CONFIRMED standing : {good}")
        print(f"     read as grounded but FELL on NOOP : {bad}")
        if good:
            edge = plat.x_min if direction == "LEFT" else plat.x_max
            lim = min(good) if direction == "LEFT" else max(good)
            print(
                f"     -> {direction} limit {lim}   "
                f"= tile edge {edge} {lim - edge:+d} px\n"
            )
        else:
            print()


if __name__ == "__main__":
    main()
