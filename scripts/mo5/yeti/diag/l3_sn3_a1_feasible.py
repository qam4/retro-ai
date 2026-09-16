#!/usr/bin/env python3
"""Is SN3->A1 achievable at all? From Lsc4_top (SN3) seeds, try scripted plans
and report if any LANDS on A1 (grounded y<=80 within A1 x-extent ram 42-45).
joystick = [vertical(1=up,2=down), horizontal(1=right,2=left), fire]."""
from __future__ import annotations

import argparse
import collections
import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)
NOOP, R, L, U, JUMP, JR, JL = (
    [0, 0, 0],
    [0, 1, 0],
    [0, 2, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
    [0, 2, 1],
)


def load(stack, ifc, e):
    ifc.load_state(e[2])
    if not stack.preprocessed.restore_frame_stack(e[3]):
        stack.preprocessed.notify_state_loaded()
        stack.gym.step(NOOP)


def run(stack, ifc, e, plan, max_steps):
    load(stack, ifc, e)
    landed_a1 = False
    best_y = 255
    for i in range(max_steps):
        x = ifc.read_ram_byte(X)
        stack.gym.step(plan(i, x))
        x, y, p = ifc.read_ram_byte(X), ifc.read_ram_byte(Y), ifc.read_ram_byte(POSE)
        if p in SURF:
            best_y = min(best_y, y)
            if y <= 80 and 42 <= x <= 46:
                landed_a1 = True
                break
        if ifc.read_ram_byte(DEATH) == 65:
            break
    return landed_a1, best_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-seeds", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=80)
    args = ap.parse_args()
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.max_steps,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    ifc = stack.base._interface
    stack.base.reset(seed=0)
    d = pickle.load(open(f"{args.run}/checkpoints.pkl", "rb"))
    seeds = list(d["waypoints"]["Lsc4_top"][0])[: args.n_seeds]
    print(f"{len(seeds)} SN3 seeds; target A1 (ram42-45,y<=80)")

    # walk left to SN3 left edge (~ram52) then jump-left up to A1, at various
    # launch columns + jump delays; plus immediate jump-left.
    plans = {}
    for launch in (48, 50, 52, 54, 56):
        plans[f"walkL_JL@{launch}"] = lambda i, x, lx=launch: (L if x > lx else JL)
    plans["JL_repeat"] = lambda i, x: JL if i % 8 < 3 else L
    plans["JL_now"] = lambda i, x: JL
    for name, plan in plans.items():
        got = collections.Counter()
        landed = 0
        for e in seeds:
            a1, by = run(stack, ifc, e, plan, args.max_steps)
            landed += int(a1)
            got[by] += 1
        print(
            f"  {name:16s} landed_A1={landed}/{len(seeds)}  "
            f"best_y_hist={dict(sorted(got.items()))}"
        )


if __name__ == "__main__":
    main()
