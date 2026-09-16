#!/usr/bin/env python3
"""Is the A1_launch -> A1 jump reliably executable? Brute-force a grid of
scripted plans from A1_launch seeds: reposition right by R steps, wait W, then
hold jump-left for H steps. Reports the best plan's landing rate.

joystick = [vertical(1=up,2=down), horizontal(1=right,2=left), fire]
"""
from __future__ import annotations

import argparse
import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)
NOOP, R, L, JL = [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 2, 1]


def attempt(stack, ifc, e, back, wait, hold, max_steps=60):
    ifc.load_state(e[2])
    if not stack.preprocessed.restore_frame_stack(e[3]):
        stack.preprocessed.notify_state_loaded()
        stack.gym.step(NOOP)
    seq = [R] * back + [NOOP] * wait + [JL] * hold
    for i in range(max_steps):
        act = seq[i] if i < len(seq) else L
        stack.gym.step(act)
        x, y, p = ifc.read_ram_byte(X), ifc.read_ram_byte(Y), ifc.read_ram_byte(POSE)
        # A1 = floor 11, standing y78, ram 42-45
        if p in SURF and abs(y - 78) <= 2 and 41 <= x <= 46:
            return True
        if ifc.read_ram_byte(DEATH) == 65:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=80,
        stall_threshold=80,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    ifc = stack.base._interface
    stack.base.reset(seed=0)
    d = pickle.load(open(f"{args.run}/checkpoints.pkl", "rb"))
    seeds = list(d["waypoints"]["A1_launch"][0])[: args.n_seeds]
    print(f"{len(seeds)} A1_launch seeds; grid back x wait x hold")
    results = []
    for back in (0, 2, 4, 6):
        for wait in (0, 2, 5):
            for hold in (2, 4, 6, 8):
                ok = sum(attempt(stack, ifc, e, back, wait, hold) for e in seeds)
                results.append((ok, back, wait, hold))
    results.sort(reverse=True)
    print("top plans (landed/n, back, wait, hold):")
    for ok, b, w, h in results[:8]:
        print(f"  {ok}/{len(seeds)}  back={b} wait={w} hold={h}")
    print(f"best={results[0][0]}/{len(seeds)}")


if __name__ == "__main__":
    main()
