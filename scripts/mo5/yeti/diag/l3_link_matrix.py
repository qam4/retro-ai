#!/usr/bin/env python3
"""Per-LINK reach diagnostic for the L3 route.

For each consecutive pair in the route order, seed from A (faithful H-AB
restore) and measure how often the policy reaches B. Pinpoints exactly which
hand-offs are broken (the reverse curriculum can master segments in isolation
while a LINK between two seeded waypoints is never learned).
"""
from __future__ import annotations

import argparse
import pickle
import random

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

X, Y, POSE, DEATH, FRUITS = 11090, 11089, 11092, 11004, yeti.FRUITS_ADDR
REACH = set(yeti.SURFACE_POSES) | {13}
TOL = 3

ROUTE = [
    "Lsc4_top",
    "A1_launch",
    "A1",
    "A2_launch",
    "A2",
    "A3_launch",
    "A3",
    "A4_launch",
    "A4",
    "A5_launch",
    "A5",
    "Lprincess_top",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    wps = dict(yeti.waypoints(3))
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.max_steps,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    gym_env, ifc = stack.gym, stack.base._interface
    stack.base.reset(seed=0)
    d = pickle.load(open(f"{args.run}/checkpoints.pkl", "rb"))
    pools = d["waypoints"]
    model = PPO.load(args.model)

    print(f"link reach ({args.episodes} eps each, max {args.max_steps} steps)")
    print(f"{'from -> to':32s} {'reach':>6s}  {'died':>5s}  pool")
    for i in range(len(ROUTE) - 1):
        a, b = ROUTE[i], ROUTE[i + 1]
        if a not in pools or not pools[a][0] or b not in wps:
            print(f"  {a} -> {b}: (no seeds)")
            continue
        seeds = list(pools[a][0])
        bx, by, _f = wps[b]
        hit = died = 0
        for _ in range(args.episodes):
            e = random.choice(seeds)
            ifc.load_state(e[2])
            if stack.preprocessed.restore_frame_stack(e[3]):
                obs = stack.preprocessed.current_observation()
            else:
                stack.preprocessed.notify_state_loaded()
                obs, _, _, _, _ = gym_env.step([0, 0, 0])
            got = False
            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, trunc, _ = gym_env.step(action)
                x, y, p = (
                    ifc.read_ram_byte(X),
                    ifc.read_ram_byte(Y),
                    ifc.read_ram_byte(POSE),
                )
                if p in REACH and abs(x - bx) <= TOL and abs(y - by) <= TOL:
                    got = True
                    break
                if ifc.read_ram_byte(DEATH) == 65:
                    died += 1
                    break
                if done or trunc:
                    break
            hit += int(got)
        n = args.episodes
        print(
            f"  {a:14s} -> {b:14s} {100*hit/n:5.0f}%  {100*died/n:4.0f}%  "
            f"{len(seeds)}"
        )


if __name__ == "__main__":
    main()
