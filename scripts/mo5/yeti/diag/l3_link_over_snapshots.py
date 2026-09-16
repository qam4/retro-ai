#!/usr/bin/env python3
"""Track ONE link's reach across training snapshots (oscillation check).

The final snapshot is often a trough on these runs; this tells us whether the
skill exists in SOME snapshot (keep-best applies) or never.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import re

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
REACH = set(yeti.SURFACE_POSES) | {13}
TOL = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--from-wp", required=True)
    ap.add_argument("--to-wp", required=True)
    ap.add_argument("--stride", type=int, default=1_500_000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    wps = dict(yeti.waypoints(3))
    bx, by, _ = wps[args.to_wp]
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
    seeds = list(d["waypoints"][args.from_wp][0])

    snaps = {}
    for p in glob.glob(os.path.join(args.run, "snapshots", "model_*_steps.zip")):
        m = re.search(r"model_(\d+)_steps", p)
        if m:
            snaps[int(m.group(1))] = p
    steps = sorted(s for s in snaps if s % args.stride == 0)
    print(
        f"{args.from_wp} -> {args.to_wp}: {len(steps)} snapshots, "
        f"{args.episodes} eps each"
    )
    for st in steps:
        model = PPO.load(snaps[st])
        hit = 0
        for _ in range(args.episodes):
            e = random.choice(seeds)
            ifc.load_state(e[2])
            if stack.preprocessed.restore_frame_stack(e[3]):
                obs = stack.preprocessed.current_observation()
            else:
                stack.preprocessed.notify_state_loaded()
                obs, _, _, _, _ = gym_env.step([0, 0, 0])
            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, trunc, _ = gym_env.step(action)
                x, y, p = (
                    ifc.read_ram_byte(X),
                    ifc.read_ram_byte(Y),
                    ifc.read_ram_byte(POSE),
                )
                if p in REACH and abs(x - bx) <= TOL and abs(y - by) <= TOL:
                    hit += 1
                    break
                if ifc.read_ram_byte(DEATH) == 65 or done or trunc:
                    break
        print(f"  {st:9d}  {100*hit/args.episodes:5.0f}%", flush=True)


if __name__ == "__main__":
    main()
