#!/usr/bin/env python3
"""Where does the agent actually cross the escalator column? Logs (x,y,pose) for
every gym step with x in the Lesc column, so we can see whether (ram33,y94) is a
point every crossing passes through, and at what y-granularity steps sample."""
from __future__ import annotations

import argparse
import collections
import pickle
import random

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()
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
    seeds = [s for k in ("Lgoat_a_top", "Lgoat_b_top") for s in d["waypoints"][k][0]]
    model = PPO.load(args.model)

    first_ride_y = []  # y at the FIRST pose-13 frame (boarding height)
    within_tol = 0  # episodes with a sample within +-2 of (33,94)
    ride_y_steps = collections.Counter()  # y granularity while riding
    crossed = 0
    for _ in range(args.episodes):
        e = random.choice(seeds)
        ifc.load_state(e[2])
        if stack.preprocessed.restore_frame_stack(e[3]):
            obs = stack.preprocessed.current_observation()
        else:
            stack.preprocessed.notify_state_loaded()
            obs, _, _, _, _ = gym_env.step([0, 0, 0])
        saw13 = None
        hit = False
        ys = []
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(action)
            x, y, p = (
                ifc.read_ram_byte(X),
                ifc.read_ram_byte(Y),
                ifc.read_ram_byte(POSE),
            )
            if p == 13:
                if saw13 is None:
                    saw13 = y
                ys.append(y)
            if (
                abs(x - 33) <= 2
                and abs(y - 94) <= 2
                and p in (set(yeti.SURFACE_POSES) | {13})
            ):
                hit = True
            if ifc.read_ram_byte(DEATH) == 65 or done or trunc:
                break
        if saw13 is not None:
            first_ride_y.append(saw13)
            crossed += 1
            for a, b in zip(ys, ys[1:]):
                ride_y_steps[b - a] += 1
        within_tol += int(hit)

    n = args.episodes
    print(
        f"episodes={n}  boarded(pose13)={crossed}  "
        f"detected within tol of (33,94)={within_tol}"
    )
    if first_ride_y:
        print(f"boarding y (first pose-13 frame): {sorted(first_ride_y)}")
        print(f"  min={min(first_ride_y)} max={max(first_ride_y)}")
        inb = sum(1 for y in first_ride_y if abs(y - 94) <= 2)
        print(f"  within +-2 of y=94: {inb}/{len(first_ride_y)}")
    print(f"y delta per STEP while riding (frame_skip=4): {dict(ride_y_steps)}")


if __name__ == "__main__":
    main()
