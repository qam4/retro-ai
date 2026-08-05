#!/usr/bin/env python3
"""Does the current v4 policy ever BOARD the escalator (enter pose 13) on its
own? This decides whether an on-distribution `Lesc_ride` seed would ever
populate without injection.

Rolls out the v4 model from goat seeds and reports, per episode: whether it
entered pose 13 (riding), the deepest y reached while in pose 13, and the
max y reached in the escalator column at all. Prints aggregate boarding rate.
"""
from __future__ import annotations

import pickle
import random

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)
RIDE_POSE = 13
MODEL = "output/mo5/yeti/training/yeti_curriculum_l3_v4_15m/final_model.zip"
SEEDS = "output/mo5/yeti/training/yeti_curriculum_l3_v4_15m/checkpoints.pkl"
EPISODES = 120
MAX_STEPS = 250


def main():
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=MAX_STEPS,
        stall_threshold=MAX_STEPS,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    base, gym_env, ifc = stack.base, stack.gym, stack.base._interface
    base.reset(seed=0)
    d = pickle.load(open(SEEDS, "rb"))
    goat = [s[2] for k in ("Lgoat_a_top", "Lgoat_b_top") for s in d["waypoints"][k][0]]
    model = PPO.load(MODEL)
    print(f"{len(goat)} goat seeds; {EPISODES} episodes", flush=True)

    boarded = 0
    ride_depths = []
    for ep in range(EPISODES):
        ifc.load_state(random.choice(goat))
        stack.preprocessed.notify_state_loaded()
        obs = None
        for _ in range(5):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])
        entered = False
        max_ride_y = 0
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(action)
            p, y = ifc.read_ram_byte(POSE), ifc.read_ram_byte(Y)
            if p == RIDE_POSE:
                entered = True
                max_ride_y = max(max_ride_y, y)
            if done or trunc or ifc.read_ram_byte(DEATH) == 65:
                break
        if entered:
            boarded += 1
            ride_depths.append(max_ride_y)
    print(
        f"\nboarded (entered pose13): {boarded}/{EPISODES} "
        f"({100 * boarded / EPISODES:.1f}%)",
        flush=True,
    )
    if ride_depths:
        ride_depths.sort()
        print(
            f"  ride depth y: min={ride_depths[0]} "
            f"median={ride_depths[len(ride_depths) // 2]} max={ride_depths[-1]} "
            f"(goat=94, eland=158)",
            flush=True,
        )


if __name__ == "__main__":
    main()
