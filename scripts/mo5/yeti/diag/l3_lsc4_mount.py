#!/usr/bin/env python3
"""Test the Lsc4 mount mechanics. Drive the v6 model from SN2 seeds until the
agent is ALIVE near the ladder column, then FORCE 'up' for a while and see if
it climbs (y drops toward SN3 y86). Sweeps the x-position at which we start
forcing up, to measure the mount window width.
"""
from __future__ import annotations

import argparse
import collections
import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)
UP = [1, 0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed-wp", default="Lsc3_top")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument(
        "--mount-x", type=int, default=70, help="force UP once alive at this x (+-1)"
    )
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
    seeds = list(d["waypoints"][args.seed_wp][0])
    model = PPO.load(args.model)

    import random

    reached_ladder = 0  # got alive to mount-x band on SN2
    climbed = 0  # y dropped below 106 while forcing up
    hit_sn3 = 0
    climb_outcomes = collections.Counter()
    for _ in range(args.episodes):
        entry = random.choice(seeds)
        ifc.load_state(entry[2])
        if stack.preprocessed.restore_frame_stack(entry[3]):
            obs = stack.preprocessed.current_observation()
        else:
            stack.preprocessed.notify_state_loaded()
            obs, _, _, _, _ = gym_env.step([0, 0, 0])
        forcing = False
        force_left = 0
        best_y_forced = 255
        got_ladder = False
        for _ in range(args.max_steps):
            x = ifc.read_ram_byte(X)
            y = ifc.read_ram_byte(Y)
            p = ifc.read_ram_byte(POSE)
            if not forcing and p in SURF and x == args.mount_x and 106 <= y <= 114:
                forcing = True
                force_left = 20
                got_ladder = True
            if forcing:
                obs, _, done, trunc, _ = gym_env.step(UP)
                force_left -= 1
                yy, pp = ifc.read_ram_byte(Y), ifc.read_ram_byte(POSE)
                if pp in SURF:
                    best_y_forced = min(best_y_forced, yy)
                if force_left <= 0:
                    break
            else:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, trunc, _ = gym_env.step(action)
            if done or trunc or ifc.read_ram_byte(DEATH) == 65:
                break
        if got_ladder:
            reached_ladder += 1
            if best_y_forced < 106:
                climbed += 1
            if best_y_forced <= 88:
                hit_sn3 += 1
            climb_outcomes[best_y_forced] += 1
    print(
        f"mount-x={args.mount_x}: reached_ladder_alive={reached_ladder}/{args.episodes}"
        f"  climbed(<106)={climbed}  hit_SN3(<=88)={hit_sn3}"
    )
    print("best_y while forcing UP (count):", dict(sorted(climb_outcomes.items())))


if __name__ == "__main__":
    main()
