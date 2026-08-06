#!/usr/bin/env python3
"""Per-waypoint REACH RATE for the v5 policy from goat seeds.

Rolls out the trained model from the goat-platform seeds and, per episode,
records which route waypoints it reaches (grounded/ride pose within tol). Prints
the fraction of episodes that reach each, in route order (BOTTOM -> up), so we
can see exactly where the climb falls off (e.g. how often SN3 / the A-floors are
reached). Also reports fruit-collect and princess-touch rates.
"""
from __future__ import annotations

import argparse
import pickle
import random

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.yeti_map import get_level_map
from stable_baselines3 import PPO

X, Y, POSE, DEATH, FRUITS, PFLAG = 11090, 11089, 11092, 11004, yeti.FRUITS_ADDR, 11050
REACH_POSES = set(yeti.SURFACE_POSES) | {13}
TOL = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=800)
    args = ap.parse_args()

    m = get_level_map(3)
    # Trackable route waypoints (id -> (x_ram, y_px)); ordered bottom -> top.
    wps = dict(yeti.waypoints(3))  # {id: (x_ram, y_px, floor)}
    targets = {}
    for wid in (
        "Ldown_bot",
        "Lsc1_top",
        "Lsc2_top",
        "Lsc3_top",
        "Lsc4_top",
        "Lprincess_bot",
        "Lprincess_top",
    ):
        if wid in wps:
            targets[wid] = (wps[wid][0], wps[wid][1])
    fx_px, fy = m.fruit_centre_px[1]
    targets["F1_fruit"] = ((fx_px - 8) // 4, fy)
    # label map (route order, bottom -> top)
    order = [
        ("Ldown_bot", "BOTTOM (floor6, post-escalator)"),
        ("Lsc1_top", "BR      (floor7, 1st snowball ladder top)"),
        ("Lsc2_top", "SN1     (floor8)"),
        ("Lsc3_top", "SN2     (floor9)"),
        ("Lsc4_top", "SN3     (floor10, last graph node)"),
        ("F1_fruit", "A4 fruit(floor14)"),
        ("Lprincess_bot", "A5    (floor15)"),
        ("Lprincess_top", "PRIN  (floor16)"),
    ]

    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.max_steps,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    base, gym_env, ifc = stack.base, stack.gym, stack.base._interface
    base.reset(seed=0)
    d = pickle.load(open(args.seeds, "rb"))
    goat = [s[2] for k in ("Lgoat_a_top", "Lgoat_b_top") for s in d["waypoints"][k][0]]
    model = PPO.load(args.model)
    print(f"{len(goat)} goat seeds; {args.episodes} episodes", flush=True)

    reach = {wid: 0 for wid in targets}
    fruit_got = 0
    princess = 0
    for ep in range(args.episodes):
        ifc.load_state(random.choice(goat))
        stack.preprocessed.notify_state_loaded()
        obs = None
        for _ in range(5):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])
        seen = set()
        pf0 = ifc.read_ram_byte(FRUITS)
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(action)
            x, y, p = (
                ifc.read_ram_byte(X),
                ifc.read_ram_byte(Y),
                ifc.read_ram_byte(POSE),
            )
            if p in REACH_POSES:
                for wid, (wx, wy) in targets.items():
                    if wid not in seen and abs(x - wx) <= TOL and abs(y - wy) <= TOL:
                        seen.add(wid)
            if ifc.read_ram_byte(PFLAG) == 1:
                princess += 1
            if ifc.read_ram_byte(FRUITS) < pf0:
                fruit_got += 1
                pf0 = ifc.read_ram_byte(FRUITS)
            if done or trunc or ifc.read_ram_byte(DEATH) == 65:
                break
        for wid in seen:
            reach[wid] += 1

    n = args.episodes
    print("\nREACH RATE (fraction of episodes reaching, route order):", flush=True)
    for wid, label in order:
        if wid in reach:
            c = reach[wid]
            print(f"  {label:44} {c:4d}/{n} = {100*c/n:5.1f}%", flush=True)
    print(
        f"\n  fruit collected in {fruit_got}/{n} eps; "
        f"princess touched in {princess}/{n} eps",
        flush=True,
    )


if __name__ == "__main__":
    main()
