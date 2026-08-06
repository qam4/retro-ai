#!/usr/bin/env python3
"""Sweep training snapshots: from-goat reach rate to key climb waypoints vs step.

Answers whether the snowball-climb skill exists in SOME snapshot (final-snapshot
oscillation, keep-best applies) or is never chained from goat. For each snapshot
(subsampled by --stride steps) rolls out N episodes from the goat seeds and
prints the fraction reaching BOTTOM / BR / SN1 / SN2 / SN3 / fruit / princess.
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
from retro_ai.training.yeti_map import get_level_map
from stable_baselines3 import PPO

X, Y, POSE, DEATH, FRUITS, PFLAG = 11090, 11089, 11092, 11004, yeti.FRUITS_ADDR, 11050
REACH_POSES = set(yeti.SURFACE_POSES) | {13}
TOL = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir")
    ap.add_argument("--stride", type=int, default=500_000)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()

    wps = dict(yeti.waypoints(3))
    key = [
        ("Ldown_bot", "BOTTOM"),
        ("Lsc1_top", "BR"),
        ("Lsc2_top", "SN1"),
        ("Lsc3_top", "SN2"),
        ("Lsc4_top", "SN3"),
    ]
    targets = {wid: (wps[wid][0], wps[wid][1]) for wid, _ in key if wid in wps}
    m = get_level_map(3)
    fx, fy = m.fruit_centre_px[1]
    targets["F1"] = ((fx - 8) // 4, fy)

    snaps = {}
    for p in glob.glob(os.path.join(args.run, "snapshots", "model_*_steps.zip")):
        mobj = re.search(r"model_(\d+)_steps", p)
        if mobj:
            snaps[int(mobj.group(1))] = p
    steps = sorted(s for s in snaps if s % args.stride == 0)
    print(
        f"{len(steps)} snapshots (stride {args.stride}); {args.episodes} eps each",
        flush=True,
    )

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
    d = pickle.load(open(os.path.join(args.run, "checkpoints.pkl"), "rb"))
    goat = [s[2] for k in ("Lgoat_a_top", "Lgoat_b_top") for s in d["waypoints"][k][0]]

    print("  step     BOTTOM  BR   SN1  SN2  SN3  fruit prin", flush=True)
    for st in steps:
        model = PPO.load(snaps[st])
        reach = {wid: 0 for wid in targets}
        fruit_got = princ = 0
        for _ in range(args.episodes):
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
                        if (
                            wid not in seen
                            and abs(x - wx) <= TOL
                            and abs(y - wy) <= TOL
                        ):
                            seen.add(wid)
                if ifc.read_ram_byte(PFLAG) == 1:
                    princ += 1
                if ifc.read_ram_byte(FRUITS) < pf0:
                    fruit_got += 1
                    pf0 = ifc.read_ram_byte(FRUITS)
                if done or trunc or ifc.read_ram_byte(DEATH) == 65:
                    break
            for wid in seen:
                reach[wid] += 1
        n = args.episodes

        def pct(wid):
            return 100 * reach.get(wid, 0) / n

        print(
            f"  {st:9d}  {pct('Ldown_bot'):4.0f}  {pct('Lsc1_top'):3.0f}  "
            f"{pct('Lsc2_top'):3.0f}  {pct('Lsc3_top'):3.0f}  {pct('Lsc4_top'):3.0f}  "
            f"{100*fruit_got/n:4.0f}  {100*princ/n:4.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
