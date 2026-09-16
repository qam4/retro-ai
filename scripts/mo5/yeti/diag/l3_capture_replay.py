#!/usr/bin/env python3
"""Minimal reproduction: capture a WP state LIVE, then reload it and REPLAY the
same actions. Survival must reproduce.

Mimics training exactly: run an episode, at the first frame within tol of the
target WP call save_state() + export_frame_stack() (as the curriculum does),
keep playing and count how many gym steps the agent survives AFTER that capture
while RECORDING the actions taken. Then load_state() + restore_frame_stack() and
replay the identical action list.

  live == replay   -> restore is faithful; any earlier discrepancy was just
                      policy stochasticity choosing different actions.
  live >> replay   -> the reload is NOT equivalent to the live moment: something
                      the emulator needs is missing from the save-state (a real
                      bug, and it would invalidate every seeded start).
"""
from __future__ import annotations

import argparse
import pickle
import random

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SEED_POSES = set(yeti.SURFACE_POSES) | {13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--seed-wp", default="Lsc3_top", help="where to START")
    ap.add_argument("--target-wp", default="Lsc4_top", help="WP to capture at")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument(
        "--follow", type=int, default=40, help="steps to follow after capture"
    )
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    wps = dict(yeti.waypoints(3))
    tx, ty, _ = wps[args.target_wp]
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=400,
        stall_threshold=400,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    gym_env, ifc = stack.gym, stack.base._interface
    stack.base.reset(seed=0)
    d = pickle.load(open(f"{args.pools}/checkpoints.pkl", "rb"))
    seeds = list(d["waypoints"][args.seed_wp][0])
    model = PPO.load(args.model)
    random.seed(0)

    def load(entry):
        ifc.load_state(entry[2])
        if entry[3] is not None and stack.preprocessed.restore_frame_stack(entry[3]):
            return stack.preprocessed.current_observation()
        stack.preprocessed.notify_state_loaded()
        obs, _, _, _, _ = gym_env.step([0, 0, 0])
        return obs

    pairs = []
    for t in range(args.trials):
        obs = load(random.choice(seeds))
        cap_state = cap_stack = None
        acts = []
        live = 0
        # phase 1: play until we touch the target WP -> capture exactly as
        # the curriculum does (state + frame stack at that moment).
        for _ in range(200):
            a, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, done, trunc, _ = gym_env.step(a)
            x, y, p = (
                ifc.read_ram_byte(X),
                ifc.read_ram_byte(Y),
                ifc.read_ram_byte(POSE),
            )
            dead = ifc.read_ram_byte(DEATH) == 65
            if dead or done or trunc:
                break
            if p in SEED_POSES and abs(x - tx) <= 2 and abs(y - ty) <= 2:
                cap_state = ifc.save_state()
                cap_stack = stack.preprocessed.export_frame_stack()
                break
        if cap_state is None:
            continue
        # phase 2: keep playing, recording actions, counting survival
        for _ in range(args.follow):
            a, _ = model.predict(obs, deterministic=args.deterministic)
            acts.append(list(int(v) for v in (a if hasattr(a, "__len__") else [a])))
            obs, _, done, trunc, _ = gym_env.step(a)
            live += 1
            if ifc.read_ram_byte(DEATH) == 65 or done or trunc:
                break
        # phase 3: reload the captured state and REPLAY the same actions
        ifc.load_state(cap_state)
        restored = cap_stack is not None and stack.preprocessed.restore_frame_stack(
            cap_stack
        )
        if not restored:
            stack.preprocessed.notify_state_loaded()
        rep = 0
        for a in acts:
            gym_env.step(a)
            rep += 1
            if ifc.read_ram_byte(DEATH) == 65:
                break
        pairs.append((live, rep, restored))

    print(
        f"{len(pairs)} captures at {args.target_wp} (from {args.seed_wp} seeds), "
        f"deterministic={args.deterministic}"
    )
    print(f"  {'live':>6s} {'replay':>7s}  restored")
    mism = 0
    for live, rep, restored in pairs[:25]:
        flag = "" if live == rep else "   <-- MISMATCH"
        mism += int(live != rep)
        print(f"  {live:6d} {rep:7d}  {restored}{flag}")
    tot_mism = sum(1 for a, b, _ in pairs if a != b)
    print(f"\n  mismatches: {tot_mism}/{len(pairs)}")
    if pairs:
        print(
            f"  mean live={sum(p[0] for p in pairs)/len(pairs):.1f} "
            f"mean replay={sum(p[1] for p in pairs)/len(pairs):.1f}"
        )


if __name__ == "__main__":
    main()
