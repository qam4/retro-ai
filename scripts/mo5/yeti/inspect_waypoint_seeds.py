#!/usr/bin/env python3
"""Inspect captured waypoint seed states to diagnose stale pools.

Motivation: in v8's checkpoints.pkl the floor-1 waypoints L12a_bot/L12a_top
sit at goal_score 0.00 (weight 1.0), so under the old sum-of-votes pick_start
they hogged ~12.6% of starts each. goal_score 0.0 is suspicious for a floor-1
waypoint (the agent gets a fruit almost always), so either the seeds are never
sampled or they are BAD states (agent dies immediately on reload).

This tool loads each waypoint pool's first few seed states, reports the
agent's position/pose/fruits/alive-ness the moment it's loaded, and then steps
NOOP for a survival window to see whether the state is a dead-on-arrival trap.

Usage:
  RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
    python scripts/mo5/yeti/inspect_waypoint_seeds.py \\
      --checkpoints .../checkpoints.pkl --level 2 --per-pool 5 --survival 60
"""
from __future__ import annotations

import argparse
import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--profile", default="yeti_fruit_level2")
    p.add_argument("--per-pool", type=int, default=5)
    p.add_argument("--survival", type=int, default=60, help="NOOP steps to test")
    p.add_argument("--only", default=None, help="comma-list of wp ids to check")
    args = p.parse_args()

    with open(args.checkpoints, "rb") as f:
        data = pickle.load(f)
    wps = data.get("waypoints", {})
    targets = yeti.waypoints(args.level)

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=10_000,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    gym_env, iface = stack.gym, stack.base._interface
    pre = getattr(stack, "preprocessed", None)
    gym_env.reset()

    only = set(args.only.split(",")) if args.only else None
    order = sorted(wps.items(), key=lambda kv: kv[1][1])  # by goal_score asc
    for wid, (states, gs) in order:
        if only and wid not in only:
            continue
        tx, ty, tfloor = targets.get(wid, (None, None, None))
        print(
            f"\n=== {wid}  goal_score={gs:.2f}  target=(x={tx},y={ty},f={tfloor}) "
            f"pool={len(states)} ==="
        )
        for i, s in enumerate(states[: args.per_pool]):
            state = bytes(s[2])
            iface.load_state(state)
            if pre is not None and hasattr(pre, "notify_state_loaded"):
                pre.notify_state_loaded()
            # settle 1 step so RAM reflects the loaded state
            gym_env.step([0, 0, 0])
            x, y = yeti.read_pos(iface)
            pose = yeti.read_pose(iface)
            fruits = yeti.read_fruits_remaining(iface)
            dead0 = yeti.is_dead(iface)
            grounded = pose in yeti.SURFACE_POSES
            # survival probe: NOOP and see if it dies / how far it drifts
            died_at = None
            for t in range(args.survival):
                gym_env.step([0, 0, 0])
                if yeti.is_dead(iface):
                    died_at = t
                    break
            xf, yf = yeti.read_pos(iface)
            print(
                f"  [{i}] load x={x} y={y} pose={pose} grounded={grounded} "
                f"fruits_left={fruits} dead_on_load={dead0} src_cp={s[0]} "
                f"| after {args.survival} NOOP: pos=({xf},{yf}) "
                f"died_at={died_at}"
            )


if __name__ == "__main__":
    main()
