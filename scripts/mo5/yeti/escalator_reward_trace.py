#!/usr/bin/env python3
"""Trace the ACTUAL training reward along goat -> descent -> land.

Rebuilds the exact L3 reward (fruit_bonus_path_progress_pbrs_grounded, the same
params as the v4 config) and feeds it the real per-frame RewardContext while we
drive the verified crossing. Logs per step: x, y, pose, airborne?, resolved
floor, potential phi, and the shaped reward. Also sums reward over the three
segments (approach / on-escalator ride / exit+land) so we can see whether
boarding+riding earns any gradient or is frozen.
"""
from __future__ import annotations

import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.rewards import RewardContext, create, reset_reward
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
REWARD_SURF = frozenset({0, 1, 2, 3, 4, 5, 8})  # reward's airborne set (no 13)
RIDE_POSE = 13
FRUITS_ADDR = yeti.FRUITS_ADDR
PRINCESS_FLAG = 11050


def main():
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=400,
        stall_threshold=400,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    base, gym_env, ifc = stack.base, stack.gym, stack.base._interface
    base.reset(seed=0)
    d = pickle.load(
        open("output/mo5/yeti/training/yeti_curriculum_l3_v4_15m/checkpoints.pkl", "rb")
    )
    seed = d["waypoints"]["Lgoat_a_top"][0][0][2]
    RIGHT, RJUMP, NOOP = [0, 1, 0], [0, 1, 1], [0, 0, 0]

    base_params = {
        "scale": 0.01,
        "fruit_scale": 0.01,
        "princess_scale": 0.05,
        "level": 3,
        "gamma": 1.0,
        "defer_fruit_credit": True,
        "waypoint_reward_tol": 2,
    }
    reward_old = create("fruit_bonus_path_progress_pbrs_grounded", dict(base_params))
    reward_new = create(
        "fruit_bonus_path_progress_pbrs_grounded",
        dict(base_params, ladder_segment_shaping=True),
    )

    def run(label, actions_fn):
        ifc.load_state(seed)
        stack.preprocessed.notify_state_loaded()
        for _ in range(5):
            gym_env.step(NOOP)
        reset_reward(reward_old)
        reset_reward(reward_new)
        pf = yeti.read_fruits_remaining(ifc)
        pb, ps, pl = yeti.read_bonus(ifc), yeti.read_score(ifc), yeti.read_lives(ifc)
        pflag = ifc.read_ram_byte(PRINCESS_FLAG)
        seg = {"old": {"ride": 0.0}, "new": {"ride": 0.0}}
        cum = {"old": 0.0, "new": 0.0}
        print(f"\n=== {label} ===", flush=True)
        print("  t   x   y pose air   r_old   r_new  phi_new", flush=True)
        for i, a in enumerate(actions_fn(ifc)):
            gym_env.step(a)
            x, y, p = (
                ifc.read_ram_byte(X),
                ifc.read_ram_byte(Y),
                ifc.read_ram_byte(POSE),
            )
            fr = yeti.read_fruits_remaining(ifc)
            bo, sc, li = (
                yeti.read_bonus(ifc),
                yeti.read_score(ifc),
                yeti.read_lives(ifc),
            )
            flag = ifc.read_ram_byte(PRINCESS_FLAG)
            dead = yeti.is_dead(ifc)
            ctx = RewardContext(
                prev_fruits=pf,
                curr_fruits=fr,
                prev_bonus=pb,
                curr_bonus=bo,
                prev_score=ps,
                curr_score=sc,
                prev_lives=pl,
                curr_lives=li,
                step_count=i,
                curr_y=y,
                curr_x=x,
                fruits_present=(fr != 0,),
                princess_touched=(flag == 1 and pflag == 0),
                pose=p,
                died=dead,
            )
            r_old = float(reward_old(ctx))
            r_new = float(reward_new(ctx))
            cum["old"] += r_old
            cum["new"] += r_new
            air = p not in REWARD_SURF
            phi_new = reward_new.prev_phi
            if p == RIDE_POSE:
                seg["old"]["ride"] += r_old
                seg["new"]["ride"] += r_new
            phistr = f"{phi_new:9.3f}" if phi_new is not None else "     None"
            print(
                f"  {i:2d} {x:3d} {y:3d}  {p:3d}  {int(air)} "
                f"{r_old:7.3f} {r_new:7.3f} {phistr}",
                flush=True,
            )
            pf, pb, ps, pl, pflag = fr, bo, sc, li, flag
            if dead:
                print("  (DEAD)", flush=True)
                break
        print(
            f"  RIDE-segment reward: OLD={seg['old']['ride']:.3f}  "
            f"NEW={seg['new']['ride']:.3f}   |   total OLD={cum['old']:.3f} "
            f"NEW={cum['new']:.3f}",
            flush=True,
        )

    def crossing(ifc):
        # entry, then NOOP until near the bottom, then jump right off
        acts = [RIGHT] * 10 + [RJUMP] * 4 + [RIGHT] * 14
        acts += [("ride_until", 148)]
        acts += [RJUMP] * 3 + [NOOP] * 12
        # expand the dynamic marker at run time
        out = []
        for a in acts:
            out.append(a)
        return _dynamic(ifc, out)

    def _dynamic(ifc, acts):
        for a in acts:
            if isinstance(a, tuple) and a[0] == "ride_until":
                while ifc.read_ram_byte(Y) < a[1]:
                    yield [0, 0, 0]
            else:
                yield a

    def ride_to_death(ifc):
        return iter([RIGHT] * 10 + [RJUMP] * 4 + [RIGHT] * 14 + [[0, 0, 0]] * 60)

    run("CROSSING (board -> ride -> exit onto eland)", crossing)
    run("RIDE-TO-DEATH (board -> ride past exit)", ride_to_death)


if __name__ == "__main__":
    main()
