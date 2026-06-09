#!/usr/bin/env python3
"""Diagnostic rollout from CP4 seed 7 — the floor-4 starting position
where v2 fails 100% of the time. Loads v2's final model, runs N
episodes from that one seed (max 500 frames each), records video,
and dumps a per-frame trace of (ram_x, ram_y, action, path-distance
to princess, lives, princess flag).

Goal: see what the policy actually does when stuck. Hypotheses to
distinguish:
  - "snowball death": agent walks left correctly but dies on snowball
  - "rightward bias": agent never tries left, oscillates around start
  - "policy doesn't understand reward gradient": agent moves left
    briefly then bounces back right
  - "stuck in a hole": agent can't physically move left from start
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build" / "ci-linux"))

from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import EnvConfig  # noqa: E402
from retro_ai.training.yeti_map import (  # noqa: E402
    agent_floor_from_pixel_y,
    build_navigation_map,
)

X_ADDR = 11090
Y_ADDR = 11089
LIVES_ADDR = 11095
FRUITS_ADDR = 11055
BONUS_HI = 11010
BONUS_LO = 11011
PRINCESS_FLAG = 11050


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model",
        type=Path,
        default=ROOT / "output/mo5/yeti/training/segment_4toP_v2/final_model.zip",
    )
    ap.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "output/mo5/yeti/seeds/v9_v5_cp4_user_seed.pkl",
    )
    ap.add_argument("--seed-idx", type=int, default=7)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--out", type=Path, default=ROOT / "debug/floor4_seed7")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=2000,
        stall_threshold=2000,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", cfg)
    base = stack.base
    gym_env = stack.gym
    base.reset(seed=0)
    iface = base._interface

    with args.pool.open("rb") as f:
        seeds = pickle.load(f)["checkpoints"][4]
    seed = seeds[args.seed_idx]
    print(
        f"loaded seed {args.seed_idx} from {args.pool} "
        f"({len(seeds)} CP4 seeds total)"
    )

    model = PPO.load(args.model)
    print(f"loaded model: {args.model}")

    nav = build_navigation_map()

    # Per-run summary.
    run_summaries = []

    for run in range(args.n_runs):
        iface.load_state(seed)
        obs = None
        for _ in range(5):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])

        # Initial state.
        x0 = iface.read_ram_byte(X_ADDR)
        y0 = iface.read_ram_byte(Y_ADDR)
        floor0 = agent_floor_from_pixel_y(y0)
        d0 = nav.path_distance_from_agent(floor0 or 4, x0 * 4 + 8, "princess")
        print(
            f"\nrun {run}: start ram=({x0},{y0}) px=({x0 * 4 + 8},{y0}) "
            f"floor={floor0} path_d_to_princess={d0}"
        )
        print(
            f"{'frame':>5} {'rx':>3} {'ry':>3} {'fl':>2} {'dP':>4} "
            f"{'lv':>2} {'bon':>5} {'flag':>4} {'fire':>4} {'dx':>3} {'dy':>3}"
        )

        prev_flag = iface.read_ram_byte(PRINCESS_FLAG)
        frames = []
        last_d = d0
        end_reason = None
        for f in range(args.max_frames):
            obs_chw = np.transpose(obs, (2, 0, 1))
            action, _ = model.predict(obs_chw, deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(action)

            x = iface.read_ram_byte(X_ADDR)
            y = iface.read_ram_byte(Y_ADDR)
            lv = iface.read_ram_byte(LIVES_ADDR)
            flag = iface.read_ram_byte(PRINCESS_FLAG)
            bonus = (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO)
            floor = agent_floor_from_pixel_y(y)
            d = (
                nav.path_distance_from_agent(floor, x * 4 + 8, "princess")
                if floor is not None
                else None
            )
            d_str = str(d) if d is not None else "?"

            # Print every frame the position or distance changes
            # (and the first 10 frames regardless).
            interesting = (
                f < 10
                or (d is not None and d != last_d)
                or x != x0
                or y != y0
                or flag != prev_flag
            )
            if interesting:
                fire, dx, dy = (int(action[0]), int(action[1]), int(action[2]))
                print(
                    f"{f:>5} {x:>3} {y:>3} "
                    f"{(floor if floor is not None else '-'):>2} {d_str:>4} "
                    f"{lv:>2} {bonus:>5} {flag:>4} {fire:>4} {dx:>3} {dy:>3}"
                )
                if d is not None:
                    last_d = d

            # Capture frame for video.
            raw = base._last_raw_obs
            if raw is not None:
                frames.append(np.asarray(raw, dtype=np.uint8))

            if flag == 1 and prev_flag == 0:
                end_reason = "princess_touched"
                print(f"  PRINCESS TOUCH at frame {f}")
                break
            prev_flag = flag
            if done or trunc:
                end_reason = "env_done" if done else "trunc"
                print(f"  episode ended at frame {f}: {end_reason}")
                break
            if lv < 1:
                end_reason = "no_lives"
                print(f"  no lives at frame {f}")
                break

        # Save video.
        out_path = args.out / f"seed{args.seed_idx}_run{run}.mp4"
        if frames:
            imageio.mimsave(out_path, frames, fps=50)
            print(f"  saved {out_path} ({len(frames)} frames)")
        run_summaries.append(
            {
                "run": run,
                "frames": len(frames),
                "end_reason": end_reason,
                "last_path_d": last_d,
            }
        )

    print("\nsummary:")
    for s in run_summaries:
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
