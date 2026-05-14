#!/usr/bin/env python3
"""Roll out a trained model from CP seeds, overlay live reward HUD.

For each step, computes the reward the configured formula would pay
given the agent's current state, and burns the result onto the video
frame: per-step reward, cumulative episode reward, target distances,
and per-fruit best-d state. Useful for "did the shaping actually
guide what we wanted?".

Usage:
  env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
    python3 scripts/rollout_with_reward_overlay.py \\
      --model output/.../final_model.zip \\
      --seeds output/mo5/yeti/seeds/v9_checkpoints_v2.pkl \\
      --cp 2 \\
      --reward fruit_bonus_path_progress \\
      --out debug/rollout_v7_overlay \\
      --seed-indices 17 36 24
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import List

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.rewards import RewardContext, create as create_reward
from retro_ai.training.rewards import reset_reward
from retro_ai.training.run_config import EnvConfig

# RAM addresses we read for context.
FRUITS_ADDR = 11055
LIVES_ADDR = 11095
X_ADDR = 11090
Y_ADDR = 11089
BONUS_HI = 11010
BONUS_LO = 11011
SCORE_HI = 11093
SCORE_LO = 11094
FRUIT_PRESENCE = {1: 0x2FAD, 2: 0x2F00, 3: 0x2E68, 4: 0x2DD8}


def _load_cp_seeds(path: str, cp: int) -> List[bytes]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "checkpoints" in data:
        return [bytes(s) for s in data["checkpoints"][cp]]
    raise SystemExit("Expected curriculum-format checkpoints file.")


def _draw_hud(
    rgb: np.ndarray,
    step: int,
    reward: float,
    cum_reward: float,
    fruits_present: tuple,
    best_d: dict,
    agent_x: int,
    agent_y: int,
) -> np.ndarray:
    """Render the per-step HUD on top of the framebuffer."""
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Black backing strip on the right side so text reads over the
    # game art without messing it up.
    draw.rectangle([(0, 0), (320, 28)], fill=(0, 0, 0))
    # Top-left: instantaneous reward + cumulative.
    color = (0, 255, 0) if reward > 0 else (200, 200, 200)
    draw.text((4, 2), f"step {step:>4}  r={reward:+.3f}  Σ={cum_reward:.2f}",
              fill=color)
    # Below: agent position + which fruits remain.
    fp_str = "".join("Y" if p else "-" for p in fruits_present)
    draw.text((4, 14), f"({agent_x:>3},{agent_y:>3})  fruits={fp_str}",
              fill=(255, 255, 255))
    # Right: best_d per fruit, lined up.
    bd_strs = []
    for fid in (1, 2, 3, 4):
        v = best_d.get(fid)
        bd_strs.append(f"{fid}:{'-' if v is None else v}")
    draw.text((180, 14), "  ".join(bd_strs), fill=(255, 220, 0))
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--cp", type=int, default=2)
    p.add_argument("--reward", required=True,
                   help="Reward formula name (matches the trained model's reward).")
    p.add_argument("--reward-params", default=None,
                   help="Optional JSON dict of reward params; defaults match training.")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--settle-frames", type=int, default=5)
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--out", required=True)
    p.add_argument("--seed-indices", type=int, nargs="*", default=None)
    args = p.parse_args()

    if args.reward_params is None:
        reward_params = {"scale": 0.01, "fruit_scale": 0.01}
    else:
        import json

        reward_params = json.loads(args.reward_params)

    seeds = _load_cp_seeds(args.seeds, args.cp)
    print(f"Loaded {len(seeds)} CP{args.cp} seeds")

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.max_steps,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    base = stack.base
    gym_env = stack.gym
    base.reset(seed=0)
    model = PPO.load(args.model)

    # Build a fresh reward function for the rollout (mirrors training).
    reward_fn = create_reward(args.reward, reward_params)

    os.makedirs(args.out, exist_ok=True)
    indices = args.seed_indices if args.seed_indices else list(range(len(seeds)))

    for i in indices:
        state = seeds[i]
        base._interface.load_state(state)
        # Settle and reset reward state.
        for _ in range(args.settle_frames):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])
        reset_reward(reward_fn)

        iface = base._interface

        def _read_state():
            fruits = iface.read_ram_byte(FRUITS_ADDR)
            lives = iface.read_ram_byte(LIVES_ADDR)
            bonus = (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO)
            score = (iface.read_ram_byte(SCORE_HI) << 8) | iface.read_ram_byte(SCORE_LO)
            x = iface.read_ram_byte(X_ADDR)
            y = iface.read_ram_byte(Y_ADDR)
            fp = tuple(
                iface.read_ram_byte(FRUIT_PRESENCE[k]) != 0 for k in (1, 2, 3, 4)
            )
            return fruits, lives, bonus, score, x, y, fp

        fruits, lives, bonus, score, sx, sy, fp = _read_state()
        prev_fruits, prev_lives, prev_bonus, prev_score = fruits, lives, bonus, score

        frames = []
        cum_reward = 0.0
        steps = 0
        done = False
        while not done and steps < args.max_steps:
            # Model expects channels-first observations.
            obs_chw = np.transpose(obs, (2, 0, 1))
            action, _ = model.predict(obs_chw, deterministic=False)
            obs, _, done, trunc, _ = gym_env.step(action)
            steps += 1

            fruits, lives, bonus, score, x, y, fp = _read_state()
            ctx = RewardContext(
                prev_fruits=prev_fruits,
                curr_fruits=fruits,
                prev_bonus=prev_bonus,
                curr_bonus=bonus,
                prev_score=prev_score,
                curr_score=score,
                prev_lives=prev_lives,
                curr_lives=lives,
                step_count=steps,
                curr_y=y,
                curr_x=x,
                fruits_present=fp,
            )
            r = float(reward_fn(ctx))
            cum_reward += r
            best_d = (
                dict(reward_fn.best_d)
                if hasattr(reward_fn, "best_d")
                else {}
            )

            raw = base._last_raw_obs
            if raw is not None:
                hud = _draw_hud(
                    np.asarray(raw, dtype=np.uint8),
                    step=steps,
                    reward=r,
                    cum_reward=cum_reward,
                    fruits_present=fp,
                    best_d=best_d,
                    agent_x=x,
                    agent_y=y,
                )
                frames.append(hud)

            prev_fruits, prev_lives, prev_bonus, prev_score = (
                fruits, lives, bonus, score,
            )
            if trunc:
                break

        advanced = fruits < prev_fruits or any(
            (sp and not p) for sp, p in zip(
                tuple(iface.read_ram_byte(FRUIT_PRESENCE[k]) != 0 for k in (1, 2, 3, 4)),
                fp,
            )
        )
        # Just check if we collected anything new vs the start.
        start_collected = sum(1 for p in (
            iface.read_ram_byte(FRUIT_PRESENCE[k]) != 0 for k in (1, 2, 3, 4)
        ))
        # Better: use start fp.
        # Re-derive start fp from the loaded seed: we already saved
        # `fp` snapshot at the START as `fp` before loop. Hmm we lost
        # it. Just use simple fruits count.
        del start_collected, advanced

        tag = "WIN" if fruits < prev_fruits else "FAIL"  # simplified
        out_name = f"seed{i:03d}_y{sy}_fp{''.join('Y' if p else '-' for p in fp)}_len{steps}_total{cum_reward:.1f}.mp4"
        out_path = os.path.join(args.out, out_name)
        imageio.mimsave(out_path, frames, fps=50)
        print(
            f"  seed {i}: y={sy} steps={steps} total_reward={cum_reward:.2f} -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
