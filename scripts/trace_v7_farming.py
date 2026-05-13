#!/usr/bin/env python3
"""Replay v7's policy from a typical floor-4-start seed and trace
per-step reward to find the farming pattern."""
from __future__ import annotations

import pickle

import numpy as np
from stable_baselines3 import PPO

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.rewards import RewardContext, create
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.yeti_map import build_navigation_map

env_cfg = EnvConfig(
    profile="yeti_fruit",
    action_mode="joystick",
    max_steps=2000,
    stall_threshold=2000,
    resize=(84, 84),
)
stack = build_training_env("yeti_fruit", env_cfg)
base = stack.base
gym_env = stack.gym
base.reset(seed=0)

model = PPO.load("output/mo5/yeti/training/segment_2to3_v7/final_model.zip")
nav = build_navigation_map()
reward_fn = create(
    "fruit_bonus_path_progress", {"scale": 0.01, "fruit_scale": 0.01}
)

import hashlib

with open("output/mo5/yeti/seeds/v9_checkpoints_v2.pkl", "rb") as f:
    seeds = pickle.load(f)["checkpoints"][2]

# Find the specific farming seed by hash.
TARGET_HASH = "6f0e246058d89138"
target_seed = None
for s in seeds:
    sb = bytes(s) if not isinstance(s, (bytes, bytearray)) else s
    h = hashlib.blake2b(sb, digest_size=8).hexdigest()
    if h == TARGET_HASH:
        target_seed = sb
        break
assert target_seed is not None, f"hash {TARGET_HASH} not found"

base._interface.load_state(target_seed)
obs = None
for _ in range(5):
    obs, _, _, _, _ = gym_env.step([0, 0, 0])

# Reset reward state.
if hasattr(reward_fn, "reset"):
    reward_fn.reset()

prev_fruits = base._interface.read_ram_byte(11055)
prev_lives = base._interface.read_ram_byte(11095)
prev_bonus = (base._interface.read_ram_byte(11010) << 8) | base._interface.read_ram_byte(11011)
prev_score = (base._interface.read_ram_byte(11093) << 8) | base._interface.read_ram_byte(11094)

total_reward = 0.0
print(
    f"{'step':>4} {'rx':>3} {'ry':>3} {'flr':>3} {'fr':>2} {'rew':>8} {'tot':>8} "
    f"{'bd1':>4} {'bd2':>4} {'bd3':>4} {'bd4':>4}"
)
for step in range(1, 1001):
    # Model expects channels-first obs (CHW); training env returns HWC.
    obs_chw = np.transpose(obs, (2, 0, 1))
    action, _ = model.predict(obs_chw, deterministic=True)
    obs, _, done, trunc, _ = gym_env.step(action)
    iface = base._interface
    fruits = iface.read_ram_byte(11055)
    lives = iface.read_ram_byte(11095)
    bonus = (iface.read_ram_byte(11010) << 8) | iface.read_ram_byte(11011)
    score = (iface.read_ram_byte(11093) << 8) | iface.read_ram_byte(11094)
    x = iface.read_ram_byte(11090)
    y = iface.read_ram_byte(11089)
    fp = tuple(iface.read_ram_byte(a) != 0 for a in (0x2FAD, 0x2F00, 0x2E68, 0x2DD8))

    ctx = RewardContext(
        prev_fruits=prev_fruits, curr_fruits=fruits,
        prev_bonus=prev_bonus, curr_bonus=bonus,
        prev_score=prev_score, curr_score=score,
        prev_lives=prev_lives, curr_lives=lives,
        step_count=step, curr_y=y, curr_x=x, fruits_present=fp,
    )
    r = reward_fn(ctx)
    total_reward += r
    floor = (200 - y) // 32

    bd = reward_fn.best_d
    if step % 20 == 0 or r > 0.5 or done or trunc:
        print(f"{step:>4} {x:>3} {y:>3} {floor:>3} {fruits:>2} {r:>8.3f} {total_reward:>8.2f} "
              f"{str(bd[1]):>4} {str(bd[2]):>4} {str(bd[3]):>4} {str(bd[4]):>4}")
    prev_fruits = fruits
    prev_lives = lives
    prev_bonus = bonus
    prev_score = score
    if done or trunc:
        break

print(f"\nTotal reward over {step} steps: {total_reward:.2f}")
print(f"Final fruits remaining: {fruits}")
