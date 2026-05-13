#!/usr/bin/env python3
"""Reproduce v7's farming episode using SegmentEnv directly.

Loads the same seed (idx 47) and walks through with a few different
action policies to see how reward accumulates.
"""
from __future__ import annotations

import pickle
import sys

import yaml
from stable_baselines3 import PPO

# Inline import: scripts/ isn't a package, but train_segment.py defines
# SegmentEnv. Hack the path.
sys.path.insert(0, "scripts")
from train_segment import SegmentEnv  # noqa: E402

from retro_ai.training.rewards import create as create_reward  # noqa: E402
from retro_ai.training.run_config import RunConfig  # noqa: E402
from retro_ai.training.run_manifest import EpisodeLogger  # noqa: E402

cfg_data = yaml.safe_load(
    open("experiments/003-yeti/configs/segment_2to3_v7.yaml")
)
cfg = RunConfig.from_dict(cfg_data)

with open(cfg.segment.checkpoints, "rb") as f:
    data = pickle.load(f)
checkpoints = data["checkpoints"]

# Use just the farming seed.
target_seed = bytes(checkpoints[2][47])

# Build a SegmentEnv with only that seed.
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    logger = EpisodeLogger(tmpdir)
    reward_fn = create_reward(cfg.reward.name, cfg.reward.params)
    env = SegmentEnv(
        cfg=cfg,
        checkpoint_states=[target_seed],
        reward_fn=reward_fn,
        env_id=0,
        episode_logger=logger,
    )
    obs, _ = env.reset()

    model = PPO.load("output/mo5/yeti/training/segment_2to3_v7/final_model.zip")

    total_reward = 0.0
    print(f"{'step':>4} {'rx':>3} {'ry':>3} {'fr':>2} {'rew':>10} {'tot':>10} "
          f"{'best_d_dict':>30}")
    import numpy as np
    # Force an action sequence that walks left-right repeatedly.
    forced_actions = None  # use trained model

    for step in range(1, 1001):
        if forced_actions is None:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = forced_actions[step - 1]
        obs, r, done, trunc, _ = env.step(action)
        total_reward += r
        bd = env._reward_fn.best_d
        x = env.iface.read_ram_byte(11090)
        y = env.iface.read_ram_byte(11089)
        if r > 0.5 or step % 50 == 0:
            print(f"{step:>4} {x:>3} {y:>3} {env._prev_fruits:>2} "
                  f"{r:>10.3f} {total_reward:>10.2f} {str(bd):>30}")
        if done or trunc:
            break

    print(f"\nTotal reward: {total_reward:.2f}, steps: {step}")
