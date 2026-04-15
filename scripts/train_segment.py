#!/usr/bin/env python3
"""Train a fresh agent on a single segment (checkpoint N -> N+1).

Loads save states from a checkpoint file and trains a new PPO policy
from scratch to reach the next fruit.

Usage:
    python scripts/train_segment.py \
        --checkpoints output/mo5/yeti/training/curriculum_v5/checkpoints.pkl \
        --segment 1 \
        --timesteps 5000000 \
        --output output/mo5/yeti/training/segment_1to2
"""

import argparse
import os
import pickle
import random

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from retro_ai.envs.base_env import BaseEnv
from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
from retro_ai.training.game_profile import GameProfileRegistry


class SegmentEnv(gym.Env):
    """Env that always starts from a specific checkpoint level."""

    metadata = {"render_modes": []}
    FRUITS_ADDR = 11055
    LIVES_ADDR = 11095
    BONUS_HI = 11010
    BONUS_LO = 11011

    def __init__(self, profile_name, checkpoint_states, max_steps=1000):
        super().__init__()
        registry = GameProfileRegistry()
        profile = registry.load(profile_name)
        config_dict = {}
        if profile.reward_params:
            config_dict["reward_params"] = profile.reward_params

        self.base = BaseEnv(
            emulator_type=profile.emulator_type,
            rom_path=profile.rom_path,
            reward_mode=profile.reward_mode,
            config=config_dict or None,
            action_mode="joystick",
        )
        self.pipeline = PreprocessingPipeline(
            grayscale=profile.grayscale,
            resize=(84, 84),
            frame_stack=profile.frame_stack,
            frame_skip=profile.frame_skip,
        )
        self.preprocessed = PreprocessedEnv(
            self.base,
            self.pipeline,
            frame_maxpool=profile.frame_maxpool,
        )
        self.gym_env = GymnasiumWrapper(self.preprocessed)

        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        self.iface = self.base._interface

        self.checkpoint_states = checkpoint_states
        self.max_steps = max_steps
        self._step_count = 0
        self._prev_fruits = 4
        self._prev_lives = 5
        self._prev_bonus = 0
        self._stall = 0
        self._initialized = False
        self.successes = 0
        self.episodes = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        state = random.choice(self.checkpoint_states)
        self.base._interface.load_state(state)
        obs, _, _, _, _ = self.gym_env.step([0, 0, 0])

        self._step_count = 0
        self._prev_fruits = self.iface.read_ram_byte(self.FRUITS_ADDR)
        self._prev_lives = self.iface.read_ram_byte(self.LIVES_ADDR)
        self._prev_bonus = (
            self.iface.read_ram_byte(self.BONUS_HI) << 8
        ) | self.iface.read_ram_byte(self.BONUS_LO)
        self._stall = 0
        return obs, {}

    def step(self, action):
        obs, _, done, truncated, info = self.gym_env.step(action)
        self._step_count += 1

        fruits = self.iface.read_ram_byte(self.FRUITS_ADDR)
        lives = self.iface.read_ram_byte(self.LIVES_ADDR)
        bonus = (
            self.iface.read_ram_byte(self.BONUS_HI) << 8
        ) | self.iface.read_ram_byte(self.BONUS_LO)

        # Reward: bonus * 0.01 per fruit collected
        reward = 0.0
        if fruits < self._prev_fruits:
            reward += bonus * 0.01
            self.successes += 1
        self._prev_fruits = fruits

        # Death detection
        if lives < self._prev_lives and self._prev_lives > 0:
            done = True
        self._prev_lives = lives

        if bonus == self._prev_bonus:
            self._stall += 1
        else:
            self._stall = 0
            self._prev_bonus = bonus
        if self._stall >= 15:
            done = True

        if self._step_count >= self.max_steps:
            truncated = True

        if done or truncated:
            self.episodes += 1

        return obs, reward, done, truncated, info


class ProgressCallback(BaseCallback):
    def __init__(self, total, env, log_interval=5000):
        super().__init__()
        self._total = total
        self._env = env
        self._log_interval = log_interval
        self._last_log = 0
        self._last_successes = 0
        self._last_episodes = 0
        import time

        self._start = time.monotonic()

    def _on_step(self):
        if self.num_timesteps - self._last_log >= self._log_interval:
            import time

            elapsed = time.monotonic() - self._start
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            pct = 100 * self.num_timesteps / self._total

            # Get success rate from envs
            total_s = 0
            total_e = 0
            if hasattr(self.training_env, "envs"):
                for e in self.training_env.envs:
                    inner = e
                    while hasattr(inner, "env"):
                        inner = inner.env
                    if hasattr(inner, "successes"):
                        total_s += inner.successes
                        total_e += inner.episodes

            new_s = total_s - self._last_successes
            new_e = total_e - self._last_episodes
            rate = f"{100*new_s/new_e:.0f}%" if new_e > 0 else "N/A"
            self._last_successes = total_s
            self._last_episodes = total_e

            print(
                f"step {self.num_timesteps}/{self._total} ({pct:.0f}%) "
                f"| emu_fps={fps*4:.0f} "
                f"| success={rate} ({new_s}/{new_e})",
                flush=True,
            )
            self._last_log = self.num_timesteps
        return True


def train(args):
    print(f"Segment Training: checkpoint {args.segment} -> {args.segment + 1}")

    with open(args.checkpoints, "rb") as f:
        data = pickle.load(f)

    states = data["checkpoints"][args.segment]
    print(f"  {len(states)} starting states")

    if not states:
        print("  No states available!")
        return

    os.makedirs(args.output, exist_ok=True)

    from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv

    def make_env(rank):
        def _init():
            env = SegmentEnv(args.profile, states, max_steps=args.max_steps)
            return Monitor(env)

        return _init

    num_envs = args.num_envs
    vec_env = ThreadedVecEnv([make_env(i) for i in range(num_envs)])

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=max(1, 128 // num_envs),
        n_epochs=4,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=os.path.join(args.output, "tb"),
        device="auto",
    )

    print(f"  {num_envs} envs, {args.timesteps} steps")
    model.learn(
        total_timesteps=args.timesteps,
        callback=ProgressCallback(args.timesteps, vec_env),
    )
    model.save(os.path.join(args.output, "final_model"))
    print(f"\nSaved to {args.output}/final_model.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument(
        "--segment",
        type=int,
        required=True,
        help="Starting checkpoint level (e.g. 1 for fruit1->fruit2)",
    )
    parser.add_argument("--profile", default="yeti_fruit")
    parser.add_argument("--timesteps", type=int, default=5000000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
