#!/usr/bin/env python3
"""Checkpoint curriculum training for Yeti.

Trains a single PPO policy progressively:
1. Start from game reset, learn to collect fruit 1
2. Save states when fruit 1 is collected
3. Mix: 50% game start, 50% fruit-1 checkpoints → learn fruit 2
4. Save states when fruit 2 is collected
5. Continue mixing all segments until all fruits + princess

Usage:
    python scripts/train_checkpoint_curriculum.py \
        --profile yeti_fruit --timesteps 10000000 \
        --output output/mo5/yeti/training/checkpoint_curriculum
"""

import argparse
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from retro_ai.envs.base_env import BaseEnv
from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
from retro_ai.training.game_profile import GameProfileRegistry


# Shared checkpoint buffers across all env instances
class CheckpointManager:
    """Manages save state buffers for each fruit checkpoint."""

    MAX_STATES_PER_CHECKPOINT = 100
    FRUITS_TOTAL = 4

    def __init__(self):
        # checkpoints[i] = deque of save states where i fruits have been collected
        self.checkpoints = [
            deque(maxlen=self.MAX_STATES_PER_CHECKPOINT)
            for _ in range(self.FRUITS_TOTAL + 1)
        ]
        self.frontier = 0  # highest checkpoint with enough states
        self.min_states_to_advance = 20
        self.stats = {
            "saves": [0] * (self.FRUITS_TOTAL + 1),
            "starts": [0] * (self.FRUITS_TOTAL + 1),
        }
        # Success tracking: episodes started from level N that reached N+1
        self.segment_attempts = [0] * (self.FRUITS_TOTAL + 1)
        self.segment_successes = [0] * (self.FRUITS_TOTAL + 1)

    def record_episode(self, start_level, reached_level):
        """Record an episode's outcome for success rate tracking."""
        if 0 <= start_level <= self.FRUITS_TOTAL:
            self.segment_attempts[start_level] += 1
            if reached_level > start_level:
                self.segment_successes[start_level] += 1

    def save_checkpoint(self, fruits_collected, state_bytes):
        """Save a state at the given checkpoint level."""
        if 0 <= fruits_collected <= self.FRUITS_TOTAL:
            self.checkpoints[fruits_collected].append(bytes(state_bytes))
            self.stats["saves"][fruits_collected] += 1
            # Update frontier
            while (
                self.frontier < self.FRUITS_TOTAL
                and len(self.checkpoints[self.frontier]) >= self.min_states_to_advance
            ):
                self.frontier = max(
                    self.frontier,
                    max(
                        i
                        for i in range(self.FRUITS_TOTAL + 1)
                        if len(self.checkpoints[i]) >= self.min_states_to_advance
                    ),
                )
                break

    def pick_start(self):
        """Pick a starting checkpoint level.

        Distribution:
        - 40% game start (keeps full-chain skill sharp)
        - 40% highest available checkpoint (pushes frontier)
        - 20% random other checkpoint (maintains intermediate skills)
        """
        # Find highest checkpoint with states
        highest = 0
        for i in range(self.FRUITS_TOTAL, 0, -1):
            if len(self.checkpoints[i]) > 0:
                highest = i
                break

        roll = random.random()
        if roll < 0.4 or highest == 0:
            # Game start
            level = 0
        elif roll < 0.8:
            # Frontier — highest checkpoint
            level = highest
        else:
            # Random intermediate (including game start)
            available = [0]
            for i in range(1, highest):
                if len(self.checkpoints[i]) > 0:
                    available.append(i)
            level = random.choice(available)

        self.stats["starts"][level] += 1

        if level == 0:
            return 0, None
        else:
            state = random.choice(self.checkpoints[level])
            return level, state

    def summary(self):
        sizes = [len(self.checkpoints[i]) for i in range(self.FRUITS_TOTAL + 1)]
        rates = []
        for i in range(self.FRUITS_TOTAL + 1):
            if self.segment_attempts[i] > 0:
                pct = 100 * self.segment_successes[i] / self.segment_attempts[i]
                rates.append(f"{i}->{i+1}:{pct:.0f}%")
            else:
                rates.append(f"{i}->{i+1}:N/A")
        return (
            f"cp={sizes} saves={self.stats['saves']} " f"success=[{', '.join(rates)}]"
        )

    def save_to_disk(self, path):
        """Persist checkpoint buffers to disk."""
        import pickle

        data = {
            "checkpoints": [
                list(self.checkpoints[i]) for i in range(self.FRUITS_TOTAL + 1)
            ],
            "stats": self.stats,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_from_disk(self, path):
        """Load checkpoint buffers from disk."""
        import pickle

        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        for i, states in enumerate(data["checkpoints"]):
            for s in states:
                self.checkpoints[i].append(s)
        self.stats = data.get("stats", self.stats)
        print(f"  Loaded checkpoints from {path}: {self.summary()}", flush=True)


# Global manager shared across envs
_manager = CheckpointManager()


class CheckpointCurriculumEnv(gym.Env):
    """Gym env with checkpoint-based curriculum for Yeti."""

    metadata = {"render_modes": []}
    FRUITS_ADDR = 11055
    LIVES_ADDR = 11095
    BONUS_HI = 11010
    BONUS_LO = 11011
    SCORE_HI = 11093
    SCORE_LO = 11094

    def __init__(self, profile_name, max_steps=1000):
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
        self.max_steps = max_steps

        self._step_count = 0
        self._prev_fruits = 4
        self._prev_lives = 5
        self._prev_bonus = 0
        self._stall = 0
        self._start_fruits = 4
        self._want_checkpoint = 0
        self._initialized = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        # Pick starting point from checkpoint manager
        level, state_bytes = _manager.pick_start()

        if state_bytes is not None:
            # Load checkpoint state and run a few frames to let game settle
            self.base._interface.load_state(state_bytes)
            for _ in range(5):
                obs, _, _, _, _ = self.gym_env.step([0, 0, 0])
        else:
            # Game start
            obs, _ = self.gym_env.reset()

        self._step_count = 0
        self._prev_fruits = self.iface.read_ram_byte(self.FRUITS_ADDR)
        self._start_fruits = self._prev_fruits
        self._prev_lives = self.iface.read_ram_byte(self.LIVES_ADDR)
        self._prev_bonus = (
            self.iface.read_ram_byte(self.BONUS_HI) << 8
        ) | self.iface.read_ram_byte(self.BONUS_LO)
        self._stall = 0
        self._want_checkpoint = 0

        return obs, {}

    def step(self, action):
        obs, _, done, truncated, info = self.gym_env.step(action)
        self._step_count += 1

        fruits = self.iface.read_ram_byte(self.FRUITS_ADDR)
        lives = self.iface.read_ram_byte(self.LIVES_ADDR)
        bonus = (
            self.iface.read_ram_byte(self.BONUS_HI) << 8
        ) | self.iface.read_ram_byte(self.BONUS_LO)

        # Reward: bonus_remaining * 0.01 per fruit collected.
        # Faster collection = higher bonus = more reward.
        # No flat reward — the bonus IS the incentive.
        reward = 0.0
        if fruits < self._prev_fruits:
            fruits_collected = self._prev_fruits - fruits
            reward += fruits_collected * bonus * 0.01
            collected_total = 4 - fruits
            state_bytes = self.base._interface.save_state()
            # WORKAROUND: ~10% of save states produce a frozen game state
            # after load (bonus/position never change). This is a crayon
            # save/restore bug where some transient CPU/emulator state isn't
            # serialized, causing the game to enter an infinite loop.
            # We validate by loading the state, running a few frames, and
            # checking if the bonus changes. Invalid states are discarded.
            # TODO: fix the root cause in crayon's state serialization.
            if self._validate_checkpoint(state_bytes):
                _manager.save_checkpoint(collected_total, state_bytes)

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

        # Record episode outcome for success rate tracking
        if done or truncated:
            start_level = 4 - self._start_fruits
            reached_level = 4 - fruits
            _manager.record_episode(start_level, reached_level)

        return obs, reward, done, truncated, info

    def _validate_checkpoint(self, state_bytes):
        """Check if a save state produces a playable game.

        WORKAROUND for crayon save/restore bug. See comment above.
        Loads the state, runs 20 noop frames, checks if bonus changes.
        Restores the original state afterward.
        """
        original = self.base._interface.save_state()
        self.base._interface.load_state(state_bytes)
        bonus_start = (
            self.iface.read_ram_byte(self.BONUS_HI) << 8
        ) | self.iface.read_ram_byte(self.BONUS_LO)
        changed = False
        for _ in range(20):
            self.base.step([0, 0, 0])
            b = (
                self.iface.read_ram_byte(self.BONUS_HI) << 8
            ) | self.iface.read_ram_byte(self.BONUS_LO)
            if b != bonus_start:
                changed = True
                break
        self.base._interface.load_state(original)
        return changed


class CurriculumCallback(BaseCallback):
    """Log curriculum progress during training."""

    def __init__(self, total_timesteps, log_interval=5000):
        super().__init__()
        self._total = total_timesteps
        self._log_interval = log_interval
        self._last_log = 0
        self._start = time.monotonic()

    def _on_step(self):
        if self.num_timesteps - self._last_log >= self._log_interval:
            elapsed = time.monotonic() - self._start
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            pct = 100 * self.num_timesteps / self._total

            # Get rolling reward from monitor
            infos = self.locals.get("infos", [])
            rewards = []
            for info in infos:
                ep = info.get("episode")
                if ep:
                    rewards.append(ep["r"])

            reward_str = f"{np.mean(rewards):.1f}" if rewards else "N/A"

            print(
                f"step {self.num_timesteps}/{self._total} ({pct:.0f}%) "
                f"| reward={reward_str} "
                f"| emu_fps={fps * 4:.0f} "
                f"| {_manager.summary()}",
                flush=True,
            )
            self._last_log = self.num_timesteps
        return True


def train(args):
    global _manager
    _manager = CheckpointManager()

    print(f"Checkpoint Curriculum Training", flush=True)
    print(f"  Profile: {args.profile}", flush=True)
    print(f"  Timesteps: {args.timesteps}", flush=True)
    print(f"  Output: {args.output}", flush=True)

    # Seed checkpoint buffers from Go-Explore archive if provided
    if args.seed_archive:
        import pickle

        print(f"  Seeding from {args.seed_archive}", flush=True)
        with open(args.seed_archive, "rb") as f:
            archive = pickle.load(f)
        for cell_key, info in archive.items():
            fruits_collected = 4 - cell_key[2]
            if fruits_collected > 0:
                _manager.save_checkpoint(fruits_collected, info["state"])
        print(f"  Seeded: {_manager.summary()}", flush=True)

    os.makedirs(args.output, exist_ok=True)

    # Multi-env
    from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv

    def make_env(rank):
        def _init():
            env = CheckpointCurriculumEnv(args.profile, max_steps=args.max_steps)
            return Monitor(env)

        return _init

    num_envs = args.num_envs
    vec_env = ThreadedVecEnv([make_env(i) for i in range(num_envs)])
    print(f"  Envs: {num_envs} threaded", flush=True)

    n_steps = max(1, 128 // num_envs)

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=n_steps,
        n_epochs=4,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=os.path.join(args.output, "tb"),
        device="auto",
    )

    if args.resume:
        print(f"  Resuming from {args.resume}", flush=True)
        model = PPO.load(
            args.resume, env=vec_env, tensorboard_log=os.path.join(args.output, "tb")
        )
        # Load persisted checkpoints if available
        ckpt_path = os.path.join(os.path.dirname(args.resume), "checkpoints.pkl")
        _manager.load_from_disk(ckpt_path)
        # Also try from the output dir
        _manager.load_from_disk(os.path.join(args.output, "checkpoints.pkl"))

    print(f"\nTraining...", flush=True)
    model.learn(
        total_timesteps=args.timesteps,
        callback=CurriculumCallback(args.timesteps),
    )
    model.save(os.path.join(args.output, "final_model"))
    _manager.save_to_disk(os.path.join(args.output, "checkpoints.pkl"))
    print(f"\nSaved model to {args.output}/final_model.zip", flush=True)
    print(f"Final: {_manager.summary()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Curriculum Training")
    parser.add_argument("--profile", default="yeti_fruit")
    parser.add_argument("--timesteps", type=int, default=10000000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--output", default="output/mo5/yeti/training/checkpoint_curriculum"
    )
    parser.add_argument("--resume", help="Path to model .zip to resume from")
    parser.add_argument(
        "--seed-archive", help="Path to Go-Explore archive.pkl to seed checkpoints"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
