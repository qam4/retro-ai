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
import hashlib
import os
import pickle
import random
import threading
import time
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from retro_ai.envs.base_env import BaseEnv
from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
from retro_ai.training.game_profile import GameProfileRegistry
from retro_ai.training.run_manifest import (
    EpisodeLogger,
    RunManifest,
    iter_inner_envs,
    seed_everything,
)


# Yeti-specific RAM addresses (kept here so this script stays self-contained;
# these mirror the constants discovered during the experiment-003 work).
FRUITS_ADDR = 11055
LIVES_ADDR = 11095
BONUS_HI = 11010
BONUS_LO = 11011
SCORE_HI = 11093
SCORE_LO = 11094
X_POS = 11090
Y_POS = 11089


# Global step counter shared across envs — ProgressCallback updates it.
# Used by each env's step() to tag episode rows with the current global step.
_global_step = 0
_global_step_lock = threading.Lock()

# Global episode ID generator, also shared across envs.
_episode_counter = 0
_episode_counter_lock = threading.Lock()


def _next_episode_id() -> int:
    global _episode_counter
    with _episode_counter_lock:
        _episode_counter += 1
        return _episode_counter


def _set_global_step(step: int) -> None:
    global _global_step
    with _global_step_lock:
        _global_step = step


def _get_global_step() -> int:
    with _global_step_lock:
        return _global_step


class SegmentEnv(gym.Env):
    """Env that always starts from a specific checkpoint level."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        profile_name: str,
        checkpoint_states,
        env_id: int,
        episode_logger: Optional[EpisodeLogger] = None,
        max_steps: int = 1000,
        stall_threshold: int = 15,
        reward_scale: float = 0.01,
    ):
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
        self.env_id = env_id
        self.episode_logger = episode_logger
        self.max_steps = max_steps
        self.stall_threshold = stall_threshold
        self.reward_scale = reward_scale

        # Per-episode state
        self._step_count = 0
        self._prev_fruits = 4
        self._prev_lives = 5
        self._prev_bonus = 0
        self._stall = 0
        self._initialized = False

        # For metrics & logging
        self._start_fruits = 4
        self._start_xy = (0, 0)
        self._start_score = 0
        self._start_bonus = 0
        self._start_state_hash = ""
        self._episode_reward = 0.0
        self._fruits_collected_this_ep = 0
        self._episode_id = 0

        # Aggregates (still kept for legacy ProgressCallback readout)
        self.successes = 0
        self.episodes = 0
        self.collected_states = deque(maxlen=100)

    # --- helpers ------------------------------------------------------

    def _read_bonus(self) -> int:
        return (self.iface.read_ram_byte(BONUS_HI) << 8) | self.iface.read_ram_byte(
            BONUS_LO
        )

    def _read_score(self) -> int:
        return (self.iface.read_ram_byte(SCORE_HI) << 8) | self.iface.read_ram_byte(
            SCORE_LO
        )

    def _read_pos(self):
        return (
            self.iface.read_ram_byte(X_POS),
            self.iface.read_ram_byte(Y_POS),
        )

    # --- gym API ------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        state = random.choice(self.checkpoint_states)
        self.base._interface.load_state(state)
        obs, _, _, _, _ = self.gym_env.step([0, 0, 0])

        # Snapshot the starting game state for this episode
        self._step_count = 0
        self._prev_fruits = self.iface.read_ram_byte(FRUITS_ADDR)
        self._start_fruits = self._prev_fruits
        self._prev_lives = self.iface.read_ram_byte(LIVES_ADDR)
        self._prev_bonus = self._read_bonus()
        self._start_bonus = self._prev_bonus
        self._start_xy = self._read_pos()
        self._start_score = self._read_score()
        # Hash of start state bytes — cheap 8-byte identifier for analysis.
        self._start_state_hash = hashlib.blake2b(state, digest_size=8).hexdigest()
        self._stall = 0
        self._episode_reward = 0.0
        self._fruits_collected_this_ep = 0
        self._episode_id = _next_episode_id()
        return obs, {}

    def step(self, action):
        obs, _, done, truncated, info = self.gym_env.step(action)
        self._step_count += 1

        fruits = self.iface.read_ram_byte(FRUITS_ADDR)
        lives = self.iface.read_ram_byte(LIVES_ADDR)
        bonus = self._read_bonus()

        # Reward: bonus_remaining * reward_scale per fruit collected.
        # Faster collection = higher bonus = higher reward.
        reward = 0.0
        if fruits < self._prev_fruits:
            collected = self._prev_fruits - fruits
            reward += collected * bonus * self.reward_scale
            self.successes += collected
            self._fruits_collected_this_ep += collected
            self.collected_states.append(self.iface.save_state())
        self._prev_fruits = fruits

        self._episode_reward += reward

        # Death detection
        end_reason = None
        if lives < self._prev_lives and self._prev_lives > 0:
            done = True
            end_reason = "death"
        self._prev_lives = lives

        if bonus == self._prev_bonus:
            self._stall += 1
        else:
            self._stall = 0
            self._prev_bonus = bonus
        if self._stall >= self.stall_threshold:
            done = True
            if end_reason is None:
                end_reason = "stall"

        if self._step_count >= self.max_steps:
            truncated = True
            if end_reason is None:
                end_reason = "max_steps"

        if done or truncated:
            self.episodes += 1
            if end_reason is None:
                end_reason = "env_done" if done else "env_truncated"
            self._log_episode(end_reason, fruits, bonus)

        return obs, reward, done, truncated, info

    def _log_episode(self, end_reason: str, fruits: int, bonus: int) -> None:
        if self.episode_logger is None:
            return
        final_xy = self._read_pos()
        start_level = 4 - self._start_fruits
        reached_level = 4 - fruits
        self.episode_logger.log(
            global_step=_get_global_step(),
            env_id=self.env_id,
            episode_id=self._episode_id,
            start_level=start_level,
            reached_level=reached_level,
            n_fruits_collected=self._fruits_collected_this_ep,
            length=self._step_count,
            total_reward=round(self._episode_reward, 4),
            end_reason=end_reason,
            start_x=self._start_xy[0],
            start_y=self._start_xy[1],
            start_score=self._start_score,
            start_bonus=self._start_bonus,
            final_x=final_xy[0],
            final_y=final_xy[1],
            final_score=self._read_score(),
            final_bonus=bonus,
            start_state_hash=self._start_state_hash,
        )


class ProgressCallback(BaseCallback):
    def __init__(self, total: int, log_interval: int = 5000):
        super().__init__()
        self._total = total
        self._log_interval = log_interval
        self._last_log = 0
        self._last_successes = 0
        self._last_episodes = 0
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        # Keep the global step counter up to date so env rows can reference it.
        _set_global_step(self.num_timesteps)

        if self.num_timesteps - self._last_log >= self._log_interval:
            elapsed = time.monotonic() - self._start
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            pct = 100 * self.num_timesteps / self._total

            total_s = 0
            total_e = 0
            for inner in iter_inner_envs(self.training_env):
                if hasattr(inner, "successes"):
                    total_s += inner.successes
                    total_e += inner.episodes

            new_s = total_s - self._last_successes
            new_e = total_e - self._last_episodes
            rate = f"{100 * new_s / new_e:.0f}%" if new_e > 0 else "N/A"
            self._last_successes = total_s
            self._last_episodes = total_e

            print(
                f"step {self.num_timesteps}/{self._total} ({pct:.0f}%) "
                f"| emu_fps={fps * 4:.0f} "
                f"| success={rate} ({new_s}/{new_e})",
                flush=True,
            )
            self._last_log = self.num_timesteps
        return True


def train(args: argparse.Namespace) -> None:
    print(f"Segment Training: checkpoint {args.segment} -> {args.segment + 1}")

    seed = seed_everything(args.seed)
    print(f"  Seed: {seed}", flush=True)

    with open(args.checkpoints, "rb") as f:
        data = pickle.load(f)

    states = data["checkpoints"][args.segment]
    print(f"  {len(states)} starting states")

    if not states:
        print("  No states available!")
        return

    os.makedirs(args.output, exist_ok=True)

    # --- persist run config + machine context -----------------------------
    ppo_hparams = {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": max(1, 128 // args.num_envs),
        "n_epochs": 4,
        "ent_coef": 0.01,
    }
    extras = {
        "script": "scripts/train_segment.py",
        "resolved_seed": seed,
        "ppo": ppo_hparams,
        "env": {
            "max_steps": args.max_steps,
            "stall_threshold": 15,
            "reward_formula": "fruits_collected * bonus_remaining * reward_scale",
            "reward_scale": 0.01,
            "action_mode": "joystick",
        },
        "checkpoints_source": args.checkpoints,
        "num_checkpoint_states": len(states),
    }
    manifest = RunManifest.capture(args, args.output, extras=extras)
    episode_logger = EpisodeLogger(args.output)

    from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv

    def make_env(rank: int):
        def _init():
            env = SegmentEnv(
                profile_name=args.profile,
                checkpoint_states=states,
                env_id=rank,
                episode_logger=episode_logger,
                max_steps=args.max_steps,
                stall_threshold=15,
                reward_scale=0.01,
            )
            return Monitor(env)

        return _init

    num_envs = args.num_envs
    vec_env = ThreadedVecEnv([make_env(i) for i in range(num_envs)])

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=ppo_hparams["learning_rate"],
        batch_size=ppo_hparams["batch_size"],
        n_steps=ppo_hparams["n_steps"],
        n_epochs=ppo_hparams["n_epochs"],
        ent_coef=ppo_hparams["ent_coef"],
        verbose=0,
        tensorboard_log=os.path.join(args.output, "tb"),
        device="auto",
        seed=seed,
    )

    print(f"  {num_envs} envs, {args.timesteps} steps")
    status = "COMPLETED"
    exit_code: Optional[int] = 0
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=ProgressCallback(args.timesteps),
        )
        model.save(os.path.join(args.output, "final_model"))

        # Save collected states (for chaining into next segment)
        all_states = []
        for inner in iter_inner_envs(vec_env):
            if hasattr(inner, "collected_states"):
                all_states.extend(inner.collected_states)
        if all_states:
            pkl_path = os.path.join(args.output, "collected_states.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"states": all_states, "segment": args.segment + 1}, f)
            print(f"\nSaved {len(all_states)} collected states to {pkl_path}")
        else:
            print("\nNo states collected (agent never reached next checkpoint)")

        print(f"\nSaved to {args.output}/final_model.zip")
    except Exception:
        status = "FAILED"
        exit_code = 1
        raise
    finally:
        episode_logger.close()
        manifest.finalize(status=status, exit_code=exit_code)


def main() -> None:
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed. If omitted, derived from current time and recorded in run.yaml.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
