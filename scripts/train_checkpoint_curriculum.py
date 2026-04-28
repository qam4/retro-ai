#!/usr/bin/env python3
"""Checkpoint curriculum training for Yeti.

Trains a single PPO policy progressively:

1. Start from game reset, learn to collect fruit 1
2. Save states when fruit 1 is collected
3. Mix game-start + frontier-checkpoint starts for the next segment
4. Continue until all fruits + princess

Configuration is YAML-driven — pass ``--config`` pointing at a file with
``training``, ``env``, ``ppo``, ``reward``, and ``curriculum`` sections.

Example::

    python scripts/train_checkpoint_curriculum.py \\
        --config experiments/003-yeti/configs/curriculum_v6.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
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

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.rewards import RewardContext, RewardFn, create as create_reward
from retro_ai.training.run_config import RunConfig
from retro_ai.training.run_manifest import (
    EpisodeLogger,
    RunManifest,
    iter_inner_envs,
    seed_everything,
)


# Yeti RAM addresses
FRUITS_ADDR = 11055
LIVES_ADDR = 11095
BONUS_HI = 11010
BONUS_LO = 11011
SCORE_HI = 11093
SCORE_LO = 11094
X_POS = 11090
Y_POS = 11089


# Shared global-step & episode-id counters across threaded envs.
_global_step = 0
_global_step_lock = threading.Lock()
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


class CheckpointManager:
    """Manages save state buffers for each fruit checkpoint.

    ``reset_fraction`` / ``frontier_fraction`` / ``earlier_fraction``
    control the start distribution. All three must sum to 1.0 (validated
    on construction).
    """

    FRUITS_TOTAL = 4

    def __init__(
        self,
        max_states_per_checkpoint: int,
        min_states_to_advance: int,
        reset_fraction: float,
        frontier_fraction: float,
        earlier_fraction: float,
    ):
        total = reset_fraction + frontier_fraction + earlier_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Curriculum fractions must sum to 1.0, got {total}: "
                f"reset={reset_fraction}, frontier={frontier_fraction}, "
                f"earlier={earlier_fraction}"
            )
        self.max_states_per_checkpoint = max_states_per_checkpoint
        self.min_states_to_advance = min_states_to_advance
        self.reset_fraction = reset_fraction
        self.frontier_fraction = frontier_fraction

        self.checkpoints = [
            deque(maxlen=max_states_per_checkpoint)
            for _ in range(self.FRUITS_TOTAL + 1)
        ]
        self.frontier = 0
        self.stats = {
            "saves": [0] * (self.FRUITS_TOTAL + 1),
            "starts": [0] * (self.FRUITS_TOTAL + 1),
        }
        self.segment_attempts = [0] * (self.FRUITS_TOTAL + 1)
        self.segment_successes = [0] * (self.FRUITS_TOTAL + 1)

    def record_episode(self, start_level, reached_level):
        if 0 <= start_level <= self.FRUITS_TOTAL:
            self.segment_attempts[start_level] += 1
            if reached_level > start_level:
                self.segment_successes[start_level] += 1

    def save_checkpoint(self, fruits_collected, state_bytes):
        if 0 <= fruits_collected <= self.FRUITS_TOTAL:
            self.checkpoints[fruits_collected].append(bytes(state_bytes))
            self.stats["saves"][fruits_collected] += 1
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

        Distribution (configurable via run config):
        - reset_fraction: game start
        - frontier_fraction: highest available checkpoint
        - earlier_fraction: random intermediate (including game start)
        """
        highest = 0
        for i in range(self.FRUITS_TOTAL, 0, -1):
            if len(self.checkpoints[i]) > 0:
                highest = i
                break

        roll = random.random()
        reset_thresh = self.reset_fraction
        frontier_thresh = self.reset_fraction + self.frontier_fraction
        if roll < reset_thresh or highest == 0:
            level = 0
        elif roll < frontier_thresh:
            level = highest
        else:
            available = [0]
            for i in range(1, highest):
                if len(self.checkpoints[i]) > 0:
                    available.append(i)
            level = random.choice(available)

        self.stats["starts"][level] += 1

        if level == 0:
            return 0, None
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
        return f"cp={sizes} saves={self.stats['saves']} success=[{', '.join(rates)}]"

    def save_to_disk(self, path):
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


# Module-level singleton. Populated in train(); shared across envs.
_manager: Optional[CheckpointManager] = None


class CheckpointCurriculumEnv(gym.Env):
    """Gym env with checkpoint-based curriculum for Yeti."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: RunConfig,
        reward_fn: RewardFn,
        env_id: int,
        episode_logger: Optional[EpisodeLogger] = None,
    ):
        super().__init__()
        stack = build_training_env(cfg.env.profile, cfg.env)
        self.base = stack.base
        self.gym_env = stack.gym
        self.observation_space = stack.gym.observation_space
        self.action_space = stack.gym.action_space
        self.iface = stack.base._interface

        self.max_steps = cfg.env.max_steps
        self.stall_threshold = cfg.env.stall_threshold
        self._reward_fn = reward_fn

        self.env_id = env_id
        self.episode_logger = episode_logger

        # Per-episode state
        self._step_count = 0
        self._prev_fruits = 4
        self._prev_lives = 5
        self._prev_bonus = 0
        self._prev_score = 0
        self._stall = 0
        self._start_fruits = 4
        self._initialized = False

        # Logging state
        self._start_xy = (0, 0)
        self._start_score = 0
        self._start_bonus = 0
        self._start_state_hash = ""
        self._episode_reward = 0.0
        self._fruits_collected_this_ep = 0
        self._episode_id = 0

    # -- helpers -------------------------------------------------------

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

    # -- gym API -------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        assert _manager is not None, "CheckpointManager not initialized"

        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        level, state_bytes = _manager.pick_start()

        if state_bytes is not None:
            self.base._interface.load_state(state_bytes)
            for _ in range(5):
                obs, _, _, _, _ = self.gym_env.step([0, 0, 0])
            self._start_state_hash = hashlib.blake2b(
                state_bytes, digest_size=8
            ).hexdigest()
        else:
            obs, _ = self.gym_env.reset()
            self._start_state_hash = ""

        self._step_count = 0
        self._prev_fruits = self.iface.read_ram_byte(FRUITS_ADDR)
        self._start_fruits = self._prev_fruits
        self._prev_lives = self.iface.read_ram_byte(LIVES_ADDR)
        self._prev_bonus = self._read_bonus()
        self._start_bonus = self._prev_bonus
        self._prev_score = self._read_score()
        self._start_score = self._prev_score
        self._start_xy = self._read_pos()
        self._stall = 0
        self._episode_reward = 0.0
        self._fruits_collected_this_ep = 0
        self._episode_id = _next_episode_id()

        return obs, {}

    def step(self, action):
        assert _manager is not None
        obs, _, done, truncated, info = self.gym_env.step(action)
        self._step_count += 1

        fruits = self.iface.read_ram_byte(FRUITS_ADDR)
        lives = self.iface.read_ram_byte(LIVES_ADDR)
        bonus = self._read_bonus()
        score = self._read_score()

        ctx = RewardContext(
            prev_fruits=self._prev_fruits,
            curr_fruits=fruits,
            prev_bonus=self._prev_bonus,
            curr_bonus=bonus,
            prev_score=self._prev_score,
            curr_score=score,
            prev_lives=self._prev_lives,
            curr_lives=lives,
            step_count=self._step_count,
        )
        reward = float(self._reward_fn(ctx))

        # Save checkpoint on fruit collection (validated via _validate_checkpoint)
        if fruits < self._prev_fruits:
            self._fruits_collected_this_ep += self._prev_fruits - fruits
            collected_total = 4 - fruits
            state_bytes = self.base._interface.save_state()
            # WORKAROUND: ~10% of save states produce a frozen game state
            # after load. Validate by running a few frames & checking bonus
            # changes. Invalid states are discarded.
            # TODO: fix the root cause in crayon's state serialization.
            if self._validate_checkpoint(state_bytes):
                _manager.save_checkpoint(collected_total, state_bytes)

        self._prev_fruits = fruits
        self._prev_score = score
        self._episode_reward += reward

        # Termination
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
            start_level = 4 - self._start_fruits
            reached_level = 4 - fruits
            _manager.record_episode(start_level, reached_level)
            if end_reason is None:
                end_reason = "env_done" if done else "env_truncated"
            self._log_episode(end_reason, fruits, bonus, score)

        return obs, reward, done, truncated, info

    def _log_episode(
        self, end_reason: str, fruits: int, bonus: int, final_score: int
    ) -> None:
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
            final_score=final_score,
            final_bonus=bonus,
            start_state_hash=self._start_state_hash,
        )

    def _validate_checkpoint(self, state_bytes) -> bool:
        """Check if a save state produces a playable game.

        WORKAROUND for crayon save/restore bug. Loads the state, runs 20
        noop frames, checks if bonus changes. Restores original state.
        """
        original = self.base._interface.save_state()
        self.base._interface.load_state(state_bytes)
        bonus_start = self._read_bonus()
        changed = False
        for _ in range(20):
            self.base.step([0, 0, 0])
            if self._read_bonus() != bonus_start:
                changed = True
                break
        self.base._interface.load_state(original)
        return changed


class CurriculumCallback(BaseCallback):
    """Log curriculum progress during training."""

    def __init__(self, total_timesteps: int, log_interval: int = 5000):
        super().__init__()
        self._total = total_timesteps
        self._log_interval = log_interval
        self._last_log = 0
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        _set_global_step(self.num_timesteps)

        if self.num_timesteps - self._last_log >= self._log_interval:
            elapsed = time.monotonic() - self._start
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            pct = 100 * self.num_timesteps / self._total

            infos = self.locals.get("infos", [])
            rewards = []
            for info in infos:
                ep = info.get("episode")
                if ep:
                    rewards.append(ep["r"])
            reward_str = f"{np.mean(rewards):.1f}" if rewards else "N/A"

            assert _manager is not None
            print(
                f"step {self.num_timesteps}/{self._total} ({pct:.0f}%) "
                f"| reward={reward_str} "
                f"| emu_fps={fps * 4:.0f} "
                f"| {_manager.summary()}",
                flush=True,
            )
            self._last_log = self.num_timesteps
        return True


def train(cfg: RunConfig, config_path: Optional[str] = None) -> None:
    global _manager

    if cfg.curriculum is None:
        raise ValueError(
            "train_checkpoint_curriculum.py requires a 'curriculum' section in the run config"
        )

    seed = seed_everything(cfg.training.seed)

    _manager = CheckpointManager(
        max_states_per_checkpoint=cfg.curriculum.max_states_per_checkpoint,
        min_states_to_advance=cfg.curriculum.min_states_to_advance,
        reset_fraction=cfg.curriculum.reset_fraction,
        frontier_fraction=cfg.curriculum.frontier_fraction,
        earlier_fraction=cfg.curriculum.earlier_fraction,
    )

    print("Checkpoint Curriculum Training", flush=True)
    print(f"  Profile: {cfg.env.profile}", flush=True)
    print(f"  Timesteps: {cfg.training.timesteps}", flush=True)
    print(f"  Output: {cfg.training.output}", flush=True)
    print(f"  Seed: {seed}", flush=True)

    if cfg.curriculum.seed_archive:
        import pickle

        print(f"  Seeding from {cfg.curriculum.seed_archive}", flush=True)
        with open(cfg.curriculum.seed_archive, "rb") as f:
            archive = pickle.load(f)
        for cell_key, info in archive.items():
            fruits_collected = 4 - cell_key[2]
            if fruits_collected > 0:
                _manager.save_checkpoint(fruits_collected, info["state"])
        print(f"  Seeded: {_manager.summary()}", flush=True)

    os.makedirs(cfg.training.output, exist_ok=True)

    reward_fn = create_reward(cfg.reward.name, cfg.reward.params)

    # Persist full, resolved config.
    manifest_extras = cfg.to_dict()
    manifest_extras["resolved_seed"] = seed
    manifest_extras["script"] = "scripts/train_checkpoint_curriculum.py"
    manifest = RunManifest.capture(
        {"config_path": config_path},
        cfg.training.output,
        extras=manifest_extras,
    )
    episode_logger = EpisodeLogger(cfg.training.output)

    from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv

    def make_env(rank: int):
        def _init():
            env = CheckpointCurriculumEnv(
                cfg=cfg,
                reward_fn=reward_fn,
                env_id=rank,
                episode_logger=episode_logger,
            )
            return Monitor(env)

        return _init

    num_envs = cfg.training.num_envs
    vec_env = ThreadedVecEnv([make_env(i) for i in range(num_envs)])
    print(f"  Envs: {num_envs} threaded", flush=True)

    n_steps = cfg.ppo.n_steps
    if n_steps is None:
        n_steps = max(1, 128 // num_envs)

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=cfg.ppo.learning_rate,
        batch_size=cfg.ppo.batch_size,
        n_steps=n_steps,
        n_epochs=cfg.ppo.n_epochs,
        ent_coef=cfg.ppo.ent_coef,
        clip_range=cfg.ppo.clip_range,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        verbose=0,
        tensorboard_log=os.path.join(cfg.training.output, "tb"),
        device="auto",
        seed=seed,
    )

    if cfg.training.resume:
        print(f"  Resuming from {cfg.training.resume}", flush=True)
        model = PPO.load(
            cfg.training.resume,
            env=vec_env,
            tensorboard_log=os.path.join(cfg.training.output, "tb"),
        )
        ckpt_path = os.path.join(
            os.path.dirname(cfg.training.resume), "checkpoints.pkl"
        )
        _manager.load_from_disk(ckpt_path)
        _manager.load_from_disk(os.path.join(cfg.training.output, "checkpoints.pkl"))

    print("\nTraining...", flush=True)
    status = "COMPLETED"
    exit_code: Optional[int] = 0
    try:
        model.learn(
            total_timesteps=cfg.training.timesteps,
            callback=CurriculumCallback(cfg.training.timesteps),
        )
        model.save(os.path.join(cfg.training.output, "final_model"))
        _manager.save_to_disk(os.path.join(cfg.training.output, "checkpoints.pkl"))
        print(f"\nSaved model to {cfg.training.output}/final_model.zip", flush=True)
        print(f"Final: {_manager.summary()}", flush=True)
    except Exception:
        status = "FAILED"
        exit_code = 1
        raise
    finally:
        episode_logger.close()
        manifest.finalize(status=status, exit_code=exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        required=True,
        help="Path to run-config YAML (must include a 'curriculum' section).",
    )
    args = parser.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    train(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
