#!/usr/bin/env python3
"""Train a fresh PPO agent on a single segment (checkpoint N -> N+1).

Loads save states from a checkpoint file and trains a new policy from
scratch. Configuration is YAML-driven — pass ``--config`` pointing at a
file with ``training``, ``env``, ``ppo``, ``reward``, and ``segment``
sections.

Example::

    python scripts/train_segment.py \\
        --config experiments/003-yeti/configs/segment_1to2_v3.yaml
"""

from __future__ import annotations

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
from retro_ai.training.callbacks import EpisodeMetricsCallback
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.rewards import RewardContext, RewardFn
from retro_ai.training.rewards import create as create_reward
from retro_ai.training.run_config import RunConfig
from retro_ai.training.run_manifest import (
    EpisodeLogger,
    RunManifest,
    iter_inner_envs,
    seed_everything,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Yeti-specific RAM addresses.
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


class SegmentEnv(gym.Env):
    """Env that always starts from a specific checkpoint level."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: RunConfig,
        checkpoint_states,
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

        self.checkpoint_states = checkpoint_states
        self.env_id = env_id
        self.episode_logger = episode_logger
        self.max_steps = cfg.env.max_steps
        self.stall_threshold = cfg.env.stall_threshold
        self._reward_fn = reward_fn

        # Per-episode state
        self._step_count = 0
        self._prev_fruits = 4
        self._prev_lives = 5
        self._prev_bonus = 0
        self._prev_score = 0
        self._stall = 0
        self._initialized = False

        # Logging state
        self._start_fruits = 4
        self._start_xy = (0, 0)
        self._start_score = 0
        self._start_bonus = 0
        self._start_state_hash = ""
        self._episode_reward = 0.0
        self._fruits_collected_this_ep = 0
        self._episode_id = 0

        # Legacy aggregates (kept for ProgressCallback readout)
        self.successes = 0
        self.episodes = 0
        self.collected_states = deque(maxlen=100)

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
        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        state = random.choice(self.checkpoint_states)
        self.base._interface.load_state(state)
        # load_state restores the emulator's RAM + framebuffer but does
        # NOT reset the preprocessing pipeline's frame_stack buffer
        # (see python/retro_ai/core/preprocessing.py). On the first step
        # after load the buffer still contains 3 pre-load frames + 1
        # post-load frame. Running 5 noop steps before returning lets
        # the frame stack refill entirely with post-load frames, so the
        # first observation the policy sees is consistent with what
        # steady-state gameplay would produce. Same rationale applies
        # in CheckpointCurriculumEnv (commit add2400 introduced the
        # value 5; any frame_stack-sized number of settle steps would
        # do).
        for _ in range(5):
            obs, _, _, _, _ = self.gym_env.step([0, 0, 0])

        self._step_count = 0
        self._prev_fruits = self.iface.read_ram_byte(FRUITS_ADDR)
        self._start_fruits = self._prev_fruits
        self._prev_lives = self.iface.read_ram_byte(LIVES_ADDR)
        self._prev_bonus = self._read_bonus()
        self._start_bonus = self._prev_bonus
        self._prev_score = self._read_score()
        self._start_score = self._prev_score
        self._start_xy = self._read_pos()
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
        score = self._read_score()

        # Named reward formula — exact behavior controlled by run config.
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

        # Side effects we still track ourselves (not part of reward)
        if fruits < self._prev_fruits:
            collected = self._prev_fruits - fruits
            self.successes += collected
            self._fruits_collected_this_ep += collected
            self.collected_states.append(self.iface.save_state())

        self._prev_fruits = fruits
        self._prev_score = score
        self._episode_reward += reward

        # Episode termination
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


def train(cfg: RunConfig, config_path: Optional[str] = None) -> None:
    if cfg.segment is None:
        raise ValueError(
            "train_segment.py requires a 'segment' section in the run config"
        )

    seed = seed_everything(cfg.training.seed)
    print(
        f"Segment Training: checkpoint {cfg.segment.segment} "
        f"-> {cfg.segment.segment + 1}"
    )
    print(f"  Seed: {seed}", flush=True)

    with open(cfg.segment.checkpoints, "rb") as f:
        data = pickle.load(f)
    states = data["checkpoints"][cfg.segment.segment]
    print(f"  {len(states)} starting states", flush=True)
    if not states:
        print("  No states available!", flush=True)
        return

    os.makedirs(cfg.training.output, exist_ok=True)

    reward_fn = create_reward(cfg.reward.name, cfg.reward.params)

    # Persist full, resolved config via the run manifest.
    manifest_extras = cfg.to_dict()
    manifest_extras["resolved_seed"] = seed
    manifest_extras["script"] = "scripts/train_segment.py"
    manifest_extras["num_checkpoint_states"] = len(states)
    manifest = RunManifest.capture(
        {"config_path": config_path},
        cfg.training.output,
        extras=manifest_extras,
    )
    episode_logger = EpisodeLogger(cfg.training.output)

    from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv

    def make_env(rank: int):
        def _init():
            env = SegmentEnv(
                cfg=cfg,
                checkpoint_states=states,
                reward_fn=reward_fn,
                env_id=rank,
                episode_logger=episode_logger,
            )
            return Monitor(env)

        return _init

    num_envs = cfg.training.num_envs
    vec_env = ThreadedVecEnv([make_env(i) for i in range(num_envs)])

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

    print(f"  {num_envs} envs, {cfg.training.timesteps} steps", flush=True)
    status = "COMPLETED"
    exit_code: Optional[int] = 0
    try:
        model.learn(
            total_timesteps=cfg.training.timesteps,
            callback=[
                ProgressCallback(cfg.training.timesteps),
                EpisodeMetricsCallback(episode_logger, log_interval=10_000),
            ],
        )
        model.save(os.path.join(cfg.training.output, "final_model"))

        # Save collected states (for chaining into next segment)
        all_states = []
        for inner in iter_inner_envs(vec_env):
            if hasattr(inner, "collected_states"):
                all_states.extend(inner.collected_states)
        if all_states:
            pkl_path = os.path.join(cfg.training.output, "collected_states.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {"states": all_states, "segment": cfg.segment.segment + 1}, f
                )
            print(f"\nSaved {len(all_states)} collected states to {pkl_path}")
        else:
            print("\nNo states collected (agent never reached next checkpoint)")

        print(f"\nSaved to {cfg.training.output}/final_model.zip")
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
        help="Path to run-config YAML (must include a 'segment' section).",
    )
    args = parser.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    train(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
