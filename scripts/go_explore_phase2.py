#!/usr/bin/env python3
"""Go-Explore Phase 2: Backward curriculum training.

Starts training from near-goal states and progressively moves the
starting point backward toward the game start. Uses save states from
Phase 1's archive.

Configuration is YAML-driven — pass ``--config`` pointing at a file
with ``training``, ``env``, ``ppo``, ``reward``, and
``backward_curriculum`` sections.

Example::

    python scripts/go_explore_phase2.py \\
        --config experiments/003-yeti/configs/go_explore_phase2_v4.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import random
import threading
import time
from typing import Optional

import gymnasium as gym
import numpy as np
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.rewards import RewardContext, RewardFn
from retro_ai.training.rewards import create as create_reward
from retro_ai.training.run_config import RunConfig
from retro_ai.training.run_manifest import (
    EpisodeLogger,
    RunManifest,
    seed_everything,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Yeti RAM addresses
FRUITS_ADDR = 11055
LIVES_ADDR = 11095
BONUS_HI = 11010
BONUS_LO = 11011
SCORE_HI = 11093
SCORE_LO = 11094
X_POS = 11090
Y_POS = 11089


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


class BackwardCurriculumEnv(gym.Env):
    """Env that starts from progressively earlier save states.

    Stage 0: start from the highest-floor states (near princess)
    Stage N: start from lower floors
    Last stage: game start

    Advances to the next stage when the agent achieves a target reward
    threshold consistently.
    """

    metadata = {"render_modes": []}

    # Class-level curriculum state, shared across all env instances.
    _global_stage = 0
    _global_rewards: list[float] = []
    _global_episode_count = 0
    _advance_threshold = 20.0
    _advance_window = 100
    _frontier_ratio = 0.5

    def __init__(
        self,
        cfg: RunConfig,
        archive: dict,
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

        self.archive = archive
        self.env_id = env_id
        self.episode_logger = episode_logger

        # Group archive cells by floor (descending — stage 0 = highest floor).
        self.cells_by_floor: dict[int, list[dict]] = {}
        for cell_key, info in archive.items():
            floor = cell_key[0]
            self.cells_by_floor.setdefault(floor, []).append(info)
        self.floors_desc = sorted(self.cells_by_floor.keys(), reverse=True)

        # Per-episode state
        self._step_count = 0
        self._prev_score = 0
        self._prev_bonus = 0
        self._prev_lives = 5
        self._prev_fruits = 4
        self._stall = 0
        self._initialized = False

        # Logging state
        self._start_xy = (0, 0)
        self._start_fruits = 4
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

    @property
    def current_stage(self):
        return BackwardCurriculumEnv._global_stage

    @property
    def num_stages(self):
        return len(self.floors_desc) + 1

    def _get_stage_floor(self):
        if self.current_stage < len(self.floors_desc):
            return self.floors_desc[self.current_stage]
        return None

    def _pick_starting_floor(self):
        cls = BackwardCurriculumEnv
        if self.current_stage == 0:
            return self.floors_desc[0]
        if self.current_stage >= len(self.floors_desc):
            if random.random() < cls._frontier_ratio:
                return None
            return random.choice(self.floors_desc)
        if random.random() < cls._frontier_ratio:
            return self.floors_desc[self.current_stage]
        earlier_idx = random.randint(0, self.current_stage - 1)
        return self.floors_desc[earlier_idx]

    # -- gym API -------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        floor = self._pick_starting_floor()
        if floor is not None and floor in self.cells_by_floor:
            cells = self.cells_by_floor[floor]
            info = random.choice(cells)
            self.base._interface.load_state(info["state"])
            obs, _, _, _, _ = self.gym_env.step([0, 0, 0])
            self._start_state_hash = hashlib.blake2b(
                info["state"], digest_size=8
            ).hexdigest()
        else:
            obs, _ = self.gym_env.reset()
            self._start_state_hash = ""

        self._step_count = 0
        self._prev_score = self._read_score()
        self._start_score = self._prev_score
        self._prev_bonus = self._read_bonus()
        self._start_bonus = self._prev_bonus
        self._prev_lives = self.iface.read_ram_byte(LIVES_ADDR)
        self._prev_fruits = self.iface.read_ram_byte(FRUITS_ADDR)
        self._stall = 0

        self._start_xy = self._read_pos()
        self._start_fruits = self._prev_fruits
        self._episode_reward = 0.0
        self._fruits_collected_this_ep = 0
        self._episode_id = _next_episode_id()

        return obs, {}

    def step(self, action):
        obs, _, done, truncated, info = self.gym_env.step(action)
        self._step_count += 1

        score = self._read_score()
        lives = self.iface.read_ram_byte(LIVES_ADDR)
        bonus = self._read_bonus()
        fruits = self.iface.read_ram_byte(FRUITS_ADDR)

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
        self._episode_reward += reward

        if fruits < self._start_fruits - self._fruits_collected_this_ep:
            self._fruits_collected_this_ep = self._start_fruits - fruits

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

        # Update other "prev" trackers after reward computation.
        self._prev_fruits = fruits
        self._prev_score = score

        if done or truncated:
            # For stage-advance scoring, use absolute score delta from start
            # (matches the legacy behavior).
            score_delta = max(0, score - self._start_score)
            BackwardCurriculumEnv._global_rewards.append(score_delta)
            BackwardCurriculumEnv._global_episode_count += 1
            if end_reason is None:
                end_reason = "env_done" if done else "env_truncated"
            self._log_episode(end_reason, fruits, bonus, score)
            self._check_advance()

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

    def _check_advance(self):
        cls = BackwardCurriculumEnv
        if len(cls._global_rewards) < cls._advance_window:
            return
        recent = cls._global_rewards[-cls._advance_window :]
        mean_reward = float(np.mean(recent))
        if (
            mean_reward >= cls._advance_threshold
            and cls._global_stage < self.num_stages - 1
        ):
            old_floor = self._get_stage_floor()
            cls._global_stage += 1
            new_floor = self._get_stage_floor()
            cls._global_rewards = []
            cls._global_episode_count = 0
            print(
                f"  STAGE ADVANCE: {cls._global_stage}/{self.num_stages} "
                f"(floor {old_floor} -> {new_floor}, mean_reward={mean_reward:.1f})",
                flush=True,
            )


def train(cfg: RunConfig, config_path: Optional[str] = None) -> None:
    from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv

    if cfg.backward_curriculum is None:
        raise ValueError(
            "go_explore_phase2.py requires a 'backward_curriculum' "
            "section in the run config"
        )

    # Apply curriculum config to the class-level knobs.
    BackwardCurriculumEnv._advance_threshold = cfg.backward_curriculum.advance_threshold
    BackwardCurriculumEnv._advance_window = cfg.backward_curriculum.advance_window
    BackwardCurriculumEnv._frontier_ratio = cfg.backward_curriculum.frontier_ratio
    BackwardCurriculumEnv._global_stage = 0
    BackwardCurriculumEnv._global_rewards = []
    BackwardCurriculumEnv._global_episode_count = 0

    seed = seed_everything(cfg.training.seed)

    print(f"Loading archive from {cfg.backward_curriculum.archive}", flush=True)
    with open(cfg.backward_curriculum.archive, "rb") as f:
        archive = pickle.load(f)
    print(f"  {len(archive)} cells loaded", flush=True)
    floors: dict[int, int] = {}
    for cell_key in archive:
        floors[cell_key[0]] = floors.get(cell_key[0], 0) + 1
    for f in sorted(floors):
        print(f"  Floor {f}: {floors[f]} cells", flush=True)
    print(f"  Seed: {seed}", flush=True)

    os.makedirs(cfg.training.output, exist_ok=True)

    reward_fn = create_reward(cfg.reward.name, cfg.reward.params)

    manifest_extras = cfg.to_dict()
    manifest_extras["resolved_seed"] = seed
    manifest_extras["script"] = "scripts/go_explore_phase2.py"
    manifest_extras["num_archive_cells"] = len(archive)
    manifest_extras["cells_by_floor"] = floors
    manifest = RunManifest.capture(
        {"config_path": config_path},
        cfg.training.output,
        extras=manifest_extras,
    )
    episode_logger = EpisodeLogger(cfg.training.output)

    def make_env(rank: int):
        def _init():
            env = BackwardCurriculumEnv(
                cfg=cfg,
                archive=archive,
                reward_fn=reward_fn,
                env_id=rank,
                episode_logger=episode_logger,
            )
            return Monitor(env)

        return _init

    num_envs = cfg.training.num_envs
    if num_envs > 1:
        vec_env = ThreadedVecEnv([make_env(i) for i in range(num_envs)])
        print(f"  Using {num_envs} threaded envs", flush=True)
    else:
        vec_env = make_env(0)()

    class ProgressCallback(BaseCallback):
        def __init__(self, total: int, log_interval: int = 1000):
            super().__init__()
            self._total = total
            self._log_interval = log_interval
            self._last_log = 0
            self._start = time.monotonic()
            self._frame_skip = 4

        def _on_step(self) -> bool:
            _set_global_step(self.num_timesteps)
            if self.num_timesteps - self._last_log >= self._log_interval:
                elapsed = time.monotonic() - self._start
                fps = self.num_timesteps / elapsed if elapsed > 0 else 0
                emu_fps = fps * self._frame_skip
                pct = 100 * self.num_timesteps / self._total
                stage = f"{BackwardCurriculumEnv._global_stage}"
                print(
                    f"step {self.num_timesteps}/{self._total} ({pct:.0f}%) "
                    f"| emu_fps={emu_fps:.0f} | stage={stage}",
                    flush=True,
                )
                self._last_log = self.num_timesteps
            return True

    n_steps = cfg.ppo.n_steps
    if n_steps is None:
        n_steps = max(1, 128 // num_envs) if num_envs > 1 else 128

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

    print(
        f"\nBackward curriculum: {cfg.training.timesteps} steps, {num_envs} envs",
        flush=True,
    )
    status = "COMPLETED"
    exit_code: Optional[int] = 0
    try:
        model.learn(
            total_timesteps=cfg.training.timesteps,
            callback=ProgressCallback(cfg.training.timesteps),
        )
        model.save(os.path.join(cfg.training.output, "final_model"))
        print(f"\nSaved model to {cfg.training.output}/final_model.zip", flush=True)
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
        help="Path to run-config YAML (must include a 'backward_curriculum' section).",
    )
    args = parser.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    train(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
